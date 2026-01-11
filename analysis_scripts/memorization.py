import os
import argparse
import re
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
login(token="your_hf_key")

# ----- Argument Parsing -----
parser = argparse.ArgumentParser(
    description="Run Min-K%++ attack on a CSV dataset with exact match accuracy"
)
parser.add_argument(
    '--csv_path',
    type=str,
    required=True,
    help='Path to CSV file with columns: base_question, type1, type2, type3, ground_truth'
)
parser.add_argument(
    '--model',
    type=str,
    default="meta-llama/Llama-3.1-8B",
    help='Model name from HuggingFace Hub'
)
parser.add_argument('--half', action='store_true', help='Use half-precision')
parser.add_argument('--int8', action='store_true', help='Use 8-bit quantization')
parser.add_argument(
    '--output_dir',
    type=str,
    default="results",
    help='Directory to save results and plots'
)
args = parser.parse_args()

# ----- Helper Functions -----
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def parse_ground_truth(gt_str):
    if isinstance(gt_str, float):
        return set()
    return {normalize_text(ans) for ans in str(gt_str).split(',')}

def load_model(name):
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        name,
        return_dict=True,
        device_map='auto',
        **int8_kwargs, **half_kwargs
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ----- Initialize Counters -----
correct_base = 0
correct_type1 = 0
correct_type2 = 0
correct_type3 = 0
question_columns = ["Base Question", "Named-Entity Indirection", 
                   "Distractor Indirection", "Contextual Overload"]

# ----- Load Model and Data -----
model, tokenizer = load_model(args.model)
data_df = pd.read_csv(args.csv_path)
total_samples = len(data_df)

# Prepare results container for Min-K%++
results = {col: defaultdict(lambda: {'scores': [], 'labels': []}) for col in question_columns}

# ----- Process Each Sample -----
for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing Samples"):
    gt_set = parse_ground_truth(row['ground_truth'])
    
    for col in question_columns:
        question = row[col] if isinstance(row[col], str) else " "
        question = "You are an assistant that answers only with the objective answer. Do not include any additional information." + question
        
        # Tokenize and process
        encoding = tokenizer(question, return_tensors="pt", padding=True)
        input_ids = encoding['input_ids'].to(model.device)
        attention_mask = encoding['attention_mask'].to(model.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Generate answer
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False
            )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer = generated_text[len(question):].strip()
        norm_answer = normalize_text(answer)
        
        # Exact match check
        exact_match = any(gt in norm_answer for gt in gt_set) if gt_set else 0
        sample_label = 1 if exact_match else 0
        
        # Update accuracy counters
        if col == "Base Question":
            correct_base += exact_match
        elif col == "Named-Entity Indirection":
            correct_type1 += exact_match
        elif col == "Distractor Indirection":
            correct_type2 += exact_match
        elif col == "Contextual Overload":
            correct_type3 += exact_match

        # Compute Min-K%++ scores
        if input_ids.shape[1] >= 2:
            shifted_input_ids = input_ids[0][1:].unsqueeze(-1)
            logits = outputs.logits[0, :-1]
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
            sigma = sigma.sqrt() + 1e-8
            
            token_log_probs = log_probs.gather(dim=-1, index=shifted_input_ids).squeeze(-1)
            mink_plus = (token_log_probs.cpu().numpy() - mu.cpu().numpy()) / sigma.cpu().numpy()
            
            for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                k = max(1, int(len(mink_plus) * ratio))
                score = np.mean(np.sort(mink_plus)[:k])
                method = f"Min-K%++_{ratio}"
                results[col][method]['scores'].append(score)
                results[col][method]['labels'].append(sample_label)

# ----- Calculate Exact Match Accuracy -----
accuracies = {
    "Base Question": correct_base / total_samples,
    "Named-Entity Indirection": correct_type1 / total_samples,
    "Distractor Indirection": correct_type2 / total_samples,
    "Contextual Overload": correct_type3 / total_samples
}
print("\nExact Match Accuracy:")
for col, acc in accuracies.items():
    print(f"{col}: {acc:.2%}")

# ----- Plot ROC Curves -----
os.makedirs(args.output_dir, exist_ok=True)

for col in question_columns:
    plt.figure(figsize=(10, 8))
    for method, data in results[col].items():
        if not data['scores']:
            continue
            
        fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f"{method} (AUC = {roc_auc:.2f})")
    
    plt.title(f'ROC Curve - {col}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f"{col}_roc_curve.png"))
    plt.close()

print("\nPlots saved to:", args.output_dir)

# ----- Save ROC AUC Scores as CSV -----
aurocs = []
for col in question_columns:
    for method, data in results[col].items():
        if not data['scores']:
            continue
        fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        roc_auc = auc(fpr, tpr)
        # Extract the ratio value from the method name, e.g., "Min-K%++_0.1" -> "0.1"
        ratio = method.split("_")[-1]
        aurocs.append({
            "Question Column": col,
            "Method": method,
            "Ratio": ratio,
            "ROC AUC": roc_auc
        })

aurocs_df = pd.DataFrame(aurocs)
csv_output_path = os.path.join(args.output_dir, "minkpp_aurocs.csv")
aurocs_df.to_csv(csv_output_path, index=False)
print("\nROC AUC CSV saved to:", csv_output_path)
