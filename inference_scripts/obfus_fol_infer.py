# This script is structured for OpenAI GPT models, but you can adapt it for other LLM APIs by replacing the relevant API client.
import os
import sys
import pandas as pd
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    print("Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-4o"

OBF_PREMISE_COL = 'obf_premises_nl'
OBF_CONCLUSION_COL = 'obf_conclusion_nl'

BASE_PREMISE_COL = 'premises_nl'
BASE_CONCLUSION_INPUT_COL = 'conclusion_nl'


def build_zeroshot_prompt(premise: str, conclusion: str) -> str:
    return (
        "Respond with exactly one word without any additional information: TRUE or FALSE.\n\n"
        f"Premise: {premise}\n"
        f"Conclusion: {conclusion}\n"
        "Answer:"
    )


def build_fewshot_prompt(premise: str, conclusion: str) -> str:
    return (
        "Here are two examples of logical reasoning problems:\n\n"
        "Example 1:\n"
        "Premise: All kids are young. All toddlers are kids. If someone is young, then they are not elderly. All pirates are seafarers. If Nancy is not a pirate, then Nancy is young. If Nancy is not a toddler, then Nancy is a seafarer.\n"
        "Conclusion: Nancy is either both a pirate and a toddler, or neither a pirate nor a toddler.\n"
        "Answer: FALSE\n\n"
        "Example 2:\n"
        "Premise: If a person is the leader of a country for life, that person has power. Leaders of a country for life are either a king or a queen. Queens are female. Kings are male. Elizabeth is a queen. Elizabeth is a leader of a country for life.\n"
        "Conclusion: Elizabeth has power.\n"
        "Answer: TRUE\n\n"
        "Now, based on these examples, evaluate the following:\n"
        f"Premise: {premise}\n"
        f"Conclusion: {conclusion}\n"
        "Respond with exactly one word without any additional information: TRUE or FALSE.\n"
        "Answer:"
    )


def build_cot_prompt(premise: str, conclusion: str) -> str:
    return (
        "You are an expert logical reasoning solver. Carefully evaluate the following premises and conclusion. "
        "Follow these steps to reason through the problem:\n\n"
        "1. Break Down Premises: Identify predicates, quantifiers (∀, ∃), and connectives (→, ∧, ∨, ¬) in each premise.\n"
        "2. Simplify Obfuscated Logic: If the logic is complex, apply transformations to clarify it. Use methods like:\n"
        "   - Contraposition: \"If P then Q\" → \"If ¬Q then ¬P\"\n"
        "   - Double Negation: \"¬¬P\" → \"P\"\n"
        "   - De Morgan’s Laws: \"¬(P ∧ Q)\" → \"¬P ∨ ¬Q\"; \"¬(P ∨ Q)\" → \"¬P ∧ ¬Q\"\n"
        "   - Conditional to Disjunction: \"P → Q\" → \"¬P ∨ Q\"\n"
        "   - Quantifier Negation: \"∀x P(x)\" → \"¬∃x ¬P(x)\"; \"∃x P(x)\" → \"¬∀x ¬P(x)\"\n"
        "   - Biconditional Expansion: \"P ↔ Q\" → \"(P → Q) ∧ (Q → P)\"\n"
        "   - Nested Transformations: Combine multiple equivalences (e.g., \"¬(P → Q)\" → \"P ∧ ¬Q\" via negation normal form).\n"
        "3. Identify Relationships: Determine how premises connect (implications, equivalences, contradictions) and test their support for the conclusion.\n"
        "4. Evaluate Conclusion: Check if the conclusion necessarily follows. Look for gaps or counterexamples.\n"
        "5. Answer: Respond TRUE if the conclusion follows, FALSE otherwise.\n\n"
        "Respond with exactly one word, without any additional information: TRUE or FALSE.\n\n"
        f"Premise: {premise}\n"
        f"Conclusion: {conclusion}\n"
        "Answer:"
    )


def call_openai_api(prompt_text: str, model: str = MODEL_NAME) -> str:
    try:
        response = client.responses.create(  # type: ignore
            model=model,
            input=prompt_text
        )

        text = None
        if hasattr(response, "output_text"):
            text = getattr(response, "output_text", None)

        if text is None and hasattr(response, 'choices') and response.choices:  # type: ignore
            choice = response.choices[0]  # type: ignore
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                text = choice.message.content
            elif hasattr(choice, 'text'):
                text = choice.text

        if text:
            answer = text.strip().split()[0].upper()
            return answer if answer in ("TRUE", "FALSE") else "FALSE"
        else:
            print(f"Warning: API response format not recognized or content is empty. Response object: {response}")
            return "ERROR_RESPONSE_FORMAT"

    except Exception as e:
        print(f"API error: {e}")
        return "ERROR_API"


def main(input_csv: str, output_csv: str, use_base_version: bool):
    if use_base_version:
        premise_col_name = BASE_PREMISE_COL
        conclusion_input_col_name = BASE_CONCLUSION_INPUT_COL
        print(f"INFO: Using BASE columns for processing: Premise='{premise_col_name}', Conclusion Input='{conclusion_input_col_name}'")
    else:
        premise_col_name = OBF_PREMISE_COL
        conclusion_input_col_name = OBF_CONCLUSION_COL
        print(f"INFO: Using OBFUSCATED columns for processing: Premise='{premise_col_name}', Conclusion Input='{conclusion_input_col_name}'")

    try:
        df_input = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV '{input_csv}': {e}")
        sys.exit(1)

    if premise_col_name not in df_input.columns or conclusion_input_col_name not in df_input.columns:
        print(f"Error: CSV must contain the required columns: '{premise_col_name}' and '{conclusion_input_col_name}'.")
        print(f"Available columns: {df_input.columns.tolist()}")
        sys.exit(1)

    zeroshot_col_name = f'{MODEL_NAME}_zeroshot'
    fewshot_col_name = f'{MODEL_NAME}_fewshot'
    cot_col_name = f'{MODEL_NAME}_cot'

    output_columns = df_input.columns.tolist() + [zeroshot_col_name, fewshot_col_name, cot_col_name]

    try:
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False, header=True, mode='w')
        print(f"Initialized output CSV '{output_csv}' with headers.")
    except Exception as e:
        print(f"Error initializing output CSV '{output_csv}': {e}")
        sys.exit(1)

    total_rows = len(df_input)
    print(f"Processing {total_rows} rows using model '{MODEL_NAME}'...")

    for idx, row_series in df_input.iterrows():
        premise_text = str(row_series[premise_col_name])
        conclusion_input_string = str(row_series[conclusion_input_col_name])

        print(f"\nProcessing Row {idx+1}/{total_rows}:")
        print(f"  Using Premise ({premise_col_name}): {premise_text[:100]}..." if len(premise_text) > 100 else f"  Using Premise ({premise_col_name}): {premise_text}")
        print(f"  Using Conclusion Input ({conclusion_input_col_name}): {conclusion_input_string[:100]}..." if len(conclusion_input_string) > 100 else f"  Using Conclusion Input ({conclusion_input_col_name}): {conclusion_input_string}")

        zeroshot_prompt = build_zeroshot_prompt(premise_text, conclusion_input_string)
        zeroshot_verdict = call_openai_api(zeroshot_prompt)
        print(f"  Zero-shot Result: {zeroshot_verdict}")

        fewshot_prompt = build_fewshot_prompt(premise_text, conclusion_input_string)
        fewshot_verdict = call_openai_api(fewshot_prompt)
        print(f"  Few-shot Result: {fewshot_verdict}")

        cot_prompt = build_cot_prompt(premise_text, conclusion_input_string)
        cot_verdict = call_openai_api(cot_prompt)
        print(f"  CoT Result: {cot_verdict}")

        output_row_data = row_series.to_dict()
        output_row_data[zeroshot_col_name] = zeroshot_verdict
        output_row_data[fewshot_col_name] = fewshot_verdict
        output_row_data[cot_col_name] = cot_verdict

        df_output_row = pd.DataFrame([output_row_data], columns=output_columns)
        try:
            df_output_row.to_csv(output_csv, mode='a', header=False, index=False)
            print(f"  Row {idx+1}/{total_rows} saved to {output_csv}")
        except Exception as e:
            print(f"Error appending row {idx+1} to output CSV '{output_csv}': {e}")

    print(f"\nAll rows processed. Results incrementally saved to {output_csv}")


if __name__ == '__main__':
    use_base_version_flag = False
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python gpt4_combined.py input.csv output.csv [base]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if len(sys.argv) == 4:
        if sys.argv[3].lower() == 'base':
            use_base_version_flag = True
        else:
            print("Error: Invalid fourth argument. If provided, it must be 'base'.")
            print("Usage: python gpt4_combined.py input.csv output.csv [base]")
            sys.exit(1)

    main(input_file, output_file, use_base_version_flag)