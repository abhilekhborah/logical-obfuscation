# This script is structured for OpenAI GPT models, but you can adapt it for other LLM APIs by replacing the relevant API client.
import os
import sys
import pandas as pd
from openai import OpenAI
import time
import csv


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

try:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error configuring OpenAI client: {e}")
    sys.exit(1)

OPENAI_MODEL_NAME = "o4-mini"

TARGET_QUESTION_COLS = ['BaseQuestion', 'ObfuscatedQuest_l1', 'ObfuscatedQuest_l2']


def build_zeroshot_prompt(question: str) -> str:
    return (
        "Respond with only the answer (one word/two word etc.). Do not output any additional information and explanation.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_fewshot_prompt(question: str) -> str:
    return (
        "Here are some examples:\n\n"
        "Example 1:\n"
        "Question: Rama tells her friend, \"The girl is the daughter of the brother of my husband's mother.\" How is Rama related to that girl?\n"
        "Answer: Aunt\n\n"
        "Example 2:\n"
        "Question: Priya says about a man, \"He is the son of my father's only daughter.\" What is the relationship between Priya and that man?\n"
        "Answer: Brother\n\n"
        "Example 3:\n"
        "Question: X is the sister of Y. Y is the father of Z. W is the mother of Y. How is X related to W?\n"
        "Answer: Daughter-in-law\n\n"
        "I have given you examples of some relationship reasoning problems above. Now you are given a new problem:\n"
        f"Question: {question}\n"
        "Please determine the answer and respond with only the answer (one word/two word etc.) without any additional information and explanation.\n"
        "Answer:"
    )


def build_cot_prompt(question: str) -> str:
    return (
        "You are an expert in solving complex family and relationship reasoning problems.\n\n"
        "Think step by step to determine the correct relationship.\n"
        "Use internal reasoning to:\n"
        "- Identify all people and their roles in the question.\n"
        "- Trace the relationships step-by-step.\n"
        "- Determine how the speaker is related to the person in question.\n\n"
        "Only output the final answer (one or two words) without any other additional information or explanation or reasoning.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def call_gpt_api(user_prompt_text: str) -> str:
    retries = 0
    max_retries = 5
    retry_delay = 60

    while retries < max_retries:
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": user_prompt_text}],
            )

            if response.choices and len(response.choices) > 0:
                text = response.choices[0].message.content
            else:
                text = ""

            cleaned_text = text.strip() if text else ""

            if not cleaned_text:
                retries += 1
                if retries >= max_retries:
                    return "ERROR_EMPTY_RESPONSE"
                time.sleep(retry_delay / 2)
                continue
            else:
                return cleaned_text.upper()

        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower() or "overloaded" in error_message.lower() or "timed out" in error_message.lower() or "connection error" in error_message.lower():
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    return "ERROR_API"
            elif isinstance(e, KeyboardInterrupt):
                sys.exit(1)
            else:
                return "ERROR_API"

    return "ERROR_API"


def main(input_csv: str, output_prefix: str):
    try:
        df_master = pd.read_csv(input_csv)
        print(f"Successfully loaded input CSV: {input_csv}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV '{input_csv}': {e}")
        sys.exit(1)

    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            sys.exit(1)

    for question_col_name in TARGET_QUESTION_COLS:
        if question_col_name not in df_master.columns:
            print(f"Warning: Column '{question_col_name}' not found in the input CSV. Skipping.")
            continue

        output_csv = f"{output_prefix}_{question_col_name}.csv"

        zeroshot_col_name = f'{OPENAI_MODEL_NAME}_zeroshot'
        fewshot_col_name = f'{OPENAI_MODEL_NAME}_fewshot'
        cot_col_name = f'{OPENAI_MODEL_NAME}_cot'

        original_cols_to_keep = [c for c in df_master.columns if c not in TARGET_QUESTION_COLS]
        output_cols_header = original_cols_to_keep + [
            question_col_name,
            zeroshot_col_name,
            fewshot_col_name,
            cot_col_name,
        ]

        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(output_cols_header)
        except Exception as e:
            print(f"Error writing header to '{output_csv}': {e}")
            continue

        total_rows = len(df_master)

        for idx, row in df_master.iterrows():
            question_text = str(row[question_col_name]) if pd.notna(row[question_col_name]) else ""

            row_data = {col: row[col] for col in original_cols_to_keep}
            row_data[question_col_name] = question_text

            if not question_text.strip():
                row_data[zeroshot_col_name] = "SKIPPED_EMPTY"
                row_data[fewshot_col_name] = "SKIPPED_EMPTY"
                row_data[cot_col_name] = "SKIPPED_EMPTY"
            else:
                zeroshot_result = call_gpt_api(build_zeroshot_prompt(question_text))
                row_data[zeroshot_col_name] = zeroshot_result
                time.sleep(1)

                fewshot_result = call_gpt_api(build_fewshot_prompt(question_text))
                row_data[fewshot_col_name] = fewshot_result
                time.sleep(1)

                cot_result = call_gpt_api(build_cot_prompt(question_text))
                row_data[cot_col_name] = cot_result

            try:
                row_values_to_write = [row_data.get(col_name, "") for col_name in output_cols_header]
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_values_to_write)
            except Exception as e:
                print(f"Error saving row {idx+1} to CSV for {question_col_name}: {e}")

        print(f"Finished processing column: {question_col_name}")
        print(f"Incremental results saved to {output_csv}")

    print("All specified columns processed.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_path> <output_file_prefix>")
        print(f"Example: python {os.path.basename(__file__)} data/my_data.csv results/experiment1_gpt")
        sys.exit(1)

    input_file_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]

    if not OPENAI_API_KEY:
        print("\nWARNING: OPENAI_API_KEY is not set.")
        print("Script will likely fail API calls.")
        print("Please set the OPENAI_API_KEY environment variable.")

    main(input_file_arg, output_prefix_arg)