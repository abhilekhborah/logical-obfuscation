# This script is structured for OpenAI GPT models, but you can adapt it for other LLM APIs by replacing the relevant API client.
import os
import sys
import pandas as pd
from openai import OpenAI
import time
import re


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

try:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error configuring OpenAI client: {e}")
    sys.exit(1)

OPENAI_MODEL_NAME = "gpt-5"

BASE_SEQUENCE_COL = 'number_series'
OBFUSCATED_SEQUENCE_COL = 'obfuscation'
GROUND_TRUTH_COL = 'answer'


def call_gpt_and_extract_number(prompt_text: str) -> str:
    if not OPENAI_CLIENT:
        return "ERROR_CLIENT_NOT_INITIALIZED"

    retries = 0
    max_retries = 3
    retry_delay = 20

    while retries < max_retries:
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}],
            )

            if response.choices and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            else:
                response_content = ""

            match = re.search(r"(?:[Oo]utput|[Nn]ext term(?: is)?|[Aa]nswer(?: is)?):\s*(-?\b\d+\b)", response_content)
            if match:
                return match.group(1)

            potential_numbers = re.findall(r"-?\b\d+\b", response_content)
            if potential_numbers:
                return potential_numbers[-1]

            cleaned_response = response_content.strip()
            if re.fullmatch(r"-?\d+", cleaned_response):
                return cleaned_response

            if not cleaned_response:
                retries += 1
                time.sleep(retry_delay)
                continue

            return f"NO_NUM_FOUND:{cleaned_response[:50]}"

        except Exception as e:
            error_message = str(e)
            is_rate_limit = "rate limit" in error_message.lower() or (hasattr(e, 'status_code') and e.status_code == 429)
            is_server_error = "server error" in error_message.lower() or (hasattr(e, 'status_code') and e.status_code >= 500)
            is_overloaded = "overloaded" in error_message.lower()
            is_timeout = "timed out" in error_message.lower()
            is_connection_error = "connection error" in error_message.lower()

            if is_rate_limit or is_server_error or is_overloaded or is_timeout or is_connection_error:
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    return "ERROR_API_MAX_RETRIES"
            elif isinstance(e, KeyboardInterrupt):
                sys.exit(1)
            else:
                return f"ERROR_API_GENERAL:{error_message[:50]}"

    return "ERROR_MAX_RETRIES_EXCEEDED"


def type1_zero_shot(sequence_string: str) -> str:
    prompt = f"""You're given a sequence that mixes integers and planet names. The planet names follow a one-to-one code between the digits 0-9 and the Sun plus the nine classical planets (Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto in that order, starting with Sun as 0).
Your task is to decode the sequence, find the arithmetic rule, and predict the next term.
Sequence:
{sequence_string}
Ouput:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type1_few_shot(sequence_string: str) -> str:
    prompt = f"""We have a mixed numeric-planetary series. Digits 0-9 encode the Sun or planets in order, starting with Sun as 0. Below are examples. For the 'Now:' section, provide ONLY the numerical output.
Example 1
Input: 7, 10, 8, 11, 9, Mercury Venus,
Output: 10
(Reason: Assuming Sun=0, Mercury=1, Venus=2 etc., Mercury Venus is 12. Sequence: 7, 10, 8, 11, 9, 12. Rule: +3, -2, +3, -2, +3. Next: 12-2=10.)
Example 2
Input: Earth Saturn, 34, 30, 28, 24,
Output: 22
(Reason: Assuming Sun=0, Earth=3, Saturn=6 etc., Earth Saturn is 36. Sequence: 36, 34, 30, 28, 24. Rule: -2, -4, -2, -4. Next: 24-2=22.)
Now:
Input: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type1_cot(sequence_string: str) -> str:
    prompt = f"""You have a mixed sequence of integers and planet names. Each digit 0-9 indexes the Sun and planets in order, with Sun representing 0. Think step-by-step to find the next term. Provide ONLY the final numerical answer for the new sequence.
Worked Mini-Example
Sequence: Earth Saturn, 34, 30, 28, 24,
1. Translate planets based on an order starting Sun=0: Earth -> 3, Saturn -> 6. So, "Earth Saturn" is 36. Sequence: 36, 34, 30, 28, 24.
2. Infer rule: 36->34 (-2), 34->30 (-4), 30->28 (-2), 28->24 (-4). Rule: alternate -2, -4.
3. Validate: Checks pass.
4. Compute next: Last is 24. Next op is -2. So, 24 - 2 = 22. Next term: 22
Now, apply this to:
Sequence: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type2_zero_shot(sequence_string: str) -> str:
    prompt = f"""You're given a sequence mixing:
1. Planet names (Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, which correspond to digits 0-9 in order).
2. Numbers that are ASCII-sum encodings of lowercase planet names (e.g., if 'mercury' has ASCII sum 775, this would correspond to Mercury, then to digit 1 based on the planet order).
If planet names (or decoded planets from ASCII sums) appear consecutively, concatenate their corresponding digits (e.g., Earth Mars -> 34 if Earth is 3rd and Mars 4th after Sun=0).
Sequence:
{sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type2_few_shot(sequence_string: str) -> str:
    prompt = f"""Hybrid series:
1. Planet names (digits 0-9 map in order to the Sun and planets, starting with Sun as 0).
2. ASCII-sum numbers (e.g., a number like 775 might be the ASCII sum of 'mercury', which then maps to its digit based on the planet order). Concatenate consecutive decoded digits.
Provide ONLY the numerical output for 'Now you:'.
Example 1
Input: Uranus, Mercury Sun, Neptune, Mercury Mercury, Pluto, 775 561
Output: 10
(Reason: Assuming Sun=0 order: Uranus(7), MercurySun(10), Neptune(8), MercuryMercury(11), Pluto(9). If 775 is 'mercury'->1 and 561 is 'venus'->2, then 12. Seq: 7,10,8,11,9,12. Rule:+3,-2,... Output: 10)
Example 2
Input: 532 669, Earth Mars, Earth Sun, Venus Neptune, Venus Mars
Output: 22
(Reason: If 532 is 'earth'->3 and 669 is 'saturn'->6, then 36. EarthMars(34), EarthSun(30), VenusNeptune(28), VenusMars(24). Seq: 36,34,30,28,24. Rule:-2,-4,... Output: 22)
Now you:
Input: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type2_cot(sequence_string: str) -> str:
    prompt = f"""Sequence has: ASCII-sums (e.g., an ASCII sum like 775 might represent 'mercury', which then maps to a digit based on planet order Sun=0) and Planet names (following an order starting with Sun as 0). Concatenate digits.
Task: decode, infer rule, validate, compute next. Provide ONLY final numerical answer.
Worked Sample
Input: 532 669, Earth Mars, Earth Sun, Venus Neptune, Venus Mars
1. Decode:
   If "532" is ASCII for 'earth' (maps to 3) and "669" for 'saturn' (maps to 6) -> 36.
   "Earth Mars" (Earth=3, Mars=4) -> 34. "Earth Sun" (Earth=3, Sun=0) -> 30. "Venus Neptune" (Venus=2, Neptune=8) -> 28. "Venus Mars" (Venus=2, Mars=4) -> 24.
2. Full sequence: [36, 34, 30, 28, 24]
3. Infer rule: 36->34(-2), 34->30(-4), 30->28(-2), 28->24(-4). Pattern: -2, -4.
4. Compute next: Last 24. Next op -2. 24-2 = 22. Next term: 22
Apply to:
Sequence: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type3_zero_shot(sequence_string: str) -> str:
    prompt = f"""Sequence mixes integers and MD5 hashes of single digits (0-9). You need to determine which hash corresponds to which digit. Decode hash(es) to digit(s) to form a number.
Sequence:
{sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type3_few_shot(sequence_string: str) -> str:
    prompt = f"""Mix of numbers and MD5 codes of single digits (0-9). Convert MD5 to digit(s). Concatenate if multiple. Find rule. Provide ONLY numerical output for 'Now you:'.
Example 1
Input: 7,10,8,11,9, c4ca4238a0b923820dcc509a6f75849b c81e728d9d4c2f636f067f89cc14862c,
Output: 10
(Reason: If c4ca... is MD5('1') and c81e... is MD5('2'), they form 12. Seq: 7,10,8,11,9,12. Rule:+3,-2,... Output: 10)
Example 2
Input: eccbc87e4b5ce2fe28308fd9f2a7baf3 1679091c5a880faf6fb5e6087eb1b2dc, 34,30,28,24,
Output: 22
(Reason: If eccb... is MD5('3') and 1679... is MD5('6'), they form 36. Seq: 36,34,30,28,24. Rule:-2,-4,... Output: 22)
Now you:
Input: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


def type3_cot(sequence_string: str) -> str:
    prompt = f"""Sequence has numbers and MD5 hashes of single digits (0-9). You must infer the digit for each hash. Decode, list sequence, infer rule, validate, find next. Provide ONLY final numerical answer.
Worked Sample
Input: eccbc87e4b5ce2fe28308fd9f2a7baf3 1679091c5a880faf6fb5e6087eb1b2dc, 34,30,28,24,
1. Decode hashes: If eccb... is MD5('3') and 1679... is MD5('6'), they form 36.
2. Full sequence: [36,34,30,28,24]
3. Infer rule: 36->34(-2), 34->30(-4), ... Pattern: -2,-4.
4. Compute next: Last 24. Next op -2. 24-2=22. Next term: 22
Apply to:
Sequence: {sequence_string}
Output:
Provide ONLY the numerical next term, and nothing else. Do not output any reasoning or explanation."""
    return call_gpt_and_extract_number(prompt)


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

    required_input_cols = [BASE_SEQUENCE_COL, OBFUSCATED_SEQUENCE_COL]
    for col in required_input_cols:
        if col not in df_master.columns:
            print(f"Error: Required input column '{col}' not found in the input CSV. Available columns: {df_master.columns.tolist()}")
            sys.exit(1)
    if GROUND_TRUTH_COL not in df_master.columns:
        print(f"Warning: Ground truth column '{GROUND_TRUTH_COL}' not found. It will not be included in the output if missing from input.")

    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            if output_dir:
                sys.exit(1)

    df_number_series_results = df_master.copy()
    df_obfuscation_results = df_master.copy()

    output_csv_number_series = f"{output_prefix}_number_series_predictions.csv"
    output_csv_obfuscation = f"{output_prefix}_obfuscation_predictions.csv"
    print(f"Number series predictions will be saved to: {output_csv_number_series}")
    print(f"Obfuscation predictions will be saved to: {output_csv_obfuscation}")

    type_functions = {
        1: {"ZeroShot": type1_zero_shot, "FewShot": type1_few_shot, "CoT": type1_cot},
        2: {"ZeroShot": type2_zero_shot, "FewShot": type2_few_shot, "CoT": type2_cot},
        3: {"ZeroShot": type3_zero_shot, "FewShot": type3_few_shot, "CoT": type3_cot},
    }
    prompt_strategies = ["ZeroShot", "FewShot", "CoT"]

    for strategy in prompt_strategies:
        col_name = f"{strategy}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
        df_number_series_results[col_name] = pd.NA
        df_obfuscation_results[col_name] = pd.NA

    total_rows_to_process = min(len(df_master), 300)
    print(f"Processing up to {total_rows_to_process} rows from '{input_csv}' using model '{OPENAI_MODEL_NAME}'...")

    api_call_delay = 0

    for idx in range(total_rows_to_process):
        current_index_label = df_master.index[idx]

        if 0 <= idx < 100:
            current_type_num = 1
        elif 100 <= idx < 200:
            current_type_num = 2
        elif 200 <= idx < 300:
            current_type_num = 3
        else:
            continue

        active_prompt_fns = type_functions[current_type_num]
        print(f"\n  Processing Row {idx+1}/{total_rows_to_process} (Index {current_index_label}) using Type {current_type_num} Prompts (Model: {OPENAI_MODEL_NAME}):")

        base_sequence_text = str(df_master.loc[current_index_label, BASE_SEQUENCE_COL])
        if pd.isna(base_sequence_text) or base_sequence_text.strip() == "":
            print(f"    Skipping empty base sequence for row {idx+1}.")
            for strategy_name in prompt_strategies:
                result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                df_number_series_results.loc[current_index_label, result_col_name] = "SKIPPED_EMPTY_SEQUENCE"
        else:
            print(f"    Base Sequence ({BASE_SEQUENCE_COL}): {(base_sequence_text[:70] + '...') if len(base_sequence_text) > 70 else base_sequence_text}")
            for strategy_name, func_to_call in active_prompt_fns.items():
                result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                try:
                    prediction = func_to_call(base_sequence_text)
                    df_number_series_results.loc[current_index_label, result_col_name] = prediction
                except Exception:
                    df_number_series_results.loc[current_index_label, result_col_name] = "ERROR_IN_FUNCTION_CALL"
                time.sleep(api_call_delay)

        obfuscated_sequence_text = str(df_master.loc[current_index_label, OBFUSCATED_SEQUENCE_COL])
        if pd.isna(obfuscated_sequence_text) or obfuscated_sequence_text.strip() == "":
            print(f"    Skipping empty obfuscated sequence for row {idx+1}.")
            for strategy_name in prompt_strategies:
                result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                df_obfuscation_results.loc[current_index_label, result_col_name] = "SKIPPED_EMPTY_SEQUENCE"
        else:
            print(f"    Obfuscated Sequence ({OBFUSCATED_SEQUENCE_COL}): {(obfuscated_sequence_text[:70] + '...') if len(obfuscated_sequence_text) > 70 else obfuscated_sequence_text}")
            for strategy_name, func_to_call in active_prompt_fns.items():
                result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                try:
                    prediction = func_to_call(obfuscated_sequence_text)
                    df_obfuscation_results.loc[current_index_label, result_col_name] = prediction
                except Exception:
                    df_obfuscation_results.loc[current_index_label, result_col_name] = "ERROR_IN_FUNCTION_CALL"
                time.sleep(api_call_delay)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_rows_to_process:
            try:
                df_number_series_results.to_csv(output_csv_number_series, index=False)
                df_obfuscation_results.to_csv(output_csv_obfuscation, index=False)
                print(f"    -- Checkpoint saved at row {idx+1} --")
            except Exception as e:
                print(f"    Error saving intermediate CSVs at row {idx+1}: {e}")

    try:
        df_number_series_results.to_csv(output_csv_number_series, index=False)
        print(f"Final number_series predictions saved to {output_csv_number_series}")
        df_obfuscation_results.to_csv(output_csv_obfuscation, index=False)
        print(f"Final obfuscation predictions saved to {output_csv_obfuscation}")
    except Exception as e:
        print(f"    Error saving final CSVs: {e}")

    print("\nProcessing complete.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_path> <output_file_prefix>")
        print(f"Example: python {os.path.basename(__file__)} data/my_sequences.csv results/{OPENAI_MODEL_NAME.replace('.', '_')}")
        print(f"         This will generate files like 'results/{OPENAI_MODEL_NAME.replace('.', '_')}_number_series_predictions.csv'")
        sys.exit(1)

    input_file_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]

    if not OPENAI_API_KEY:
        print("\nCRITICAL WARNING: OPENAI_API_KEY is not set.")
        print("Script will fail API calls.")
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    main(input_file_arg, output_prefix_arg)
