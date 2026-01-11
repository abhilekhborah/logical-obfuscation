# This script is structured for OpenAI GPT models, but you can adapt it for other LLM APIs by replacing the relevant API client.
import os
import sys
import pandas as pd
from openai import OpenAI
import time
import re
import csv

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PASTE YOUR API KEY HERE")

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is not set.")
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

try:
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error configuring OpenAI client: {e}")
    sys.exit(1)

OPENAI_MODEL_NAME = "gpt-5"

BASE_QUESTION_COL = 'Base'
OBFUSCATED_QUESTION_COL = 'Obfus_l'
GROUND_TRUTH_COL = 'Answer'


def find_incomplete_rows(df: pd.DataFrame, result_columns: list) -> list:
    incomplete_rows = []
    
    for idx, row in df.iterrows():
        is_incomplete = False
        for col in result_columns:
            if col in df.columns:
                value = row[col]
                if (pd.isna(value) or 
                    value == "" or 
                    value is None):
                    is_incomplete = True
                    break
                elif isinstance(value, str):
                    value_str = str(value).strip()
                    if (value_str.startswith("ERROR_") or 
                        value_str.startswith("NO_PATTERN_FOUND") or
                        "NO_PATTERN_FOUND" in value_str or
                        value_str == "SKIPPED_EMPTY"):
                        is_incomplete = True
                        break
        
        if is_incomplete:
            incomplete_rows.append(idx)
    
    return incomplete_rows


def load_existing_results(output_file: str, master_df: pd.DataFrame, result_columns: list, correct_columns: list) -> pd.DataFrame:
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            print(f"Found existing results file: {output_file}")
            print(f"Existing rows: {len(existing_df)}")
            
            for col in result_columns + correct_columns:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NA
            
            if len(existing_df) < len(master_df):
                print(f"Extending from {len(existing_df)} to {len(master_df)} rows")
                additional_data = master_df.iloc[len(existing_df):].copy()
                
                for col in result_columns + correct_columns:
                    additional_data[col] = pd.NA
                
                existing_df = pd.concat([existing_df, additional_data], ignore_index=True)
            
            return existing_df
            
        except Exception as e:
            print(f"Error reading existing results file: {e}")
            print("Creating new results DataFrame")
            
    df_results = master_df.copy()
    for col in result_columns + correct_columns:
        df_results[col] = pd.NA
    
    return df_results


def normalize_direction_answer(answer: str) -> str:
    if pd.isna(answer) or not answer:
        return "NO_ANSWER"
    
    answer_str = str(answer).strip().upper()
    answer_str = re.sub(r'\s+', ' ', answer_str)
    answer_str = re.sub(r'[.,;]+$', '', answer_str)
    
    direction_map = {
        'NORTH-EAST': 'NORTHEAST', 'NORTH EAST': 'NORTHEAST', 'NE': 'NORTHEAST',
        'NORTH-WEST': 'NORTHWEST', 'NORTH WEST': 'NORTHWEST', 'NW': 'NORTHWEST',
        'SOUTH-EAST': 'SOUTHEAST', 'SOUTH EAST': 'SOUTHEAST', 'SE': 'SOUTHEAST',
        'SOUTH-WEST': 'SOUTHWEST', 'SOUTH WEST': 'SOUTHWEST', 'SW': 'SOUTHWEST',
    }
    
    for old_format, new_format in direction_map.items():
        answer_str = answer_str.replace(old_format, new_format)
    
    away_match = re.search(r'(\w+),?\s*(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)\s*AWAY', answer_str)
    if away_match:
        direction, distance = away_match.groups()
        return f"{distance} KM, {direction}"
    
    dist_dir_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?),?\s*(\w+)', answer_str)
    if dist_dir_match:
        distance, direction = dist_dir_match.groups()
        return f"{distance} KM, {direction}"
    
    direction_only = re.search(r'\b(NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|NORTH|SOUTH|EAST|WEST)\b', answer_str)
    if direction_only:
        return direction_only.group(1)
    
    distance_only = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)\b', answer_str)
    if distance_only:
        return f"{distance_only.group(1)} KM"
    
    return answer_str


def answers_match(predicted: str, ground_truth: str) -> bool:
    if pd.isna(predicted) or pd.isna(ground_truth):
        return False
    
    pred_str = str(predicted).strip().upper()
    truth_str = str(ground_truth).strip().upper()
    
    if pred_str.startswith("ERROR") or pred_str.startswith("NO_"):
        return False
    
    if pred_str == truth_str:
        return True
    
    def extract_all_components(answer):
        components = {
            'distances': [],
            'directions': [],
            'letters': [],
            'special_terms': []
        }
        
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'[.,;]+$', '', answer)
        
        direction_map = {
            'NORTH-EAST': 'NORTHEAST', 'NORTH EAST': 'NORTHEAST', 'NE': 'NORTHEAST',
            'EAST OF NORTH': 'NORTHEAST', 'EASTOFNORTH': 'NORTHEAST',
            'NORTH-WEST': 'NORTHWEST', 'NORTH WEST': 'NORTHWEST', 'NW': 'NORTHWEST',
            'WEST OF NORTH': 'NORTHWEST', 'WESTOFNORTH': 'NORTHWEST',
            'SOUTH-EAST': 'SOUTHEAST', 'SOUTH EAST': 'SOUTHEAST', 'SE': 'SOUTHEAST',
            'EAST OF SOUTH': 'SOUTHEAST', 'EASTOFSOUTH': 'SOUTHEAST',
            'SOUTH-WEST': 'SOUTHWEST', 'SOUTH WEST': 'SOUTHWEST', 'SW': 'SOUTHWEST',
            'WEST OF SOUTH': 'SOUTHWEST', 'WESTOFSOUTH': 'SOUTHWEST',
            'NORTH OF EAST': 'NORTHEAST', 'NORTHOFEAST': 'NORTHEAST',
            'NORTH OF WEST': 'NORTHWEST', 'NORTHOFWEST': 'NORTHWEST',
            'SOUTH OF EAST': 'SOUTHEAST', 'SOUTHOFEAST': 'SOUTHEAST',
            'SOUTH OF WEST': 'SOUTHWEST', 'SOUTHOFWEST': 'SOUTHWEST',
        }
        
        for old_format, new_format in direction_map.items():
            answer = answer.replace(old_format, new_format)
        
        dist_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)', answer)
        components['distances'] = [float(d) for d in dist_matches]
        
        if not components['distances'] and re.search(r'\b0\b', answer):
            components['distances'] = [0.0]
        
        dir_matches = re.findall(r'\b(NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|NORTH|SOUTH|EAST|WEST)\b', answer)
        components['directions'] = dir_matches
        
        letter_combo_matches = re.findall(r'\b([A-Z]{2,})\b', answer)
        single_letter_matches = re.findall(r'\b([A-Z])\b', answer)
        components['letters'] = letter_combo_matches + single_letter_matches
        
        special_terms = re.findall(r'\b(FARTHEST|CLOSEST|NEAREST|MAXIMUM|MINIMUM|SHORTEST|LONGEST)\b', answer)
        components['special_terms'] = special_terms
        
        return components
    
    pred_comp = extract_all_components(pred_str)
    truth_comp = extract_all_components(truth_str)
    
    def components_overlap(pred_comp, truth_comp):
        distance_match = False
        if pred_comp['distances'] and truth_comp['distances']:
            for pred_dist in pred_comp['distances']:
                for truth_dist in truth_comp['distances']:
                    if abs(pred_dist - truth_dist) < 0.01:
                        distance_match = True
                        break
                if distance_match:
                    break
        elif not truth_comp['distances'] and not pred_comp['distances']:
            distance_match = True
        elif not truth_comp['distances']:
            distance_match = True
        
        direction_match = False
        if pred_comp['directions'] and truth_comp['directions']:
            direction_match = bool(set(pred_comp['directions']) & set(truth_comp['directions']))
        elif not truth_comp['directions'] and not pred_comp['directions']:
            direction_match = True
        elif not truth_comp['directions']:
            direction_match = True
        
        letter_match = False
        if pred_comp['letters'] and truth_comp['letters']:
            letter_match = bool(set(pred_comp['letters']) & set(truth_comp['letters']))
        elif not truth_comp['letters'] and not pred_comp['letters']:
            letter_match = True
        elif not truth_comp['letters']:
            letter_match = True
        
        is_multipart = ('.' in truth_str and any(term in truth_str for term in ['FARTHEST', 'DISTANCE BETWEEN', 'CLOSEST', 'WHO IS'])) or \
                       ('=' in truth_str and len(truth_comp['letters']) > 2)
        
        matches = []
        if truth_comp['distances']:
            matches.append(distance_match)
        if truth_comp['directions']:
            matches.append(direction_match)
        if truth_comp['letters']:
            matches.append(letter_match)
            
        if not matches:
            return distance_match and direction_match
        
        if is_multipart:
            return any(matches)
        else:
            if len([m for m in [truth_comp['distances'], truth_comp['directions'], truth_comp['letters']] if m]) == 1:
                return all(matches)
            else:
                required_matches = []
                if truth_comp['distances']:
                    required_matches.append(distance_match)
                if truth_comp['directions']:
                    required_matches.append(direction_match)
                return all(required_matches) if required_matches else all(matches)
    
    return components_overlap(pred_comp, truth_comp)


def extract_direction_answer(response_content: str) -> str:
    if not response_content:
        return "ERROR_EMPTY_RESPONSE"
    
    response = response_content.strip()
    
    answer_match = re.search(r"(?:Answer|The answer is):\s*(.+)", response, re.IGNORECASE)
    if answer_match:
        response = answer_match.group(1).strip()
    else:
        answer_is_match = re.search(r"The answer is\s+(.+)", response, re.IGNORECASE)
        if answer_is_match:
            response = answer_is_match.group(1).strip()
    
    distance_direction = re.search(r"(\d+(?:\.\d+)?\s*(?:km|m|blocks?)),?\s*([A-Za-z]+(?:-[A-Za-z]+)*)", response)
    if distance_direction:
        return f"{distance_direction.group(1)}, {distance_direction.group(2)}"
    
    direction_distance = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)*),?\s*(\d+(?:\.\d+)?\s*(?:km|m|blocks?))\s*(?:away)?", response)
    if direction_distance:
        return f"{direction_distance.group(2)}, {direction_distance.group(1)}"
    
    coordinate_match = re.search(r"(\d+(?:\.\d+)?\s*(?:km|m))\s+(East|West|North|South),\s*(\d+(?:\.\d+)?\s*(?:km|m))\s+(East|West|North|South)", response, re.IGNORECASE)
    if coordinate_match:
        return f"{coordinate_match.group(1)} {coordinate_match.group(2)}, {coordinate_match.group(3)} {coordinate_match.group(4)} of the starting point"
    
    direction_only = re.search(r"\b(North-?East|North-?West|South-?East|South-?West|North|South|East|West)\b", response, re.IGNORECASE)
    if direction_only:
        return direction_only.group(1)
    
    distance_only = re.search(r"\b(\d+(?:\.\d+)?\s*(?:km|m|blocks?))\b", response)
    if distance_only:
        return distance_only.group(1)
    
    descriptive = re.search(r"(East|West|North|South)\s+of\s+(North|South|East|West)", response, re.IGNORECASE)
    if descriptive:
        return f"{descriptive.group(1)} of {descriptive.group(2)}"
    
    facing_match = re.search(r"facing\s+([A-Za-z][A-Za-z\-]*)", response, re.IGNORECASE)
    if facing_match:
        return f"facing {facing_match.group(1)}"
    
    letter_combo = re.search(r"\b([A-Z]{2})\b", response)
    if letter_combo:
        return letter_combo.group(1)
    
    single_letter = re.search(r"\b([A-Z])\b", response)
    if single_letter:
        return single_letter.group(1)
    
    if re.search(r"square", response, re.IGNORECASE):
        return "The path forms a square"
    
    if re.search(r"same\s+point|origin|starting\s+point", response, re.IGNORECASE):
        return "0"
    
    if re.search(r"no\s+direction|at\s+the\s+starting\s+point", response, re.IGNORECASE):
        return "0"
    
    words = response.strip().split()
    if words:
        if len(response.strip()) <= 50:
            return response.strip()
    
    cleaned = re.sub(r'[^\w\s\.\-,]', '', response)[:100]
    return f"NO_PATTERN_FOUND:{cleaned}"


def call_gpt_and_extract_answer(prompt_text: str) -> str:
    retries = 0
    max_retries = 3
    retry_delay = 10

    while retries < max_retries:
        try:
            print(f"    API call attempt {retries + 1}...", end="", flush=True)
            
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}],
            )
            
            print(" Done")

            if response.choices and len(response.choices) > 0:
                response_content = response.choices[0].message.content
            else:
                response_content = ""

            return extract_direction_answer(response_content)

        except Exception as e:
            print(" Failed")
            error_message = str(e)
            
            retryable_keywords = ["rate limit", "server error", "overloaded", "timed out", "connection error"]
            if any(keyword in error_message.lower() for keyword in retryable_keywords):
                retries += 1
                print(f"    API error (attempt {retries}/{max_retries}): {error_message}")
                if retries < max_retries:
                    print(f"    Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    return "ERROR_API_MAX_RETRIES"
            elif isinstance(e, KeyboardInterrupt):
                print("\nScript interrupted by user. Exiting.")
                sys.exit(1)
            else:
                print(f"    Non-retryable API error: {error_message}")
                return f"ERROR_API_GENERAL:{error_message[:50]}"

    return "ERROR_MAX_RETRIES_EXCEEDED"


def build_zeroshot_prompt(question: str) -> str:
    return (
        "Solve this direction and navigation problem step by step.\n"
        "Provide only the final answer (distance and/or direction) without any additional explanation.\n"
        "Use standard compass directions: North, South, East, West, North-East, North-West, South-East, South-West.\n"
        "For distances, include the unit (km, m, blocks) as given in the problem.\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )


def build_fewshot_prompt(question: str) -> str:
    return (
        "Here are examples of direction and navigation problems:\n\n"
        "Example 1:\n"
        "Problem: A person walks 10 km North, then 6 km East. What is his distance and direction from the starting point?\n"
        "Answer: 11.66 km, North-East\n\n"
        "Example 2:\n"
        "Problem: From point A, a traveler moves 5 km North. From there, he proceeds 3 km East. Relative to point A, what is the final direction and distance of the traveler?\n"
        "Answer: North-East, 5.83 km away\n\n"
        "Example 3:\n"
        "Problem: A person goes 7 km South, 3 km East, 4 km North, then 6 km West. What is the shortest distance from the starting point and in which quadrant/direction is the final position?\n"
        "Answer: 4.24 km, South-West\n\n"
        "Example 4:\n"
        "Problem: At 6 PM, a watch's hour hand points North. In which direction will the minute hand point at 9:15 PM?\n"
        "Answer: West\n\n"
        "Example 5:\n"
        "Problem: A person walks 5 km towards south and then turns to the right. After walking 3 km, he turns to the left and walks 4 km. Then he goes back 10 km straight. Now, in which direction is he from the starting place?\n"
        "Answer: North-West\n\n"
        "Now solve this problem using the same approach:\n"
        f"Problem: {question}\n"
        "Provide only the final answer (distance and/or direction) without any additional explanation.\n"
        "Answer:"
    )


def build_cot_prompt(question: str) -> str:
    return (
        "You are an expert in solving direction and navigation problems. Follow these steps:\n\n"
        "1. Parse the Problem: Identify starting point, movements (distances and directions), and what needs to be found.\n"
        "2. Handle Obfuscated Directions: If the problem uses complex rotational descriptions or clock references:\n"
        "   - Convert clock positions: 12 o'clock = North, 3 o'clock = East, 6 o'clock = South, 9 o'clock = West\n"
        "   - Simplify rotation sequences: Multiple spins that cancel each other out\n"
        "   - Focus on net direction changes, ignore theatrical elements\n"
        "3. Track Position: Use coordinate system with starting point as origin (0,0)\n"
        "   - North = +Y, South = -Y, East = +X, West = -X\n"
        "   - Update coordinates after each movement\n"
        "4. Calculate Final Result:\n"
        "   - Distance: Use Pythagorean theorem sqrt(x^2 + y^2)\n"
        "   - Direction: Use coordinate quadrants and angles\n"
        "   - Handle special cases (same point = 0 distance, cardinal directions)\n"
        "5. Provide Answer: Give final distance and/or direction as requested\n\n"
        "Think step by step, but provide only the final answer (distance and/or direction) without showing your work.\n"
        "Use standard compass directions: North, South, East, West, North-East, North-West, South-East, South-West.\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )


def main(input_csv: str, output_prefix: str):
    try:
        df_master = pd.read_csv(input_csv)
        df_master = df_master.loc[:, ~df_master.columns.str.contains('^Unnamed')]
        print(f"Loaded input CSV: {input_csv} ({len(df_master)} rows)")
        print(f"Columns: {df_master.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV '{input_csv}': {e}")
        sys.exit(1)

    required_cols = [BASE_QUESTION_COL, OBFUSCATED_QUESTION_COL, GROUND_TRUTH_COL]
    for col in required_cols:
        if col not in df_master.columns:
            print(f"Error: Required column '{col}' not found. Available: {df_master.columns.tolist()}")
            sys.exit(1)

    output_csv_base = f"{output_prefix}_base_questions.csv"
    output_csv_obfuscated = f"{output_prefix}_obfuscated_questions.csv"
    
    print("\nStarting Direction Sense GPT-5 Evaluation...")
    print(f"Base questions file: {output_csv_base}")
    print(f"Obfuscated questions file: {output_csv_obfuscated}")

    prompt_strategies = ["ZeroShot", "FewShot", "CoT"]
    prompt_functions = {
        "ZeroShot": build_zeroshot_prompt,
        "FewShot": build_fewshot_prompt,
        "CoT": build_cot_prompt,
    }
    
    result_columns = [f"{strategy}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}" for strategy in prompt_strategies]
    correct_columns = [f"{strategy}_Correct_{OPENAI_MODEL_NAME.replace('.', '_')}" for strategy in prompt_strategies]

    for file_type, (output_csv, question_col) in [
        ("BASE", (output_csv_base, BASE_QUESTION_COL)),
        ("OBFUSCATED", (output_csv_obfuscated, OBFUSCATED_QUESTION_COL))
    ]:
        
        print("\n" + "="*60)
        print(f"Processing {file_type} Questions")
        print("="*60)

        df_results = load_existing_results(output_csv, df_master, result_columns, correct_columns)
        
        incomplete_rows = find_incomplete_rows(df_results, result_columns)
        
        if len(incomplete_rows) == 0:
            print(f"All rows already completed in {output_csv}")
            continue
        
        print(f"Found {len(incomplete_rows)} incomplete rows to process")
        
        if not os.path.exists(output_csv):
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(df_results.columns.tolist())

        processed_count = 0
        for idx in incomplete_rows:
            row = df_results.iloc[idx]
            question_text = str(row[question_col])
            
            if pd.isna(question_text) or question_text.strip() == "":
                print(f"\nRow {idx+1}: Skipping empty question")
                for strategy_name in prompt_strategies:
                    result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                    correct_col_name = f"{strategy_name}_Correct_{OPENAI_MODEL_NAME.replace('.', '_')}"
                    df_results.loc[idx, result_col_name] = "SKIPPED_EMPTY"
                    df_results.loc[idx, correct_col_name] = False
                
                df_results.to_csv(output_csv, index=False)
                processed_count += 1
                continue

            question_preview = (question_text[:80] + '...') if len(question_text) > 80 else question_text
            print(f"\nRow {idx+1}: {question_preview}")
            print(f"Ground truth: {row[GROUND_TRUTH_COL]}")

            row_needs_processing = False
            for strategy_name in prompt_strategies:
                result_col_name = f"{strategy_name}_Result_{OPENAI_MODEL_NAME.replace('.', '_')}"
                correct_col_name = f"{strategy_name}_Correct_{OPENAI_MODEL_NAME.replace('.', '_')}"
                ground_truth = row[GROUND_TRUTH_COL]
                
                current_value = df_results.loc[idx, result_col_name]
                if (not pd.isna(current_value) and 
                    current_value != "" and 
                    current_value is not None):
                    
                    if isinstance(current_value, str):
                        current_str = str(current_value).strip()
                        if (current_str.startswith("ERROR_") or 
                            current_str.startswith("NO_PATTERN_FOUND") or
                            "NO_PATTERN_FOUND" in current_str or
                            current_str == "SKIPPED_EMPTY"):
                            print(f"{strategy_name} needs retry: {current_str[:50]}...")
                        else:
                            print(f"{strategy_name} already completed: {current_value}")
                            continue
                    else:
                        print(f"{strategy_name} already completed: {current_value}")
                        continue
                
                row_needs_processing = True
                print(f"Running {strategy_name}...")
                try:
                    prompt_function = prompt_functions[strategy_name]
                    prompt_text = prompt_function(question_text)
                    prediction = call_gpt_and_extract_answer(prompt_text)
                    df_results.loc[idx, result_col_name] = prediction
                    
                    is_correct = answers_match(prediction, ground_truth)
                    df_results.loc[idx, correct_col_name] = is_correct
                    
                    correctness_indicator = "CORRECT" if is_correct else "WRONG"
                    print(f"Result: {prediction} ({correctness_indicator})")
                except Exception as e:
                    print(f"Error: {e}")
                    df_results.loc[idx, result_col_name] = "ERROR_FUNCTION_CALL"
                    df_results.loc[idx, correct_col_name] = False
                
                time.sleep(1)

            if row_needs_processing:
                try:
                    df_results.to_csv(output_csv, index=False)
                    print(f"Saved row {idx+1} results")
                except Exception as e:
                    print(f"Error saving row: {e}")

            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Progress: {processed_count}/{len(incomplete_rows)} rows processed")

        print(f"\n{file_type} processing complete!")
        print(f"Final file: {output_csv}")

    print("\nDirection Sense evaluation completed successfully!")

    print("\n" + "="*60)
    print("ACCURACY ANALYSIS")
    print("="*60)
    
    calculate_and_display_accuracy(output_csv_base, output_csv_obfuscated, prompt_strategies, OPENAI_MODEL_NAME)


def calculate_and_display_accuracy(base_file: str, obfuscated_file: str, strategies: list, model_name: str):
    def get_accuracy_metrics(df: pd.DataFrame, strategies: list, model_name: str):
        results = {}
        model_suffix = model_name.replace('.', '_')
        
        for strategy in strategies:
            correct_col = f"{strategy}_Correct_{model_suffix}"
            result_col = f"{strategy}_Result_{model_suffix}"
            
            if correct_col not in df.columns or result_col not in df.columns:
                continue
                
            total = len(df[df[result_col].notna()])
            correct = df[correct_col].sum()
            accuracy = correct / total if total > 0 else 0
            
            error_responses = len(df[df[result_col].str.startswith("ERROR", na=False)])
            skipped = len(df[df[result_col] == "SKIPPED_EMPTY"])
            valid_responses = total - error_responses - skipped
            valid_accuracy = correct / valid_responses if valid_responses > 0 else 0
            
            results[strategy] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy,
                'valid_responses': valid_responses,
                'valid_accuracy': valid_accuracy,
                'error_responses': error_responses,
                'skipped': skipped
            }
        
        return results
    
    if os.path.exists(base_file):
        print("\nBASE QUESTIONS ACCURACY")
        print("-" * 40)
        
        try:
            df_base = pd.read_csv(base_file)
            base_results = get_accuracy_metrics(df_base, strategies, model_name)
            
            for strategy, metrics in base_results.items():
                print(f"\n{strategy}:")
                print(f"Total: {metrics['total']}, Correct: {metrics['correct']}")
                print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
                print(f"Valid accuracy: {metrics['valid_accuracy']:.3f} ({metrics['valid_accuracy']*100:.1f}%)")
                if metrics['error_responses'] > 0:
                    print(f"Errors: {metrics['error_responses']}")
                    
        except Exception as e:
            print(f"Error analyzing base file: {e}")
            base_results = {}
    else:
        print(f"Base file not found: {base_file}")
        base_results = {}
    
    if os.path.exists(obfuscated_file):
        print("\nOBFUSCATED QUESTIONS ACCURACY")
        print("-" * 40)
        
        try:
            df_obfuscated = pd.read_csv(obfuscated_file)
            obf_results = get_accuracy_metrics(df_obfuscated, strategies, model_name)
            
            for strategy, metrics in obf_results.items():
                print(f"\n{strategy}:")
                print(f"Total: {metrics['total']}, Correct: {metrics['correct']}")
                print(f"Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
                print(f"Valid accuracy: {metrics['valid_accuracy']:.3f} ({metrics['valid_accuracy']*100:.1f}%)")
                if metrics['error_responses'] > 0:
                    print(f"Errors: {metrics['error_responses']}")
                    
        except Exception as e:
            print(f"Error analyzing obfuscated file: {e}")
            obf_results = {}
    else:
        print(f"Obfuscated file not found: {obfuscated_file}")
        obf_results = {}
    
    if base_results and obf_results:
        print("\nOBFUSCATION ROBUSTNESS ANALYSIS")
        print("-" * 40)
        print(f"{'Strategy':<12} {'Base %':<8} {'Obf %':<8} {'Diff':<8} {'Impact'}")
        print("-" * 50)
        
        total_base_acc = 0
        total_obf_acc = 0
        valid_comparisons = 0
        
        for strategy in strategies:
            if strategy in base_results and strategy in obf_results:
                base_acc = base_results[strategy]['accuracy']
                obf_acc = obf_results[strategy]['accuracy']
                diff = base_acc - obf_acc
                
                if abs(diff) < 0.02:
                    impact = "Robust"
                elif diff > 0.05:
                    impact = "Degraded"
                elif diff > 0.02:
                    impact = "Minor Drop"
                else:
                    impact = "Improved"
                
                print(f"{strategy:<12} {base_acc*100:>6.1f}%  {obf_acc*100:>6.1f}%  {diff*100:>+6.1f}%  {impact}")
                
                total_base_acc += base_acc
                total_obf_acc += obf_acc
                valid_comparisons += 1
        
        if valid_comparisons > 0:
            avg_base = total_base_acc / valid_comparisons
            avg_obf = total_obf_acc / valid_comparisons
            overall_impact = avg_base - avg_obf
            
            print("\nOVERALL SUMMARY:")
            print(f"Average Base Accuracy: {avg_base*100:.1f}%")
            print(f"Average Obfuscated Accuracy: {avg_obf*100:.1f}%")
            print(f"Obfuscation Impact: {overall_impact*100:+.1f} percentage points")
            
            if overall_impact > 0.1:
                print("Significant degradation - Models struggle with obfuscation")
            elif overall_impact > 0.05:
                print("Moderate degradation - Some obfuscation impact")
            elif abs(overall_impact) <= 0.05:
                print("Robust performance - Models handle obfuscation well")
            else:
                print("Improved performance - Obfuscation unexpectedly helped")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Direction Sense GPT-5 Evaluation Script")
        print("Processes base and obfuscated direction problems with Zero-shot, Few-shot, and CoT prompts.\n")
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_path> <output_file_prefix>")
        print(f"Example: python {os.path.basename(__file__)} direction_sense.csv results")
        print("This will create:")
        print("- results_base_questions.csv")
        print("- results_obfuscated_questions.csv")
        print("\nThe script will:")
        print("1. Process both base and obfuscated question columns")
        print("2. Apply Zero-shot, Few-shot, and CoT prompting strategies")
        print("3. Extract answers using regex patterns for various formats")
        print("4. Save results incrementally to avoid data loss")
        sys.exit(1)

    input_file_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]
    
    main(input_file_arg, output_prefix_arg)
