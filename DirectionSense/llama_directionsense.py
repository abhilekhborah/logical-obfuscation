# -*- coding: utf-8 -*-
"""
Direction Sense Llama Evaluation Script
Processes base and obfuscated direction sense problems using Zero-shot, Few-shot, and CoT approaches.
Uses Llama-4-Scout-17B-16E-Instruct via Olakrutrim cloud API.

Usage: python llama_directionsense.py input.csv output_prefix
Example: python llama_directionsense.py direction_sense.csv results/llama_ds
This will generate:
- results/llama_ds_Base.csv
- results/llama_ds_Obfus_l.csv
"""
import os
import sys
import pandas as pd
import requests
import json
import time
import csv
import re

# --- Llama API Configuration ---
LLAMA4_API_KEY = "jk5c8YsP3loxL-0W57w7Vxwy"
LLAMA4_API_URL = 'https://cloud.olakrutrim.com/v1/chat/completions'
LLAMA4_MODEL_NAME = "Llama-4-Scout-17B-16E-Instruct"

if not LLAMA4_API_KEY or LLAMA4_API_KEY == "jk5c8YsP3loxL-0W57w7Vxwy":
    print("Warning: Using the placeholder API Key.")
    print("Ensure this is your correct key or replace it.")

# --- Target Column Names for Direction Sense Dataset ---
TARGET_QUESTION_COLS = ['Base', 'Obfus_l']

# --- Direction Sense Prompt Templates ---

def build_zeroshot_prompt(question: str) -> str:
    """Builds a zero-shot prompt for direction and navigation problems."""
    return (
        "Solve this direction and navigation problem step by step.\n"
        "Provide only the final answer (distance and/or direction) without any additional explanation.\n"
        "Use standard compass directions: North, South, East, West, North-East, North-West, South-East, South-West.\n"
        "For distances, include the unit (km, m, blocks) as given in the problem.\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )

def build_fewshot_prompt(question: str) -> str:
    """Builds a few-shot prompt for direction and navigation problems with examples."""
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
    """Builds a Chain-of-Thought prompt for direction and navigation problems."""
    return (
        "You are an expert in solving direction and navigation problems. Follow these steps:\n\n"
        "1. **Parse the Problem**: Identify starting point, movements (distances and directions), and what needs to be found.\n"
        "2. **Handle Obfuscated Directions**: If the problem uses complex rotational descriptions or clock references:\n"
        "   - Convert clock positions: 12 o'clock = North, 3 o'clock = East, 6 o'clock = South, 9 o'clock = West\n"
        "   - Simplify rotation sequences: Multiple spins that cancel each other out\n"
        "   - Focus on net direction changes, ignore theatrical elements\n"
        "3. **Track Position**: Use coordinate system with starting point as origin (0,0)\n"
        "   - North = +Y, South = -Y, East = +X, West = -X\n"
        "   - Update coordinates after each movement\n"
        "4. **Calculate Final Result**:\n"
        "   - Distance: Use Pythagorean theorem √(x² + y²)\n"
        "   - Direction: Use coordinate quadrants and angles\n"
        "   - Handle special cases (same point = 0 distance, cardinal directions)\n"
        "5. **Provide Answer**: Give final distance and/or direction as requested\n\n"
        "Think step by step, but provide only the final answer (distance and/or direction) without showing your work.\n"
        "Use standard compass directions: North, South, East, West, North-East, North-West, South-East, South-West.\n\n"
        f"Problem: {question}\n"
        "Answer:"
    )

# --- Answer Normalization and Comparison Functions ---
def normalize_direction_answer(answer: str) -> str:
    """Normalize direction answers for comparison"""
    if pd.isna(answer) or not answer:
        return "NO_ANSWER"

    answer_str = str(answer).strip().upper()

    # Handle various formatting inconsistencies
    answer_str = re.sub(r'\s+', ' ', answer_str)  # Normalize whitespace
    answer_str = re.sub(r'[.,;]+$', '', answer_str)  # Remove trailing punctuation

    # Standardize direction names
    direction_map = {
        'NORTH-EAST': 'NORTHEAST', 'NORTH EAST': 'NORTHEAST', 'NE': 'NORTHEAST',
        'NORTH-WEST': 'NORTHWEST', 'NORTH WEST': 'NORTHWEST', 'NW': 'NORTHWEST',
        'SOUTH-EAST': 'SOUTHEAST', 'SOUTH EAST': 'SOUTHEAST', 'SE': 'SOUTHEAST',
        'SOUTH-WEST': 'SOUTHWEST', 'SOUTH WEST': 'SOUTHWEST', 'SW': 'SOUTHWEST',
    }

    for old_format, new_format in direction_map.items():
        answer_str = answer_str.replace(old_format, new_format)

    # Handle "X km away" vs "X km, Direction" formats
    away_match = re.search(r'(\w+),?\s*(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)\s*AWAY', answer_str)
    if away_match:
        direction, distance = away_match.groups()
        return f"{distance} KM, {direction}"

    # Handle "Distance, Direction" format
    dist_dir_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?),?\s*(\w+)', answer_str)
    if dist_dir_match:
        distance, direction = dist_dir_match.groups()
        return f"{distance} KM, {direction}"

    # Handle direction only
    direction_only = re.search(r'\b(NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|NORTH|SOUTH|EAST|WEST)\b', answer_str)
    if direction_only:
        return direction_only.group(1)

    # Handle distance only
    distance_only = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)\b', answer_str)
    if distance_only:
        return f"{distance_only.group(1)} KM"

    return answer_str

def answers_match(predicted: str, ground_truth: str) -> bool:
    """Compare normalized predicted answer with ground truth"""
    if pd.isna(predicted) or pd.isna(ground_truth):
        return False

    pred_str = str(predicted).strip().upper()
    truth_str = str(ground_truth).strip().upper()

    # Handle error responses
    if pred_str.startswith("ERROR") or pred_str.startswith("NO_"):
        return False

    # Exact match after normalization
    if pred_str == truth_str:
        return True

    # Extract all numeric values and key information from both answers
    def extract_all_components(answer):
        components = {
            'distances': [],  # All distance values found
            'directions': [],  # All direction values found
            'letters': [],    # Single letters (like A, B, C, D)
            'special_terms': []  # Special terms like FARTHEST, CLOSEST, etc.
        }

        # Normalize common variations
        answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
        answer = re.sub(r'[.,;]+$', '', answer)  # Remove trailing punctuation

        # Standardize direction names (including equivalent descriptions)
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

        # Extract all distance values
        dist_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:KM|M|BLOCKS?)', answer)
        components['distances'] = [float(d) for d in dist_matches]

        # Special case: handle standalone "0" which means zero distance
        if not components['distances'] and re.search(r'\b0\b', answer):
            components['distances'] = [0.0]

        # Extract all directions
        dir_matches = re.findall(r'\b(NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST|NORTH|SOUTH|EAST|WEST)\b', answer)
        components['directions'] = dir_matches

        # Extract single letters and letter combinations (A, B, C, D, LP, SP, etc.)
        # First try letter combinations (2+ letters)
        letter_combo_matches = re.findall(r'\b([A-Z]{2,})\b', answer)
        # Then try single letters
        single_letter_matches = re.findall(r'\b([A-Z])\b', answer)
        # Combine both (prioritize combinations over singles)
        components['letters'] = letter_combo_matches + single_letter_matches

        # Extract special terms
        special_terms = re.findall(r'\b(FARTHEST|CLOSEST|NEAREST|MAXIMUM|MINIMUM|SHORTEST|LONGEST)\b', answer)
        components['special_terms'] = special_terms

        return components

    pred_comp = extract_all_components(pred_str)
    truth_comp = extract_all_components(truth_str)

    # For multi-part answers, check if prediction contains key components from ground truth
    def components_overlap(pred_comp, truth_comp):
        # Check if predicted distances overlap with truth distances (with tolerance)
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
            distance_match = True  # Both have no distances
        elif not truth_comp['distances']:  # Only truth has no distance
            distance_match = True

        # Check direction overlap
        direction_match = False
        if pred_comp['directions'] and truth_comp['directions']:
            direction_match = bool(set(pred_comp['directions']) & set(truth_comp['directions']))
        elif not truth_comp['directions'] and not pred_comp['directions']:
            direction_match = True  # Both have no directions
        elif not truth_comp['directions']:  # Only truth has no directions
            direction_match = True

        # Check letter overlap (for questions like "Farthest = D")
        letter_match = False
        if pred_comp['letters'] and truth_comp['letters']:
            letter_match = bool(set(pred_comp['letters']) & set(truth_comp['letters']))
        elif not truth_comp['letters'] and not pred_comp['letters']:
            letter_match = True  # Both have no letters
        elif not truth_comp['letters']:  # Only truth has no letters
            letter_match = True

        # Determine if this is a multi-part question vs single answer with multiple components
        # Multi-part: "Farthest = D (15 km). Distance between A and B = 2 km." (two separate questions)
        # Single answer: "North-East, 5.83 km away" (one answer with distance AND direction)

        # Check if ground truth has multiple sentences or explicit multi-part structure
        is_multipart = ('.' in truth_str and any(term in truth_str for term in ['FARTHEST', 'DISTANCE BETWEEN', 'CLOSEST', 'WHO IS'])) or \
                       ('=' in truth_str and len(truth_comp['letters']) > 2)

        matches = []
        if truth_comp['distances']:
            matches.append(distance_match)
        if truth_comp['directions']:
            matches.append(direction_match)
        if truth_comp['letters']:
            matches.append(letter_match)

        # If no specific components in truth, check basic distance/direction match
        if not matches:
            return distance_match and direction_match

        if is_multipart:
            # For multi-part questions, any component match is sufficient
            return any(matches)
        else:
            # For single answers with multiple components, ALL components must match
            # Exception: if only one component type exists in truth, just check that
            if len([m for m in [truth_comp['distances'], truth_comp['directions'], truth_comp['letters']] if m]) == 1:
                return all(matches)
            else:
                # Both distance and direction required for single coordinate answers
                required_matches = []
                if truth_comp['distances']:
                    required_matches.append(distance_match)
                if truth_comp['directions']:
                    required_matches.append(direction_match)
                return all(required_matches) if required_matches else all(matches)

    return components_overlap(pred_comp, truth_comp)

# --- Answer Extraction Function (All 11 Regex Patterns from GPT-5 Script) ---
def extract_direction_answer(response_content: str) -> str:
    """Extract direction/distance answer from Llama response using regex patterns"""
    if not response_content:
        return "ERROR_EMPTY_RESPONSE"

    # Clean the response
    response = response_content.strip()

    # Pattern 1: Look for "Answer:" or "The answer is" followed by the answer
    answer_match = re.search(r"(?:Answer|The answer is):\s*(.+)", response, re.IGNORECASE)
    if answer_match:
        response = answer_match.group(1).strip()
    else:
        # Also try "The answer is X" format
        answer_is_match = re.search(r"The answer is\s+(.+)", response, re.IGNORECASE)
        if answer_is_match:
            response = answer_is_match.group(1).strip()

    # Pattern 2: Distance + Direction formats
    # Examples: "11.66 km, North-East", "4.24 km,South-West", "5 km, North-East."
    distance_direction = re.search(r"(\d+(?:\.\d+)?\s*(?:km|m|blocks?)),?\s*([A-Za-z]+(?:-[A-Za-z]+)*)", response)
    if distance_direction:
        return f"{distance_direction.group(1)}, {distance_direction.group(2)}"

    # Pattern 3: Direction + Distance formats
    # Examples: "North-East, 5.83 km away", "West, 3 km"
    # Fixed to handle compound directions and the word "away"
    direction_distance = re.search(r"([A-Za-z]+(?:-[A-Za-z]+)*),?\s*(\d+(?:\.\d+)?\s*(?:km|m|blocks?))\s*(?:away)?", response)
    if direction_distance:
        return f"{direction_distance.group(2)}, {direction_distance.group(1)}"

    # Pattern 4: Complex coordinate descriptions
    # Examples: "5 km East, 6 km South of the starting point"
    coordinate_match = re.search(r"(\d+(?:\.\d+)?\s*(?:km|m))\s+(East|West|North|South),\s*(\d+(?:\.\d+)?\s*(?:km|m))\s+(East|West|North|South)", response, re.IGNORECASE)
    if coordinate_match:
        return f"{coordinate_match.group(1)} {coordinate_match.group(2)}, {coordinate_match.group(3)} {coordinate_match.group(4)} of the starting point"

    # Pattern 5: Direction only
    # Examples: "North", "South-West", "East", "North-East"
    direction_only = re.search(r"\b(North-?East|North-?West|South-?East|South-?West|North|South|East|West)\b", response, re.IGNORECASE)
    if direction_only:
        return direction_only.group(1)

    # Pattern 6: Distance only
    # Examples: "3km", "6.708 km", "0 m"
    distance_only = re.search(r"\b(\d+(?:\.\d+)?\s*(?:km|m|blocks?))\b", response)
    if distance_only:
        return distance_only.group(1)

    # Pattern 7: Special descriptive formats
    # Examples: "East of North", "facing North-East"
    descriptive = re.search(r"(East|West|North|South)\s+of\s+(North|South|East|West)", response, re.IGNORECASE)
    if descriptive:
        return f"{descriptive.group(1)} of {descriptive.group(2)}"

    facing_match = re.search(r"facing\s+([A-Za-z][A-Za-z\-]*)", response, re.IGNORECASE)
    if facing_match:
        return f"facing {facing_match.group(1)}"

    # Pattern 8: Simple letter combinations (for relative position questions)
    # Examples: "SP", "LA", "LP" - common in point relation problems
    letter_combo = re.search(r"\b([A-Z]{2})\b", response)
    if letter_combo:
        return letter_combo.group(1)

    # Pattern 9: Single letters (like A, B, C, D)
    single_letter = re.search(r"\b([A-Z])\b", response)
    if single_letter:
        return single_letter.group(1)

    # Pattern 10: Special cases and common phrases
    if re.search(r"square", response, re.IGNORECASE):
        return "The path forms a square"

    if re.search(r"same\s+point|origin|starting\s+point", response, re.IGNORECASE):
        return "0"

    if re.search(r"no\s+direction|at\s+the\s+starting\s+point", response, re.IGNORECASE):
        return "0"

    # Pattern 11: Try to extract any reasonable answer from the response
    # Look for the last meaningful word or phrase
    words = response.strip().split()
    if words:
        # If the response is short and might be a valid answer, return it
        if len(response.strip()) <= 50:
            return response.strip()

    # If no pattern matches, return the first reasonable substring
    cleaned = re.sub(r'[^\w\s\.\-,]', '', response)[:100]
    return f"NO_PATTERN_FOUND:{cleaned}"

# --- API Call Function ---
def call_llama_api(user_prompt_text: str) -> str:
    """
    Calls the Llama API via HTTP POST, includes retry logic.
    Returns the extracted direction/distance answer or error codes.
    """
    retries = 0
    max_retries = 5
    retry_delay = 60  # seconds

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LLAMA4_API_KEY}'
    }

    payload = {
        "model": LLAMA4_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text}
                ]
            }
        ],
        "max_tokens": 150,  # Allow longer responses for direction answers
        "temperature": 0.0,
    }

    while retries < max_retries:
        try:
            response = requests.post(LLAMA4_API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            response_data = response.json()

            # Parse the response
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0].get('message', {})
                text = message.get('content', '')
            else:
                print(f"Warning: Unexpected API response format. Keys: {response_data.keys()}")
                return "ERROR_RESPONSE_FORMAT"

            cleaned_text = text.strip() if text else ""

            if not cleaned_text:
                print(f"Warning: API returned an empty response content. Raw JSON: {response_data}")
                retries += 1
                if retries >= max_retries:
                    return "ERROR_EMPTY_RESPONSE"
                print(f"Retrying empty response (attempt {retries}/{max_retries})...")
                time.sleep(retry_delay / 2)
                continue
            else:
                # Extract answer using all 11 regex patterns
                return extract_direction_answer(cleaned_text)

        except requests.exceptions.Timeout as e:
            retries += 1
            print(f"API Timeout Error (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                print(f"Waiting for {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                print("Max retries reached for timeout error. Returning ERROR_API.")
                return "ERROR_API"
        except requests.exceptions.HTTPError as e:
            retries += 1
            print(f"HTTP Error (attempt {retries}/{max_retries}): {e.response.status_code} - {e.response.text}")
            if e.response.status_code in [429, 500, 502, 503, 504]:
                if retries < max_retries:
                    print(f"Waiting for {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
                else:
                    print("Max retries reached for HTTP error. Returning ERROR_API.")
                    return "ERROR_API"
            else:
                print("Non-retryable HTTP error.")
                return "ERROR_API"
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"Request Exception (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                print(f"Waiting for {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                print("Max retries reached for request exception. Returning ERROR_API.")
                return "ERROR_API"
        except json.JSONDecodeError as e:
            print(f"Error decoding API response JSON: {e}")
            print(f"Raw response text: {response.text}")
            return "ERROR_RESPONSE_FORMAT"
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                print("\nScript interrupted by user (KeyboardInterrupt). Exiting.")
                sys.exit(1)
            retries += 1
            print(f"General error during API call (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                print(f"Waiting for {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached for general error. Returning ERROR_API.")
                return "ERROR_API"

    print("Max retries reached without success (loop exit).")
    return "ERROR_API"


def main(input_csv: str, output_prefix: str):
    """
    Processes direction sense questions from the input CSV using Llama API,
    runs inference, and saves results incrementally to separate CSV files.

    Args:
        input_csv (str): Path to the input CSV file.
        output_prefix (str): Prefix for the output CSV files.
    """
    try:
        df_master = pd.read_csv(input_csv)
        print(f"✅ Successfully loaded input CSV: {input_csv}")
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV '{input_csv}': {e}")
        sys.exit(1)

    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            sys.exit(1)

    # Process each target question column
    for question_col_name in TARGET_QUESTION_COLS:
        print(f"\n{'='*60}")
        print(f"🧭 Processing Column: {question_col_name}")
        print(f"{'='*60}")

        if question_col_name not in df_master.columns:
            print(f"⚠️ Warning: Column '{question_col_name}' not found in the input CSV. Skipping.")
            print(f"Available columns: {df_master.columns.tolist()}")
            continue

        # Define output CSV path for the current column
        output_csv = f"{output_prefix}_{question_col_name}.csv"
        print(f"📊 Output for this column will be saved incrementally to: {output_csv}")

        # Define result column names based on the model used
        safe_model_name = LLAMA4_MODEL_NAME.replace("/", "_").replace(".", "_").replace("-", "_")
        zeroshot_col_name = f'{safe_model_name}_zeroshot'
        fewshot_col_name = f'{safe_model_name}_fewshot'
        cot_col_name = f'{safe_model_name}_cot'

        # Prepare list of columns for the output CSV
        original_cols_to_keep = [c for c in df_master.columns if c not in TARGET_QUESTION_COLS]
        output_cols_header = original_cols_to_keep + [question_col_name, zeroshot_col_name, fewshot_col_name, cot_col_name]

        # Write Header Row
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(output_cols_header)
            print(f"Initialized output file with header: {output_csv}")
        except Exception as e:
            print(f"Error writing header to '{output_csv}': {e}")
            print(f"Skipping processing for column '{question_col_name}' due to file error.")
            continue

        total_rows = len(df_master)
        print(f"Processing {total_rows} rows for column '{question_col_name}' using model '{LLAMA4_MODEL_NAME}'...\n")

        # Process Rows Incrementally
        for idx, row in df_master.iterrows():
            question_text = str(row[question_col_name]) if pd.notna(row[question_col_name]) else ""

            # Prepare data for the current row
            row_data = {col: row[col] for col in original_cols_to_keep}
            row_data[question_col_name] = question_text

            if not question_text.strip():
                print(f"  📝 Row {idx+1}/{total_rows} (Index {idx}): Skipping empty question.")
                row_data[zeroshot_col_name] = "SKIPPED_EMPTY"
                row_data[fewshot_col_name] = "SKIPPED_EMPTY"
                row_data[cot_col_name] = "SKIPPED_EMPTY"
            else:
                print(f"\n  📝 Processing Row {idx+1}/{total_rows} (Index {idx}) for '{question_col_name}':")
                question_print = (question_text[:90] + '...') if len(question_text) > 90 else question_text
                print(f"    Question: {question_print}")

                # Run Zero-shot
                print("    🎯 Running Zero-shot...")
                zeroshot_user_prompt = build_zeroshot_prompt(question_text)
                zeroshot_result = call_llama_api(zeroshot_user_prompt)
                row_data[zeroshot_col_name] = zeroshot_result
                print(f"    Zero-shot Result: {zeroshot_result}")
                time.sleep(1)

                # Run Few-shot
                print("    🎯 Running Few-shot...")
                fewshot_user_prompt = build_fewshot_prompt(question_text)
                fewshot_result = call_llama_api(fewshot_user_prompt)
                row_data[fewshot_col_name] = fewshot_result
                print(f"    Few-shot Result: {fewshot_result}")
                time.sleep(1)

                # Run CoT
                print("    🎯 Running CoT...")
                cot_user_prompt = build_cot_prompt(question_text)
                cot_result = call_llama_api(cot_user_prompt)
                row_data[cot_col_name] = cot_result
                print(f"    CoT Result: {cot_result}")

            # Append the processed row to the CSV
            try:
                row_values_to_write = [row_data.get(col_name, "") for col_name in output_cols_header]
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_values_to_write)
                if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                    print(f"    💾 Row {idx+1} saved to {os.path.basename(output_csv)}")

            except Exception as e:
                print(f"    ❌ Error saving row {idx+1} to CSV for {question_col_name}: {e}")

        print(f"\n{'='*60}")
        print(f"✅ Finished processing column: {question_col_name}")
        print(f"📁 Incremental results saved to {output_csv}")
        print(f"{'='*60}")

    print("\n🎉 All specified columns processed.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Direction Sense Llama Evaluation Script")
        print("Processes direction sense problems with Zero-shot, Few-shot, and CoT prompts.\n")
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_path> <output_file_prefix>")
        print(f"Example: python {os.path.basename(__file__)} direction_sense.csv results/llama_ds")
        print(f"         This will create:")
        print(f"         - results/llama_ds_Base.csv")
        print(f"         - results/llama_ds_Obfus_l.csv")
        print("\nThe script will:")
        print("  1. Process both base and obfuscated question columns")
        print("  2. Apply Zero-shot, Few-shot, and CoT prompting strategies")
        print("  3. Extract answers using all 11 regex patterns from GPT-5 script")
        print("  4. Save results incrementally to avoid data loss")
        sys.exit(1)

    input_file_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]

    print("🧭 Direction Sense Llama-4 Evaluation Script")
    print("=" * 60)

    main(input_file_arg, output_prefix_arg)
