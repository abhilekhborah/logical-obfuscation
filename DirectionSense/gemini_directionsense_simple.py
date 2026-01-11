#!/usr/bin/env python3
"""
Gemini 2.5 Pro Direction Sense Evaluation Script
Processes base and obfuscated direction sense problems using Zero-shot, Few-shot, and CoT approaches.
Handles complex answer formats with regex parsing and resume functionality.
"""

import os
import sys
import pandas as pd
import google.generativeai as genai
import time
import re
import csv

# --- Google Gemini API Configuration ---
GOOGLE_API_KEY = "AIzaSyChM0rYAx-2-9dcaTCeNiFZT3lf4MzQg0Q"

if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("AIzaSy..."):
    print("Error: GOOGLE_API_KEY is not set or is still the placeholder.")
    print("Please replace the placeholder in the script or set the GOOGLE_API_KEY environment variable.")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error configuring Google Gemini client: {e}")
    sys.exit(1)

GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"
try:
    GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Gemini model '{GEMINI_MODEL_NAME}': {e}")
    sys.exit(1)

# --- Input Column Names ---
BASE_QUESTION_COL = 'Base'
OBFUSCATED_QUESTION_COL = 'Obfus_l'
GROUND_TRUTH_COL = 'Answer'

# --- Resume Logic Functions ---
def find_incomplete_rows(df: pd.DataFrame, result_columns: list) -> list:
    """Find rows with missing, empty, or error predictions that need to be retried"""
    incomplete_rows = []

    for idx, row in df.iterrows():
        is_incomplete = False
        for col in result_columns:
            if col in df.columns:
                value = row[col]
                # Check if value is missing, empty, or has error patterns
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
    """Load existing results CSV if it exists, otherwise create new DataFrame"""
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            print(f"📂 Found existing results file: {output_file}")
            print(f"   Existing rows: {len(existing_df)}")

            # Ensure all required columns exist
            for col in result_columns + correct_columns:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NA

            # If existing file has fewer rows than master, extend it
            if len(existing_df) < len(master_df):
                print(f"   Extending from {len(existing_df)} to {len(master_df)} rows")
                # Create additional rows from master_df
                additional_rows_needed = len(master_df) - len(existing_df)
                additional_data = master_df.iloc[len(existing_df):].copy()

                # Add empty result columns to additional rows
                for col in result_columns + correct_columns:
                    additional_data[col] = pd.NA

                # Concatenate existing with additional rows
                existing_df = pd.concat([existing_df, additional_data], ignore_index=True)

            return existing_df

        except Exception as e:
            print(f"⚠️ Error reading existing results file: {e}")
            print("   Creating new results DataFrame")

    # Create new DataFrame with all required columns
    df_results = master_df.copy()
    for col in result_columns + correct_columns:
        df_results[col] = pd.NA

    return df_results

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

# --- Answer Extraction Function (All 11 Regex Patterns) ---
def extract_direction_answer(response_content: str) -> str:
    """Extract direction/distance answer from Gemini response using regex patterns"""
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
def call_gemini_api(prompt_text: str) -> str:
    """
    Calls Google Gemini API and extracts direction/distance answer with retry logic.
    """
    retries = 0
    max_retries = 5
    retry_delay = 60  # seconds

    # Configuration for the generation
    generation_config = genai.GenerationConfig(
        max_output_tokens=150,
        temperature=0.0,
    )

    # Safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    while retries < max_retries:
        try:
            print(f"    API call attempt {retries + 1}...", end="", flush=True)

            response = GEMINI_MODEL.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            print(" ✓")

            # Parse the response
            if not response.candidates:
                print(f" ✗ No candidates in response (safety block?)")
                return "ERROR_SAFETY"

            # Access text from the first candidate
            text = ""
            if response.parts:
                text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'text'):
                text = response.text
            else:
                try:
                    text = response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError):
                    print(f" ✗ Could not extract text")
                    return "ERROR_RESPONSE_FORMAT"

            cleaned_text = text.strip() if text else ""

            if not cleaned_text:
                retries += 1
                if retries >= max_retries:
                    return "ERROR_EMPTY_RESPONSE"
                print(f" Empty response, retrying...")
                time.sleep(retry_delay / 2)
                continue
            else:
                # Extract answer using the comprehensive regex patterns
                return extract_direction_answer(cleaned_text)

        except Exception as e:
            print(" ✗")
            error_message = str(e)

            if isinstance(e, KeyboardInterrupt):
                print("\nScript interrupted by user. Exiting.")
                sys.exit(1)

            retries += 1
            print(f"    API error (attempt {retries}/{max_retries}): {error_message}")

            is_retryable = "quota" in error_message.lower() or \
                          "resource exhausted" in error_message.lower() or \
                          "503" in error_message

            if is_retryable and retries < max_retries:
                print(f"    Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
            else:
                return "ERROR_API"

    return "ERROR_MAX_RETRIES_EXCEEDED"

# --- Prompt Templates ---
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

def main(input_csv: str, output_prefix: str):
    """
    Process direction sense problems with Gemini using different prompt strategies.
    """
    try:
        df_master = pd.read_csv(input_csv)
        # Remove any unnamed columns caused by trailing commas
        df_master = df_master.loc[:, ~df_master.columns.str.contains('^Unnamed')]
        print(f"✅ Loaded input CSV: {input_csv} ({len(df_master)} rows)")
        print(f"📋 Columns: {df_master.columns.tolist()}")
    except FileNotFoundError:
        print(f"❌ Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV '{input_csv}': {e}")
        sys.exit(1)

    # Verify necessary columns
    required_cols = [BASE_QUESTION_COL, OBFUSCATED_QUESTION_COL, GROUND_TRUTH_COL]
    for col in required_cols:
        if col not in df_master.columns:
            print(f"❌ Error: Required column '{col}' not found. Available: {df_master.columns.tolist()}")
            sys.exit(1)

    # Define output files
    output_csv_base = f"{output_prefix}_base_questions.csv"
    output_csv_obfuscated = f"{output_prefix}_obfuscated_questions.csv"

    print(f"\n🚀 Starting Direction Sense Gemini ({GEMINI_MODEL_NAME}) Evaluation...")
    print(f"📊 Base questions file: {output_csv_base}")
    print(f"📊 Obfuscated questions file: {output_csv_obfuscated}")

    # Prompt strategies and corresponding functions
    prompt_strategies = ["ZeroShot", "FewShot", "CoT"]
    prompt_functions = {
        "ZeroShot": build_zeroshot_prompt,
        "FewShot": build_fewshot_prompt,
        "CoT": build_cot_prompt,
    }

    result_columns = [f"{strategy}_Result_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}" for strategy in prompt_strategies]
    correct_columns = [f"{strategy}_Correct_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}" for strategy in prompt_strategies]

    # Process both base and obfuscated questions
    for file_type, (output_csv, question_col) in [
        ("BASE", (output_csv_base, BASE_QUESTION_COL)),
        ("OBFUSCATED", (output_csv_obfuscated, OBFUSCATED_QUESTION_COL))
    ]:

        print(f"\n{'='*60}")
        print(f"🧭 Processing {file_type} Questions")
        print(f"{'='*60}")

        # Load existing results or create new DataFrame
        df_results = load_existing_results(output_csv, df_master, result_columns, correct_columns)

        # Find incomplete rows that need processing
        incomplete_rows = find_incomplete_rows(df_results, result_columns)

        if len(incomplete_rows) == 0:
            print(f"✅ All rows already completed in {output_csv}")
            continue

        print(f"🔄 Found {len(incomplete_rows)} incomplete rows to process")

        # If file doesn't exist, create it with headers
        if not os.path.exists(output_csv):
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(df_results.columns.tolist())

        # Process only incomplete rows
        processed_count = 0
        for idx in incomplete_rows:
            row = df_results.iloc[idx]
            question_text = str(row[question_col])

            if pd.isna(question_text) or question_text.strip() == "":
                print(f"\n  📝 Row {idx+1}: Skipping empty question")
                # Write skip results
                for strategy_name in prompt_strategies:
                    result_col_name = f"{strategy_name}_Result_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}"
                    correct_col_name = f"{strategy_name}_Correct_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}"
                    df_results.loc[idx, result_col_name] = "SKIPPED_EMPTY"
                    df_results.loc[idx, correct_col_name] = False

                # Save updated results
                df_results.to_csv(output_csv, index=False)
                processed_count += 1
                continue

            question_preview = (question_text[:80] + '...') if len(question_text) > 80 else question_text
            print(f"\n  📝 Row {idx+1}: {question_preview}")
            print(f"    💡 Ground truth: {row[GROUND_TRUTH_COL]}")

            # Process each prompt strategy for this row
            row_needs_processing = False
            for strategy_name in prompt_strategies:
                result_col_name = f"{strategy_name}_Result_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}"
                correct_col_name = f"{strategy_name}_Correct_{GEMINI_MODEL_NAME.replace('/', '_').replace('.', '_').replace('-', '_')}"
                ground_truth = row[GROUND_TRUTH_COL]

                # Check if this strategy already has a valid result
                current_value = df_results.loc[idx, result_col_name]
                if (not pd.isna(current_value) and
                    current_value != "" and
                    current_value is not None):

                    # Check if it's an error that should be retried
                    if isinstance(current_value, str):
                        current_str = str(current_value).strip()
                        if (current_str.startswith("ERROR_") or
                            current_str.startswith("NO_PATTERN_FOUND") or
                            "NO_PATTERN_FOUND" in current_str or
                            current_str == "SKIPPED_EMPTY"):
                            print(f"    🔄 {strategy_name} needs retry: {current_str[:50]}...")
                        else:
                            print(f"    ✅ {strategy_name} already completed: {current_value}")
                            continue
                    else:
                        print(f"    ✅ {strategy_name} already completed: {current_value}")
                        continue

                row_needs_processing = True
                print(f"    🎯 Running {strategy_name}...")
                try:
                    prompt_function = prompt_functions[strategy_name]
                    prompt_text = prompt_function(question_text)
                    prediction = call_gemini_api(prompt_text)
                    df_results.loc[idx, result_col_name] = prediction

                    # Check if prediction is correct
                    is_correct = answers_match(prediction, ground_truth)
                    df_results.loc[idx, correct_col_name] = is_correct

                    correctness_indicator = "✅" if is_correct else "❌"
                    print(f"       Result: {prediction} {correctness_indicator}")
                except Exception as e:
                    print(f"       Error: {e}")
                    df_results.loc[idx, result_col_name] = "ERROR_FUNCTION_CALL"
                    df_results.loc[idx, correct_col_name] = False

                time.sleep(1)  # Rate limiting

            # Save results immediately after each row (live saving)
            if row_needs_processing:
                try:
                    df_results.to_csv(output_csv, index=False)
                    print(f"       💾 Saved row {idx+1} results")
                except Exception as e:
                    print(f"       ❌ Error saving row: {e}")

            processed_count += 1

            # Progress checkpoint
            if processed_count % 10 == 0:
                print(f"    📊 Progress: {processed_count}/{len(incomplete_rows)} rows processed")

        print(f"\n✅ {file_type} processing complete!")
        print(f"   Final file: {output_csv}")

    print(f"\n🎉 Direction Sense evaluation completed successfully!")

    # Calculate and display accuracy results
    print(f"\n{'='*60}")
    print("📊 ACCURACY ANALYSIS")
    print(f"{'='*60}")

    calculate_and_display_accuracy(output_csv_base, output_csv_obfuscated, prompt_strategies, GEMINI_MODEL_NAME)

def calculate_and_display_accuracy(base_file: str, obfuscated_file: str, strategies: list, model_name: str):
    """Calculate and display accuracy results for both base and obfuscated questions"""

    def get_accuracy_metrics(df: pd.DataFrame, strategies: list, model_name: str):
        """Calculate accuracy metrics for a dataframe"""
        results = {}
        model_suffix = model_name.replace('/', '_').replace('.', '_').replace('-', '_')

        for strategy in strategies:
            correct_col = f"{strategy}_Correct_{model_suffix}"
            result_col = f"{strategy}_Result_{model_suffix}"

            if correct_col not in df.columns or result_col not in df.columns:
                continue

            total = len(df[df[result_col].notna()])
            correct = df[correct_col].sum()
            accuracy = correct / total if total > 0 else 0

            # Count different response types
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

    # Analyze base questions
    if os.path.exists(base_file):
        print(f"\n📊 BASE QUESTIONS ACCURACY")
        print("-" * 40)

        try:
            df_base = pd.read_csv(base_file)
            base_results = get_accuracy_metrics(df_base, strategies, model_name)

            for strategy, metrics in base_results.items():
                print(f"\n🎯 {strategy}:")
                print(f"   Total: {metrics['total']}, Correct: {metrics['correct']}")
                print(f"   Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
                print(f"   Valid accuracy: {metrics['valid_accuracy']:.3f} ({metrics['valid_accuracy']*100:.1f}%)")
                if metrics['error_responses'] > 0:
                    print(f"   Errors: {metrics['error_responses']}")

        except Exception as e:
            print(f"❌ Error analyzing base file: {e}")
            base_results = {}
    else:
        print(f"❌ Base file not found: {base_file}")
        base_results = {}

    # Analyze obfuscated questions
    if os.path.exists(obfuscated_file):
        print(f"\n📊 OBFUSCATED QUESTIONS ACCURACY")
        print("-" * 40)

        try:
            df_obfuscated = pd.read_csv(obfuscated_file)
            obf_results = get_accuracy_metrics(df_obfuscated, strategies, model_name)

            for strategy, metrics in obf_results.items():
                print(f"\n🎯 {strategy}:")
                print(f"   Total: {metrics['total']}, Correct: {metrics['correct']}")
                print(f"   Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
                print(f"   Valid accuracy: {metrics['valid_accuracy']:.3f} ({metrics['valid_accuracy']*100:.1f}%)")
                if metrics['error_responses'] > 0:
                    print(f"   Errors: {metrics['error_responses']}")

        except Exception as e:
            print(f"❌ Error analyzing obfuscated file: {e}")
            obf_results = {}
    else:
        print(f"❌ Obfuscated file not found: {obfuscated_file}")
        obf_results = {}

    # Comparison analysis
    if base_results and obf_results:
        print(f"\n📈 OBFUSCATION ROBUSTNESS ANALYSIS")
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

            print(f"\n🎯 OVERALL SUMMARY:")
            print(f"   Average Base Accuracy: {avg_base*100:.1f}%")
            print(f"   Average Obfuscated Accuracy: {avg_obf*100:.1f}%")
            print(f"   Obfuscation Impact: {overall_impact*100:+.1f} percentage points")

            if overall_impact > 0.1:
                print("   🔴 Significant degradation - Model struggles with obfuscation")
            elif overall_impact > 0.05:
                print("   🟡 Moderate degradation - Some obfuscation impact")
            elif abs(overall_impact) <= 0.05:
                print("   🟢 Robust performance - Model handles obfuscation well")
            else:
                print("   🔵 Improved performance - Obfuscation unexpectedly helped")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Direction Sense Gemini Evaluation Script")
        print("Processes base and obfuscated direction problems with Zero-shot, Few-shot, and CoT prompts.\n")
        print(f"Usage: python {os.path.basename(__file__)} <input_csv_path> <output_file_prefix>")
        print(f"Example: python {os.path.basename(__file__)} direction_sense.csv results")
        print(f"         This will create:")
        print(f"         - results_base_questions.csv")
        print(f"         - results_obfuscated_questions.csv")
        print("\nThe script will:")
        print("  1. Process both base and obfuscated question columns")
        print("  2. Apply Zero-shot, Few-shot, and CoT prompting strategies")
        print("  3. Extract answers using all 11 regex patterns")
        print("  4. Save results incrementally to avoid data loss")
        print("  5. Resume from incomplete runs automatically")
        sys.exit(1)

    input_file_arg = sys.argv[1]
    output_prefix_arg = sys.argv[2]

    print("🧭 Direction Sense Gemini Evaluation Script")
    print("=" * 60)

    main(input_file_arg, output_prefix_arg)
