
import logging
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
# Make sure this import path is correct
from DataLoader import DataLoader
# Make sure this import path is correct and InstanceLoader is compatible
from get_example_from_row import InstanceLoader
from tqdm import tqdm

THRESHOLD = 0.1
log_file = "analysis_log.txt"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('fsspec').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for analysis pipeline with immutable attributes."""
    repo_id: str       # Added repository ID
    model_name: str    # Full model name
    language: str      # Added language
    shots_selected: int # Number of shots
    benchmark_file: str # Benchmark file name

    @property
    def output_path(self) -> Path:
        """Generate standardized output path for results, including repo and language."""
        base_dir = Path("../app/results_local")
        repo_part = self.repo_id.replace('/', '_')
        model_folder_name = self.model_name.split('/')[-1]
        benchmark_part = Path(self.benchmark_file).stem
        return base_dir / repo_part / self.language / f"Shots_{self.shots_selected}" / \
            model_folder_name.replace('/', '_') / benchmark_part / f'low_performance_questions_{THRESHOLD}.parquet'


class AnswerMapper:
    """Maps multiple choice answer strings to valid positions (1-4)."""

    POSITION_MAPPINGS = {
        'greek': "αβγδεζηθικ",
        'keyboard': "!@#$%^₪*)(",
        'capitals': "ABCDEFGHIJ",
        'lowercase': "abcdefghij",
        'numbers': [str(i + 1) for i in range(10)],
        'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    }

    VALID_POSITIONS: Set[int] = {1, 2, 3, 4} # Only interested in positions 1-4

    @classmethod
    def get_position(cls, answer: str) -> Optional[int]:
        """
        Maps an answer to a valid multiple choice position (1-4) by extracting the leading label.
        Adds debug logging for input and output.

        Args:
            answer: The answer string to map (e.g., "A.", "1)", "I. Full text", "Answer: B").

        Returns:
            Optional[int]: Position (1-4) if valid, None otherwise.
        """
        logging.debug(f"AnswerMapper: Attempting to map answer: '{answer}' (Type: {type(answer)})")
        if not isinstance(answer, str):
            logging.debug(f"AnswerMapper: Input is not a string ({type(answer)}). Returning None.")
            return None

        cleaned_answer = answer.strip()
        if not cleaned_answer:
             logging.debug("AnswerMapper: Input string is empty after strip. Returning None.")
             return None

        # --- New Logic: Attempt to extract a leading label ---
        # Look for common separators after the label (., ), :)
        label_end_index = -1
        for sep in ['.', ')', ':']:
            idx = cleaned_answer.find(sep)
            if idx != -1 and (label_end_index == -1 or idx < label_end_index):
                label_end_index = idx

        prefix_to_map = cleaned_answer # Default to the whole string if no separator found

        if label_end_index != -1:
            # Extract the potential label before the separator
            potential_label = cleaned_answer[:label_end_index].strip()
            # Check if the character *after* the separator is a space (common in "Label. Text")
            # This helps distinguish "1. Text" from "1.23" or "I.V. drip"
            if len(cleaned_answer) > label_end_index + 1 and cleaned_answer[label_end_index + 1].isspace():
                 if potential_label: # Ensure the extracted label isn't empty
                     prefix_to_map = potential_label
                     logging.debug(f"AnswerMapper: Extracted potential label '{prefix_to_map}' before separator '{cleaned_answer[label_end_index]}'.")
                 else:
                     # Separator found early, but no label before it (e.g., ".  Text")
                     logging.debug(f"AnswerMapper: Found separator '{cleaned_answer[label_end_index]}', but no label before it.")
                     # Fall through to try mapping the whole string or other patterns
            # else:
                 # Separator found, but not followed by space (e.g., "1.23", "A.B."), treat as part of the text initially
                 # The logic will now default to prefix_to_map = cleaned_answer and try matching the whole thing or handle other patterns

        # --- Keep existing cleaning for known prefixes like "Answer: " if no leading label was clearly extracted ---
        # This might catch cases like "Answer: A" or "The answer is II"
        lower_prefix_to_map = prefix_to_map.lower()
        if lower_prefix_to_map.startswith("answer: "):
            prefix_to_map = prefix_to_map[len("answer: "):].strip()
            logging.debug(f"AnswerMapper: Cleaned 'Answer: ' prefix. New prefix to map: '{prefix_to_map}'")
        elif lower_prefix_to_map.startswith("the answer is "):
             prefix_to_map = prefix_to_map[len("the answer is "):].strip()
             logging.debug(f"AnswerMapper: Cleaned 'The answer is ' prefix. New prefix to map: '{prefix_to_map}'")
        elif lower_prefix_to_map.startswith("the correct answer is "):
             prefix_to_map = prefix_to_map[len("the correct answer is "):].strip()
             logging.debug(f"AnswerMapper: Cleaned 'The correct answer is ' prefix. New prefix to map: '{prefix_to_map}'")
         # Handle single character answers potentially surrounded by quotes/backticks after other cleaning
        if len(prefix_to_map) == 3 and prefix_to_map[0] in ["'", '"', '`'] and prefix_to_map[2] in ["'", '"', '`']:
             prefix_to_map = prefix_to_map[1]
             logging.debug(f"AnswerMapper: Cleaned quotes/backticks around single char. New prefix to map: '{prefix_to_map}'")


        logging.debug(f"AnswerMapper: Final prefix to map: '{prefix_to_map}'")

        try:
            if not prefix_to_map:
                logging.debug("AnswerMapper: Final prefix to map is empty. Returning None.")
                return None

            # --- Existing Mapping Logic (using the extracted/cleaned prefix) ---
            for mapping_name, mapping in cls.POSITION_MAPPINGS.items():
                 logging.debug(f"AnswerMapper: Checking mapping '{mapping_name}' against '{prefix_to_map}'")
                 # Handle list mappings (like numbers or Roman) vs string mappings (like ABC)
                 if isinstance(mapping, list):
                     # Check for exact match in list
                     if prefix_to_map in mapping:
                         position = mapping.index(prefix_to_map) + 1
                         logging.debug(f"AnswerMapper: Found exact match in list mapping '{mapping_name}'. Position: {position}. Valid: {position in cls.VALID_POSITIONS}")
                         return position if position in cls.VALID_POSITIONS else None # Return None on invalid position index
                 elif isinstance(mapping, str):
                     # Check if the prefix is a single character present in the string mapping
                     if len(prefix_to_map) == 1 and prefix_to_map in mapping:
                          position = mapping.index(prefix_to_map) + 1
                          logging.debug(f"AnswerMapper: Found single char match in string mapping '{mapping_name}'. Position: {position}. Valid: {position in cls.VALID_POSITIONS}")
                          return position if position in cls.VALID_POSITIONS else None


            # Check if the prefix itself is a valid single character/number position (handles cases like just "A" or "1")
            if prefix_to_map in "ABCD":
                 position = ord(prefix_to_map) - ord('A') + 1
                 logging.debug(f"AnswerMapper: Found as direct Capital letter. Position: {position}. Valid: {position in cls.VALID_POSITIONS}")
                 return position if position in cls.VALID_POSITIONS else None
            if prefix_to_map in "abcd":
                 position = ord(prefix_to_map) - ord('a') + 1
                 logging.debug(f"AnswerMapper: Found as direct Lowercase letter. Position: {position}. Valid: {position in cls.VALID_POSITIONS}")
                 return position if position in cls.VALID_POSITIONS else None
            if prefix_to_map in "1234":
                 position = int(prefix_to_map)
                 logging.debug(f"AnswerMapper: Found as direct Number. Position: {position}. Valid: {position in cls.VALID_POSITIONS}")
                 return position if position in cls.VALID_POSITIONS else None

            logging.debug(f"AnswerMapper: Final prefix '{prefix_to_map}' did not match any valid mapping. Returning None.")
            return None # Return None if no mapping found

        except (AttributeError, IndexError, ValueError) as e:
             # Log the specific error for parsing issues
             logging.debug(f"AnswerMapper: Error processing prefix '{prefix_to_map}' from original answer '{answer}': {e}", exc_info=True) # Log exception details
             logging.warning(f"Could not parse position from answer: '{answer}' (Prefix was '{prefix_to_map}')")
             return None


# Keep the rest of the main.py code as is, including PerformanceAnalyzer and run_analysis_pipeline
# ... (rest of the code remains the same)

class PerformanceAnalyzer:
    # ... (find_low_performers method remains the same)
    """Analyzes model performance on questions."""

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold

    def find_low_performers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f"Starting find_low_performers. Input df shape: {df.shape}")

        if df.empty:
            logging.debug("Input df is empty for find_low_performers. Returning empty DataFrame.")
            return pd.DataFrame()

        if 'sample_index' not in df.columns or 'score' not in df.columns:
             logging.error("DataFrame missing 'sample_index' or 'score' column for performance analysis. Skipping.")
             return pd.DataFrame()

        logging.debug("Calculating accuracy stats per sample_index.")
        accuracy_stats = (
            df.groupby('sample_index')
            .agg({
                'score': ['sum', 'count']
            })
            .reset_index()
        )

        accuracy_stats.columns = ['sample_index', 'correct_count', 'total_count']
        logging.debug(f"Accuracy stats calculated. Shape: {accuracy_stats.shape}")
        # logging.debug(f"Sample accuracy_stats:\n{accuracy_stats.head()}") # Too verbose


        accuracy_stats['accuracy'] = (
                accuracy_stats['correct_count'] /
                accuracy_stats['total_count'].replace(0, pd.NA)
        ).round(3)


        logging.debug(f"Filtering for accuracy < {self.threshold}. Initial rows: {len(accuracy_stats)}")
        low_performers_stats = accuracy_stats[accuracy_stats['accuracy'] < self.threshold].dropna(subset=['accuracy'])
        logging.debug(f"Filtered low_performers_stats rows: {len(low_performers_stats)}")
        if not low_performers_stats.empty:
             logging.debug(f"Sample low_performers_stats:\n{low_performers_stats.head()}")


        if not low_performers_stats.empty:
            logging.info(f"Found {len(low_performers_stats)} low-performing samples (accuracy < {self.threshold:.1%})")
            for _, row in low_performers_stats.head(10).iterrows():
                sample_index = row['sample_index']
                correct = row['correct_count']
                total = row['total_count']
                accuracy = row['accuracy']

                logging.info(
                    f"  Sample {sample_index}: Accuracy {accuracy:.1%} ({correct}/{total} correct)"
                )
            if len(low_performers_stats) > 10:
                 logging.info(f"  ... and {len(low_performers_stats) - 10} more low-performing samples.")
        else:
             logging.info("No low-performing samples found.")

        logging.debug("Filtering original DataFrame for low_performers_stats sample_indices.")
        df_low_performers = df[df['sample_index'].isin(low_performers_stats['sample_index'])].copy()
        logging.debug(f"Filtered df_low_performers shape: {df_low_performers.shape}")

        if not df_low_performers.empty:
            logging.debug(f"Sample df_low_performers before mapping:\n{df_low_performers[['sample_index', 'closest_answer', 'score']].head()}") # Log relevant columns
        else:
             logging.debug("df_low_performers is empty before mapping.")


        return df_low_performers

def process_configuration(config: AnalysisConfig) -> Dict[str, any]:
    # ... (rest of process_configuration remains the same)
    """
    Processes a single configuration of repository, model, language, shots, and benchmark file.
    Adds extensive debug logging.

    Args:
        config: Configuration parameters for analysis

    Returns:
        Dict containing processing status and metrics
    """
    current_config_str = f"{config.repo_id}/{config.model_name.split('/')[-1]}/{config.language}/{config.shots_selected}_shot/{config.benchmark_file}"
    try:
        print(f"\n--- Starting processing for {current_config_str} ---")
        logging.info(f"Starting processing for config: {config}")

        data_loader = DataLoader(repo_id=config.repo_id)
        logging.debug(f"DataLoader instantiated for repo_id: {config.repo_id}")

        logging.debug(f"Loading and processing data with model: {config.model_name}, language: {config.language}, shots: {config.shots_selected}, benchmark: {config.benchmark_file}")
        df = data_loader.load_and_process_data(
            model_name=config.model_name,
            language=config.language,
            shots=config.shots_selected,
            benchmark_files=[config.benchmark_file],
            drop_columns=['cumulative_logprob', 'generated_text', 'ground_truth']
        )
        logging.debug(f"Initial data load and process completed. DataFrame shape: {df.shape}")
        if not df.empty:
            logging.debug(f"Sample of initial DataFrame (first 5 rows):\n{df.head()}")


        if df.empty:
            logging.info(f"No data found or remaining after initial processing for config: {config}. Skipping further analysis.")
            return {"status": "empty", "config": config}

        initial_rows_before_filter = len(df)
        # Apply any filters needed BEFORE finding low performers or mapping
        # This filter logic might need adjustment based on requirements for 5-shot
        logging.debug(f"Applying 5-shot choices_order filter. Initial rows: {initial_rows_before_filter}")
        if 'dimensions_5: shots' in df.columns and 'dimensions_3: choices_order' in df.columns:
             mask = (df['dimensions_5: shots'] != 5) | (~df['dimensions_3: choices_order'].isin(["correct_first", "correct_last"]))
             df = df[mask].copy()
             if len(df) < initial_rows_before_filter:
                  logging.info(f"Filtered out {initial_rows_before_filter - len(df)} rows based on 5-shot choices_order filter. Remaining rows: {len(df)}")
             else:
                 logging.debug("No rows filtered by 5-shot choices_order filter.")
        else:
             logging.warning("Skipping 5-shot choices_order filter: Required columns ('dimensions_5: shots' or 'dimensions_3: choices_order') not found in DataFrame.")

        logging.debug(f"DataFrame shape after custom filters: {df.shape}")
        if not df.empty:
            logging.debug(f"Sample of DataFrame after custom filters:\n{df.head()}")


        if df.empty:
             logging.info(f"DataFrame is empty after custom filters for config: {config}. Skipping further analysis.")
             return {"status": "empty_after_filters", "config": config}


        if not df.empty:
            if 'sample_index' in df.columns and 'score' in df.columns:
                logging.debug("Calculating question counts and scores.")
                question_counts = df.groupby("sample_index").size()
                correct_counts = df.groupby("sample_index")["score"].sum()
                logging.info(f"Data summary after filters: {len(df)} rows, {len(question_counts)} unique questions.")
                sample_indices_to_log = question_counts.index.tolist()[:min(5, len(question_counts))]
                if sample_indices_to_log:
                    logging.info("Sample question configurations and scores:")
                    for question_id in sample_indices_to_log:
                         total_configs = question_counts.get(question_id, 0)
                         correct_sum = correct_counts.get(question_id, 0)
                         logging.info(
                            f"  Question {question_id}: {total_configs} configurations, "
                            f"{correct_sum} total score"
                        )
            else:
                 logging.warning("Skipping question summary logging: 'sample_index' or 'score' column not found in DataFrame.")


        analyzer = PerformanceAnalyzer()
        logging.debug(f"Instantiated PerformanceAnalyzer with threshold: {analyzer.threshold}")
        df_low_performers = analyzer.find_low_performers(df)

        logging.debug(f"find_low_performers completed. df_low_performers shape: {df_low_performers.shape}")
        if not df_low_performers.empty:
            logging.debug(f"Sample df_low_performers returned by find_low_performers:\n{df_low_performers[['sample_index', 'closest_answer', 'score']].head()}")
        else:
             logging.debug("df_low_performers returned by find_low_performers is empty.")


        if df_low_performers.empty:
            logging.info(f"No low-performing questions found for config: {config}")
            return {"status": "no_low_performers", "config": config}

        logging.debug("Attempting to map answers to positions.")
        if 'closest_answer' in df_low_performers.columns:
            logging.debug(f"'closest_answer' column found. Mapping {len(df_low_performers)} answers.")

            if not df_low_performers.empty:
                 logging.debug(f"Sample 'closest_answer' values before mapping: {df_low_performers['closest_answer'].dropna().sample(min(10, len(df_low_performers['closest_answer'].dropna()))) if not df_low_performers['closest_answer'].dropna().empty else []}")


            df_low_performers['chosen_position'] = df_low_performers['closest_answer'].apply(AnswerMapper.get_position)

            logging.debug(f"Mapping applied. df_low_performers shape: {df_low_performers.shape}")

            if not df_low_performers.empty:
                 non_null_positions = df_low_performers['chosen_position'].dropna()
                 logging.debug(f"Number of non-null 'chosen_position' values after mapping: {len(non_null_positions)}")
                 if not non_null_positions.empty:
                     logging.debug(f"Sample non-null 'chosen_position' values: {non_null_positions.sample(min(5, len(non_null_positions))).tolist()}")
                 null_positions = df_low_performers[df_low_performers['chosen_position'].isnull()]
                 if not null_positions.empty:
                      logging.debug(f"Number of null 'chosen_position' values after mapping: {len(null_positions)}")
                      logging.debug(f"Sample rows with null 'chosen_position' after mapping:\n{null_positions[['sample_index', 'closest_answer', 'chosen_position', 'score']].head()}")


        else:
             logging.error("'closest_answer' column not found in df_low_performers. Cannot map positions. Adding 'chosen_position' column with None.")
             df_low_performers['chosen_position'] = None
             return {
                "status": "mapping_skipped",
                "config": config,
                "error": "'closest_answer' column missing for mapping.",
                "initial_rows": len(df),
                "low_performer_rows": len(df_low_performers)
            }


        if df_low_performers.empty or df_low_performers['chosen_position'].isnull().all():
            logging.warning(f"df_low_performers is empty or all 'chosen_position' values are null after mapping attempt for config: {config}")
            # Decide on the status based on whether 'closest_answer' was missing or mapping failed for all rows
            status_after_mapping = "invalid_mappings"
            return {
                "status": status_after_mapping,
                "config": config,
                "initial_rows": len(df),
                "final_rows": len(df_low_performers) # Log how many rows *were* in df_low_performers before check
            }


        output_path = config.output_path
        logging.debug(f"Attempting to save results to: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Only save rows where chosen_position is NOT null, as Graphs.py will drop them anyway
            df_to_save = df_low_performers.dropna(subset=['chosen_position']).copy()
            logging.debug(f"Saving {len(df_to_save)} rows (non-null chosen_position) to parquet.")
            df_to_save.to_parquet(output_path, index=False)
            logging.info(f"Successfully saved {len(df_to_save)} rows to {output_path}")
        except Exception as save_error:
            logging.error(f"Error saving results to {output_path}: {save_error}")
            return {
                "status": "save_error",
                "config": config,
                "error": str(save_error),
                "final_rows": len(df_low_performers) # Report rows before dropping nulls for saving
            }


        return {
            "status": "success",
            "config": config,
            "initial_rows": len(df),
            "low_performer_rows": len(df_low_performers) # Report rows before dropping nulls for saving
        }

    except Exception as e:
        logging.exception(f"An unexpected error occurred processing config: {current_config_str}")
        return {
            "status": "error",
            "config": config,
            "error": str(e)
        }


def run_analysis_pipeline(num_processes: int = 3) -> None:
    # ... (run_analysis_pipeline function remains the same)
    """
    Runs the complete analysis pipeline with parallel processing.
    Iterates through specified configurations of repo, model, language, shots, and benchmark file.

    Args:
        num_processes: Number of worker processes to use
    """
    # Set the logging level back to INFO for the overall summary if desired
    # logging.getLogger().setLevel(logging.INFO)


    # Define the configurations to run
    repo_ids = [
        "nlphuji/DOVE_Lite",
        # "nlphuji/DOVE",
    ]
    languages = ["en"]
    models = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct'
    ]
    shots = [0, 5]

    benchmark_files = [
        # "mmlu.anatomy.parquet",
        # "mmlu.business_ethics.parquet",
        # "mmlu.clinical_knowledge.parquet",
        # # Add other specific files you saw issues with
        # "mmlu.medical_genetics.parquet",
        # "mmlu.miscellaneous.parquet",
        # "mmlu.nutrition.parquet",
        # "mmlu.philosophy.parquet",
        # "mmlu.prehistory.parquet",
        # "mmlu.professional_accounting.parquet",
        # "mmlu.professional_medicine.parquet",
        # "mmlu.professional_psychology.parquet",

        # Add back others once debugging is successful
        "ai2_arc.arc_challenge.parquet",
        "ai2_arc.arc_easy.parquet",
        "hellaswag.parquet",
        "openbook_qa.parquet",
        "social_iqa.parquet",
        "mmlu.abstract_algebra.parquet",
        "mmlu.astronomy.parquet",
        "mmlu.college_biology.parquet",
        "mmlu.college_chemistry.parquet",
        "mmlu.college_computer_science.parquet",
        "mmlu.college_mathematics.parquet",
        "mmlu.college_physics.parquet",
        "mmlu.sociology.parquet",
        "mmlu.world_religions.parquet",
    ]


    configs = [
        AnalysisConfig(repo_id, model, language, shots_val, benchmark_file)
        for repo_id in repo_ids
        for model in models
        for language in languages
        for shots_val in shots
        for benchmark_file in benchmark_files
    ]

    logging.info(f"Generated {len(configs)} configurations to process.")
    print(f"Generated {len(configs)} configurations to process.")


    processed_count = 0
    success_count = 0
    empty_count = 0
    no_low_performers_count = 0
    invalid_mappings_count = 0
    mapping_skipped_count = 0
    error_count = 0
    save_error_count = 0
    empty_after_filters_count = 0 # New status for empty after custom filters

    run_sequentially = True

    results = []
    if run_sequentially:
         print("Running configurations sequentially for easier debugging...")
         for config in tqdm(configs, desc="Processing configurations sequentially"):
             results.append(process_configuration(config))
    else:
        print(f"Running configurations with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_configuration, configs),
                total=len(configs),
                desc="Processing configurations"
            ))

    print("\n--- Analysis Pipeline Summary ---")
    logging.info("\n--- Analysis Pipeline Summary ---")

    for res in results:
        processed_count += 1
        status = res.get("status", "unknown")
        config = res.get("config", "N/A")
        config_str_summary = f"{config.repo_id.replace('/', '_')}/{config.model_name.split('/')[-1]}/{config.language}/Shots_{config.shots_selected}/{Path(config.benchmark_file).stem}"


        if status == "success":
            success_count += 1
            logging.info(f"SUCCESS: {config_str_summary} (Rows: Initial {res['initial_rows']}, Saved Low Performers {res['low_performer_rows']})") # Updated log message
        elif status == "empty":
            empty_count += 1
            logging.warning(f"EMPTY DATA: {config_str_summary} (No data loaded initially)")
        elif status == "empty_after_filters": # Log the new status
             empty_after_filters_count += 1
             logging.warning(f"EMPTY AFTER FILTERS: {config_str_summary}")
        elif status == "no_low_performers":
            no_low_performers_count += 1
            logging.info(f"NO LOW PERFORMERS: {config_str_summary}")
        elif status == "invalid_mappings":
             invalid_mappings_count += 1
             logging.warning(f"INVALID MAPPINGS (post-mapping empty/all null): {config_str_summary} (Initial rows: {res.get('initial_rows', 'N/A')}, Final rows before saving: {res.get('final_rows', 'N/A')})") # Updated log message
        elif status == "mapping_skipped":
             mapping_skipped_count += 1
             logging.warning(f"MAPPING SKIPPED ('closest_answer' missing): {config_str_summary} (Low Performer rows: {res.get('low_performer_rows', 'N/A')})")
        elif status == "save_error":
             save_error_count += 1
             logging.error(f"SAVE ERROR: {config_str_summary} - {res.get('error', 'N/A')}")
        elif status == "error":
            error_count += 1
            logging.error(f"ERROR: {config_str_summary} - {res.get('error', 'N/A')}")
        else:
            logging.error(f"UNKNOWN STATUS for config: {config_str_summary} - Result: {res}")

    print(f"\nTotal Configurations Attempted: {processed_count}")
    print(f"Successful Processes: {success_count}")
    print(f"Configs with Empty Data (Initial Load): {empty_count}")
    print(f"Configs with Empty Data (After Filters): {empty_after_filters_count}") # Report the new status
    print(f"Configs with No Low Performers: {no_low_performers_count}")
    print(f"Configs with Invalid Mappings (post-mapping empty/all null): {invalid_mappings_count}")
    print(f"Configs where Mapping Skipped ('closest_answer' missing): {mapping_skipped_count}")
    print(f"Configs with Save Errors: {save_error_count}")
    print(f"Configs with Unexpected Errors: {error_count}")
    print("--- Summary End ---")

    logging.info(f"\nTotal Configurations Attempted: {processed_count}")
    logging.info(f"Successful Processes: {success_count}")
    logging.info(f"Configs with Empty Data (Initial Load): {empty_count}")
    logging.info(f"Configs with Empty Data (After Filters): {empty_after_filters_count}")
    logging.info(f"Configs with No Low Performers: {no_low_performers_count}")
    logging.info(f"Configs with Invalid Mappings (post-mapping empty/all null): {invalid_mappings_count}")
    logging.info(f"Configs where Mapping Skipped ('closest_answer' missing): {mapping_skipped_count}")
    logging.info(f"Configs with Save Errors: {save_error_count}")
    logging.info(f"Configs with Unexpected Errors: {error_count}")
    logging.info("--- Summary End ---")


if __name__ == "__main__":
    # Keep batch size small for debugging if needed
    run_analysis_pipeline(num_processes=1) # Set to 1 for sequential debugging
