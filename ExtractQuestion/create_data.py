import json
import os
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from src.ImproveRobustness.create_data.BaseDataLoader import BaseDataLoader


# -------------------------------
# Configuration Processing Function
# -------------------------------
def process_configuration(params):
    """Processes a single configuration (model, shots, dataset)."""
    model_name, dataset = params
    # Create a safe filename from model and dataset names
    model_short_name = model_name.split('/')[-1]
    dataset_safe_name = dataset.replace('/', '_').replace('.', '_')
    output_filename = f"{model_short_name}_{dataset_safe_name}.parquet"

    # Save individual dataset results
    output_path = os.path.join("..", "data", "raw_data", output_filename)
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"Skipping existing dataset: {output_path}")
        return None, 0, 0, 0
    data_loader = BaseDataLoader()

    try:
        df_partial = data_loader.load_and_process_data(
            model_name=model_name,
            datasets=[dataset],
            shots=None,
            max_samples=None
        )
    except Exception as e:
        print(f"Warning: Failed to load dataset '{dataset}' for model '{model_name}'")
        print(f"Error: {e}")
        return None, 0, 0, 0  # Skip this dataset

    if df_partial.empty:
        return None, 0, 0, 0  # Skip empty dataframes

    df_partial.to_parquet(output_path, index=False)
    print(f"Saved dataset results to: {output_path}")
    return df_partial, len(df_partial), len(df_partial), len(df_partial)


def immediate_error_callback(error, params):
    """Handles errors and prints useful debug info."""
    print("\n" + "=" * 50)
    print(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Failed Parameters: {json.dumps(params, indent=2)}")
    print(f"Error Message: {str(error)}")
    print("Traceback:")
    print(traceback.format_exc())
    print("=" * 50 + "\n")


def process_configuration_with_error_handling(params):
    """Wrapper to process a configuration and catch exceptions."""
    try:
        return process_configuration(params)
    except Exception as e:
        immediate_error_callback(e, params)
        return None


def run_configuration_analysis(model_name, num_processes=None, output_filepath="results.parquet"):
    """Runs parallelized configuration analysis for a given model."""
    if num_processes is None:
        num_processes = max(1, cpu_count() // 2)  # Use all but one CPU core

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    interesting_datasets = [
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa",
    ]

    subtasks = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    pro_subtasks = [
        "history",
        "law",
        "health",
        "physics",
        "business",
        "other",
        "philosophy",
        "psychology",
        "economics",
        "math",
        "biology",
        "chemistry",
        "computer_science",
        "engineering",
    ]

    # sample 2 mmul datasets
    # import random
    # mmlu_datasets = random.sample(subtasks, 2)
    # interesting_datasets.extend(["mmlu." + name for name in mmlu_datasets])
    #
    interesting_datasets.extend(["mmlu." + name for name in subtasks])
    interesting_datasets.extend(["mmlu_pro." + name for name in pro_subtasks])

    # Create a list of all parameter combinations
    params_list = [
        (model_name, dataset)
        for dataset in interesting_datasets
    ]

    # results = []
    total_raw_rows = 0
    total_processed_rows = 0
    total_unique_rows = 0

    print(f"\n🔄 Starting parallel processing with {num_processes} processes...")

    with Pool(processes=num_processes) as pool:
        # Process in parallel with a progress bar
        for res, raw_rows, processed_rows, unique_rows in tqdm(
                pool.imap_unordered(process_configuration_with_error_handling, params_list, chunksize=10),
                total=len(params_list),
                desc=f"Processing {model_name} Configurations"
        ):
            if res is not None:
                # results.append(res)
                total_raw_rows += raw_rows
                total_processed_rows += processed_rows
                total_unique_rows += unique_rows

    # if not results:
    #     print("⚠ No results to aggregate!")
    #     return

    # Concatenate all processed DataFrames
    # aggregated_df = pd.concat(results, ignore_index=True)
    # print(f"\n✅ Aggregated results shape: {aggregated_df.shape}")

    # Save combined results as Parquet (keeping this for backward compatibility)
    # aggregated_df.to_parquet(output_filepath, index=False)
    print(f"\n📂 Aggregated results saved to: {output_filepath}")

    # Print dataset size statistics
    print("\n===== 📊 Dataset Size Summary =====")
    print(f"Total raw rows: {total_raw_rows}")
    print(f"Total processed rows: {total_processed_rows}")
    print(f"Total unique rows: {total_unique_rows}")
    print("===================================")


if __name__ == "__main__":
    models_to_evaluate = [
        # 'meta-llama/Llama-3.2-1B-Instruct',
        # 'allenai/OLMoE-1B-7B-0924-Instruct',
        # 'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        # 'mistralai/Mistral-7B-Instruct-v0.3',
    ]

    base_folder_name = "data"
    os.makedirs(base_folder_name, exist_ok=True)

    for model in models_to_evaluate:
        output_filepath = os.path.join(base_folder_name, f"results_{model.split('/')[1]}.parquet")
        run_configuration_analysis(model, output_filepath=output_filepath, num_processes=1)
