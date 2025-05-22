# data_loader.py
import time
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa # Import pyarrow to use pa.array()
from datasets import load_dataset, Dataset

# The HfFileSystem approach is no longer needed as we construct the exact path
# from huggingface_hub import HfFileSystem

# repo_name = "eliyahabba/llm-evaluation-analysis-split" # Old repo name

class DataLoader:
    def __init__(self, repo_id: str, split: str = "train", batch_size: int = 10000):
        """
        Initializes the DataLoader with dataset details.

        Args:
            repo_id (str): Name of the dataset repository to load from HuggingFace
                           (e.g., "nlphuji/DOVE" or "nlphuji/DOVE_Lite").
            split (str): Dataset split to use (usually "train" for these benchmarks).
            batch_size (int): Number of samples per batch (used in batched map if needed,
                              but extract_data uses pyarrow which is often faster).
        """
        self.dataset: Optional[Dataset] = None # Store the loaded dataset
        self.repo_id = repo_id
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(self,
                              model_name: str, # This will be the full name like "allenai/OLMoE-1B-7B-0924-Instruct"
                              language: str,
                              shots: int,
                              benchmark_files: List[str],
                              template: Optional[str] = None,
                              separator: Optional[str] = None,
                              enumerator: Optional[str] = None,
                              choices_order: Optional[str] = None,
                              instruction_phrasing_text: Optional[str] = None, # Added instruction phrasing filter
                              drop_columns: Optional[List[str]] = None # List of columns to drop after loading
                              ) -> pd.DataFrame:
        """
        Loads data for specific model, language, shots, and benchmark files,
        then filters and processes it.

        Args:
            model_name (str): The full model name (e.g., "allenai/OLMoE-1B-7B-0924-Instruct").
            language (str): The language code (e.g., "en").
            shots (int): The number of shots (e.g., 0, 5).
            benchmark_files (List[str]): List of benchmark file names (e.g., ["mmlu.global_facts.parquet"]).
            template (Optional[str]): Optional template filter.
            separator (Optional[str]): Optional separator filter.
            enumerator (Optional[str]): Optional enumerator filter.
            choices_order (Optional[str]): Optional choices order filter.
            instruction_phrasing_text (Optional[str]): Optional instruction phrasing filter.
            drop_columns (Optional[List[str]]): Optional list of column names to drop.

        Returns:
            pd.DataFrame: Filtered and processed data as a pandas DataFrame.
        """
        start_time = time.time()
        print(f"Loading and processing data for model: {model_name}, language: {language}, shots: {shots}, benchmarks: {benchmark_files}")

        # Step 1: Load data using the new path structure
        self._load_data_from_structure(model_name, language, shots, benchmark_files)

        if self.dataset is None or len(self.dataset) == 0:
            print(f"No data found for model {model_name}, language {language}, shots {shots}, benchmarks {benchmark_files}")
            return pd.DataFrame()

        # Step 2: Drop specified columns early
        if drop_columns:
             columns_to_drop_exist = [col for col in drop_columns if col in self.dataset.column_names]
             if columns_to_drop_exist:
                 self.dataset = self.dataset.remove_columns(columns_to_drop_exist)
                 # print(f"Dropped columns: {columns_to_drop_exist}")
             # else:
                 # print("No specified columns found to drop.")

        # Step 3: Extract/filter data based on content (model, shots, etc. already loaded, but good to confirm and apply optional filters)
        # Note: The load_data_from_structure loads files that SHOULD match model, lang, shots,
        # but the extract_data step is still useful for optional filters (template, separator, etc.)
        # and as a safeguard.
        full_results = self.extract_data(
            model_name=model_name, # Keep these filters even though file path implies them
            shots=shots,         # as the loaded parquet might contain other data
            datasets=benchmark_files, # Filter by benchmark filenames (from the list)
            template=template,
            separator=separator,
            enumerator=enumerator,
            choices_order=choices_order,
            instruction_phrasing_text=instruction_phrasing_text # Pass instruction phrasing filter
        )

        if full_results.empty:
             print(f"No data remaining after content filtering for model {model_name}, language {language}, shots {shots}, benchmarks {benchmark_files} with specified content filters.")
             return pd.DataFrame()

        # Step 4: Remove duplicates
        clean_df = self.remove_duplicates(full_results)

        load_time = time.time()
        print(f"The size of the data after loading, filtering and removing duplicates is: {len(clean_df)}")
        print(f"Total data loading and initial processing completed in {load_time - start_time:.2f} seconds")
        return clean_df

    def _load_data_from_structure(self,
                                  model_name: str, # This is the full model name (e.g., "allenai/OLMoE-1B-7B-0924-Instruct")
                                  language: str,
                                  shots: int,
                                  benchmark_files: List[str]):
        """
        Constructs file paths based on the DOVE repository structure and loads data.
        Corrects for model folder name not including organization prefix.
        """
        data_files = []
        # Extract just the model name part for the folder structure path
        model_folder_name = model_name.split('/')[-1] # <--- EXTRACT FOLDER NAME HERE

        # Construct paths for each benchmark file
        for benchmark_file in benchmark_files:
            # Ensure the benchmark file has the .parquet extension if it doesn't already
            if not benchmark_file.endswith('.parquet'):
                # This might be unnecessary if benchmark_files list already contains .parquet
                # Based on your main.py, it seems they already have .parquet
                pass # No change needed if already .parquet

            # Construct the file path using the extracted model folder name
            file_path = f"{model_folder_name}/{language}/{shots}_shot/{benchmark_file}" # <--- USE EXTRACTED NAME

            data_files.append(file_path)
            print(f"Attempting to load file: {file_path}") # Optional: for debugging paths

        if not data_files:
            print("No benchmark files specified.")
            self.dataset = None
            return

        try:
            # Load the dataset from the specified files
            # USING token=True HERE TO AUTHENTICATE
            self.dataset = load_dataset(
                self.repo_id,
                data_files=data_files,
                split=self.split,
                cache_dir=None, # Use default cache or specify one
                token=True      # <--- ADD THIS LINE
            )
            print(f"Successfully loaded {len(self.dataset)} examples from specified files.")

        except Exception as e:
            print(f"Error loading dataset from {self.repo_id} with data_files {data_files}: {e}")
            self.dataset = None


    # Using extract_data with pyarrow for filtering is generally efficient
    # Keeping extract_data2 commented out unless needed for specific cases
    # def extract_data2(...): # Keep the original extract_data2 logic if needed later


    def extract_data(
            self,
            model_name: str, # This is the full model name used for filtering the 'model' column
            shots: int,
            datasets: List[str], # Expecting a list of benchmark filenames here
            template: Optional[str] = None,
            separator: Optional[str] = None,
            enumerator: Optional[str] = None,
            choices_order: Optional[str] = None,
            instruction_phrasing_text: Optional[str] = None # Added instruction phrasing filter
    ) -> pd.DataFrame:
        """
        Filters the loaded dataset using PyArrow based on specified column values.

        Args:
            model_name: Name of the model to filter by (full name).
            shots: Number of shots to filter by.
            datasets: List of benchmark filenames (used to filter the 'dataset' column).
            template: Optional template filter.
            separator: Optional separator filter.
            enumerator: Optional enumerator filter.
            choices_order: Optional choices order filter.
            instruction_phrasing_text: Optional instruction phrasing filter.

        Returns:
            pd.DataFrame: Filtered dataset as a pandas DataFrame.
        """
        if self.dataset is None:
            print("No dataset loaded to extract data from.")
            return pd.DataFrame()

        try:
            print("Starting data filtering process using PyArrow...")
            arrow_table = self.dataset.data.table

            conditions = [
                # Filter by the full model name as it appears in the 'model' column
                pc.equal(arrow_table['model'], model_name),
                # Filter by the 'dimensions_5: shots' column
                pc.equal(arrow_table['dimensions_5: shots'], shots) # <--- CORRECTED COLUMN NAME
            ]

            # Filter by dataset name (which corresponds to the benchmark file name without .parquet)
            # We assume the 'dataset' column in the parquet contains the benchmark filename stem.
            # Need to remove .parquet for comparison if benchmark_files includes it.
            dataset_names_to_filter = [Path(f).stem for f in datasets]
            # Use pa.array() instead of pc.array()
            conditions.append(pc.is_in(arrow_table['dataset'], pa.array(dataset_names_to_filter))) # <--- CORRECTED HERE


            optional_filters = {
                'dimensions_1: enumerator': enumerator, # <--- CORRECTED COLUMN NAMES
                'dimensions_2: separator': separator,   # <--- CORRECTED COLUMN NAMES
                'dimensions_3: choices_order': choices_order, # <--- CORRECTED COLUMN NAMES
                'dimensions_4: instruction_phrasing_text': instruction_phrasing_text # <--- ADDED FILTER
            }

            for column, value in optional_filters.items():
                # Check if the column exists before adding the condition
                if value is not None and column in arrow_table.column_names:
                     conditions.append(pc.equal(arrow_table[column], value))
                 # elif value is not None:
                 #     print(f"Warning: Filter column '{column}' not found in dataset.")


            # Combine all conditions using AND
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = pc.and_(combined_condition, condition)

            # Apply the filter
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()

            print(f"Filtering results: {len(df_filtered)} rows remaining.")
            if not df_filtered.empty and 'model' in df_filtered.columns:
                 print("\nModel distribution in filtered data:")
                 print(df_filtered['model'].value_counts())

            return df_filtered

        except Exception as e:
            # Catch potential errors during PyArrow operations (e.g., column not found)
            print(f"Error during data filtering with PyArrow: {str(e)}")
            # Return empty DataFrame or re-raise, depending on desired error handling
            # Returning empty dataframe allows the pipeline to continue with other configs
            return pd.DataFrame()


    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates from the DataFrame based on 'evaluation_id'.
        """
        if df.empty:
            return df
        # Assuming 'evaluation_id' is the correct column for identifying unique instances
        # print(f"Initial rows before duplicate removal: {len(df)}")
        if 'evaluation_id' in df.columns:
            df_unique = df.drop_duplicates(subset=['evaluation_id'])
            # print(f"Rows after removing duplicates: {len(df_unique)}")
            return df_unique
        else:
            print("Warning: 'evaluation_id' column not found for duplicate removal.")
            return df # Return original df if column is missing

