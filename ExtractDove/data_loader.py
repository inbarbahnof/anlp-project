# ============================================================================
# data_loader.py
"""Data loading and filtering functionality."""

import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import pyarrow.compute as pc
import pyarrow as pa
from datasets import load_dataset, Dataset
import logging


class DataLoader:
    """Handles loading and filtering of dataset from HuggingFace repositories."""

    def __init__(self, repo_id: str, split: str = "train", batch_size: int = 10000):
        """
        Initialize the DataLoader.

        Args:
            repo_id: Name of the dataset repository (e.g., "nlphuji/DOVE_Lite")
            split: Dataset split to use
            batch_size: Number of samples per batch
        """
        self.dataset: Optional[Dataset] = None
        self.repo_id = repo_id
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(
        self,
        model_name: str,
        language: str,
        shots: int,
        benchmark_files: List[str],
        template: Optional[str] = None,
        separator: Optional[str] = None,
        enumerator: Optional[str] = None,
        choices_order: Optional[str] = None,
        instruction_phrasing_text: Optional[str] = None,
        drop_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load and process data for specific configuration.

        Args:
            model_name: Full model name
            language: Language code
            shots: Number of shots
            benchmark_files: List of benchmark file names
            template: Optional template filter
            separator: Optional separator filter
            enumerator: Optional enumerator filter
            choices_order: Optional choices order filter
            instruction_phrasing_text: Optional instruction phrasing filter
            drop_columns: Optional list of columns to drop

        Returns:
            Processed DataFrame
        """
        start_time = time.time()
        logging.info(
            f"Loading data for model: {model_name}, language: {language}, "
            f"shots: {shots}, benchmarks: {benchmark_files}"
        )

        # Load data from repository structure
        self._load_data_from_structure(model_name, language, shots, benchmark_files)

        if self.dataset is None or len(self.dataset) == 0:
            logging.warning(f"No data found for specified configuration")
            return pd.DataFrame()

        # Drop specified columns early for efficiency
        if drop_columns:
            columns_to_drop = [
                col for col in drop_columns if col in self.dataset.column_names
            ]
            if columns_to_drop:
                self.dataset = self.dataset.remove_columns(columns_to_drop)

        # Extract and filter data
        filtered_df = self._extract_and_filter_data(
            model_name=model_name,
            shots=shots,
            datasets=benchmark_files,
            template=template,
            separator=separator,
            enumerator=enumerator,
            choices_order=choices_order,
            instruction_phrasing_text=instruction_phrasing_text,
        )

        if filtered_df.empty:
            logging.warning("No data remaining after filtering")
            return pd.DataFrame()

        # Remove duplicates
        clean_df = self._remove_duplicates(filtered_df)

        load_time = time.time()
        logging.info(
            f"Loaded {len(clean_df)} rows in {load_time - start_time:.2f} seconds"
        )
        return clean_df

    def _load_data_from_structure(
        self, model_name: str, language: str, shots: int, benchmark_files: List[str]
    ) -> None:
        """Load data files based on repository structure."""
        model_folder_name = model_name.split("/")[-1]

        data_files = [
            f"{model_folder_name}/{language}/{shots}_shot/{benchmark_file}"
            for benchmark_file in benchmark_files
        ]

        if not data_files:
            logging.warning("No benchmark files specified")
            self.dataset = None
            return

        try:
            self.dataset = load_dataset(
                self.repo_id,
                data_files=data_files,
                split=self.split,
                cache_dir=None,
                token=True,
            )
            logging.info(f"Successfully loaded {len(self.dataset)} examples")
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            self.dataset = None

    def _extract_and_filter_data(
        self,
        model_name: str,
        shots: int,
        datasets: List[str],
        template: Optional[str] = None,
        separator: Optional[str] = None,
        enumerator: Optional[str] = None,
        choices_order: Optional[str] = None,
        instruction_phrasing_text: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter dataset using PyArrow for efficiency."""
        if self.dataset is None:
            return pd.DataFrame()

        try:
            arrow_table = self.dataset.data.table

            # Base conditions
            conditions = [
                pc.equal(arrow_table["model"], model_name),
                pc.equal(arrow_table["dimensions_5: shots"], shots),
            ]

            # Filter by dataset names
            dataset_names = [Path(f).stem for f in datasets]
            conditions.append(pc.is_in(arrow_table["dataset"], pa.array(dataset_names)))

            # Optional filters
            optional_filters = {
                "dimensions_1: enumerator": enumerator,
                "dimensions_2: separator": separator,
                "dimensions_3: choices_order": choices_order,
                "dimensions_4: instruction_phrasing_text": instruction_phrasing_text,
            }

            for column, value in optional_filters.items():
                if value is not None and column in arrow_table.column_names:
                    conditions.append(pc.equal(arrow_table[column], value))

            # Combine conditions
            combined_condition = conditions[0]
            for condition in conditions[1:]:
                combined_condition = pc.and_(combined_condition, condition)

            # Apply filter
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()

            logging.info(f"Filtering results: {len(df_filtered)} rows remaining")
            return df_filtered

        except Exception as e:
            logging.error(f"Error during data filtering: {e}")
            return pd.DataFrame()

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates based on evaluation_id."""
        if df.empty or "evaluation_id" not in df.columns:
            return df

        initial_count = len(df)
        df_unique = df.drop_duplicates(subset=["evaluation_id"])
        logging.debug(f"Removed {initial_count - len(df_unique)} duplicates")
        return df_unique
