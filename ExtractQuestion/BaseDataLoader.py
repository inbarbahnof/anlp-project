# data_loader.py
import time
from typing import Dict, List, Any
from typing import Optional

import pandas as pd
import pyarrow.compute as pc
from datasets import load_dataset

from config.get_config import Config

repo_name = "nlphuji/DOVE_Lite"
from huggingface_hub import HfApi, login

config = Config()
TOKEN = config.config_values.get("hf_access_token_dove", "")
# First, authenticate with your Hugging Face token
login(token=TOKEN)


class BaseDataLoader:
    def __init__(self, repo_name=repo_name, split="train", batch_size=10000):
        """
        Initializes the DataLoader with dataset details.

        Args:
            repo_name (str): Name of the dataset to load from HuggingFace.
            split (str): Dataset split to use.
            batch_size (int): Number of samples per batch.
        """
        self.dataset = None
        self.repo_name = repo_name
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(self, model_name, shots,
                              datasets=None, template=None, separator=None, enumerator=None, choices_order=None,
                              max_samples=None, drop=False):
        start_time = time.time()
        # print(f"Processing model: {model_name}")
        self.load_data_with_filter(max_samples, drop, model_name, shots, datasets)
        if len(self.dataset) == 0:
            print(f"No data found for model {model_name} and shots {shots}")
            return pd.DataFrame()
        # full_results = self.extract_data(model_name, shots,
        #                                  dataset=datasets, template=template, separator=separator,
        #                                  enumerator=enumerator,
        #                                  choices_order=choices_order)
        # clean_df = self.remove_duplicates(full_results)
        # clean_df = full_results.drop(['quantization', 'closest_answer'], axis=1)
        load_time = time.time()
        # print(f"The size of the data after removing duplicates is: {len(clean_df)}")
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return self.dataset.to_pandas()

    def load_data_with_filter_local(self, max_samples=None, drop=True, model_name=None, shots=None, dataset=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded
        file = f"{model_name}_shot{shots}_{dataset}.parquet"
        data_files = [file]
        if len(data_files) > 0:
            if self.dataset is None:
                # split = split_with_filter
                self.dataset = load_dataset(self.repo_name, data_files=data_files, cache_dir=None)
            else:
                self.dataset = load_dataset(self.repo_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['family', 'generated_text', 'ground_truth'])

    def load_data_with_filter(self, max_samples=None, drop=True, model_name=None, shots=None, datasets=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded

        api = HfApi()
        all_files = api.list_repo_files(
            repo_id=self.repo_name,
            repo_type="dataset",
        )
        existing_files = [file for file in all_files if file.endswith('.parquet')]
        # split only the file name that contains the model name and shots and dataset
        if model_name is not None:
            model_name = model_name.split("/")[-1]
            existing_files = [file for file in existing_files if model_name in file]
        if shots is not None:
            shots = "shots" + str(shots)
            existing_files = [file for file in existing_files if shots in file]
        if datasets is not None:
            if isinstance(datasets, str):
                datasets = [datasets]
            datasets = [dataset.split("/")[-1] for dataset in datasets]
            existing_files = [file for file in existing_files if any(dataset in file for dataset in datasets)]

        if len(existing_files) > 0:
            if self.dataset is None:
                # split = split_with_filter
                self.dataset = load_dataset(self.repo_name, data_files=existing_files, split=self.split, cache_dir=None)
            else:
                self.dataset = load_dataset(self.repo_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['cumulative_logprob', 'generated_text', 'ground_truth'])

    def extract_data2(
            self,
            model_name: str,
            shots: int,
            dataset: Optional[str] = None,
            template: Optional[str] = None,
            separator: Optional[str] = None,
            enumerator: Optional[str] = None,
            choices_order: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Efficiently filter a large dataset using batched processing and parallel execution.

        Args:
            model_name: Name of the model to filter by
            shots: Number of shots to filter by
            dataset: Optional dataset name filter
            template: Optional template filter
            separator: Optional separator filter
            enumerator: Optional enumerator filter
            choices_order: Optional choices order filter

        Returns:
            pd.DataFrame: Filtered dataset as a pandas DataFrame
        """
        try:
            print("Starting data filtering process...")

            # Step 1: Define the columns we need to keep
            # Add any additional columns you need in the final output
            required_columns = [
                'model',
                'shots',
                'dataset',
                'template',
                'separator',
                'enumerator',
                'choices_order'
            ]

            # Step 2: Select only necessary columns to reduce memory usage
            dataset_subset = self.dataset.select_columns(required_columns)

            # Step 3: Define the batch filtering function
            def filter_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[bool]]:
                """
                Process a batch of examples and return a boolean mask for filtering.

                Args:
                    examples: Dictionary of column names to lists of values

                Returns:
                    Dictionary with a single 'keep' key containing boolean mask
                """
                batch_size = len(examples['model'])
                # Initialize all rows as True
                mask = [True] * batch_size

                # Apply each filter condition if it exists
                # Required filters
                mask = [m and (mod == model_name) for m, mod in zip(mask, examples['model'])]
                mask = [m and (s == shots) for m, s in zip(mask, examples['shots'])]

                # Optional filters
                if dataset is not None:
                    mask = [m and (d == dataset) for m, d in zip(mask, examples['dataset'])]

                if template is not None:
                    mask = [m and (t == template) for m, t in zip(mask, examples['template'])]

                if separator is not None:
                    mask = [m and (sep == separator) for m, sep in zip(mask, examples['separator'])]

                if enumerator is not None:
                    mask = [m and (e == enumerator) for m, e in zip(mask, examples['enumerator'])]

                if choices_order is not None:
                    mask = [m and (c == choices_order) for m, c in zip(mask, examples['choices_order'])]

                return {'keep': mask}

            # Step 4: Apply the batched filtering
            print("Applying filters...")
            filtered_dataset = dataset_subset.map(
                filter_batch,
                batched=True,
                batch_size=100000,  # Adjust based on available memory
                num_proc=4,
                remove_columns=dataset_subset.column_names,
                load_from_cache_file=True,
                desc="Filtering dataset"
            )

            # Step 5: Convert to pandas DataFrame
            print("Converting to pandas DataFrame...")
            df_filtered = filtered_dataset.to_pandas()

            # Step 6: Print summary statistics
            print("\nFiltering results:")
            print(f"Total rows after filtering: {len(df_filtered)}")
            print("\nModel distribution:")
            print(df_filtered['model'].value_counts())

            return df_filtered

        except Exception as e:
            raise Exception(f"Error during data filtering: {str(e)}")

    def extract_data(self, model_name, shots, dataset=None, template=None, separator=None, enumerator=None,
                     choices_order=None):
        arrow_table = self.dataset.data.table

        conditions = [
            pc.equal(arrow_table['model'], model_name),
            pc.equal(arrow_table['shots'], shots)
        ]

        optional_filters = {
            'template': template,
            'separator': separator,
            'enumerator': enumerator,
            'choices_order': choices_order
        }

        for column, value in optional_filters.items():
            if value is not None:
                conditions.append(pc.equal(arrow_table[column], value))

        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = pc.and_(combined_condition, condition)

        try:
            print("Filtering data...")
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()
            print(df_filtered['model'].value_counts())
            return df_filtered
        except Exception as e:
            raise Exception(f"Error filtering data: {str(e)}")

    def remove_duplicates(self, df):
        """
        Removes duplicates from the DataFrame.
        """
        cols = ['sample_index', 'model', 'dataset', 'template', 'separator', 'enumerator', 'choices_order', 'shots']
        df_unique = df.drop_duplicates(subset=['evaluation_id'])
        return df_unique
