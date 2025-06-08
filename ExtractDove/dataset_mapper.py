# ============================================================================
# dataset_mapper.py
"""Dataset mapping functionality for retrieving original questions."""

import pandas as pd
from datasets import load_dataset
from typing import Dict, List
import logging


class DatasetMapper:
    """Maps dataset names to HuggingFace dataset configurations."""

    @staticmethod
    def load_dataset_by_name(dataset_name: str):
        """
        Load a dataset by name using the appropriate HuggingFace configuration.

        Args:
            dataset_name: Name of the dataset (e.g., "mmlu.abstract_algebra")

        Returns:
            Loaded dataset
        """
        dataset_mappings = {
            # MMLU datasets
            "mmlu": lambda subset: load_dataset("cais/mmlu", "all", split="test"),
            # MMLU Pro datasets
            "mmlu_pro": lambda subset: load_dataset(
                "TIGER-Lab/MMLU-Pro", subset, split="test[:110]"
            ),
            # Specific datasets
            "ai2_arc.arc_challenge": lambda _: load_dataset(
                "allenai/ai2_arc", "ARC-Challenge", split="test[:110]"
            ),
            "ai2_arc.arc_easy": lambda _: load_dataset(
                "allenai/ai2_arc", "ARC-Easy", split="test[:110]"
            ),
            "hellaswag": lambda _: load_dataset("Rowan/hellaswag", split="test[:110]"),
            "social_iqa": lambda _: load_dataset(
                "allenai/social_i_qa", split="validation[:110]", trust_remote_code=True
            ),
            "openbook_qa": lambda _: load_dataset(
                "allenai/openbookqa", split="test[:110]"
            ),
            "race.high": lambda _: load_dataset(
                "ehovy/race", "high", split="test[:110]"
            ),
            "race.middle": lambda _: load_dataset(
                "ehovy/race", "middle", split="test[:110]"
            ),
        }

        # Handle prefixed datasets (e.g., "mmlu.abstract_algebra")
        for prefix, loader_func in dataset_mappings.items():
            if dataset_name.startswith(prefix + "."):
                subset = dataset_name.split(".", 1)[1] if "." in dataset_name else None
                return loader_func(subset)
            elif dataset_name == prefix:
                return loader_func(None)

        raise ValueError(f"Unknown dataset mapping for {dataset_name}")

    @staticmethod
    def extract_answer_choices(example: dict, dataset_name: str) -> List[str]:
        """
        Extract answer choices from a dataset example.

        Args:
            example: Dataset example dictionary
            dataset_name: Name of the dataset

        Returns:
            List of answer choices
        """
        choice_mappings = {
            "ai2_arc": lambda ex: ex["choices"]["text"],
            "openbook_qa": lambda ex: ex["choices"]["text"],
            "hellaswag": lambda ex: ex["endings"],
            "mmlu_pro": lambda ex: ex["options"],
            "race": lambda ex: ex["options"],
            "social_iqa": lambda ex: [ex["answerA"], ex["answerB"], ex["answerC"]],
            "mmlu": lambda ex: ex["choices"],
        }

        for prefix, extractor in choice_mappings.items():
            if dataset_name.startswith(prefix):
                return extractor(example)

        raise ValueError(
            f"Unknown dataset for answer choices extraction: {dataset_name}"
        )


class InstanceLoader:
    """Loads and processes dataset instances for answer mapping."""

    @staticmethod
    def get_example_from_index(dataset_name: str, df: pd.DataFrame) -> pd.Series:
        """
        Extract chosen answer positions from dataset examples.

        Args:
            dataset_name: Name of the dataset
            df: DataFrame with 'sample_index' and 'closest_answer' columns

        Returns:
            Series of chosen answer positions
        """
        try:
            dataset = DatasetMapper.load_dataset_by_name(dataset_name)
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            return pd.Series(dtype=int)

        chosen_positions = []
        valid_indices = []

        for i, row in df.iterrows():
            try:
                position = InstanceLoader._get_position_for_row(
                    row, dataset, dataset_name
                )
                if position is not None:
                    chosen_positions.append(position)
                    valid_indices.append(i)
                else:
                    logging.debug(f"Could not map answer for row {i}")
            except Exception as e:
                logging.debug(f"Error processing row {i}: {e}")

        return pd.Series(chosen_positions, index=valid_indices)

    @staticmethod
    def _get_position_for_row(
        row: pd.Series, dataset, dataset_name: str
    ) -> Optional[int]:
        """Get the answer position for a specific row."""
        sample_index = row["sample_index"]
        closest_answer = row["closest_answer"]

        if pd.isna(closest_answer) or not isinstance(closest_answer, str):
            return None

        example = dataset[sample_index]

        # Extract answer text from "A. answer_text" format
        try:
            answer_text = closest_answer.split(". ", 1)[1]
            answer_choices = DatasetMapper.extract_answer_choices(example, dataset_name)

            # Find position in choices (1-based indexing)
            return answer_choices.index(answer_text) + 1

        except (IndexError, ValueError, AttributeError):
            return None
