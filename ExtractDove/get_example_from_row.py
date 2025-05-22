import pandas as pd
from datasets import load_dataset

class InstanceLoader:
    @staticmethod
    def load_dataset_by_name(dataset_name: str):
        """
        Map a dataset name (as in your get_all_datasets list) to the correct load_dataset call.

        The mapping is as follows:
          - For mmlu datasets (e.g., "mmlu.abstract_algebra"), use the repository "cais/mmlu"
            with the subset name (e.g., "abstract_algebra").
          - For mmlu_pro datasets (e.g., "mmlu_pro.history"), use "TIGER-Lab/MMLU-Pro" with the subset.
          - For specific base datasets, we map them manually.
          - For a "race" dataset (if needed), assume the format "race.high" or "race.middle".
        """
        if dataset_name.startswith("mmlu."):
            # e.g., "mmlu.abstract_algebra"
            subset = dataset_name.split(".", 1)[1]
            ds = load_dataset("cais/mmlu","all", split="test")
        elif dataset_name.startswith("mmlu_pro."):
            # e.g., "mmlu_pro.history"
            subset = dataset_name.split(".", 1)[1]
            ds = load_dataset("TIGER-Lab/MMLU-Pro", subset, split="test[:110]")
        elif dataset_name == "ai2_arc.arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:110]")
        elif dataset_name == "ai2_arc.arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test[:110]")
        elif dataset_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="test[:110]")
        elif dataset_name == "social_iqa":
            ds = load_dataset("allenai/social_i_qa", split="validation[:110]", trust_remote_code=True)
        elif dataset_name == "openbook_qa":
            ds = load_dataset("allenai/openbookqa", split="test[:110]")
        elif dataset_name.startswith("race.high"):
            # e.g., "race.high" or "race.middle"
            ds = load_dataset("ehovy/race", "high", split="test[:110]")
        elif dataset_name.startswith("race.middle"):
            ds = load_dataset("ehovy/race", "middle", split="test[:110]")
        else:
            raise ValueError(f"Unknown dataset mapping for {dataset_name}")
        return ds

    @staticmethod
    def extract_answer_choices(example: dict, dataset_name: str) -> list:
        """
        Given an example and its dataset, return the list of answer choices.

        The rules are:
          - For ai2_arc (ARC) and openbook_qa: choices are in example['choices']['text']
          - For hellaswag: choices are in example['endings']
          - For mmlu: choices are in example['choices']
          - For mmlu_pro and race: choices are in example['options']
          - For social_iqa: choices are in the separate fields "answerA", "answerB", "answerC"
        """
        if dataset_name.startswith("ai2_arc") or dataset_name == "openbook_qa":
            return example['choices']['text']
        elif dataset_name == "hellaswag":
            return example['endings']
        elif dataset_name.startswith("mmlu_pro.") or dataset_name.startswith("race"):
            return example['options']
        elif dataset_name.startswith("social_iqa"):
            print(example)
            return [example['answerA'], example['answerB'], example['answerC']]
        elif dataset_name.startswith("mmlu."):
            return example['choices']
        else:
            raise ValueError(f"Unknown dataset mapping for answer choices extraction: {dataset_name}")

    @staticmethod
    def get_example_from_index(dataset_name: str, df: pd.DataFrame):
        """
        Given a DataFrame with 'sample_index' and 'closest_answer',
        extract the chosen answer position from the dataset.

        Returns:
            pd.Series: The chosen answer positions for each row
        """
        ds = InstanceLoader.load_dataset_by_name(dataset_name)

        chosen_positions = []
        rows_to_drop = []

        for i, row in df.iterrows():
            sample_index = row["sample_index"]
            example = ds[sample_index]

            # Extract the answer text from "A. answer_text" format
            try:
                answer_text = row['closest_answer'].split(". ", 1)[1]
                answer_choices = InstanceLoader.extract_answer_choices(example, dataset_name)

                # Find the position of this answer in the choices
                try:
                    answer_position = answer_choices.index(answer_text) + 1  # Convert to 1-based indexing
                    chosen_positions.append(answer_position)
                except ValueError:
                    # If answer not found in choices, add row to be dropped
                    rows_to_drop.append(i)
            except (IndexError, AttributeError):
                # Handle cases where closest_answer doesn't have expected format
                rows_to_drop.append(i)

        # Remove rows where answer couldn't be mapped
        df_copy = df.copy()
        df_copy.drop(rows_to_drop, inplace=True)

        # Create a Series with the same index as the filtered DataFrame
        result = pd.Series(chosen_positions, index=df_copy.index)
        return result


if __name__ == "__main__":
    pass
    # Example usage:
    # Assuming you have a pandas DataFrame 'df' where each row has a 'dataset' and 'sample_index'
    # e.g. df.iloc[0] might be:
    #    evaluation_id    dataset              sample_index
    # 0  1b5dc69c933e396... ai2_arc.arc_challenge      3

    # filter_df_with_closest_answer_ind = InstanceLoader.get_example_from_index("ai2_arc.arc_challenge", df)


