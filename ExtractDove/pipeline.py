# ============================================================================
# pipeline.py
"""Main analysis pipeline orchestration."""

import logging
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from answer_mapper import AnswerMapper
from config import DEFAULT_PIPELINE_SETTINGS, AnalysisConfig, PipelineSettings
from data_loader import DataLoader
from performance_analyzer import PerformanceAnalyzer


class AnalysisPipeline:
    """Main pipeline for analyzing model performance on multiple choice questions."""

    def __init__(self, settings: PipelineSettings = None):
        """
        Initialize the pipeline.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings or DEFAULT_PIPELINE_SETTINGS
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("analysis_log.txt", mode="w"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        # Reduce noise from external libraries
        logging.getLogger("fsspec").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)

    def run(self) -> None:
        """Run the complete analysis pipeline."""
        configs = self._generate_configurations()
        logging.info(f"Generated {len(configs)} configurations to process")
        print(f"Generated {len(configs)} configurations to process")

        results = self._process_configurations(configs)
        self._summarize_results(results)

    def _generate_configurations(self) -> List[AnalysisConfig]:
        """Generate all configuration combinations."""
        configs = []
        for repo_id in self.settings.repo_ids:
            for model in self.settings.models:
                for language in self.settings.languages:
                    for shots_val in self.settings.shots:
                        for benchmark_file in self.settings.benchmark_files:
                            configs.append(
                                AnalysisConfig(
                                    repo_id=repo_id,
                                    model_name=model,
                                    language=language,
                                    shots_selected=shots_val,
                                    benchmark_file=benchmark_file,
                                )
                            )
        return configs

    def _process_configurations(
        self, configs: List[AnalysisConfig]
    ) -> List[Dict[str, Any]]:
        """Process all configurations either sequentially or in parallel."""
        if self.settings.run_sequentially:
            print("Running configurations sequentially...")
            return [
                self._process_single_configuration(config)
                for config in tqdm(configs, desc="Processing configurations")
            ]
        else:
            print(f"Running with {self.settings.num_processes} processes...")
            with Pool(processes=self.settings.num_processes) as pool:
                return list(
                    tqdm(
                        pool.imap_unordered(
                            self._process_single_configuration, configs
                        ),
                        total=len(configs),
                        desc="Processing configurations",
                    )
                )

    def _process_single_configuration(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Process a single configuration."""
        config_str = self._get_config_string(config)

        try:
            print(f"\n--- Processing {config_str} ---")
            logging.info(f"Starting processing for config: {config}")

            # Load and process data
            df = self._load_data(config)
            if df.empty:
                return {"status": "empty", "config": config}

            # Apply filters
            df = self._apply_filters(df, config)
            if df.empty:
                return {"status": "empty_after_filters", "config": config}

            # Group by sample_index and calculate average scores
            df_grouped = self._group_by_sample_index(df)
            if df_grouped.empty:
                return {"status": "no_samples_after_grouping", "config": config}

            # Find low performers (now based on averaged scores)
            df_low_performers = self._find_low_performers_by_average(df_grouped)
            if df_low_performers.empty:
                return {"status": "no_low_performers", "config": config}

            # Save results
            self._save_results(df_low_performers, config)

            return {
                "status": "success",
                "config": config,
                "initial_rows": len(df),
                "grouped_samples": len(df_grouped),
                "low_performer_samples": len(df_low_performers),
            }

        except Exception as e:
            logging.exception(f"Error processing {config_str}")
            return {"status": "error", "config": config, "error": str(e)}

    def _load_data(self, config: AnalysisConfig) -> pd.DataFrame:
        """Load data for a configuration."""
        data_loader = DataLoader(repo_id=config.repo_id)
        return data_loader.load_and_process_data(
            model_name=config.model_name,
            language=config.language,
            shots=config.shots_selected,
            benchmark_files=[config.benchmark_file],
            drop_columns=["cumulative_logprob", "generated_text", "ground_truth"],
        )

    def _apply_filters(self, df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        """Apply custom filters to the data."""
        initial_rows = len(df)

        # Apply 5-shot choices_order filter
        required_columns = ["dimensions_5: shots", "dimensions_3: choices_order"]
        if all(col in df.columns for col in required_columns):
            mask = (df["dimensions_5: shots"] != 5) | (
                ~df["dimensions_3: choices_order"].isin(
                    ["correct_first", "correct_last"]
                )
            )
            df = df[mask].copy()

            filtered_count = initial_rows - len(df)
            if filtered_count > 0:
                logging.info(
                    f"Filtered out {filtered_count} rows. Remaining: {len(df)}"
                )
        else:
            logging.warning("Skipping 5-shot filter: required columns not found")

        return df

    def _group_by_sample_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by sample_index and calculate average scores.
        
        Args:
            df: DataFrame with sample_index, score, and raw_input columns
            
        Returns:
            DataFrame with sample_index, average_score, and question columns
        """
        logging.info("Grouping by sample_index and calculating average scores")
        
        if 'sample_index' not in df.columns or 'score' not in df.columns:
            logging.error("Required columns 'sample_index' and 'score' not found")
            return pd.DataFrame()
        
        # Group by sample_index and calculate statistics
        grouped = df.groupby('sample_index').agg({
            'score': ['sum', 'count', 'mean'],
            'raw_input': 'first'  # Take the first occurrence (should be same for all instances of same sample)
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['sample_index', 'total_score', 'instance_count', 'average_score', 'question']
        
        # Round average score to 3 decimal places
        grouped['average_score'] = grouped['average_score'].round(3)
        
        # Keep only the columns we need
        result_df = grouped[['sample_index', 'average_score', 'question']].copy()
        
        logging.info(f"Grouped {len(df)} rows into {len(result_df)} unique samples")
        logging.info(f"Average score range: {result_df['average_score'].min():.3f} - {result_df['average_score'].max():.3f}")
        
        return result_df

    def _find_low_performers_by_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find samples with average scores below the threshold.
        
        Args:
            df: DataFrame with sample_index, average_score, and question columns
            
        Returns:
            DataFrame containing only low-performing samples
        """
        threshold = getattr(self, 'threshold', 0.5)  # Default threshold if not set
        
        logging.info(f"Finding low performers with average score < {threshold}")
        
        if df.empty:
            logging.debug("Input DataFrame is empty")
            return pd.DataFrame()
        
        # Filter for low performers based on average score
        low_performers = df[df['average_score'] < threshold].copy()
        
        if low_performers.empty:
            logging.info("No low-performing samples found")
            return pd.DataFrame()
        
        # Log results
        total_samples = len(df)
        low_performer_count = len(low_performers)
        logging.info(f"Found {low_performer_count}/{total_samples} low-performing samples ({low_performer_count/total_samples:.1%})")
        
        # Log some examples
        sample_size = min(5, low_performer_count)
        for _, row in low_performers.head(sample_size).iterrows():
            logging.info(f"  Sample {row['sample_index']}: Average score {row['average_score']:.3f}")
        
        if low_performer_count > sample_size:
            logging.info(f"  ... and {low_performer_count - sample_size} more")
            
        return low_performers

    def _save_results(self, df: pd.DataFrame, config: AnalysisConfig) -> None:
        """Save results to parquet file."""
        output_path = config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)
        logging.info(f"Saved {len(df)} samples to {output_path}")
        logging.info(f"Columns saved: {list(df.columns)}")

    def _get_config_string(self, config: AnalysisConfig) -> str:
        """Generate a readable string representation of the configuration."""
        model_short = config.model_name.split("/")[-1]
        benchmark_short = Path(config.benchmark_file).stem
        return f"{model_short}_{config.language}_{config.shots_selected}shot_{benchmark_short}"

    def _summarize_results(self, results: List[Dict[str, Any]]) -> None:
        """Summarize the results of all processed configurations."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)

        # Count results by status
        status_counts = {}
        for result in results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"Total configurations processed: {len(results)}")
        print("\nStatus breakdown:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        # Show successful configurations with row counts
        successful_results = [r for r in results if r["status"] == "success"]
        if successful_results:
            print(f"\nSuccessful configurations ({len(successful_results)}):")
            for result in successful_results:
                config_str = self._get_config_string(result["config"])
                initial_rows = result.get("initial_rows", "N/A")
                grouped_samples = result.get("grouped_samples", "N/A")
                low_performer_samples = result.get("low_performer_samples", "N/A")
                print(f"  {config_str}: {initial_rows} rows → {grouped_samples} samples → {low_performer_samples} low performers")

        # Show failed configurations
        failed_results = [r for r in results if r["status"] == "error"]
        if failed_results:
            print(f"\nFailed configurations ({len(failed_results)}):")
            for result in failed_results:
                config_str = self._get_config_string(result["config"])
                error = result.get("error", "Unknown error")
                print(f"  {config_str}: {error}")

        print("=" * 80)


def main():
    """Main entry point for the analysis pipeline."""
    print("Starting Analysis Pipeline")
    print("=" * 50)

    # Use default settings
    pipeline = AnalysisPipeline(DEFAULT_PIPELINE_SETTINGS)
    pipeline.run()

    print("Pipeline execution completed!")


if __name__ == "__main__":
    main()
