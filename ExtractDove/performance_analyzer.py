# ============================================================================
# performance_analyzer.py
"""Performance analysis functionality."""

import logging

import pandas as pd

from config import PERFORMANCE_THRESHOLD


class PerformanceAnalyzer:
    """Analyzes model performance and identifies low-performing questions."""

    def __init__(self, threshold: float = PERFORMANCE_THRESHOLD):
        """
        Initialize the analyzer.

        Args:
            threshold: Accuracy threshold below which questions are considered low-performing
        """
        self.threshold = threshold

    def find_low_performers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find questions with accuracy below the threshold.

        This method now works with pre-grouped data that has average_score column.

        Args:
            df: DataFrame with 'sample_index', 'average_score', and 'question' columns

        Returns:
            DataFrame containing only low-performing questions
        """
        logging.debug(f"Starting performance analysis. Input shape: {df.shape}")

        if df.empty:
            logging.debug("Input DataFrame is empty")
            return pd.DataFrame()

        # Check if we have the new grouped format
        if "average_score" in df.columns:
            return self._find_low_performers_grouped(df)

        # Fall back to original method for backwards compatibility
        required_columns = ["sample_index", "score"]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns: {required_columns}")
            return pd.DataFrame()

        # Calculate accuracy statistics per question
        accuracy_stats = self._calculate_accuracy_stats(df)

        # Filter for low performers
        low_performers_stats = self._filter_low_performers(accuracy_stats)

        if low_performers_stats.empty:
            logging.info("No low-performing questions found")
            return pd.DataFrame()

        # Log results
        self._log_low_performers(low_performers_stats)

        # Filter original DataFrame for low-performing questions
        low_performer_indices = low_performers_stats["sample_index"]
        df_low_performers = df[df["sample_index"].isin(low_performer_indices)].copy()

        logging.debug(f"Filtered to {len(df_low_performers)} rows for low performers")
        return df_low_performers

    def _find_low_performers_grouped(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find low performers from pre-grouped data with average scores.

        Args:
            df: DataFrame with 'sample_index', 'average_score', and 'question' columns

        Returns:
            DataFrame containing only low-performing samples
        """
        required_columns = ["sample_index", "average_score"]
        if not all(col in df.columns for col in required_columns):
            logging.error(
                f"Missing required columns for grouped analysis: {required_columns}"
            )
            return pd.DataFrame()

        # Filter for low performers based on average score
        low_performers = df[df["average_score"] < self.threshold].copy()

        if low_performers.empty:
            logging.info("No low-performing samples found")
            return pd.DataFrame()

        # Log results
        total_samples = len(df)
        low_performer_count = len(low_performers)
        logging.info(
            f"Found {low_performer_count}/{total_samples} low-performing samples ({low_performer_count/total_samples:.1%})"
        )

        # Log some examples
        sample_size = min(10, low_performer_count)
        for _, row in low_performers.head(sample_size).iterrows():
            logging.info(
                f"  Sample {row['sample_index']}: Average score {row['average_score']:.3f}"
            )

        if low_performer_count > sample_size:
            logging.info(f"  ... and {low_performer_count - sample_size} more")

        return low_performers

    def _calculate_accuracy_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate accuracy statistics per sample_index (legacy method)."""
        accuracy_stats = (
            df.groupby("sample_index").agg({"score": ["sum", "count"]}).reset_index()
        )

        accuracy_stats.columns = ["sample_index", "correct_count", "total_count"]
        accuracy_stats["accuracy"] = (
            accuracy_stats["correct_count"]
            / accuracy_stats["total_count"].replace(0, pd.NA)
        ).round(3)

        return accuracy_stats

    def _filter_low_performers(self, accuracy_stats: pd.DataFrame) -> pd.DataFrame:
        """Filter for questions below the accuracy threshold (legacy method)."""
        return accuracy_stats[accuracy_stats["accuracy"] < self.threshold].dropna(
            subset=["accuracy"]
        )

    def _log_low_performers(self, low_performers_stats: pd.DataFrame) -> None:
        """Log information about low-performing questions (legacy method)."""
        total_count = len(low_performers_stats)
        logging.info(
            f"Found {total_count} low-performing questions "
            f"(accuracy < {self.threshold:.1%})"
        )

        # Log details for first 10 questions
        sample_size = min(10, total_count)
        for _, row in low_performers_stats.head(sample_size).iterrows():
            logging.info(
                f"  Question {row['sample_index']}: "
                f"Accuracy {row['accuracy']:.1%} "
                f"({row['correct_count']}/{row['total_count']} correct)"
            )

        if total_count > sample_size:
            logging.info(f"  ... and {total_count - sample_size} more")
