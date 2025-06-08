# config.py
"""Configuration management for the analysis pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

# Global constants
PERFORMANCE_THRESHOLD = 0.1
LOG_FILE = "analysis_log.txt"


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for analysis pipeline with immutable attributes."""

    repo_id: str
    model_name: str
    language: str
    shots_selected: int
    benchmark_file: str

    @property
    def output_path(self) -> Path:
        """Generate standardized output path for results."""
        base_dir = Path("../app/results_local")
        repo_part = self.repo_id.replace("/", "_")
        model_folder_name = self.model_name.split("/")[-1]
        benchmark_part = Path(self.benchmark_file).stem

        return (
            base_dir
            / repo_part
            / self.language
            / f"Shots_{self.shots_selected}"
            / model_folder_name.replace("/", "_")
            / benchmark_part
            / f"low_performance_questions_{PERFORMANCE_THRESHOLD}.parquet"
        )


@dataclass
class PipelineSettings:
    """Settings for the entire analysis pipeline."""

    repo_ids: List[str]
    languages: List[str]
    models: List[str]
    shots: List[int]
    benchmark_files: List[str]
    num_processes: int = 3
    run_sequentially: bool = True


# Default pipeline configuration
DEFAULT_PIPELINE_SETTINGS = PipelineSettings(
    repo_ids=["nlphuji/DOVE_Lite"],
    languages=["en"],
    models=["allenai/OLMoE-1B-7B-0924-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"],
    shots=[0, 5],
    benchmark_files=[
        "mmlu.abstract_algebra.parquet",
        "mmlu.anatomy.parquet",
        "mmlu.astronomy.parquet",
        "mmlu.business_ethics.parquet",
        "mmlu.clinical_knowledge.parquet",
        "mmlu.college_biology.parquet",
        "mmlu.college_chemistry.parquet",
        "mmlu.college_computer_science.parquet",
        "mmlu.college_mathematics.parquet",
        "mmlu.college_medicine.parquet",
        "mmlu.college_physics.parquet",
        "mmlu.computer_security.parquet",
        "mmlu.conceptual_physics.parquet",
        "mmlu.econometrics.parquet",
        "mmlu.electrical_engineering.parquet",
        "mmlu.elementary_mathematics.parquet",
        "mmlu.formal_logic.parquet",
        "mmlu.global_facts.parquet",
        "mmlu.high_school_biology.parquet",
        "mmlu.high_school_chemistry.parquet",
        "mmlu.high_school_computer_science.parquet",
        "mmlu.high_school_european_history.parquet",
        "mmlu.high_school_geography.parquet",
        "mmlu.high_school_government_and_politics.parquet",
        "mmlu.high_school_macroeconomics.parquet",
        "mmlu.high_school_mathematics.parquet",
        "mmlu.high_school_microeconomics.parquet",
        "mmlu.high_school_physics.parquet",
        "mmlu.high_school_psychology.parquet",
        "mmlu.high_school_statistics.parquet",
        "mmlu.high_school_us_history.parquet",
        "mmlu.high_school_world_history.parquet",
        "mmlu.human_aging.parquet",
        "mmlu.human_sexuality.parquet",
        "mmlu.international_law.parquet",
        "mmlu.jurisprudence.parquet",
        "mmlu.logical_fallacies.parquet",
        "mmlu.machine_learning.parquet",
        "mmlu.management.parquet",
        "mmlu.marketing.parquet",
        "mmlu.medical_genetics.parquet",
        "mmlu.miscellaneous.parquet",
        "mmlu.moral_disputes.parquet",
        "mmlu.moral_scenarios.parquet",
        "mmlu.nutrition.parquet",
        "mmlu.philosophy.parquet",
        "mmlu.prehistory.parquet",
        "mmlu.professional_accounting.parquet",
        "mmlu.professional_law.parquet",
        "mmlu.professional_medicine.parquet",
        "mmlu.professional_psychology.parquet",
        "mmlu.public_relations.parquet",
        "mmlu.security_studies.parquet",
        "mmlu.sociology.parquet",
        "mmlu.us_foreign_policy.parquet",
        "mmlu.virology.parquet",
        "mmlu.world_religions.parquet",
    ],
)
