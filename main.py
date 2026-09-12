"""
Module: main.py
Description: Master orchestrator for the ParaSol MLOps pipeline.

Sequentially executes the data engineering and modelling modules
(01_ingest_raw → 09_jv_mppt_trajectory_forecasting), enforcing strict
memory isolation between stages via subprocess execution.

Auxiliary scripts (viz_*, xai_*, rul_xgb_audit) are intentionally excluded
from the main pipeline: they are diagnostic/visualisation tools meant to be
run on demand, not production pipeline stages.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List

# Professional logging configuration (MLOps standard).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("PipelineOrchestrator")


def execute_pipeline_module(script_path: Path) -> None:
    """
    Execute an isolated Python module as a subprocess to ensure clean memory
    garbage collection and strict sequential enforcement.

    Args:
        script_path: Path to the target Python script.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code or the
                    script is missing on disk.
    """
    if not script_path.exists():
        logger.error(f"Target module missing: {script_path}. Halting pipeline execution.")
        sys.exit(1)

    logger.info(f"--- Initializing execution topology for: {script_path.name} ---")

    try:
        # sys.executable guarantees the active virtual environment is inherited.
        subprocess.run([sys.executable, str(script_path)], check=True)
        logger.info(f"✅ Module '{script_path.name}' terminated successfully.\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Critical runtime failure in module '{script_path.name}'. Pipeline aborted.")
        sys.exit(e.returncode)


# Strict chronological sequence of the production pipeline.
PIPELINE_TOPOLOGY: List[Path] = [
    Path("src/01_ingest_raw.py"),
    Path("src/02_jv_filtering.py"),
    Path("src/03_jv_clustering.py"),
    Path("src/04_mppt_aggregation.py"),
    Path("src/05_merge_jv_mppt.py"),
    Path("src/06_jv_mppt_t80_tracker.py"),
    Path("src/07_jv_mppt_early_screening.py"),
    Path("src/08_mppt_rul_forecasting.py"),
    Path("src/09_jv_mppt_trajectory_forecasting.py"),
]

# Auxiliary scripts that are NOT part of the main pipeline and are executed
# on demand. Listed here as documentation only.
AUXILIARY_SCRIPTS: List[Path] = [
    Path("src/viz_filtering.py"),
    Path("src/viz_clustering.py"),
    Path("src/xai_digital_twin.py"),
    Path("src/xai_physical_forensic.py"),
    Path("src/rul_xgb_audit.py"),
]


if __name__ == "__main__":
    logger.info("=== INITIATING END-TO-END MLOPS PIPELINE ===")

    for module_path in PIPELINE_TOPOLOGY:
        execute_pipeline_module(module_path)

    logger.info("=== PIPELINE EXECUTION SUCCESSFULLY COMPLETED ===")
    logger.info(
        f"Auxiliary scripts ({len(AUXILIARY_SCRIPTS)}) are not part of the main "
        f"pipeline. Run them individually when needed."
    )