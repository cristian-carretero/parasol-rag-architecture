"""
Module: main.py
Description: Master orchestrator for the MLOps data engineering pipeline.
Sequentially executes data ingestion, physical filtering, and clustering modules,
ensuring strict memory isolation and sequential dependency validation.
"""

import subprocess
import sys
import logging
from pathlib import Path
from typing import List

# Professional logging configuration (MLOps standard)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("PipelineOrchestrator")


def execute_pipeline_module(script_path: Path) -> None:
    """
    Executes an isolated Python module as a subprocess to ensure clean memory 
    garbage collection and strict sequential enforcement.
    
    Args:
        script_path (Path): Path to the target Python script.
        
    Raises:
        SystemExit: If the subprocess returns a non-zero exit code or the file is missing.
    """
    if not script_path.exists():
        logger.error(f"Target module missing: {script_path}. Halting pipeline execution.")
        sys.exit(1)
        
    logger.info(f"--- Initializing execution topology for: {script_path.name} ---")
    
    try:
        # sys.executable guarantees the active virtual environment context is maintained
        subprocess.run([sys.executable, str(script_path)], check=True)
        logger.info(f"✅ Module '{script_path.name}' terminated successfully.\n")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Critical runtime failure in module '{script_path.name}'. Pipeline aborted.")
        sys.exit(e.returncode)


if __name__ == "__main__":
    logger.info("=== INITIATING END-TO-END DATA ENGINEERING PIPELINE ===")
    
    # Define the strict chronological sequence of the pipeline architecture
    pipeline_topology: List[Path] = [
        Path("src/data_processing.py"),
        Path("src/filtering.py"),
        Path("src/clustering.py")
    ]
    
    for module_path in pipeline_topology:
        execute_pipeline_module(module_path)
        
    logger.info("=== PIPELINE EXECUTION SUCCESSFULLY COMPLETED ===")