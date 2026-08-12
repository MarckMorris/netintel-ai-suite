"""Make the three component packages importable from the repository root.

Each component lives in its own top-level directory rather than a single
installed package, so the module directories are put on sys.path here. That
keeps `pytest` working from a clean clone with no editable install step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for relative in (
    "project1_anomaly_detection",
    "project1_anomaly_detection/models",
    "project1_anomaly_detection/data",
    "project2_alert_correlation/engine",
    "project3_sdwan_optimizer/simulator",
    "project3_sdwan_optimizer/agents",
):
    path = ROOT / relative
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
