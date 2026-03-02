#!/usr/bin/env python3
"""
Wrapper script for the selection stage.

This stays intentionally small: most logic lives in occlusion_selection_tools.py
so it can be imported from notebooks as well.
"""

from pathlib import Path
from occlusion_selection_tools import SelectionConfig, run_selection_pipeline


PROJECT = Path("/home/hep/an1522/dark_tridents_wspace")
INFER_BASE = PROJECT / "outputs" / "inference"
OUT_BASE = INFER_BASE / "_occlusion_selections"

SAMPLES_FILES_BY_RUN = {
    "run1": [
        "run1_NuMI_dirt_larcv_cropped_scores.csv",
        "run1_NuMI_nu_overlay_larcv_cropped_scores.csv",
        "run1_offbeam_larcv_cropped_full_set_scores.csv",
    ]
}


def main() -> None:
    cfg = SelectionConfig(infer_base=INFER_BASE, out_base=OUT_BASE)
    run_selection_pipeline(cfg, samples_files_by_run=SAMPLES_FILES_BY_RUN)


if __name__ == "__main__":
    main()
