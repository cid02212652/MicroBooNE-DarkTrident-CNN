#!/usr/bin/env python3
"""
Wrapper script for the task stage (occlusion_tasks.csv + occlusion_tasks.list).
"""

from pathlib import Path
from occlusion_selection_tools import TaskConfig, build_occlusion_tasks, write_method_task_lists


PROJECT = Path("/home/hep/an1522/dark_tridents_wspace")
SEL_DIR = PROJECT / "outputs" / "inference" / "_occlusion_selections"
LARCV_BASE = Path("/vols/sbn/uboone/darkTridents/data/larcv_files")

WEIGHTS = {
    "mpid": Path(
        "/home/hep/an1522/dark_tridents_wspace/outputs/weights/DM-CNN_model_20260116-10_22_PM_epoch_4_batch_id_1961_labels_2_title_0.001_AG_GN_LM_TRAINING_step_9821.pwf"
    ),
    "resnet34_gn": Path(
        "/home/hep/an1522/dark_tridents_wspace/outputs/weights/resnet34_gn/resnet34_gn_model_20260123-12_20_AM_epoch_4_batch_id_1961_labels_2_step_9821.pwf"
    ),
}

TASKS_CSV = SEL_DIR / "occlusion_tasks.csv"
TASKS_LIST = SEL_DIR / "occlusion_tasks.list"

# Extra method list files (same tasks, different output folders)
EXTRA_METHODS = ["integrated_gradients", "gradcam", "gradcampp", "scorecam"]


def main() -> None:
    cfg = TaskConfig(
        project_dir=PROJECT,
        selection_dir=SEL_DIR,
        larcv_base=LARCV_BASE,
        weights=WEIGHTS,
        tasks_csv=TASKS_CSV,
        tasks_list=TASKS_LIST,
    )
    tasks = build_occlusion_tasks(cfg)
    print("Wrote:", TASKS_CSV, "rows=", len(tasks))
    print("Wrote:", TASKS_LIST)

    written = write_method_task_lists(TASKS_LIST, EXTRA_METHODS)
    for out_path in written:
        print("Wrote:", out_path)


if __name__ == "__main__":
    main()
