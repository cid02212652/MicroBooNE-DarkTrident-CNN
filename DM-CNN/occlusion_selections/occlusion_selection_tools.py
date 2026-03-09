"""
Utilities for selecting interesting events for occlusion studies and for quickly
querying inference score tables.

The pipeline is intentionally split into two stages:

1) Selection stage (from inference score CSVs)
   - Scan inference folders and load all *_scores.csv files.
   - Clean sentinel values, optionally restrict signal to a specific mass point.
   - Pick a compact, diverse set of events (tails, borderline, low-pixel oddities, busy controls).
   - Optionally add "model disagreement" events between a reference model and another folder.
   - Write:
       - master__{run}_{kind}.csv (one master table per dataset)
       - to_occlude__{run}_{kind}__{folder}.csv (one table per model folder)
       - matching_report.csv

2) Task stage (from to_occlude CSVs)
   - Convert per-folder selection tables into a flat task list:
       - occlusion_tasks.csv
       - occlusion_tasks.list (space-separated, convenient for GNU parallel)

The functions below are written to avoid reliance on global variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import re

import pandas as pd

# Event identity columns present in the score CSVs.
KEYS: List[str] = ["run_number", "subrun_number", "event_number"]

# Columns required from each *_scores.csv file.
NEEDED_COLS: List[str] = KEYS + ["signal_score", "entry_number", "n_pixels"]

# Sentinel / invalid values observed in score tables.
DEFAULT_BAD_SCORE = -999999.9
DEFAULT_BAD_PIXELS = -1


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for selection and cleaning.

    Attributes
    ----------
    infer_base:
        Directory containing inference folders (each folder holds *_scores.csv files).
    out_base:
        Output directory for selection CSVs.
    signal_keep_substr:
        For signal datasets, keep only rows whose score CSV filename contains this substring.
        Set to None to keep all signal files.
    bad_score, bad_pixels:
        Sentinel values to drop.
    exclude_suffixes:
        Inference folder suffixes to ignore (e.g. *_pdf, *_png).
    exclude_prefixes:
        Inference folder prefixes to ignore (e.g. folders starting with "_").
    """

    infer_base: Path
    out_base: Path
    larcv_base: Path = Path("/vols/sbn/uboone/darkTridents/data/larcv_files")
    signal_keep_substr: Optional[str] = "dt_ratio_0.6_ma_0.05_pi0"
    bad_score: float = DEFAULT_BAD_SCORE
    bad_pixels: int = DEFAULT_BAD_PIXELS
    exclude_suffixes: Tuple[str, ...] = ("_pdf", "_png")
    exclude_prefixes: Tuple[str, ...] = ("_",)


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for task list generation."""

    project_dir: Path
    selection_dir: Path
    larcv_base: Path
    weights: Dict[str, Path]
    tasks_csv: Path
    tasks_list: Path


# -------------------------
# Folder parsing and loading
# -------------------------


_FOLDER_RE = re.compile(r"^(run\d+)_(samples|signal)(?:_(.+))?$")


def parse_inference_folder_name(folder_name: str) -> Optional[Tuple[str, str, str]]:
    """Parse an inference folder name.

    Expected formats
    ----------------
    - run1_samples
    - run1_samples_resnet34_gn

    Returns
    -------
    (run, kind, model_tag) or None if the name doesn't match.
    model_tag defaults to "mpid" when the folder has no explicit suffix.
    """
    m = _FOLDER_RE.match(folder_name)
    if not m:
        return None
    run, kind, suffix = m.group(1), m.group(2), m.group(3)
    if suffix in ("resnet34_bn", "resnet18_bn", "resnet18_gn"):
        return None
    model_tag = suffix if suffix else "mpid"
    return run, kind, model_tag


def list_inference_folders(
    infer_base: Path,
    exclude_suffixes: Sequence[str] = ("_pdf", "_png"),
    exclude_prefixes: Sequence[str] = ("_",),
) -> List[Path]:
    """List inference folders (one level under infer_base)."""
    out: List[Path] = []
    for p in infer_base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if any(name.endswith(suf) for suf in exclude_suffixes):
            continue
        if any(name.startswith(pre) for pre in exclude_prefixes):
            continue
        if parse_inference_folder_name(name) is None:
            continue
        out.append(p)
    return sorted(out)


def read_scores_csv(
    path: Path, needed_cols: Sequence[str] = NEEDED_COLS
) -> pd.DataFrame:
    """Read one *_scores.csv file and standardize types."""
    df = pd.read_csv(path)
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    df = df[list(needed_cols)].copy()
    df["entry_number"] = df["entry_number"].astype(int)
    df["n_pixels"] = df["n_pixels"].astype(int)
    return df


def read_scores_folder(folder: Path) -> Optional[pd.DataFrame]:
    """Read all *_scores.csv files from a folder into one dataframe."""
    files = sorted(folder.glob("*_scores.csv"))
    if not files:
        return None

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = read_scores_csv(f)
        except Exception:
            continue
        df["__file"] = f.name
        df["__folder"] = folder.name
        parsed = parse_inference_folder_name(folder.name)
        if parsed is None:
            continue
        run, kind, model_tag = parsed
        df["__run"] = run
        df["__kind"] = kind
        df["__model"] = model_tag
        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_all_scores(cfg: SelectionConfig) -> pd.DataFrame:
    """Load all inference score tables under cfg.infer_base."""
    folders = list_inference_folders(
        cfg.infer_base,
        exclude_suffixes=cfg.exclude_suffixes,
        exclude_prefixes=cfg.exclude_prefixes,
    )
    dfs: List[pd.DataFrame] = []
    for folder in folders:
        df = read_scores_folder(folder)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(
            columns=NEEDED_COLS + ["__file", "__folder", "__run", "__kind", "__model"]
        )

    return pd.concat(dfs, ignore_index=True)


def clean_scores(
    all_scores: pd.DataFrame,
    *,
    bad_score: float = DEFAULT_BAD_SCORE,
    bad_pixels: int = DEFAULT_BAD_PIXELS,
    signal_keep_substr: Optional[str] = "dt_ratio_0.6_ma_0.05_pi0",
) -> pd.DataFrame:
    """Drop sentinel rows and optionally restrict signal files to a substring."""
    if all_scores.empty:
        return all_scores.copy()

    mask_good = (
        (all_scores["signal_score"] != bad_score)
        & (all_scores["n_pixels"] != bad_pixels)
        & (all_scores["n_pixels"] >= 0)
    )
    clean = all_scores.loc[mask_good].copy()

    if signal_keep_substr:
        clean = clean[
            (clean["__kind"] == "samples")
            | (
                (clean["__kind"] == "signal")
                & clean["__file"].str.contains(signal_keep_substr, na=False)
            )
        ].copy()

    return clean


# -------------------------
# Selection logic
# -------------------------


def _borderline_mask(scores: pd.Series, target: float, frac: float) -> pd.Series:
    dist = (scores - target).abs()
    thr = dist.quantile(frac)
    return dist <= thr


def select_occlusion_set(
    df_folder: pd.DataFrame,
    *,
    kind: str,
    A: int = 1,
    B: int = 1,
    C: int = 1,
    D: int = 2,
    # A: int = 10,
    # B: int = 10,
    # C: int = 10,
    # D: int = 5,
    samples_high_q: float = 0.995,
    signal_low_q: float = 0.005,
    borderline_target: float = 0.5,
    border_frac: float = 0.01,
    weird_lowpix_q: float = 0.10,
) -> pd.DataFrame:
    """Select a compact set of events for occlusion for a single folder.

    The selection is deliberately mixed:
    - A: score tail (high tail for samples, low tail for signal)
    - B: borderline events (closest to borderline_target)
    - C: low-pixel oddities within the tail
    - D: "busy" high-pixel controls (low-score busy samples or high-score busy signal)

    The output is deduped by KEYS and annotated with a compact `pick_reason` tag.
    """
    if df_folder.empty:
        return df_folder.copy()

    df = df_folder.copy()

    if kind not in ("samples", "signal"):
        raise ValueError(f"kind must be 'samples' or 'signal', got {kind}")

    # A) Tail events (different for samples vs signal)
    if kind == "samples":
        q = df["signal_score"].quantile(samples_high_q)
        A_df = (
            df[df["signal_score"] >= q]
            .sort_values("signal_score", ascending=False)
            .head(A)
        )
    else:
        q = df["signal_score"].quantile(signal_low_q)
        A_df = (
            df[df["signal_score"] <= q]
            .sort_values("signal_score", ascending=True)
            .head(A)
        )

    # B) Borderline events
    df2 = df.assign(__dist=(df["signal_score"] - borderline_target).abs())
    k = max(1, int(border_frac * len(df2)))
    B_df = (
        df2.sort_values("__dist", ascending=True)
        .head(max(B, k))
        .drop(columns="__dist")
        .head(B)
    )

    # C) Low-pixel oddities within A_df
    if not A_df.empty:
        pix_cut = A_df["n_pixels"].quantile(weird_lowpix_q)
        C_df = (
            A_df[A_df["n_pixels"] <= pix_cut]
            .sort_values(
                ["n_pixels", "signal_score"], ascending=[True, (kind == "signal")]
            )
            .head(C)
        )
    else:
        C_df = df.iloc[0:0].copy()

    # D) Busy controls (upper quartile by pixels)
    pix_hi = df["n_pixels"].quantile(0.75)
    busy = df[df["n_pixels"] >= pix_hi].copy()
    if kind == "samples":
        D_df = busy.sort_values("signal_score", ascending=True).head(D)
    else:
        D_df = busy.sort_values("signal_score", ascending=False).head(D)

    pick = pd.concat([A_df, B_df, C_df, D_df], ignore_index=True)
    pick = pick.drop_duplicates(subset=KEYS, keep="first").copy()

    border_mask = _borderline_mask(df["signal_score"], borderline_target, border_frac)

    def reason(row: pd.Series) -> str:
        tags: List[str] = []
        if kind == "samples":
            if row["signal_score"] >= df["signal_score"].quantile(samples_high_q):
                tags.append("A_high_tail")
        else:
            if row["signal_score"] <= df["signal_score"].quantile(signal_low_q):
                tags.append("A_low_tail")

        if border_mask.loc[row.name] if row.name in border_mask.index else False:
            tags.append("B_border")

        if (not A_df.empty) and (
            row["n_pixels"] <= A_df["n_pixels"].quantile(weird_lowpix_q)
        ):
            tags.append("C_weird_lowpix")

        if row["n_pixels"] >= pix_hi:
            tags.append("D_busy")

        return "+".join(sorted(set(tags))) if tags else "picked"

    # Using original index inside `pick` is not reliable; compute tags with direct checks.
    # The tag rules are simple enough to re-evaluate against the source df.
    def reason_row(row: pd.Series) -> str:
        tags: List[str] = []
        if kind == "samples":
            if row["signal_score"] >= df["signal_score"].quantile(samples_high_q):
                tags.append("A_high_tail")
        else:
            if row["signal_score"] <= df["signal_score"].quantile(signal_low_q):
                tags.append("A_low_tail")

        if abs(row["signal_score"] - borderline_target) <= (
            df["signal_score"] - borderline_target
        ).abs().quantile(border_frac):
            tags.append("B_border")

        if (not A_df.empty) and (
            row["n_pixels"] <= A_df["n_pixels"].quantile(weird_lowpix_q)
        ):
            tags.append("C_weird_lowpix")

        if row["n_pixels"] >= pix_hi:
            tags.append("D_busy")

        return "+".join(sorted(set(tags))) if tags else "picked"

    pick["pick_reason"] = pick.apply(reason_row, axis=1)

    # Propagate metadata columns (all rows within df_folder share them).
    for col in ["__file", "__folder", "__run", "__kind", "__model"]:
        if col in df_folder.columns and col not in pick.columns:
            pick[col] = df_folder[col].iloc[0]

    return pick


def select_occlusion_set_stratified(
    df_folder: pd.DataFrame,
    *,
    kind: str,
    files: Optional[Sequence[str]] = None,
    per_file_counts: Tuple[int, int, int, int] = (1, 1, 1, 1),
    # per_file_counts: Tuple[int, int, int, int] = (4, 4, 4, 2),
    **kwargs,
) -> pd.DataFrame:
    """Select events per score-file to enforce representation (useful for samples)."""
    if df_folder.empty:
        return df_folder.copy()

    df = df_folder.copy()
    if files is not None:
        df = df[df["__file"].isin(files)].copy()
    if df.empty:
        return df

    A, B, C, D = per_file_counts
    picks: List[pd.DataFrame] = []
    for _, g in df.groupby("__file"):
        p = select_occlusion_set(g, kind=kind, A=A, B=B, C=C, D=D, **kwargs)
        if p is not None and not p.empty:
            picks.append(p)

    if not picks:
        return df.iloc[0:0].copy()

    out = pd.concat(picks, ignore_index=True)
    out = out.drop_duplicates(subset=KEYS, keep="first")
    return out


def add_disagreement_picks(
    clean: pd.DataFrame,
    *,
    picks_ref: pd.DataFrame,
    run: str,
    kind: str,
    other_folder: str,
    E_abs: int = 2,
    E_flip: int = 2,
    # E_abs: int = 10,
    # E_flip: int = 10,
    thr: float = 0.5,
    margin: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Append model-disagreement events to an existing pick set.

    Disagreement candidates are constructed from the overlap on KEYS between:
    - the reference folder (taken from picks_ref["__folder"].iloc[0])
    - other_folder

    The method selects:
    - top E_abs rows by |delta|
    - top E_flip rows that are confidently on opposite sides of thr
    """
    if picks_ref.empty:
        return picks_ref.copy(), pd.DataFrame()

    if "__folder" not in picks_ref.columns:
        raise ValueError("picks_ref must include __folder")

    ref_folder = str(picks_ref["__folder"].iloc[0])

    subset = clean[(clean["__run"] == run) & (clean["__kind"] == kind)].copy()
    df_ref = subset[subset["__folder"] == ref_folder].copy()
    df_other = subset[subset["__folder"] == other_folder].copy()

    if df_other.empty or df_ref.empty:
        return picks_ref.copy(), pd.DataFrame()

    df_ref = df_ref[
        KEYS + ["signal_score", "n_pixels", "__file", "__folder", "entry_number"]
    ].rename(columns={"signal_score": "score_ref"})
    df_other = df_other[KEYS + ["signal_score"]].rename(
        columns={"signal_score": "score_other"}
    )

    m = df_ref.merge(df_other, on=KEYS, how="inner")
    if m.empty:
        return picks_ref.copy(), pd.DataFrame()

    m["delta"] = m["score_other"] - m["score_ref"]
    m["abs_delta"] = m["delta"].abs()
    m["flip_strong"] = (
        (m["score_ref"] >= thr + margin) & (m["score_other"] <= thr - margin)
    ) | ((m["score_other"] >= thr + margin) & (m["score_ref"] <= thr - margin))

    top_abs = m.sort_values("abs_delta", ascending=False).head(E_abs)
    top_flip = (
        m[m["flip_strong"]].sort_values("abs_delta", ascending=False).head(E_flip)
    )

    extra = pd.concat([top_abs, top_flip], ignore_index=True).drop_duplicates(
        subset=KEYS, keep="first"
    )

    extra_rows = subset.merge(extra[KEYS], on=KEYS, how="inner")

    def mk_reason(row: pd.Series) -> str:
        rr = extra.loc[
            (extra["run_number"] == row["run_number"])
            & (extra["subrun_number"] == row["subrun_number"])
            & (extra["event_number"] == row["event_number"])
        ].iloc[0]
        tags = ["E_disagree", f"E_vs_{other_folder}"]
        if bool(rr["flip_strong"]):
            tags.append(f"E_strong_flip_m{margin}")
        tags.append("E_other_gt_ref" if rr["delta"] > 0 else "E_ref_gt_other")
        return "+".join(tags)

    extra_rows = extra_rows.copy()
    extra_rows["pick_reason"] = extra_rows.apply(mk_reason, axis=1)

    out = pd.concat([picks_ref, extra_rows], ignore_index=True).drop_duplicates(
        subset=KEYS, keep="first"
    )
    return out, m


# -------------------------
# Master tables and outputs
# -------------------------


def scores_to_root(scores_file: str) -> str:
    """Convert a score CSV filename to the corresponding larcv root filename."""
    return re.sub(r"_scores\.csv$", ".root", Path(scores_file).name)


def build_master(
    clean: pd.DataFrame,
    *,
    run: str,
    kind: str,
    reference_folder: Optional[str] = None,
    samples_files: Optional[Sequence[str]] = None,
    disagreement_against: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Build the master table for a dataset (run + kind)."""
    subset = clean[(clean["__run"] == run) & (clean["__kind"] == kind)].copy()
    folders = sorted(subset["__folder"].unique().tolist())
    if not folders:
        return None

    ref = reference_folder if (reference_folder in folders) else folders[0]
    df_ref = subset[subset["__folder"] == ref].copy()
    ref_clean = ref.replace(f"{run}_{kind}_", "")

    if kind == "samples":
        picks = select_occlusion_set_stratified(
            df_ref,
            kind="samples",
            files=samples_files,
            per_file_counts=(1, 1, 1, 1),
            # per_file_counts=(4, 4, 4, 2),
        )
    else:
        picks = select_occlusion_set(df_ref, kind="signal")

    if picks is None or picks.empty:
        return None

    if disagreement_against:
        picks, _ = add_disagreement_picks(
            clean,
            picks_ref=picks,
            run=run,
            kind=kind,
            other_folder=disagreement_against,
            E_abs=1,
            E_flip=1,
            # E_abs=10,
            # E_flip=10,
            margin=0.20,
        )

    # Base columns for the master
    master = picks[["pick_reason", "entry_number", "__file"]].copy()

    # Merge signal scores from all folders into master
    merge_on = ["entry_number", "__file"]
    for folder in folders:
        folder_clean = folder.replace(f"{run}_{kind}_", "")
        df_folder = subset[subset["__folder"] == folder][
            merge_on + ["signal_score"]
        ].rename(columns={"signal_score": f"{folder_clean}__signal_score"})
        master = master.merge(df_folder, on=merge_on, how="left")

    master["n_pixels"] = picks["n_pixels"].astype(int).values
    master = master.rename(columns={"__file": "scores_file"})
    master["reference_folder"] = ref_clean
    master["dataset"] = f"{run}_{kind}"
    master[KEYS] = picks[KEYS].copy()
    return master


def write_to_occlude_files(
    clean: pd.DataFrame,
    *,
    master: pd.DataFrame,
    out_base: Path,
    larcv_base: Path,
) -> pd.DataFrame:
    """Write one to_occlude CSV per model folder and return a matching report dataframe."""
    dataset = str(master["dataset"].iloc[0])
    run, kind = dataset.split("_", 1)

    subset = clean[(clean["__run"] == run) & (clean["__kind"] == kind)].copy()
    folders = sorted(subset["__folder"].unique().tolist())

    report_rows: List[dict] = []
    for folder in folders:
        df_f = subset[subset["__folder"] == folder].copy()

        lookup = df_f[
            KEYS + ["entry_number", "signal_score", "n_pixels", "__file"]
        ].rename(columns={"__file": "scores_file"})
        lookup = lookup.drop_duplicates(subset=KEYS, keep="first")

        merged = master.merge(lookup, on=KEYS, how="left", suffixes=("_ref", "_folder"))

        matched = int(merged["entry_number_folder"].notna().sum())
        total = int(len(merged))

        to_occ = merged.dropna(subset=["entry_number_folder"]).copy()
        to_occ["entry_number"] = to_occ["entry_number_folder"].astype(int)
        to_occ["scores_file"] = to_occ["scores_file_folder"]

        # Root file is deterministic from scores_file; keep the full path here so downstream steps
        # don't need to re-derive it.
        root_dir = larcv_base / f"{run}_{kind}"
        to_occ["root_file"] = to_occ["scores_file"].apply(
            lambda s: str(root_dir / scores_to_root(s))
        )

        folder_clean = folder.replace(f"{run}_{kind}_", "")

        keep_cols = (
            KEYS
            + ["pick_reason", "entry_number"]
            + [c for c in to_occ.columns if c.endswith("__signal_score")]
            + [
                "n_pixels_folder",
                "scores_file",
                "root_file",
                "reference_folder",
                "dataset",
            ]
        )
        to_occ = to_occ[keep_cols].rename(columns={"n_pixels_folder": "n_pixels"})

        out = out_base / f"to_occlude__{dataset}__{folder_clean}.csv"
        to_occ.to_csv(out, index=False)

        report_rows.append(
            {
                "dataset": dataset,
                "folder": folder_clean,
                "total_master_events": total,
                "matched_events": matched,
                "wrote_csv": str(out),
            }
        )

    return pd.DataFrame(report_rows)


def run_selection_pipeline(
    cfg: SelectionConfig,
    *,
    runs: Sequence[str] = ("run1", "run3"),
    kinds: Sequence[str] = ("samples", "signal"),
    samples_files_by_run: Optional[Dict[str, Sequence[str]]] = None,
    disagreement_model: str = "resnet34_gn",
) -> Dict[str, pd.DataFrame]:
    """Run the selection stage and write master + to_occlude tables."""
    all_scores = load_all_scores(cfg)
    clean = clean_scores(
        all_scores,
        bad_score=cfg.bad_score,
        bad_pixels=cfg.bad_pixels,
        signal_keep_substr=cfg.signal_keep_substr,
    )

    cfg.out_base.mkdir(parents=True, exist_ok=True)

    masters: Dict[str, pd.DataFrame] = {}
    reports: List[pd.DataFrame] = []

    for run in runs:
        for kind in kinds:
            key = f"{run}_{kind}"
            samples_files = None
            if samples_files_by_run and (run in samples_files_by_run):
                samples_files = samples_files_by_run[run]

            disagreement_against = f"{run}_{kind}_{disagreement_model}"
            master = build_master(
                clean,
                run=run,
                kind=kind,
                samples_files=samples_files,
                disagreement_against=disagreement_against,
            )
            if master is None or master.empty:
                continue

            out = cfg.out_base / f"master__{run}_{kind}.csv"
            master.to_csv(out, index=False)
            masters[key] = master

            rep = write_to_occlude_files(
                clean,
                master=master,
                out_base=cfg.out_base,
                larcv_base=cfg.larcv_base,
            )
            reports.append(rep)

    if reports:
        report = pd.concat(reports, ignore_index=True)
        report_path = cfg.out_base / "matching_report.csv"
        report.to_csv(report_path, index=False)

    return masters


# -------------------------
# Task list generation
# -------------------------


def sanitize_tag(tag: str) -> str:
    """Make a string safe-ish as a single path component."""
    tag = str(tag).strip()
    tag = re.sub(r"\s+", "_", tag)
    tag = re.sub(r"[^0-9A-Za-z._+\-]+", "_", tag)
    return tag or "picked"


def folder_to_model_key(folder_name: str) -> str:
    """Map folder name -> weights key."""
    # Extend as needed; default to mpid baseline.
    if "resnet34_gn" in folder_name:
        return "resnet34_gn"
    return "mpid"


def outdir_for(project_dir: Path, tag: str, dataset: str, folder: str) -> Path:
    """Directory where occlusion outputs for this task should be written."""
    return project_dir / "outputs" / "occlusion" / tag / dataset / folder


def build_occlusion_tasks(task_cfg: TaskConfig) -> pd.DataFrame:
    """Create occlusion_tasks.csv and occlusion_tasks.list from to_occlude tables."""
    sel_dir = task_cfg.selection_dir
    to_occlude_files = sorted(sel_dir.glob("to_occlude__*.csv"))

    rows: List[dict] = []
    for f in to_occlude_files:
        parts = f.stem.split("__")
        if len(parts) < 3:
            continue
        dataset = parts[1]
        folder = "__".join(parts[2:])

        df = pd.read_csv(f)
        needed = {"root_file", "entry_number", "n_pixels", "pick_reason"}
        if not needed.issubset(df.columns):
            continue

        model_key = folder_to_model_key(folder)
        wpath = task_cfg.weights.get(model_key)
        if wpath is None:
            raise KeyError(f"Missing weights entry for model_key='{model_key}'")

        for r in df.itertuples(index=False):
            tag = sanitize_tag(getattr(r, "pick_reason", "picked"))
            rows.append(
                {
                    "dataset": dataset,
                    "folder": folder,
                    "model_key": model_key,
                    "weight_file": str(wpath),
                    "root_file": str(getattr(r, "root_file")),
                    "entry_number": int(getattr(r, "entry_number")),
                    "n_pixels": int(getattr(r, "n_pixels")),
                    "out_dir": str(
                        outdir_for(task_cfg.project_dir, tag, dataset, folder)
                    ),
                    "tag": tag,
                }
            )

    tasks = pd.DataFrame(rows)
    task_cfg.tasks_csv.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(task_cfg.tasks_csv, index=False)

    cols = ["root_file", "weight_file", "out_dir", "entry_number", "n_pixels", "tag"]
    with open(task_cfg.tasks_list, "w") as f:
        for r in tasks[cols].itertuples(index=False):
            f.write(
                f"{r.root_file} {r.weight_file} {r.out_dir} {int(r.entry_number)} {int(r.n_pixels)} {r.tag}\n"
            )

    return tasks


def write_method_task_lists(
    base_tasks_list: Path,
    methods: Sequence[str],
    *,
    src_method: str = "occlusion",
    outputs_dirname: str = "outputs",
    out_dir: Optional[Path] = None,
    name_template: str = "{method}_tasks.list",
) -> List[Path]:
    """Create task-list files for other explanation methods by rewriting the out_dir field.

    The task runner list format is expected to be:

        root_file weight_file out_dir entry_number n_pixels tag

    Only the *out_dir* field (3rd token) is rewritten, to avoid accidental changes in
    root/weight paths.

    The rewrite is a simple path-component swap:

        /<outputs_dirname>/<src_method>/  ->  /<outputs_dirname>/<method>/

    Parameters
    ----------
    base_tasks_list:
        Path to the reference list file (usually occlusion_tasks.list).
    methods:
        Method directory names to generate (e.g. ["gradcam", "gradcampp"]).
    src_method:
        The method directory in the base list (default: "occlusion").
    outputs_dirname:
        Directory name used in the out_dir path (default: "outputs").
    out_dir:
        Where to write the new list files. Defaults to base_tasks_list.parent.
    name_template:
        Output filename template. Must include "{method}".

    Returns
    -------
    list[Path]
        Paths to the written list files.
    """
    out_dir = base_tasks_list.parent if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = base_tasks_list.read_text().splitlines()
    needle = f"/{outputs_dirname}/{src_method}/"

    written: List[Path] = []
    for method in methods:
        repl = f"/{outputs_dirname}/{method}/"
        out_lines: List[str] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 6:
                # Preserve malformed lines verbatim rather than guessing.
                out_lines.append(ln)
                continue

            out_path = parts[2]
            if needle not in out_path:
                raise ValueError(
                    f"Expected '{needle}' in out_dir field, got: {out_path}"
                )

            parts[2] = out_path.replace(needle, repl)
            out_lines.append(" ".join(parts))

        out_path = out_dir / name_template.format(method=method)
        out_path.write_text("\n".join(out_lines) + "\n")
        written.append(out_path)

    return written


# -------------------------
# Quick score querying helpers
# -------------------------


def load_scores_any(path: Path) -> pd.DataFrame:
    """Load a scores CSV or a folder of scores CSVs into a single dataframe.

    Adds `source_file` for folder loads.
    """
    if path.is_dir():
        dfs: List[pd.DataFrame] = []
        for p in sorted(path.glob("*_scores.csv")):
            df = read_scores_csv(p)
            df["source_file"] = p.name
            dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=NEEDED_COLS + ["source_file"])
        return pd.concat(dfs, ignore_index=True)
    return read_scores_csv(path)


def select_top(
    df: pd.DataFrame,
    *,
    by: str = "signal_score",
    n: int = 20,
    min_pixels: Optional[int] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """Return the top-N rows by `by` with simple optional cuts."""
    x = df.copy()
    if min_pixels is not None:
        x = x[x["n_pixels"] >= int(min_pixels)]
    if min_score is not None:
        x = x[x["signal_score"] >= float(min_score)]

    ascending = by != "signal_score"
    cols = [
        "run_number",
        "subrun_number",
        "event_number",
        "entry_number",
        "signal_score",
        "n_pixels",
    ]
    if "source_file" in x.columns:
        cols = ["source_file"] + cols
    return x.sort_values(by, ascending=ascending).head(int(n))[cols]


def nearest(
    df: pd.DataFrame,
    *,
    col: str,
    target: float,
    n: int = 10,
    min_pixels: Optional[int] = None,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """Return the N rows closest to `target` in column `col`."""
    x = df.copy()
    if min_pixels is not None:
        x = x[x["n_pixels"] >= int(min_pixels)]
    if min_score is not None:
        x = x[x["signal_score"] >= float(min_score)]
    if x.empty:
        return x

    d = (x[col] - float(target)).abs()
    out = x.loc[d.nsmallest(int(n)).index]
    cols = [
        "run_number",
        "subrun_number",
        "event_number",
        "entry_number",
        "signal_score",
        "n_pixels",
    ]
    if "source_file" in out.columns:
        cols = ["source_file"] + cols
    return out.sort_values(col)[cols]


def filter_range(
    df: pd.DataFrame,
    *,
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
    pixels_min: Optional[int] = None,
    pixels_max: Optional[int] = None,
) -> pd.DataFrame:
    """Filter a score table by score and pixel ranges."""
    x = df.copy()
    if score_min is not None:
        x = x[x["signal_score"] >= float(score_min)]
    if score_max is not None:
        x = x[x["signal_score"] <= float(score_max)]
    if pixels_min is not None:
        x = x[x["n_pixels"] >= int(pixels_min)]
    if pixels_max is not None:
        x = x[x["n_pixels"] <= int(pixels_max)]
    cols = [
        "run_number",
        "subrun_number",
        "event_number",
        "entry_number",
        "signal_score",
        "n_pixels",
    ]
    if "source_file" in x.columns:
        cols = ["source_file"] + cols
    return x[cols]


def occlusion_cmd(entry_number: int) -> str:
    """Small helper to build an occlusion script command snippet."""
    return f"python3 ./uboone/occlusion_analysis_CNN.py -n {int(entry_number)}"


# -------------------------
# Minimal CLIs
# -------------------------


def _cli_select() -> None:
    ap = argparse.ArgumentParser(
        description="Generate master/to_occlude selection tables."
    )
    ap.add_argument("--infer-base", type=Path, required=True)
    ap.add_argument("--out-base", type=Path, required=True)
    ap.add_argument("--signal-keep", type=str, default="dt_ratio_0.6_ma_0.05_pi0")
    args = ap.parse_args()

    cfg = SelectionConfig(
        infer_base=args.infer_base,
        out_base=args.out_base,
        signal_keep_substr=args.signal_keep if args.signal_keep else None,
    )

    samples_files_by_run = {
        "run1": [
            "run1_NuMI_dirt_larcv_cropped_scores.csv",
            "run1_NuMI_nu_overlay_larcv_cropped_scores.csv",
            "run1_offbeam_larcv_cropped_full_set_scores.csv",
        ]
    }
    run_selection_pipeline(cfg, samples_files_by_run=samples_files_by_run)


def _cli_tasks() -> None:
    ap = argparse.ArgumentParser(
        description="Build occlusion task lists from to_occlude tables."
    )
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--selection-dir", type=Path, required=True)
    ap.add_argument("--larcv-base", type=Path, required=True)
    ap.add_argument("--tasks-csv", type=Path, required=True)
    ap.add_argument("--tasks-list", type=Path, required=True)
    ap.add_argument("--mpid-weight", type=Path, required=True)
    ap.add_argument("--resnet34-gn-weight", type=Path, required=True)
    args = ap.parse_args()

    weights = {"mpid": args.mpid_weight, "resnet34_gn": args.resnet34_gn_weight}
    task_cfg = TaskConfig(
        project_dir=args.project_dir,
        selection_dir=args.selection_dir,
        larcv_base=args.larcv_base,
        weights=weights,
        tasks_csv=args.tasks_csv,
        tasks_list=args.tasks_list,
    )

    build_occlusion_tasks(task_cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["select", "tasks"], help="Which stage to run.")
    args, _ = ap.parse_known_args()
    if args.command == "select":
        _cli_select()
    else:
        _cli_tasks()


if __name__ == "__main__":
    main()
