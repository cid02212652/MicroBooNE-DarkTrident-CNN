#!/usr/bin/env python3
import os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- I/O paths (same defaults as before)
OUTDIR_INF = os.path.expanduser("~/dark_tridents_wspace/outputs/inference/run1")
PLOT_OUT   = os.path.expanduser("~/dark_tridents_wspace/outputs/inference/run1/run1_stacked_hist3.png")

# ---- Column name guesses (case-insensitive)
SCORE_CANDIDATES = ["score","signal_score","y_pred","pred","prob","prob_signal"]
LABEL_CANDIDATES = ["label","y_true","target"]  # unused in the plot but left for parity

def classify(path):
    name = os.path.basename(path)
    low  = name.lower()

    if "dt_ratio_0.6_ma_0.05" in low:
        return "Dark trident"
    if "offbeam" in low:
        return "Beam-off"
    if "numi_dirt" in low:          # covers run1_dirt_* and numi_dirt
        return "Out of cryo"
    if "numi_nu_overlay" in low:
        return "In cryo ν"
    return None

def find_score_key(header_lower):
    """Return the actual header key (original case) for a score-like column."""
    # header_lower: dict {lower_name: original_name}
    for cand in SCORE_CANDIDATES:
        if cand in header_lower:
            return header_lower[cand]
    # fallback: choose the first column that can be parsed as float later
    # (we’ll discover this in read loop if needed)
    return None

def read_scores_from_csv(path):
    """Read the score column from CSV into a 1D numpy array (clipped to [0,1])."""
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            return np.empty((0,), dtype=float)

        # map lower->original case for header lookup
        header_lower = {h.strip().lower(): h for h in header}
        score_key = find_score_key(header_lower)

        # If we didn't find a known score column, we’ll pick the first numeric column we see.
        numeric_col_idx = None
        numeric_vals = []

        # Prepare index for fast access
        idx_map = {name: i for i, name in enumerate(header)}
        score_idx = idx_map.get(score_key) if score_key is not None else None

        # Read rows
        for row in rdr:
            if not row:
                continue
            # Try preferred score column first
            if score_idx is not None and score_idx < len(row):
                try:
                    v = float(row[score_idx])
                    numeric_vals.append(v)
                    continue
                except (ValueError, TypeError):
                    pass  # fall through to numeric discovery

            # Discover a numeric column if not chosen yet
            if numeric_col_idx is None:
                for i, cell in enumerate(row):
                    try:
                        _ = float(cell)
                        numeric_col_idx = i
                        break
                    except (ValueError, TypeError):
                        continue
            if numeric_col_idx is not None and numeric_col_idx < len(row):
                try:
                    v = float(row[numeric_col_idx])
                    numeric_vals.append(v)
                except (ValueError, TypeError):
                    # skip non-numeric rows
                    pass

        if not numeric_vals:
            return np.empty((0,), dtype=float)

        vals = np.asarray(numeric_vals, dtype=float)
        return np.clip(vals, 0.0, 1.0)

def main():
    # collect scores per class
    scores = {"In cryo ν":[], "Out of cryo":[], "Beam-off":[], "Dark trident":[]}

    csvs = sorted(glob.glob(os.path.join(OUTDIR_INF, "*DM-CNN_scores_*.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs found in {OUTDIR_INF}. Did inference finish?")

    for f in csvs:
        cat = classify(f)
        if cat is None:
            continue
        vals = read_scores_from_csv(f)
        if vals.size:
            scores[cat].append(vals)

    # flatten to single arrays
    for k in list(scores.keys()):
        scores[k] = np.concatenate(scores[k]) if scores[k] else np.empty((0,), dtype=float)
        
    # ---- plotting
    plt.figure(figsize=(7.0,5.0), dpi=200)
    bins = np.linspace(0,1,15)

    stack_order = ["In cryo ν","Out of cryo","Beam-off"]  # backgrounds
    bg_data = [scores[k] for k in stack_order]
    labels  = ["In cryo ν","Out of cryo","Beam-off"]
    colors  = ["#7b1fa2","#0b0b6b","#71c7ec"]
    # weights = [np.full(len(x), w) for x, w in zip(bg_data, [0.09, 0.10, 0.66])]
    weights = [np.full(len(x), w) for x, w in zip(bg_data, [1, 1, 1])]

    # stacked backgrounds (force exact range)
    plt.hist(bg_data, bins=bins, stacked=True, label=labels, color=colors,
             weights=weights, range=(0, 1))

    # signal outline (same range)
    plt.hist(scores["Dark trident"], bins=bins, histtype="step", linewidth=1.8,
             label="Dark trident", color="#e31a1c", range=(0, 1))

    # clamp axes + remove x padding
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.margins(x=0)

    plt.title("MicroBooNE Simulation Run 1")
    plt.xlabel("Classifier score")
    plt.ylabel("Events")
    plt.legend(framealpha=1, loc='upper center')

    # cut line + marker (compute after axes are fixed)
    cut_val = 0.5
    plt.axvline(cut_val, linestyle="--", color="k")
    y_top = ax.get_ylim()[1] * 0.9
    plt.plot([cut_val], [y_top], markersize=8, color="red")

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_OUT), exist_ok=True)
    plt.savefig(PLOT_OUT)
    print("Wrote:", PLOT_OUT)

if __name__ == "__main__":
    main()
