#!/usr/bin/env python3
import os, re, shutil, subprocess, sys, time
from datetime import datetime

BASE   = os.path.expanduser("~/dark_tridents_wspace")
CFG    = os.path.join(BASE, "DM-CNN", "cfg", "inference_config_binary.cfg")
JOBDIR = os.path.join(BASE, "DM-CNN")
JOBFILE= "inference_dmcnn.job"

# dataset_stem -> name (used for name=str("..."))
PAIRS = [
    ("cosmics_corsika_test_set", "inference_cosmics_corsika"),
    ("dm_corsika_test_set",      "inference_dm_corsika"),
    ("dm_signal_only_test_set",  "inference_dm_signal_only"),
    ("ncpi0_corsika_test_set",   "inference_ncpi0_corsika"),
    ("ncpi0_only_test_set",      "inference_ncpi0_only"),
]

DRYRUN  = ("--dry-run" in sys.argv)
NOWAIT  = ("--no-wait" in sys.argv)  # optional: keep old behavior

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def write_text(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def edit_cfg(text, dataset_stem, tag):
    """Replace only name, input_file, input_csv lines."""
    patterns = [
        (r'^\s*name\s*=.*$',       f'name=str("{tag}")'),
        (r'^\s*input_file\s*=.*$', f'input_file = "/workspace/cnn_datasets/{dataset_stem}.root"'),
        (r'^\s*input_csv\s*=.*$',  f'input_csv = "/workspace/cnn_datasets/{dataset_stem}.csv"'),
    ]
    new = text
    for pat, repl in patterns:
        if re.search(pat, new, flags=re.MULTILINE):
            new = re.sub(pat, repl, new, flags=re.MULTILINE)
        else:
            new = new.rstrip() + "\n" + repl + "\n"
    return new

def show_cfg_preview(text):
    for ln in text.splitlines():
        if re.match(r'^\s*(name|input_file|input_csv)\s*=', ln):
            print("[cfg]", ln)

def submit_job():
    if DRYRUN:
        print(f"[dry-run] Would run: (cd {JOBDIR} && condor_submit {JOBFILE})")
        return 0, "1 job(s) submitted to cluster 0000000."  # fake
    print(f"[submit] condor_submit {JOBFILE}")
    proc = subprocess.run(["condor_submit", JOBFILE], cwd=JOBDIR, text=True,
                          capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
    return proc.returncode, proc.stdout

def parse_cluster_id(submit_stdout):
    # Typical line: "1 job(s) submitted to cluster 3566777."
    m = re.search(r'cluster\s+(\d+)', submit_stdout or "")
    return m.group(1) if m else None

def read_log_template(job_text):
    # Find the 'log = ...' line from your existing job
    for line in job_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.lower().startswith("log"):
            _, val = line.split("=", 1)
            return val.strip()
    # fallback to a common pattern if not present
    return "logs/infer_dmcnn.$(CLUSTER).log"

def condor_wait_on(cluster_id, log_template):
    # Replace $(CLUSTER) with actual cluster id; condor_wait handles $(PROCESS)
    log_path = log_template.replace("$(CLUSTER)", cluster_id)
    print(f"[wait] looking for job log: {log_path}")

    # Wait for the log file to be created (jobs that idle won't make it immediately)
    wait_secs = 0
    max_wait = 1800       # 30 minutes total
    interval = 10         # check every 10s

    while not os.path.exists(log_path) and wait_secs < max_wait:
        time.sleep(interval)
        wait_secs += interval
        print(f"[wait] log not present yet ({wait_secs}s)… still waiting")

    if not os.path.exists(log_path):
        print(f"[warn] log never appeared after {max_wait}s: {log_path}", file=sys.stderr)
        return 1

    print(f"[wait] condor_wait {log_path}")
    proc = subprocess.run(["condor_wait", log_path], text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.returncode

def main():
    if not os.path.isfile(CFG):
        print(f"[error] config not found: {CFG}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(JOBDIR):
        print(f"[error] job dir not found: {JOBDIR}", file=sys.stderr)
        sys.exit(1)

    # Read once
    original_cfg = read_text(CFG)
    job_text     = read_text(os.path.join(JOBDIR, JOBFILE))
    log_template = read_log_template(job_text)

    # Backup config
    backup = f"{CFG}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(CFG, backup)
    print(f"[info] Backed up config -> {backup}")

    try:
        for dataset_stem, tag in PAIRS:
            print(f"\n[run] dataset={dataset_stem}  name={tag}")

            edited = edit_cfg(original_cfg, dataset_stem, tag)
            write_text(CFG, edited)
            show_cfg_preview(edited)

            rc, out = submit_job()
            if rc != 0:
                print("[warn] submit failed, skipping wait.", file=sys.stderr)
                continue

            cluster_id = parse_cluster_id(out)
            if cluster_id:
                if NOWAIT:
                    print(f"[info] submitted {cluster_id}; not waiting (--no-wait).")
                else:
                    condor_wait_on(cluster_id, log_template)
            else:
                print("[warn] could not parse cluster id; sleeping 10s and continuing.", file=sys.stderr)
                time.sleep(10)

            # brief pause to avoid log stamp collisions
            time.sleep(2)

        print("\n[info] All submissions handled.")

    finally:
        # Restore original config so your repo stays unchanged
        shutil.move(backup, CFG)
        print(f"[info] Restored original config from {backup}")

if __name__ == "__main__":
    main()
