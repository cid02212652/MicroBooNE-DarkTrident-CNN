#!/usr/bin/env python3
import os, re, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

# --- Paths you already use
BASE   = os.path.expanduser("~/dark_tridents_wspace")
CFG    = os.path.join(BASE, "DM-CNN", "cfg", "inference_config_binary.cfg")
JOBDIR = os.path.join(BASE, "DM-CNN")
JOBFILE= "inference_dmcnn.job"  

# --- Where the RUN-1 files live (outside your repo)
RUN1_BASE = "/vols/sbn/uboone/darkTridents/data/larcv_files"
RUN_DIRS  = ["run1_signal", "run1_samples"]  # both categories
# CSVs you generated earlier:
CSV_DIR   = os.path.join(BASE, "run1_datasets")

# Optional flags
DRYRUN = False

def read(p):  return Path(p).read_text(encoding="utf-8")
def write(p,s): Path(p).write_text(s, encoding="utf-8")

def edit_cfg(text, file_root, file_csv, tag):
    repls = [
        (r'^\s*name\s*=.*$',       f'name=str("{tag}")'),
        (r'^\s*input_file\s*=.*$', f'input_file = "{file_root}"'),
        (r'^\s*input_csv\s*=.*$',  f'input_csv = "{file_csv}"'),
    ]
    new = text
    for pat, rep in repls:
        new = re.sub(pat, rep, new, flags=re.MULTILINE) if re.search(pat, new, flags=re.MULTILINE) else (new.rstrip() + "\n" + rep + "\n")
    return new

def submit():
    proc = subprocess.run(["condor_submit", JOBFILE], cwd=JOBDIR, text=True, capture_output=True)
    print(proc.stdout.strip())
    if proc.returncode != 0 and proc.stderr: print(proc.stderr.strip())
    return proc.returncode, proc.stdout

def parse_cluster_id(s):
    m = re.search(r"cluster\s+(\d+)", s or "")
    return m.group(1) if m else None

def find_job_log_template(job_text):
    for ln in job_text.splitlines():
        if ln.strip().lower().startswith("log"):
            return ln.split("=",1)[1].strip()
    return "logs/infer_dmcnn.$(CLUSTER).log"

def condor_wait(log_template, cluster_id):
    log_path = log_template.replace("$(CLUSTER)", cluster_id)
    # wait until log appears (idle jobs may delay)
    for _ in range(180):
        if os.path.exists(log_path): break
        time.sleep(5)
    if os.path.exists(log_path):
        subprocess.run(["condor_wait", log_path], check=False)
    else:
        print(f"[warn] log not found yet: {log_path}")

def main():
    assert os.path.isfile(CFG), f"Missing cfg: {CFG}"
    assert os.path.isdir(JOBDIR), f"Missing jobdir: {JOBDIR}"

    cfg0 = read(CFG)
    job_txt = read(os.path.join(JOBDIR, JOBFILE))
    log_tmpl = find_job_log_template(job_txt)

    # backup cfg
    backup = f"{CFG}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(CFG, backup)
    print(f"[info] Backed up -> {backup}")

    try:
        # discover all *cropped.root files in both folders
        roots = []
        for sub in RUN_DIRS:
            here = os.path.join(RUN1_BASE, sub)
            for p in sorted(Path(here).glob("*_cropped.root")):
                roots.append((sub, str(p)))

        if not roots:
            print("[warn] No cropped ROOT files found.")
            return

        for sub, file_root in roots:
            base = Path(file_root).stem                    # e.g. run1_dt_ratio_0.6_ma_0.01_pi0_larcv_cropped
            file_csv = os.path.join(CSV_DIR, f"{sub}__{base}.csv")
            if not os.path.isfile(file_csv):
                print(f"[skip] CSV not found for {base} -> {file_csv}")
                continue

            tag = f"infer_{sub}__{base}"
            print(f"\n[run] {tag}")
            edited = edit_cfg(cfg0, file_root, file_csv, tag)
            write(CFG, edited)
            for ln in edited.splitlines():
                if re.match(r'^\s*(name|input_file|input_csv)\s*=', ln):
                    print("[cfg]", ln)

            if DRYRUN:
                print("[dry-run] condor_submit", JOBFILE)
                continue

            rc, out = submit()
            cid = parse_cluster_id(out)
            if cid: condor_wait(log_tmpl, cid)
            time.sleep(1)

        print("\n[info] Submissions done.")

    finally:
        shutil.move(backup, CFG)
        print(f"[info] Restored cfg from {backup}")

if __name__ == "__main__":
    main()
