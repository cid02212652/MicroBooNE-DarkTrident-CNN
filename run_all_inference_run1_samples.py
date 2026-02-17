#!/usr/bin/env python3
import os, re, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

# --- Base paths
BASE    = os.path.expanduser("~/dark_tridents_wspace")
CFG     = os.path.join(BASE, "DM-CNN", "cfg", "inference_config_binary_mpid_1.cfg")
JOBDIR  = os.path.join(BASE, "DM-CNN")
JOBFILE = "inference_dmcnn_1.job"

# --- Inputs
RUN1_BASE = "/vols/sbn/uboone/darkTridents/data/larcv_files"
ROOT_SUB  = "run1_samples"                 # ROOTs live in RUN1_BASE/ROOT_SUB/
CSV_DIR   = os.path.join(BASE, "run1_samples")  # your 3 CSVs live here

# --- Outputs (write directly here by editing output_dir in cfg)
OUT_DIR = os.path.join(BASE, "outputs", "inference", "run1_samples_mpid")

DRYRUN = False


def read(p):  return Path(p).read_text(encoding="utf-8")
def write(p,s): Path(p).write_text(s, encoding="utf-8")


def edit_cfg(text, file_root, file_csv, tag, out_dir):
    repls = [
        (r'^\s*name\s*=.*$',       f'name=str("{tag}")'),
        (r'^\s*input_file\s*=.*$', f'input_file = "{file_root}"'),
        (r'^\s*input_csv\s*=.*$',  f'input_csv = "{file_csv}"'),
        (r'^\s*output_dir\s*=.*$', f'output_dir = "{out_dir}/"'),
    ]
    new = text
    for pat, rep in repls:
        if re.search(pat, new, flags=re.MULTILINE):
            new = re.sub(pat, rep, new, flags=re.MULTILINE)
        else:
            new = new.rstrip() + "\n" + rep + "\n"
    return new


def submit():
    proc = subprocess.run(["condor_submit", JOBFILE], cwd=JOBDIR, text=True, capture_output=True)
    print(proc.stdout.strip())
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr.strip())
    return proc.returncode, proc.stdout


def parse_cluster_id(s):
    m = re.search(r"cluster\s+(\d+)", s or "")
    return m.group(1) if m else None


def find_job_log_template(job_text):
    for ln in job_text.splitlines():
        if ln.strip().lower().startswith("log"):
            return ln.split("=", 1)[1].strip()
    return "logs/infer_dmcnn.$(CLUSTER).log"


def condor_wait(log_template, cluster_id):
    log_path = log_template.replace("$(CLUSTER)", cluster_id)
    for _ in range(180):
        if os.path.exists(log_path):
            break
        time.sleep(5)
    if os.path.exists(log_path):
        subprocess.run(["condor_wait", log_path], check=False)
    else:
        print(f"[warn] log not found yet: {log_path}")


def main():
    assert os.path.isfile(CFG), f"Missing cfg: {CFG}"
    assert os.path.isdir(JOBDIR), f"Missing jobdir: {JOBDIR}"
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg0 = read(CFG)
    job_txt = read(os.path.join(JOBDIR, JOBFILE))
    log_tmpl = find_job_log_template(job_txt)

    backup = f"{CFG}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(CFG, backup)
    print(f"[info] Backed up -> {backup}")

    try:
        csvs = sorted(Path(CSV_DIR).glob("*.csv"))
        if not csvs:
            print(f"[warn] No CSVs found in {CSV_DIR}")
            return

        for csv_path in csvs:
            base = csv_path.stem
            file_csv = str(csv_path)

            file_root = os.path.join(RUN1_BASE, ROOT_SUB, base + ".root")
            if not os.path.isfile(file_root):
                print(f"[skip] ROOT not found for {base} -> {file_root}")
                continue

            tag = f"infer_{ROOT_SUB}__{base}"
            print(f"\n[run] {tag}")

            edited = edit_cfg(cfg0, file_root, file_csv, tag, OUT_DIR)
            write(CFG, edited)

            for ln in edited.splitlines():
                if re.match(r'^\s*(name|input_file|input_csv|output_dir)\s*=', ln):
                    print("[cfg]", ln)

            if DRYRUN:
                print("[dry-run] condor_submit", JOBFILE)
                continue

            rc, out = submit()
            cid = parse_cluster_id(out)
            if cid:
                condor_wait(log_tmpl, cid)

            time.sleep(1)

        print("\n[info] Done.")

    finally:
        shutil.move(backup, CFG)
        print(f"[info] Restored cfg from {backup}")


if __name__ == "__main__":
    main()
