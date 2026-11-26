"""Plans B and C training queue (docs/analiz_plani_v3.md). Resumable and safe to run as several instances:
a job is claimed by creating experiments/run_logs/<name>.claim (atomic create); finished jobs have results/cv/<name>.json.
Usage: python src/run_stage3.py [--workers 1] [--threads 3]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
LOGD = ROOT / "experiments" / "run_logs"
MODELS = ["deq", "unet_kan", "cnn_bilstm"]
VARIANTS = {"s3pw": ["--dissoc", "0.5", "--p_weight", "3", "--p_dice", "0.5"],
            "s3xdb": ["--dissoc", "0.5", "--xdb_qtdb", "4"],
            "s4multi": ["--dissoc", "0.5", "--multi", "1"]}   # plan v4, Plan D


def run(tag, model, fold, lr, threads):
    nm = f"{tag}_{model}_f{fold}_s1"
    if (CV / f"{nm}.json").exists():
        return
    claim = LOGD / f"{nm}.claim"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return
    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", model, "--fold", str(fold), "--lr", lr,
           "--seed", "1", "--threads", str(threads), "--tag", tag] + VARIANTS[tag]
    t = time.time()
    with open(LOGD / f"{nm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        claim.unlink(missing_ok=True)   # allow a retry
    print(time.strftime("%H:%M:%S"), nm, f"rc={rc} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    jobs = [(t, m, f, chosen[m], a.threads) for f in range(5) for t in VARIANTS for m in MODELS]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: run(*j), jobs))
    print("STAGE3 QUEUE DONE", flush=True)
