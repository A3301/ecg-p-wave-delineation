"""Resumable orchestrator for Phase 3/4 training (frozen plan, section 6).

Stage A (tuning): every model x lr in {1e-3, 3e-3} on fold 0.
Stage B (CV)    : chosen lr (best fold-0 val mean F1) on folds 1..4 (fold 0 run is reused).
Runs whose results/cv/<name>.json exists are skipped. Two workers x 3 threads by default.

Usage: python src/run_all.py [--workers 2] [--threads 3] [--models unet,cnn_bilstm,...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODELS = ["unet", "cnn_bilstm", "transformer", "unet_kan", "convkan", "explicit_stack", "deq_unrolled", "deq"]
LRS = ["1e-3", "3e-3"]
CV = ROOT / "results" / "cv"
LOGD = ROOT / "experiments" / "run_logs"


def name(model, lr, fold, seed=1):
    return f"lr{lr}_{model}_f{fold}_s{seed}"


def run(model, lr, fold, threads, seed=1):
    nm = name(model, lr, fold, seed)
    if (CV / f"{nm}.json").exists():
        return nm, "skip"
    LOGD.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(ROOT / "src" / "train.py"), "--model", model, "--fold", str(fold), "--lr", lr,
           "--seed", str(seed), "--threads", str(threads), "--tag", f"lr{lr}"]
    t = time.time()
    with open(LOGD / f"{nm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    msg = f"rc={rc} {time.time() - t:.0f}s"
    print(time.strftime("%H:%M:%S"), nm, msg, flush=True)
    return nm, msg


def val_f1(nm):
    p = CV / f"{nm}.json"
    return json.loads(p.read_text(encoding="utf-8")).get("val_meanF1", -1) if p.exists() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    models = a.models.split(",")
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda job: run(*job, a.threads), [(m, lr, 0) for m in models for lr in LRS]))
    chosen = {}
    for m in models:
        scores = {lr: val_f1(name(m, lr, 0)) for lr in LRS}
        chosen[m] = max(scores, key=scores.get)
        print("chosen lr", m, chosen[m], scores, flush=True)
    (ROOT / "configs" / "chosen_lr.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    jobs = [(m, chosen[m], f) for f in range(1, 5) for m in models]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda job: run(*job, a.threads), jobs))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
