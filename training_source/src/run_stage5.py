"""Training queue for the additional analyses (docs/analiz_plani_v5.md). Same claim-file mechanism as run_stage3.py; safe to run as
several detached instances. Usage: python src/run_stage5.py [--threads 3]"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
LOGD = ROOT / "experiments" / "run_logs"
CD = ["--dissoc", "0.5", "--xdb_qtdb", "4", "--multi", "1"]
JOBS = ([("s5cd", m, 1, CD) for m in ["unet_kan", "cnn_bilstm"]] +
        [("s5ins", "unet_kan", 1, ["--dissoc", "0.5", "--dissoc_mode", "insert"]),
         ("s5del", "unet_kan", 1, ["--dissoc", "0.5", "--dissoc_mode", "delete"]),
         ("s5u18", "deq_unrolled", 1, ["--kw", '{"unroll": 18}'])] +
        [("s4multi", "unet_kan", 2, ["--dissoc", "0.5", "--multi", "1"]),
         ("s3xdb", "unet_kan", 2, ["--dissoc", "0.5", "--xdb_qtdb", "4"]),
         ("s2dissoc", "unet_kan", 2, ["--dissoc", "0.5"]),
         ("STAGE1", "unet_kan", 2, []), ("STAGE1", "unet", 2, []),
         ("STAGE1", "deq_unrolled", 2, []), ("STAGE1", "deq", 2, [])])


def run(tag, model, seed, extra, fold, lr, threads):
    tag = f"lr{lr}" if tag == "STAGE1" else tag
    nm = f"{tag}_{model}_f{fold}_s{seed}"
    if (CV / f"{nm}.json").exists():
        return
    claim = LOGD / f"{nm}.claim"
    try:
        os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return
    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", model, "--fold", str(fold), "--lr", lr,
           "--seed", str(seed), "--threads", str(threads), "--tag", tag] + extra
    t = time.time()
    with open(LOGD / f"{nm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        claim.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), nm, f"rc={rc} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    for tag, m, seed, extra in JOBS:
        for f in range(5):
            run(tag, m, seed, extra, f, chosen[m], a.threads)
    print("STAGE5 QUEUE DONE", flush=True)
