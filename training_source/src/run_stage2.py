"""Stage 2 training queue (docs/analiz_plani_v2.md): 4 models x 5 folds with P-QRS dissociation augmentation
(p = 0.5), Stage-1 learning rates and budget. Resumable. Tag: s2dissoc.
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
LOGD = ROOT / "experiments" / "run_logs"
MODELS = ["deq", "unet_kan", "cnn_bilstm", "unet"]   # long jobs first for load balance
TAG, P = "s2dissoc", "0.5"


def run(model, fold, lr, threads=3):
    nm = f"{TAG}_{model}_f{fold}_s1"
    if (CV / f"{nm}.json").exists():
        return
    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", model, "--fold", str(fold), "--lr", lr,
           "--seed", "1", "--threads", str(threads), "--tag", TAG, "--dissoc", P]
    t = time.time()
    with open(LOGD / f"{nm}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    print(time.strftime("%H:%M:%S"), nm, f"rc={rc} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    chosen = json.loads((ROOT / "configs" / "chosen_lr.json").read_text())
    jobs = [(m, f, chosen[m]) for f in range(5) for m in MODELS]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: run(*j), jobs))
    print("STAGE2 TRAINING DONE", flush=True)
