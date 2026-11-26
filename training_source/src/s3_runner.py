"""Evaluate every Plan B/C/D ensemble as soon as its five folds exist (detached helper next to eval_pending.py).
Per model: final_eval (LUDB/QTDB/BUT PDB/pwave), p_tol_secondary (+-50 ms), extra_eval (Zhejiang/ISP),
first_lead_eval (Saclova protocol), coupling_analysis (QRS-coupled vs dissociated P). Each step is skipped when its
output exists; a model is claimed via results/final/<name>.s3claim so that two runners never work on the same model.
Log: experiments/run_logs/s3_runner.log
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "experiments" / "run_logs" / "s3_runner.log"
R = ROOT / "results"
S3 = [f"{m}@{t}" for t in ["s3pw", "s3xdb", "s4multi"] for m in ["unet_kan", "cnn_bilstm", "deq"]]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def run(args):
    log("RUN " + " ".join(args))
    with open(LOG, "a", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    log(f"END rc={rc} " + args[0])
    return rc


def name(m):
    return m.replace("@", "__")


def folds_done(m):
    arch, tag = m.split("@")
    return all((R / "cv" / f"{tag}_{arch}_f{f}_s1.json").exists() for f in range(5))


def steps(m):
    n = name(m)
    tol = json.loads((R / "p_tol50.json").read_text()) if (R / "p_tol50.json").exists() else {}
    return [((R / "final" / f"{n}.json").exists(), ["src/final_eval.py", "--models", m, "--threads", "2"]),
            (n in tol, ["src/p_tol_secondary.py", m]),
            ((R / "final_extra" / f"{n}.json").exists(), ["src/extra_eval.py", m, "--threads", "2"]),
            ((R / "final_firstlead" / f"{n}.json").exists(), ["src/first_lead_eval.py", m, "--threads", "2"]),
            ((R / "mechanism" / f"{n}.pkl").exists(), ["src/coupling_analysis.py", m, "--threads", "2"])]


def main():
    log("s3_runner started")
    done = set()
    while len(done) < len(S3):
        progressed = False
        for m in S3:
            if m in done or not folds_done(m):
                continue
            claim = R / "final" / f"{name(m)}.s3claim"
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            for i in range(5):            # re-read state before each step (another process may have produced it)
                ok, args = steps(m)[i]
                if not ok:
                    run(args)
            claim.unlink(missing_ok=True)
            done.add(m)
            progressed = True
        if not progressed:
            time.sleep(300)
    log("s3_runner ALL DONE")


if __name__ == "__main__":
    main()
