"""Evaluate the plan-v5 ensembles as soon as their five folds exist (detached helper; claim files allow several
instances). Full evaluation (final_eval, +-50 ms, Zhejiang/ISP, first lead, coupling detail on BUT PDB and MIT-BIH
P-wave) for the new configurations; final_eval only for the second-seed replicates of Stage-1/2 models.
Log: experiments/run_logs/s5_runner.log
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
LOG = ROOT / "experiments" / "run_logs" / "s5_runner.log"
R = ROOT / "results"
FULL = ["unet_kan@s5cd", "cnn_bilstm@s5cd", "unet_kan@s5ins", "unet_kan@s5del", "unet_kan@s5sham", "unet_kan@s4multi~2", "unet_kan@s3xdb~2"]
BASIC = ["deq_unrolled@s5u18", "unet_kan@abl-mlphead", "unet_kan@abl-mlphead~2", "unet_kan@abl-headonly~2",
         "unet_kan@s2dissoc~2", "unet_kan~2", "unet~2", "deq_unrolled~2", "deq~2"]


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def run(args):
    log("RUN " + " ".join(args))
    with open(LOG, "a", encoding="utf-8") as fh:
        rc = subprocess.call([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    log(f"END rc={rc} " + args[0])


def name(m):
    return m.replace("@", "__").replace("~", "__seed")


def folds_done(m):
    m2, _, seed = m.partition("~")
    seed = seed or "1"
    arch, _, tag = m2.partition("@")
    if not tag:
        tag = "lr" + json.loads((ROOT / "configs" / "chosen_lr.json").read_text())[arch]
    return all((R / "cv" / f"{tag}_{arch}_f{f}_s{seed}.json").exists() for f in range(5))


def steps(m):
    n = name(m)
    st = [((R / "final" / f"{n}.json").exists(), ["src/final_eval.py", "--models", m, "--threads", "3"])]
    if m in FULL:
        st += [((R / "final_extra" / f"{n}.json").exists(), ["src/extra_eval.py", m, "--threads", "3"]),
               ((R / "final_firstlead" / f"{n}.json").exists(), ["src/first_lead_eval.py", m, "--threads", "3"]),
               ((R / "mechanism_detail" / f"{n}.pkl").exists(), ["src/coupling_detail.py", m, "--threads", "3"]),
               ((R / "mechanism_detail_pwave" / f"{n}.pkl").exists(), ["src/coupling_detail.py", m, "--threads", "3", "--db", "pwave"])]
    return st


def main():
    log("s5_runner started")
    done = set()
    allm = FULL + BASIC
    while len(done) < len(allm):
        progressed = False
        for m in allm:
            if m in done or not folds_done(m):
                continue
            claim = R / "final" / f"{name(m)}.s5claim"
            try:
                os.close(os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            for i in range(len(steps(m))):
                ok, args = steps(m)[i]
                if not ok:
                    run(args)
            claim.unlink(missing_ok=True)
            done.add(m)
            progressed = True
        if not progressed:
            time.sleep(300)
    log("s5_runner ALL DONE")


if __name__ == "__main__":
    main()
