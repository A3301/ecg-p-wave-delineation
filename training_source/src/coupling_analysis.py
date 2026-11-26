"""Mechanism analysis: P-wave sensitivity for QRS-coupled versus QRS-dissociated P waves (BUT PDB, each lead, ±150 ms).

Every reference P peak of BUT PDB is classified with the database's own reference QRS annotations (.qrs), so the
classification does not depend on any model:
  QRS-coupled    : a reference QRS follows the P peak within [60, 450] ms and no other reference P lies between them
                   (the conducted P wave of a beat, including first-degree AV block)
  QRS-dissociated: every other P (non-conducted / blocked P, P waves of complete AV block not followed by their own
                   QRS, P waves superimposed on T or ST, atrial ectopy without conduction)
The shortcut hypothesis predicts that LUDB-trained models lose sensitivity selectively on QRS-dissociated P waves,
and that the QRS-independent augmentation closes that gap.
Statistics: Se per class pooled over subjects; subject-level bootstrap (10 000) 95% CI of the paired Se difference
(augmented minus Stage 1, same architecture) on the dissociated class.
Output: results/mechanism/<name>.pkl (per-subject counts) and results/tables/coupling_sensitivity.md/.csv
Usage: python src/coupling_analysis.py wavelet,cnn_bilstm,cnn_bilstm@s2dissoc,... [--threads 2]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from evaluate import predict_proba  # noqa: E402
from final_eval import load_models, out_name  # noqa: E402
from metrics import match  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

OUT = RESULTS / "mechanism"
TOL = int(round(150 * data.FS / 1000))
PR_MIN, PR_MAX = int(round(0.060 * data.FS)), int(round(0.450 * data.FS))
LAB = {"wavelet": "Wavelet", "unet": "U-Net", "cnn_bilstm": "CNN-BiLSTM", "transformer": "Transformer",
       "unet_kan": "U-Net-KAN", "convkan": "ConvKAN", "deq": "DEQ", "deq_unrolled": "DEQ-unrolled",
       "explicit_stack": "Explicit stack"}
TAGS = {"": "", "s2dissoc": " + aug", "s3pw": " + aug + P-loss", "s3xdb": " + aug + QTDB", "s4multi": " + aug + multi-source"}


def label(m):
    arch, _, tag = m.partition("@")
    return LAB.get(arch, arch) + TAGS.get(tag, " " + tag)


def coupled_mask(p, q):
    """p: sorted reference P peaks, q: sorted reference QRS. True where P is QRS-coupled."""
    out = np.zeros(len(p), bool)
    for i, t in enumerate(p):
        k = np.searchsorted(q, t + PR_MIN)
        if k < len(q) and q[k] - t <= PR_MAX:
            nxt = p[i + 1] if i + 1 < len(p) else np.inf
            out[i] = nxt > q[k]
    return out


def run_model(m, samples):
    if m == "wavelet":
        from wavelet_baseline import delineate
        pred_fn = lambda x: delineate(x)["P_peak"]  # noqa: E731
    else:
        ms = load_models(m)[0]
        pred_fn = lambda x, ms=ms: mask_to_points(predict_proba(ms, x).argmax(0), x)["P_peak"]  # noqa: E731
    subj = {}
    for s in samples:
        ref = np.sort(s.points["P_peak"]).astype(int)
        cm = coupled_mask(ref, np.sort(s.points["QRS_ref"]).astype(int))
        mref, _, _, _ = match(ref, pred_fn(s.x), TOL)
        found = np.isin(ref, mref)
        d = subj.setdefault(s.subject, {"c_tp": 0, "c_n": 0, "d_tp": 0, "d_n": 0})
        d["c_tp"] += int(found[cm].sum())
        d["c_n"] += int(cm.sum())
        d["d_tp"] += int(found[~cm].sum())
        d["d_n"] += int((~cm).sum())
    return subj


def mcnemar_bound(tp_a, tp_b, n):
    """Exact two-sided McNemar p for the same n P waves found tp_a times by model A and tp_b times by model B.
    Only the totals are stored, so the discordant split is unknown; the largest p over every feasible split is
    returned (a valid upper bound): b - c = tp_a - tp_b, 0 <= c <= min(tp_b, n - tp_a)."""
    from scipy.stats import binom
    delta = tp_a - tp_b
    worst = 0.0
    for c in range(0, min(tp_b, n - tp_a) + 1):
        b = c + delta
        if b < 0:
            continue
        m = b + c
        if m == 0:
            return 1.0
        worst = max(worst, min(1.0, 2 * binom.cdf(min(b, c), m, 0.5)))
    return worst


def se(subj, keys, cls):
    tp = np.array([subj[k][f"{cls}_tp"] for k in keys], float)
    n = np.array([subj[k][f"{cls}_n"] for k in keys], float)
    return tp, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    samples = data.get("butpdb")
    for m in a.models.split(","):
        f = OUT / f"{out_name(m)}.pkl"
        if not f.exists() or f.stat().st_size == 0:
            res_m = run_model(m, samples)
            with open(f, "wb") as fh:
                pickle.dump(res_m, fh)
            print("done", m, flush=True)
    # table over every model with a result file
    res = {p.stem: pickle.load(open(p, "rb")) for p in sorted(OUT.glob("*.pkl")) if p.stat().st_size > 0}
    keys = sorted(next(iter(res.values())).keys())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (10000, len(keys)))
    rows = []
    order = [n for n in [out_name(x) for x in
             ["wavelet", "unet", "cnn_bilstm", "transformer", "unet_kan", "convkan", "deq", "deq_unrolled",
              "explicit_stack"] + [f"{arch}@{t}" for t in ["s2dissoc", "s3pw", "s3xdb", "s4multi"]
                                   for arch in ["unet", "cnn_bilstm", "unet_kan", "deq"]]] if n in res]
    for n in order:
        subj = res[n]
        ctp, cn = se(subj, keys, "c")
        dtp, dn = se(subj, keys, "d")
        row = {"model": label(n.replace("__", "@")), "coupled P found/n": f"{int(ctp.sum())}/{int(cn.sum())}",
               "Se coupled": 100 * ctp.sum() / cn.sum(),
               "dissociated P found/n": f"{int(dtp.sum())}/{int(dn.sum())}", "Se dissociated": 100 * dtp.sum() / dn.sum()}
        if "__" in n:
            ref = n.split("__")[0]
            if ref in res:
                rtp, rn = se(res[ref], keys, "d")
                row["dissociated Se change vs Stage 1"] = 100 * (dtp.sum() / dn.sum() - rtp.sum() / rn.sum())
                pm = mcnemar_bound(int(dtp.sum()), int(rtp.sum()), int(dn.sum()))
                row["exact McNemar p (upper bound)"] = f"{pm:.1e}" if pm > 0 else "<1e-300"
        rows.append(row)
    df = pd.DataFrame(rows)
    T = RESULTS / "tables"
    df.round(2).to_csv(T / "coupling_sensitivity.csv", index=False)
    md = ["### BUT PDB: P-wave sensitivity for QRS-coupled vs QRS-dissociated P waves (each lead, ±150 ms)", "",
          "Classification from the reference P and QRS annotations: coupled = a reference QRS follows within "
          "60–450 ms with no intervening P; dissociated = all other P waves. "
          f"Dissociated P waves per subject: " + ", ".join(f"{k.split(':')[-1]} {v}" for k, v in sorted(
              {k: res[order[0]][k]['d_n'] for k in keys if res[order[0]][k]['d_n']}.items(), key=lambda kv: -kv[1])) + ".", "", df.round(2).to_markdown(index=False), ""]
    (T / "coupling_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
