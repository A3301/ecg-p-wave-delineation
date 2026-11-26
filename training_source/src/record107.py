"""Revision analysis (Paper A, reviewer A-3): the second AV-block patient in the public P-annotated data.

BUT PDB record 03 is a 120-s segment of MIT-BIH record 107 (complete AV block with ventricular pacing; BUT PDB
pathology code 'BIII'). In complete block no P wave is conducted, so every annotated P wave is non-conducted,
whatever its timing relative to the paced QRS (the timing rule of Table 3 classifies 55 of its 56 P waves as
QRS-coupled because the paced ventricular rate places most P waves within 60-450 ms of a following QRS).
Uses the per-P detection flags of src/coupling_detail.py (both leads, +-150 ms). Sensitivity per model, exact McNemar
against the Stage-1 model of the same architecture on the same P waves (both leads pooled).
Output: results/tables/record107.md, results/record107.json
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "mechanism_detail"
ORDER = ["wavelet", "unet", "cnn_bilstm", "transformer", "unet_kan", "deq",
         "unet__s2dissoc", "cnn_bilstm__s2dissoc", "unet_kan__s2dissoc", "deq__s2dissoc",
         "unet_kan__s5ins", "unet_kan__s5del", "unet_kan__s5sham",
         "cnn_bilstm__s3xdb", "unet_kan__s3xdb", "deq__s3xdb", "cnn_bilstm__s4multi", "unet_kan__s4multi", "deq__s4multi",
         "cnn_bilstm__s5cd", "unet_kan__s5cd", "unet_kan__seed2", "unet_kan__s2dissoc__seed2", "unet_kan__s4multi__seed2"]
NICE = {"s2dissoc": "augmentation", "s5ins": "insertion only", "s5del": "deletion only", "s5sham": "QRS-coupled sham",
        "s3xdb": "aug + QTDB (Plan C)", "s4multi": "aug + multi-source (Plan D)", "s5cd": "aug + C + D"}


def flags(name):
    f = D / f"{name}.pkl"
    if not f.exists():
        return None
    rows = [r for r in pickle.load(open(f, "rb")) if r["subject"].endswith(":107")]
    if not rows:
        return None
    rows.sort(key=lambda r: r["lead"])
    return {r["lead"]: np.asarray(r["found"], bool) for r in rows}


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    m = n10 + n01
    return n10, n01, (1.0 if m == 0 else float(min(1, 2 * binom.cdf(min(n10, n01), m, 0.5))))


def nice(name):
    parts = name.split("__")
    arch = {"unet": "U-Net", "cnn_bilstm": "CNN–BiLSTM", "transformer": "Transformer", "unet_kan": "U-Net-KAN", "deq": "DEQ",
            "wavelet": "Wavelet"}[parts[0]]
    rest = [NICE.get(p, p) for p in parts[1:] if p != "seed2"]
    lab = arch + (" + " + " + ".join(rest) if rest else " (Stage 1)")
    return lab + (", seed 2" if name.endswith("seed2") else "")


def main():
    res, md = {}, ["### Complete AV block with ventricular pacing (BUT PDB record 03, from MIT-BIH 107; 56 annotated P waves per lead, 2 leads)", "",
                   "All annotated P waves are non-conducted (complete block). Se: share detected within ±150 ms. "
                   "McNemar: exact, same P waves, against the Stage-1 model of the same architecture and seed.", "",
                   "| Model | Se lead 1 | Se lead 2 | Se both leads (n = 112) | vs Stage 1: n10 / n01, p |", "|---|---|---|---|---|"]
    for name in ORDER:
        fl = flags(name)
        if fl is None:
            continue
        leads = sorted(fl)
        both = np.concatenate([fl[k] for k in leads])
        base = name.split("__")[0] + ("__seed2" if name.endswith("seed2") else "")
        cmp = "–"
        if name != base and base != "wavelet":
            fb = flags(base)
            if fb is not None:
                n10, n01, p = mcnemar(both, np.concatenate([fb[k] for k in leads]))
                cmp = f"{n10} / {n01}, {p:.2g}"
        res[name] = {"se": [float(100 * fl[k].mean()) for k in leads], "se_both": float(100 * both.mean()), "n": int(len(both))}
        md.append(f"| {nice(name)} | {100 * fl[leads[0]].mean():.1f} | {100 * fl[leads[1]].mean():.1f} | {100 * both.mean():.1f} | {cmp} |")
    (ROOT / "results" / "record107.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (ROOT / "results" / "tables" / "record107.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written", len(res))


if __name__ == "__main__":
    main()
