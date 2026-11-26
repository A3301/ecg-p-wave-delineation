"""Revision analysis (plan v5, section 3.1): P-wave-level detection flags on BUT PDB for the coupling analysis.

Stores, for every model, every lead and every reference P peak: QRS-coupled flag, detected flag (+-150 ms, greedy
one-to-one), and a near-duplicate flag (another reference P within 100 ms). This enables subject-clustered statistics,
first-lead-only results and a sensitivity analysis without near-duplicate reference labels.
Output: results/mechanism_detail/<name>.pkl ; summary via src/revision_analyses.py
Usage: python src/coupling_detail.py wavelet,cnn_bilstm,... [--threads 2]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from coupling_analysis import TOL, coupled_mask  # noqa: E402
from evaluate import predict_proba  # noqa: E402
from final_eval import load_models, out_name  # noqa: E402
from metrics import match  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

OUT = RESULTS / "mechanism_detail"
DUP = int(round(0.100 * data.FS))
BEATS = set("NLRAaJSVFejE/fQ")          # MIT-BIH beat symbols (non-beat annotations such as 'x', '+', '~' excluded)


def pwave_samples():
    """MIT-BIH P-wave annotation records with the MIT-BIH reference beat annotations (data/raw/mitdb_atr)."""
    import wfdb
    from data import to_fs
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "mitdb_atr"
    out = []
    for s in data.get("pwave"):
        a = wfdb.rdann(str(root / s.record), "atr")
        keep = np.array([c in BEATS for c in a.symbol])
        s.points["QRS_ref"] = np.sort(to_fs(a.sample[keep], 360))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--db", default="butpdb", choices=["butpdb", "pwave"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out_dir = OUT if a.db == "butpdb" else RESULTS / "mechanism_detail_pwave"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = data.get("butpdb") if a.db == "butpdb" else pwave_samples()
    for m in a.models.split(","):
        f = out_dir / f"{out_name(m)}.pkl"
        if f.exists() and f.stat().st_size > 0:
            continue
        if m == "wavelet":
            from wavelet_baseline import delineate
            pred_fn = lambda x: delineate(x)["P_peak"]  # noqa: E731
        else:
            ms = load_models(m)[0]
            pred_fn = lambda x, ms=ms: mask_to_points(predict_proba(ms, x).argmax(0), x)["P_peak"]  # noqa: E731
        rows = []
        for s in samples:
            ref = np.sort(s.points["P_peak"]).astype(int)
            cm = coupled_mask(ref, np.sort(s.points["QRS_ref"]).astype(int))
            mref, _, _, _ = match(ref, pred_fn(s.x), TOL)
            gap = np.minimum(np.r_[np.inf, np.diff(ref)], np.r_[np.diff(ref), np.inf])
            rows.append({"record": s.record, "subject": s.subject, "lead": s.lead, "coupled": cm,
                         "found": np.isin(ref, mref), "near_dup": gap < DUP})
        res = rows
        with open(f, "wb") as fh:
            pickle.dump(res, fh)
        print("done", m, flush=True)


if __name__ == "__main__":
    main()
