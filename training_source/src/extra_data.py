"""Loaders for the additional public databases (downloaded 2026-11-23, see docs/ek_veri_manifest.md).

All are converted to the Stage-1 representation (data.LeadSample at data.FS, band-pass + robust standardisation,
mask 0 bg / 1 P / 2 QRS / 3 T / -1 ignore outside [first, last] annotation) and cached in data/processed/.

  rdb    RDB (Chapman/Shaoxing resting ECG), 500 Hz, per-lead manual P/QRS/T on-off   -> training (Plan D)
  me     ISP-MED Multi-Expert, 500/1000 Hz, lead II manual P/QRS/T on-peak-off (WFDB) -> training (Plan D)
  isp    ISP delineation dataset, 1000 Hz, global (all-lead) manual P/QRS/T on-off    -> external test
  zj     Zhejiang PVC/VT (noise-reduced signals PVCVTECGData, 2000 Hz) + Jimenez-Perez global P/QRS/T on-off
         -> external test. NOTE: the labels do NOT align with the raw version (different crops in many records).
Global labels are applied to every lead (as in the original evaluations).
"""
from __future__ import annotations

import ast
import csv
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from data import LeadSample, clean, resample, to_fs  # noqa: E402
from paths import PROCESSED, ROOT  # noqa: E402

EXTRA = ROOT / "data" / "raw" / "extra"
KIND = {0: 1, 1: 2, 2: 3}            # file type code (0 P, 1 QRS, 2 T) -> mask class
NAME = {1: "P", 2: "QRS", 3: "T"}


def _build(x_list, triples, fs_in, db, record, subject, leads, meta=None, peaks=None):
    """x_list: list of 1-D raw signals (one per lead); triples: list (per lead) of (cls, on, off) at fs_in."""
    out = []
    for lead, x, tri in zip(leads, x_list, triples):
        xr = resample(np.asarray(x, float), fs_in)
        n = len(xr)
        y = np.full(n, -1, dtype=np.int8)
        pts = {f"{NAME[c]}_{k}": [] for c in NAME for k in ("on", "off", "peak")}
        if len(tri):
            on_all = [int(to_fs([o], fs_in)[0]) for _, o, _ in tri]
            off_all = [int(to_fs([e], fs_in)[0]) for _, _, e in tri]
            s0, s1 = max(0, min(on_all)), min(n, max(off_all) + 1)
            y[s0:s1] = 0
            for (c, o, e), oo, ee in zip(tri, on_all, off_all):
                y[oo:ee + 1] = c
                pts[f"{NAME[c]}_on"].append(oo)
                pts[f"{NAME[c]}_off"].append(ee)
        else:
            s0 = s1 = 0
        if peaks:
            for k, v in peaks.items():
                pts[k] = list(to_fs(v, fs_in))
        pts = {k: np.sort(np.asarray(v, int)) for k, v in pts.items()}
        if not peaks:   # no peak annotations in this database -> drop peak keys (evaluation skips them)
            pts = {k: v for k, v in pts.items() if not k.endswith("peak")}
        out.append(LeadSample(db, record, subject, lead, clean(xr), y, pts, (s0, s1), dict(meta or {})))
    return out


def _read_rdb_ann(path):
    tri = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if r and r[0].strip()[:1].isdigit():
                c, o, e = (int(float(v)) for v in r[:3])
                tri.append((KIND[c], o, e))
    return tri


def load_rdb():
    base = EXTRA / "RDB" / "extracted"
    leads = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
    out = []
    for f in sorted(glob.glob(str(base / "dat_csv" / "*.csv"))):
        rec = Path(f).stem
        sig = np.loadtxt(f, delimiter=",", ndmin=2)
        tris = [_read_rdb_ann(base / "ann_txt" / f"{rec}.{ld}.txt") if (base / "ann_txt" / f"{rec}.{ld}.txt").exists()
                else [] for ld in leads]
        out += _build([sig[:, i] for i in range(12)], tris, 500, "RDB", rec, f"RDB:{rec}", leads,
                      {"rhythm": "".join(ch for ch in rec if ch.isalpha())})
    return out


def load_me():
    import pandas as pd
    import wfdb
    from annotations import parse_waves
    base = EXTRA / "MultiExpert" / "extracted" / "ecg_dataset"
    m = pd.read_csv(base / "map_file.csv", sep=";")
    m = m[m.segmented.astype(str) == "True"]
    out = []
    for _, row in m.iterrows():
        r = str(row.ID)
        sig, f = wfdb.rdsamp(str(base / "data" / r))
        li = [s.lower() for s in f["sig_name"]].index("ii")
        a = wfdb.rdann(str(base / "data" / r), "ii")
        waves = parse_waves(a.sample, list(a.symbol))
        tri = [({"P": 1, "QRS": 2, "T": 3}[w.kind], w.on, w.off) for w in waves
               if w.kind in ("P", "QRS", "T") and w.on is not None and w.off is not None]
        peaks = {f"{w}_peak": [x.peak for x in waves if x.kind == w] for w in ("P", "QRS", "T")}
        out += _build([sig[:, li]], [tri], f["fs"], "ME", r, f"ME:{r}", ["ii"],
                      {"diag": ast.literal_eval(row.unified_decision), "device": int(row.device_id)}, peaks)
    return out


def load_isp():
    import wfdb
    base = EXTRA / "ISP" / "extracted" / "isp_delineation_dataset"
    out = []
    for split in ("train", "test"):
        with open(base / f"{split}_isp_delineation_data.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            r = str(row["file_name"])
            sig, f = wfdb.rdsamp(str(base / f"{split}_data" / r))
            tri = [(KIND[c], o, e) for c, o, e in ast.literal_eval(row["target"])]
            leads = [s.lower() for s in f["sig_name"]]
            out += _build([sig[:, i] for i in range(sig.shape[1])], [tri] * len(leads), f["fs"], "ISP",
                          f"{split}{r}", f"ISP:{split}{r}", leads, {"split": split, "age": row["age"]})
    return out


def _read_zj(name):
    lab = {}
    with open(EXTRA / "ZJ_labels" / f"{name}.csv", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if "###" not in parts[0]:
                continue
            rec, lead = parts[0].split("###")
            v = [int(float(p)) for p in parts[1:] if p.strip()]
            lab.setdefault(rec, {})[lead.lower()] = list(zip(v[0::2], v[1::2]))
    return lab


def load_zj(version="PVCVTECGData"):  # labels align with the noise-reduced version (verified 2026-11-23)
    import pandas as pd
    labs = {k: _read_zj(k) for k in ("P", "QRS", "T")}
    recs = sorted(set(labs["P"]) | set(labs["QRS"]) | set(labs["T"]))
    out = []
    for rec in recs:
        f = EXTRA / "ZJ_signals" / "extracted" / version / f"{rec}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        leads = [c.lower() for c in df.columns]
        tris = []
        for ld in leads:
            tri = []
            for k, c in (("P", 1), ("QRS", 2), ("T", 3)):
                tri += [(c, o, e) for o, e in labs[k].get(rec, {}).get(ld, [])]
            tris.append(sorted(tri, key=lambda t: t[1]))
        out += _build([df.iloc[:, i].to_numpy(float) for i in range(df.shape[1])], tris, 2000, "ZJ", rec, f"ZJ:{rec}",
                      leads)
    return out


LOADERS = {"rdb": load_rdb, "me": load_me, "isp": load_isp, "zj": load_zj}


def get(name, rebuild=False):
    path = PROCESSED / f"extra_{name}_fs{data.FS}.pkl"
    if path.exists() and not rebuild:
        class _U(pickle.Unpickler):
            def find_class(self, module, nm):
                return LeadSample if nm == "LeadSample" else super().find_class(module, nm)
        with open(path, "rb") as fh:
            return _U(fh).load()
    d = LOADERS[name]()
    with open(path, "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return d


if __name__ == "__main__":
    for nm in (sys.argv[1].split(",") if len(sys.argv) > 1 else LOADERS):
        d = get(nm, rebuild=True)
        n_p = sum(len(s.points.get("P_on", [])) for s in d)
        print(nm, len(d), "lead-samples;", len({s.subject for s in d}), "records; P annotations:", n_p,
              "; example", d[0].record, d[0].lead, len(d[0].x), d[0].span, flush=True)
