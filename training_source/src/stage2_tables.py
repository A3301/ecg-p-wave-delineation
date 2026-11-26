"""Stage 2 results (docs/analiz_plani_v2.md): H4 primary (augmentation effect per architecture on BUT PDB P_peak F1,
Holm over 4), H5 LUDB non-inferiority, H6 difference-in-differences vs CNN-BiLSTM, secondary endpoints, subgroups.
Inputs: results/final/<arch>.json|_records.pkl (Stage 1) and <arch>__s2dissoc.* (Stage 2).
Output: results/tables/stage2_*.md/.csv, results/tables/stage2_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS  # noqa: E402

T = RESULTS / "tables"
F = RESULTS / "final"
ARCH = ["unet", "cnn_bilstm", "unet_kan", "deq"]
NICE = {"unet": "U-Net", "cnn_bilstm": "CNN-BiLSTM", "unet_kan": "U-Net-KAN", "deq": "DEQ", "wavelet": "Wavelet"}
ONOFF = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]
AUG = "__s2dissoc"


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fmt(v, d=2):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def write(df, name, title):
    df.to_csv(T / f"{name}.csv", index=False)
    (T / f"{name}.md").write_text(f"### {title}\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")


def h4(summary):
    tests = [stats.compare(a + AUG, a, "butpdb_each_lead", "P_peak") for a in ARCH]
    adj = stats.holm([t["p_boot"] for t in tests])
    rows = []
    for a, t, h in zip(ARCH, tests, adj):
        ra, rb = jl(F / f"{a}{AUG}.json")["butpdb_each_lead"]["P_peak"], jl(F / f"{a}.json")["butpdb_each_lead"]["P_peak"]
        rows.append({"architecture": NICE[a], "F1 no-aug": fmt(t["F1_B"]), "F1 aug": fmt(t["F1_A"]),
                     "diff [95% CI]": f"{fmt(t['diff'])} [{fmt(t['diff_CI95'][0])}, {fmt(t['diff_CI95'][1])}]",
                     "p (bootstrap)": fmt(t["p_boot"], 4), "p (Holm)": fmt(h, 4), "p (Wilcoxon)": fmt(t["p_wilcoxon"], 4),
                     "Se no-aug -> aug": f"{fmt(rb['Se'], 1)} -> {fmt(ra['Se'], 1)}",
                     "PPV no-aug -> aug": f"{fmt(rb['PPV'], 1)} -> {fmt(ra['PPV'], 1)}"})
        t["p_holm"] = h
    summary["H4"] = tests
    write(pd.DataFrame(rows), "stage2_H4_primary", "Stage 2 primary (H4): effect of P-QRS dissociation augmentation "
                                                   "on BUT PDB P_peak F1 (±150 ms, each lead), Holm over 4")


def mean_f1_boot(a, b, part="ludb_test"):
    ra, rb = stats.load_records(a)[part]["subj"], stats.load_records(b)[part]["subj"]
    keys = sorted(ra["P_on"])
    idx = np.random.default_rng(0).integers(0, len(keys), (stats.B, len(keys)))
    fa, fb, da, db = [], [], [], []
    for q in ONOFF:
        ta, na, pa, _ = stats.subj_arrays(ra[q], keys)
        tb, nb, pb, _ = stats.subj_arrays(rb[q], keys)
        fa.append(stats.f1(ta.sum(), na.sum(), pa.sum()))
        fb.append(stats.f1(tb.sum(), nb.sum(), pb.sum()))
        da.append(stats.f1(ta[idx].sum(1), na[idx].sum(1), pa[idx].sum(1)))
        db.append(stats.f1(tb[idx].sum(1), nb[idx].sum(1), pb[idx].sum(1)))
    d = np.mean(da, 0) - np.mean(db, 0)
    return float(np.mean(fa)), float(np.mean(fb)), np.percentile(d, [2.5, 97.5])


def h5(summary):
    rows = []
    for a in ARCH:
        fa, fb, ci = mean_f1_boot(a + AUG, a)
        rows.append({"architecture": NICE[a], "LUDB mean F1 no-aug": fmt(fb), "aug": fmt(fa),
                     "diff [95% CI]": f"{fmt(fa - fb)} [{fmt(ci[0])}, {fmt(ci[1])}]",
                     "non-inferior (lower CI > -1)": "yes" if ci[0] > -1 else "no"})
    summary["H5"] = rows
    write(pd.DataFrame(rows), "stage2_H5_ludb", "Stage 2 H5: LUDB internal test mean F1 (6 on/off), aug vs no-aug")


def h6(summary):
    """Difference-in-differences of BUT PDB P_peak F1 gain, proposed vs CNN-BiLSTM, paired subject bootstrap."""
    def arrays(m):
        s = stats.load_records(m)["butpdb_each_lead"]["subj"]["P_peak"]
        return s
    base = {k: arrays(k) for k in ["cnn_bilstm", "cnn_bilstm" + AUG]}
    keys = sorted(base["cnn_bilstm"])
    idx = np.random.default_rng(0).integers(0, len(keys), (stats.B, len(keys)))

    def f1b(m):
        t, n, p, _ = stats.subj_arrays(arrays(m), keys)
        return stats.f1(t.sum(), n.sum(), p.sum()), stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1))
    g0, gb0 = f1b("cnn_bilstm")
    g1, gb1 = f1b("cnn_bilstm" + AUG)
    rows = []
    for a in ["unet_kan", "deq", "unet"]:
        x0, xb0 = f1b(a)
        x1, xb1 = f1b(a + AUG)
        dd = (xb1 - xb0) - (gb1 - gb0)
        p = float(min(1, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
        rows.append({"architecture": NICE[a], "gain": fmt(x1 - x0), "gain CNN-BiLSTM": fmt(g1 - g0),
                     "diff-in-diff [95% CI]": f"{fmt((x1 - x0) - (g1 - g0))} [{fmt(np.percentile(dd, 2.5))}, "
                                              f"{fmt(np.percentile(dd, 97.5))}]", "p (bootstrap)": fmt(p, 4)})
    summary["H6"] = rows
    write(pd.DataFrame(rows), "stage2_H6_interaction", "Stage 2 H6 (exploratory): does KAN/DEQ gain more from "
                                                        "augmentation than CNN-BiLSTM? (BUT PDB P_peak F1)")


def secondary():
    rows = []
    for m in ["wavelet"] + [x for a in ARCH for x in (a, a + AUG)]:
        if m == "wavelet":
            w = jl(RESULTS / "baselines" / "wavelet_neurokit_dwt.json")
            get = lambda part, q: w[part]["table"][q] if part in w else None  # noqa: E731
        else:
            r = jl(F / f"{m}.json")
            get = lambda part, q, r=r: r.get(part, {}).get(q)  # noqa: E731
        row = {"model": NICE[m.split("__")[0]] + (" + aug" if AUG in m else "")}
        for part, q, lab in [("butpdb_each_lead", "P_peak", "BUT PDB"), ("butpdb_nooverlap_each_lead", "P_peak",
                                                                          "BUT PDB no-overlap"),
                             ("pwave_each_lead", "P_peak", "pwave"), ("qtdb_each_lead", "P_on", "QTDB P_on"),
                             ("qtdb_each_lead", "P_off", "QTDB P_off")]:
            v = get(part, q)
            row[lab] = "" if v is None else f"{fmt(v['F1'])} ({fmt(v['Se'], 1)}/{fmt(v['PPV'], 1)})"
        rows.append(row)
    write(pd.DataFrame(rows), "stage2_secondary", "Stage 2 secondary: F1 (Se/PPV), ±150 ms, each lead")
    tol = jl(RESULTS / "p_tol50.json") if (RESULTS / "p_tol50.json").exists() else {}
    rows = []
    for m in ["wavelet"] + [x for a in ARCH for x in (a, a + AUG)]:
        if m in tol:
            rows.append({"model": NICE[m.split("__")[0]] + (" + aug" if AUG in m else ""),
                         "BUT PDB ±50 ms F1": fmt(tol[m]["butpdb_each_lead"]["F1"]),
                         "pwave ±50 ms F1": fmt(tol[m]["pwave_each_lead"]["F1"])})
    if rows:
        write(pd.DataFrame(rows), "stage2_tol50", "Stage 2: P_peak at ±50 ms")


def subgroups():
    import data
    but = {s.record: s.meta["pathology"] for s in data.get("butpdb")}
    subj_of = {s.record: s.subject for s in data.get("butpdb")}
    groups = {"AV block (BI/BII/BIII)": ["BI", "BII", "BIII"], "Atrial ectopy (A/a)": ["A", "a"],
              "Ventricular ectopy": ["V", "B", "T", "VT", "VP", "E", "F"], "AF/AFL": ["AFIB", "AFL", "FIB"],
              "Nodal/SVTA": ["NOD", "J", "SVTA"], "Paced/BBB/other": ["P", "L", "R", "PREX", "IVR", "NA", "VFL"]}
    rows = []
    for m in ["wavelet"] + [x for a in ARCH for x in (a, a + AUG)]:
        subj = stats.load_records(m)["butpdb_each_lead"]["subj"]["P_peak"]
        for g, toks in groups.items():
            recs = [r for r, pth in but.items() if any(t in [x.strip() for x in str(pth).split(",")] for t in toks)]
            ss = sorted({subj_of[r] for r in recs})
            tp = sum(subj[s]["tp"] for s in ss if s in subj)
            fn = sum(subj[s]["fn"] for s in ss if s in subj)
            fp = sum(subj[s]["fp"] for s in ss if s in subj)
            rows.append({"model": NICE[m.split("__")[0]] + (" + aug" if AUG in m else ""), "subgroup": g,
                         "Se": fmt(100 * tp / max(tp + fn, 1)), "PPV": fmt(100 * tp / max(tp + fp, 1)),
                         "F1": fmt(200 * tp / max(2 * tp + fn + fp, 1))})
    df = pd.DataFrame(rows)
    write(df, "stage2_subgroups", "Stage 2: BUT PDB P_peak by pathology subgroup")
    write(df.pivot(index="subgroup", columns="model", values="F1").reset_index(), "stage2_subgroups_F1",
          "Stage 2: BUT PDB P_peak F1 by subgroup (columns = models)")


def main():
    summary = {}
    h4(summary)
    h5(summary)
    h6(summary)
    secondary()
    subgroups()
    (T / "stage2_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    for f in ["stage2_H4_primary", "stage2_H5_ludb", "stage2_H6_interaction", "stage2_secondary", "stage2_subgroups_F1"]:
        print((T / f"{f}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
