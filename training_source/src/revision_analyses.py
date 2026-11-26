"""Revision analyses from existing outputs (docs/analiz_plani_v5.md, section 3). Output: results/tables/revision_*.md

A. Plan C/D gain on BUT PDB stratified by source database (MIT-BIH 360 Hz vs LTAFDB/SVDB 128 Hz).
B. Sensitivity of the configuration selection to the selection rule.
C. DEQ iteration count as a difficulty signal: subject-level tests, partial correlation given entropy, AUROC.
D. Architecture comparisons on the independent sets (paired subject bootstrap, Holm).
E. Zhejiang vs Lim et al. with one F1 definition (F1 from mean Se and mean PPV over six points).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import RESULTS  # noqa: E402

F, X, T = RESULTS / "final", RESULTS / "final_extra", RESULTS / "tables"
B = 10_000
SIX = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]


def rec(d, n):
    with open(d / f"{n}_records.pkl", "rb") as fh:
        return pickle.load(fh)


def pooled(subj, keys):
    t, n, p, _ = stats.subj_arrays(subj, keys)
    tp, fn, fp = t.sum(), n.sum(), p.sum()   # subj_arrays returns TP, FN, FP
    return stats.f1(tp, fn, fp), 100 * tp / max(tp + fn, 1), 100 * tp / max(tp + fp, 1)


def paired(sa_list, sb_list, seed):
    keys = sorted(set(sa_list[0]) | set(sb_list[0]))
    idx = np.random.default_rng(seed).integers(0, len(keys), (B, len(keys)))

    def f(sl):
        pt, bt = [], []
        for s in sl:
            t, n, p, _ = stats.subj_arrays(s, keys)
            pt.append(stats.f1(t.sum(), n.sum(), p.sum()))
            bt.append(stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1)))
        return float(np.mean(pt)), np.mean(bt, 0)
    a, ab = f(sa_list)
    b, bb = f(sb_list)
    d = ab - bb
    return a, b, a - b, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def part_a():
    md = ["### A. BUT PDB P_peak by source database (each lead, ±150 ms): F1 (Se/PPV)", "",
          "| model | MIT-BIH Arrhythmia (360 Hz) | LTAFDB + SVDB (128 Hz) |", "|---|---|---|"]
    for n in ["cnn_bilstm", "cnn_bilstm__s2dissoc", "cnn_bilstm__s3xdb", "cnn_bilstm__s4multi",
              "unet_kan", "unet_kan__s2dissoc", "unet_kan__s3xdb", "unet_kan__s4multi",
              "deq", "deq__s2dissoc", "deq__s3xdb", "deq__s4multi"]:
        subj = rec(F, n)["butpdb_each_lead"]["subj"]["P_peak"]
        cells = []
        for grp in [lambda k: k.startswith("MIT-BIH Arrhythmia"), lambda k: not k.startswith("MIT-BIH Arrhythmia")]:
            keys = sorted(k for k in subj if grp(k))
            f1, se, ppv = pooled(subj, keys)
            cells.append(f"{f1:.2f} ({se:.1f}/{ppv:.1f}), n={len(keys)}")
        md.append(f"| {n} | " + " | ".join(cells) + " |")
    return md


def part_b():
    tol = json.loads((RESULTS / "p_tol50.json").read_text())
    rows = []
    for p in sorted(F.glob("*.json")):
        if p.stem == "wavelet" or p.stem.startswith("unet_kan__abl") or "__seed" in p.stem or "__s5" in p.stem:
            continue
        r = json.loads(p.read_text())
        if "pwave_each_lead" not in r:
            continue
        rows.append({"model": p.stem, "pwave150": r["pwave_each_lead"]["P_peak"]["F1"],
                     "pwave50": tol.get(p.stem, {}).get("pwave_each_lead", {}).get("F1", np.nan),
                     "butnoov": r["butpdb_nooverlap_each_lead"]["P_peak"]["F1"]})
    d = pd.DataFrame(rows)
    zj = {}
    for n in d.model:
        if (X / f"{n}.json").exists():
            z = json.loads((X / f"{n}.json").read_text())["zj_each_lead"]
            zj[n] = (z["P_on"]["F1"] + z["P_off"]["F1"]) / 2
    d["zj_Pmean"] = d.model.map(zj)
    md = ["### B. Selection-rule sensitivity (development set = MIT-BIH P-wave)", ""]
    for rule in ["pwave150", "pwave50"]:
        top = d.sort_values(rule, ascending=False).head(3)
        md.append(f"- Rule max {rule}: " + "; ".join(f"{r.model} ({r[rule]:.2f}; BUT no-overlap {r.butnoov:.2f}; "
                                                    f"Zhejiang P {r.zj_Pmean:.2f})" for _, r in top.iterrows()))
    md += ["", "Baseline CNN-BiLSTM (Stage 1): BUT no-overlap {:.2f}; Zhejiang P {:.2f}".format(
        float(d[d.model == "cnn_bilstm"].butnoov.iloc[0]), zj.get("cnn_bilstm", np.nan)), ""]
    return md


def part_c():
    w = pd.read_csv(RESULTS / "deq_windows.csv")
    l = w[w.db == "LUDB"].copy()
    md = ["### C. DEQ solver iterations as a difficulty signal (LUDB internal test, fold-0 DEQ)", ""]
    # subject (record) level
    g = l.groupby("record").agg(iters=("iters", "mean"), mae=("mae_ms", "mean"), ent=("entropy", "mean"),
                                group=("group", "first"))
    r_s, p_s = ss.spearmanr(g.iters, g.mae)
    path, sin = g[g.group != "sinus"].iters, g[g.group == "sinus"].iters
    u = ss.mannwhitneyu(path, sin, alternative="greater")
    md += [f"- Record level (n = {len(g)} records, iterations averaged over 12 leads): Spearman ρ(iterations, mean |error|) "
           f"= {r_s:.2f} (p = {p_s:.2g}); pathological vs sinus records median {path.median():.1f} vs {sin.median():.1f} "
           f"iterations, one-sided Mann–Whitney p = {u.pvalue:.3g} (n = {len(path)} vs {len(sin)})."]
    # partial correlation (rank-based) iterations ~ error | entropy, lead level and record level
    def partial(df, x, y, z):
        rx = ss.rankdata(df[x]); ry = ss.rankdata(df[y]); rz = ss.rankdata(df[z])
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return ss.pearsonr(ex, ey)
    pl = partial(l, "iters", "mae_ms", "entropy")
    pg = partial(g.reset_index(), "iters", "mae", "ent")
    md.append(f"- Partial rank correlation of iterations with |error| given entropy: lead level ρ = {pl[0]:.2f} "
              f"(p = {pl[1]:.2g}); record level ρ = {pg[0]:.2f} (p = {pg[1]:.2g}).")
    # AUROC for flagging the worst quartile of leads by error
    from sklearn.metrics import roc_auc_score
    y = (l.mae_ms >= l.mae_ms.quantile(0.75)).astype(int)
    a_it, a_en = roc_auc_score(y, l.iters), roc_auc_score(y, l.entropy)
    a_both = roc_auc_score(y, ss.rankdata(l.iters) + ss.rankdata(l.entropy))
    rng = np.random.default_rng(0)
    recs = l.record.unique()
    diffs = []
    for _ in range(2000):
        pick = rng.choice(recs, len(recs))
        s = pd.concat([l[l.record == r] for r in pick])
        yy = (s.mae_ms >= l.mae_ms.quantile(0.75)).astype(int)
        if yy.nunique() == 2:
            diffs.append(roc_auc_score(yy, s.iters) - roc_auc_score(yy, s.entropy))
    md.append(f"- AUROC for flagging the worst-quartile leads (|error| ≥ {l.mae_ms.quantile(0.75):.1f} ms): iterations "
              f"{a_it:.3f}, predictive entropy {a_en:.3f}, rank sum of both {a_both:.3f}; difference iterations − entropy "
              f"{a_it - a_en:+.3f} [95% CI {np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}] "
              f"(record-level bootstrap).")
    md.append(f"- Leads at the 30-iteration cap: {(l.iters >= 30).mean() * 100:.1f}%.")
    return md + [""]


def part_d():
    tests = []
    base = rec(X, "cnn_bilstm")
    for n, set_, pts, lab in [("convkan", "zj_each_lead", ["P_on"], "ConvKAN vs CNN-BiLSTM, Zhejiang P onset"),
                              ("convkan", "zj_each_lead", SIX, "ConvKAN vs CNN-BiLSTM, Zhejiang mean of 6"),
                              ("deq", "isp_each_lead", SIX, "DEQ vs CNN-BiLSTM, ISP mean of 6"),
                              ("deq_unrolled", "isp_each_lead", SIX, "DEQ-unrolled vs CNN-BiLSTM, ISP mean of 6")]:
        r = rec(X, n)
        a = [r[set_]["subj"][q] for q in pts]
        b = [base[set_]["subj"][q] for q in pts]
        tests.append((lab,) + paired(a, b, seed=len(tests) + 1))
    adj = stats.holm([t[-1] for t in tests])
    md = ["### D. Architecture comparisons on the independent sets (each lead, ±150 ms; paired subject bootstrap; "
          "Holm over the four tests)", "", "| comparison | F1 A | F1 B | diff [95% CI] | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for t, h in zip(tests, adj):
        md.append(f"| {t[0]} | {t[1]:.2f} | {t[2]:.2f} | {t[3]:+.2f} [{t[4]:+.2f}, {t[5]:+.2f}] | {t[6]:.4f} | {h:.4f} |")
    return md + [""]


def part_e():
    md = ["### E. Zhejiang, mean of six onset/offset points: F1 from mean Se and mean PPV (definition of Lim et al.)", "",
          "| model | tolerance | mean Se | mean PPV | F1 from means | mean of per-point F1 |", "|---|---|---|---|---|---|"]
    for n in ["wavelet", "cnn_bilstm", "unet_kan__s3xdb", "unet_kan__s4multi", "cnn_bilstm__s4multi"]:
        if not (X / f"{n}.json").exists():
            continue
        r = json.loads((X / f"{n}.json").read_text())
        for key, tol in [("zj_each_lead", "±150 ms"), ("zj_each_lead_tol40", "±40 ms")]:
            t = r[key]
            se = np.mean([t[q]["Se"] for q in SIX]); ppv = np.mean([t[q]["PPV"] for q in SIX])
            md.append(f"| {n} | {tol} | {se:.2f} | {ppv:.2f} | {2 * se * ppv / (se + ppv):.2f} | "
                      f"{np.mean([t[q]['F1'] for q in SIX]):.2f} |")
    return md + [""]


def main():
    md = part_a() + [""] + part_b() + part_c() + part_d() + part_e()
    (T / "revision_analyses.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
