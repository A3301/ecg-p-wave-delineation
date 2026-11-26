"""Paper A figures (all numbers read from results/; variants included automatically when their files exist).

figures/paperA/
  figA1_augmentation.png      examples of the in-place QRS-independent augmentation (copied from figures/stage2)
  figA2_failure_examples.png  BUT PDB windows with QRS-dissociated P waves: reference P vs Stage-1 and augmented detections
  figA3_effect_forest.png     P F1 with 95% CI per architecture: Stage 1 vs augmentation (and B/C/D variants)
                              on BUT PDB, BUT PDB no-overlap, MIT-BIH P-wave, QTDB P_on, Zhejiang (mean P_on/P_off)
  figA4_subgroups.png         BUT PDB P_peak F1 by rhythm subgroup (heatmap)
  figA5_avblock_subjects.png  sensitivity per AV-block subject (MIT-BIH 231, 107, 111)
"""
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402
from paths import FIGURES, RESULTS  # noqa: E402

OUT = FIGURES / "paperA"
OUT.mkdir(parents=True, exist_ok=True)
F, X = RESULTS / "final", RESULTS / "final_extra"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})
ARCH = {"unet": "U-Net", "cnn_bilstm": "CNN-BiLSTM", "unet_kan": "U-Net-KAN", "deq": "DEQ"}
VAR = [("", "Stage 1", "#9e9e9e"), ("__s2dissoc", "+ aug", "#1f77b4"), ("__s3pw", "+ aug + P-loss", "#ff7f0e"),
       ("__s3xdb", "+ aug + QTDB", "#2ca02c"), ("__s4multi", "+ aug + multi-source", "#d62728")]


def ci_from(dirp, name, part, points):
    with open(dirp / f"{name}_records.pkl", "rb") as fh:
        rec = pickle.load(fh)[part]["subj"]
    subjs = [rec[q] for q in points]
    keys = sorted(set().union(*[set(s) for s in subjs]))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (5000, len(keys)))
    pt, bt = [], []
    for s in subjs:
        t, n, p, _ = stats.subj_arrays(s, keys)
        pt.append(stats.f1(t.sum(), n.sum(), p.sum()))
        bt.append(stats.f1(t[idx].sum(1), n[idx].sum(1), p[idx].sum(1)))
    b = np.mean(bt, 0)
    return float(np.mean(pt)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fig_failure(arch="unet_kan"):
    """Two BUT PDB windows (first lead): the 8-s window of record 01/13 (MIT-BIH 231, 2nd-degree AV block) with the
    most QRS-dissociated reference P waves, and record 03 (MIT-BIH 107, complete AV block, paced)."""
    import torch
    import data
    from coupling_analysis import coupled_mask
    from evaluate import predict_proba
    from final_eval import load_models
    from postprocess import mask_to_points
    torch.set_num_threads(2)
    S = {s.record: s for s in data.get("butpdb") if s.lead == "ch0"}
    W = 8 * data.FS
    cands = []
    for r in ["01", "13"]:
        s = S[r]
        p = np.sort(s.points["P_peak"]).astype(int)
        d = p[~coupled_mask(p, np.sort(s.points["QRS_ref"]).astype(int))]
        for st in range(0, len(s.x) - W, data.FS):
            cands.append((((d >= st) & (d < st + W)).sum(), r, st))
    _, r1, st1 = max(cands)
    s3 = S["03"]
    p3 = np.sort(s3.points["P_peak"]).astype(int)
    st3 = max(int(p3[0]) - data.FS, 0)
    models = {"Stage 1": load_models(arch)[0], "+ aug": load_models(arch + "@s2dissoc")[0]}
    fig, axs = plt.subplots(2, 1, figsize=(11, 5.2))
    for ax, (r, st, title) in zip(axs, [(r1, st1, f"BUT PDB {r1} (MIT-BIH 231, second-degree AV block), first lead"),
                                         ("03", st3, "BUT PDB 03 (MIT-BIH 107, complete AV block, ventricular pacing), first lead")]):
        s = S[r]
        x = s.x
        t = np.arange(len(x)) / data.FS
        sl = slice(st, st + W)
        ax.plot(t[sl], x[sl], color="k", lw=0.8)
        lo, hi = x[sl].min(), x[sl].max()
        span = hi - lo
        ref = s.points["P_peak"]
        ref = ref[(ref >= st) & (ref < st + W)]
        cm = coupled_mask(np.sort(s.points["P_peak"]).astype(int), np.sort(s.points["QRS_ref"]).astype(int))
        allp = np.sort(s.points["P_peak"]).astype(int)
        dis = allp[~cm]
        dis = dis[(dis >= st) & (dis < st + W)]
        ax.plot(ref / data.FS, np.full(len(ref), lo - 0.08 * span), "^", color="#1f77b4", ms=6, label="reference P")
        ax.plot(dis / data.FS, np.full(len(dis), lo - 0.08 * span), "^", color="#d62728", ms=6,
                label="reference P, QRS-dissociated")
        for k, (lab, ms) in enumerate(models.items()):
            det = mask_to_points(predict_proba(ms, x).argmax(0), x)["P_peak"]
            det = det[(det >= st) & (det < st + W)]
            ax.plot(det / data.FS, np.full(len(det), hi + (0.08 + 0.1 * k) * span), "v",
                    color=["#9e9e9e", "#2ca02c"][k], ms=6, label=f"{ARCH[arch]} {lab}")
        ax.set_ylim(lo - 0.18 * span, hi + 0.32 * span)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks([])
    axs[0].legend(fontsize=7, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.28))
    axs[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "figA2_failure_examples.png")
    plt.close(fig)


def fig_coupling():
    """Se for QRS-coupled vs QRS-dissociated reference P waves (BUT PDB, each lead) with subject-bootstrap 95% CI."""
    M = RESULTS / "mechanism"
    rows = [("wavelet", "Wavelet")] + [(a + suf, f"{an}{lab}") for a, an in ARCH.items()
                                        for suf, lab in [("", " Stage 1"), ("__s2dissoc", " + aug")]]
    rows = [(n, l) for n, l in rows if (M / f"{n}.pkl").exists() and (M / f"{n}.pkl").stat().st_size > 0]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.2 + 0.62 * len(rows), 3.4))
    for k, (cls, col, lab) in enumerate([("c", "#9e9e9e", "QRS-coupled P"), ("d", "#d62728", "QRS-dissociated P")]):
        for i, (n, _) in enumerate(rows):
            d = pickle.load(open(M / f"{n}.pkl", "rb"))
            keys = sorted(d)
            tp = np.array([d[q][f"{cls}_tp"] for q in keys], float)
            nn = np.array([d[q][f"{cls}_n"] for q in keys], float)
            idx = rng.integers(0, len(keys), (5000, len(keys)))
            b = 100 * tp[idx].sum(1) / np.maximum(nn[idx].sum(1), 1)
            v = 100 * tp.sum() / nn.sum()
            ax.bar(i + (k - 0.5) * 0.38, v, 0.36, color=col, label=lab if i == 0 else None,
                   yerr=[[v - np.percentile(b, 2.5)], [np.percentile(b, 97.5) - v]], capsize=2,
                   edgecolor="k" if "__" in n else None)
            if cls == "d":
                ax.text(i + 0.19, 101, f"{int(tp.sum())}/{int(nn.sum())}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(rows)), [l for _, l in rows], rotation=50, ha="right", fontsize=7.5)
    ax.set_ylabel("Sensitivity (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figA6_coupling.png")
    plt.close(fig)


def fig_forest():
    panels = [(F, "butpdb_each_lead", ["P_peak"], "BUT PDB P peak"),
              (F, "butpdb_nooverlap_each_lead", ["P_peak"], "BUT PDB (no overlap) P peak"),
              (F, "pwave_each_lead", ["P_peak"], "MIT-BIH P-wave P peak"),
              (F, "qtdb_each_lead", ["P_on"], "QTDB P onset"),
              (X, "zj_each_lead", ["P_on", "P_off"], "Zhejiang P on/off (mean)")]
    rows = [(arch, an, suf, vl, col) for arch, an in ARCH.items() for suf, vl, col in VAR
            if any((d / f"{arch + suf}_records.pkl").exists() for d in (F, X))]
    pos, y, prev = [], 0.0, None
    for r in rows:
        if prev is not None and r[0] != prev:
            y += 0.6
        pos.append(y)
        y += 1
        prev = r[0]
    fig, axs = plt.subplots(1, len(panels), figsize=(14, 0.9 + 0.22 * len(rows)), sharey=True)
    for ax, (dirp, part, pts, title) in zip(axs, panels):
        for yy, (arch, an, suf, vl, col) in zip(pos, rows):
            nm = arch + suf
            if part.startswith("qtdb") and suf == "__s3xdb":   # Plan C trained on 77 QTDB records -> not external
                continue
            if (dirp / f"{nm}_records.pkl").exists():
                try:
                    v, lo, hi = ci_from(dirp, nm, part, pts)
                    ax.errorbar(v, yy, xerr=[[v - lo], [hi - v]], fmt="o", ms=3.5, color=col, capsize=2)
                except KeyError:
                    pass
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("F1 (%)")
        ax.grid(axis="x", alpha=.3)
    axs[0].set_yticks(pos, [f"{an} {vl}" for _, an, _, vl, _ in rows], fontsize=7)
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "figA3_effect_forest.png")
    plt.close(fig)


def fig_subgroups():
    import data
    but = {s.record: s.meta["pathology"] for s in data.get("butpdb")}
    subj_of = {s.record: s.subject for s in data.get("butpdb")}
    groups = {"AV block": ["BI", "BII", "BIII"], "Atrial ectopy": ["A", "a"],
              "Ventricular ectopy": ["V", "B", "T", "VT", "VP", "E", "F"], "AF/AFL": ["AFIB", "AFL", "FIB"],
              "Nodal/SVTA": ["NOD", "J", "SVTA"], "Paced/BBB/other": ["P", "L", "R", "PREX", "IVR", "NA", "VFL"]}
    cols, mat = [], []
    for arch, an in [("wavelet", "Wavelet")] + list(ARCH.items()):
        for suf, vl, _ in ([("", "", None)] if arch == "wavelet" else VAR):
            nm = arch + suf
            if not (F / f"{nm}_records.pkl").exists():
                continue
            with open(F / f"{nm}_records.pkl", "rb") as fh:
                subj = pickle.load(fh)["butpdb_each_lead"]["subj"]["P_peak"]
            colv = []
            for g, toks in groups.items():
                ss = {subj_of[r] for r, p in but.items() if any(t in [x.strip() for x in str(p).split(",")] for t in toks)}
                tp = sum(subj[s]["tp"] for s in ss if s in subj)
                fn = sum(subj[s]["fn"] for s in ss if s in subj)
                fp = sum(subj[s]["fp"] for s in ss if s in subj)
                colv.append(200 * tp / max(2 * tp + fn + fp, 1))
            cols.append(f"{an} {vl}".strip())
            mat.append(colv)
    m = np.array(mat).T
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(cols), 3.4))
    im = ax.imshow(m, cmap="viridis", vmin=60, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(groups)), list(groups), fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="w" if m[i, j] < 82 else "k")
    plt.colorbar(im, ax=ax, label="P peak F1 (%)")
    ax.set_title("BUT PDB, P peak F1 by rhythm subgroup (each lead, ±150 ms)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "figA4_subgroups.png")
    plt.close(fig)


def fig_avblock():
    d = pd.read_csv(RESULTS / "tables" / "butpdb_avblock_subjects.csv")
    subj = {"MIT-BIH Arrhythmia Database:231": "231: 2nd-degree AV block",
            "MIT-BIH Arrhythmia Database:111": "111: 1st-degree AV block + LBBB",
            "MIT-BIH Arrhythmia Database:107": "107: 3rd-degree AV block, paced"}
    models = [m for m in d.model.unique()]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (s, lab) in zip(axs, subj.items()):
        g = d[d.subject == s].set_index("model").reindex(models)
        colors = ["#000000" if m == "wavelet" else ("#1f77b4" if "__" in m else "#9e9e9e") for m in models]
        ax.bar(range(len(models)), g.Se.values, color=colors)
        names = {"wavelet": "Wavelet", **ARCH}
        ax.set_xticks(range(len(models)), [names[m.split("__")[0]] + (" + aug" if "__" in m else "") for m in models],
                      rotation=60, ha="right", fontsize=7)
        ax.set_title(lab, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("P-wave sensitivity (%)")
    fig.tight_layout()
    fig.savefig(OUT / "figA5_avblock_subjects.png")
    plt.close(fig)


def main():
    shutil.copy(FIGURES / "stage2" / "dissoc_aug_examples.png", OUT / "figA1_augmentation.png")
    fig_failure()
    fig_forest()
    fig_subgroups()
    if (RESULTS / "mechanism").exists():
        fig_coupling()
    if (RESULTS / "tables" / "butpdb_avblock_subjects.csv").exists():
        fig_avblock()
    print("Paper A figures written to", OUT)


if __name__ == "__main__":
    main()
