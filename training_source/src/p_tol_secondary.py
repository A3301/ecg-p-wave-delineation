"""Secondary analysis (plan section 4): P_peak at a strict ±50 ms tolerance on BUT PDB and pwave (each-lead
policy), for all ensembles and the wavelet baseline. Also stores per-subject counts for CIs.
Output: results/p_tol50.json, results/final/<model>_tol50_records.pkl
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from evaluate import p_only_eval  # noqa: E402
from final_eval import OVERLAP_BUT, load_models, out_name  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402

MODELS = ["unet", "cnn_bilstm", "transformer", "unet_kan", "convkan", "deq", "deq_unrolled", "explicit_stack"]


def main(threads=3, models=None):
    torch.set_num_threads(threads)
    but, pw = data.get("butpdb"), data.get("pwave")
    out = {}
    path = RESULTS / "p_tol50.json"
    if path.exists():
        out = json.loads(path.read_text())
    for m in models or (["wavelet"] + MODELS):
        if out_name(m) in out:
            continue
        recs = {}
        if m == "wavelet":
            from wavelet_baseline import delineate
            tol = int(round(50 * data.FS / 1000))
            for nm, ds in [("butpdb", but), ("pwave", pw)]:
                acc = Accumulator()
                for s in ds:
                    acc.add("P_peak", s.subject, s.points["P_peak"], delineate(s.x)["P_peak"], tol,
                            fp_windows=[s.span])
                recs[f"{nm}_each_lead"] = {"table": acc.table(), "subj": {"P_peak": acc.c["P_peak"]["subj"]}}
        else:
            ms, _ = load_models(m)
            for nm, ds, ex in [("butpdb", but, ()), ("butpdb_nooverlap", but, OVERLAP_BUT), ("pwave", pw, ())]:
                r = p_only_eval(ms, ds, 50, exclude_records=ex)["each_lead"]
                recs[f"{nm}_each_lead"] = {"table": r.table(), "subj": {"P_peak": r.c["P_peak"]["subj"]}}
        out[out_name(m)] = {k: v["table"]["P_peak"] for k, v in recs.items()}
        with open(RESULTS / "final" / f"{out_name(m)}_tol50_records.pkl", "wb") as fh:
            pickle.dump(recs, fh)
        path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        print(m, {k: round(v["F1"], 2) for k, v in out[out_name(m)].items()}, flush=True)


if __name__ == "__main__":
    main(models=sys.argv[1].split(",") if len(sys.argv) > 1 else None)
