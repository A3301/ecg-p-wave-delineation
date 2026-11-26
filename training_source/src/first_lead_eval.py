"""P_peak on BUT PDB and MIT-BIH P-wave using ONLY the first lead (channel 0; MLII in MIT-BIH), matching the
protocol of Saclova et al. 2022 ("In all databases, the first lead was used for algorithm testing", PMC9023481).
Output: results/final_firstlead/<name>.json and <name>_records.pkl  (P_peak, ±150 ms and ±50 ms)
Usage: python src/first_lead_eval.py wavelet,cnn_bilstm,cnn_bilstm@s2dissoc,... [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
from evaluate import predict_proba  # noqa: E402
from final_eval import load_models, out_name  # noqa: E402
from metrics import Accumulator  # noqa: E402
from paths import RESULTS  # noqa: E402
from postprocess import mask_to_points  # noqa: E402

OUT = RESULTS / "final_firstlead"


def first_leads(samples):
    seen, out = set(), []
    for s in samples:
        if s.record not in seen:
            seen.add(s.record)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"butpdb": first_leads(data.get("butpdb")), "pwave": first_leads(data.get("pwave"))}
    for m in a.models.split(","):
        if (OUT / f"{out_name(m)}.json").exists():
            continue
        if m == "wavelet":
            from wavelet_baseline import delineate
            pred_fn = lambda x: delineate(x)["P_peak"]  # noqa: E731
        else:
            ms = load_models(m)[0]
            pred_fn = lambda x, ms=ms: mask_to_points(predict_proba(ms, x).argmax(0), x)["P_peak"]  # noqa: E731
        res, recs = {"model": m, "leads": {k: sorted({s.lead for s in v}) for k, v in sets.items()}}, {}
        for nm, smp in sets.items():
            accs = {150: Accumulator(), 50: Accumulator()}
            for s in smp:
                det = pred_fn(s.x)
                for tol_ms, acc in accs.items():
                    acc.add("P_peak", s.subject, s.points["P_peak"], det, int(round(tol_ms * data.FS / 1000)),
                            fp_windows=[s.span])
            for tol_ms, acc in accs.items():
                res[f"{nm}_tol{tol_ms}"] = acc.table()["P_peak"]
                recs[f"{nm}_tol{tol_ms}"] = {"subj": acc.c["P_peak"]["subj"]}
        (OUT / f"{out_name(m)}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        with open(OUT / f"{out_name(m)}_records.pkl", "wb") as fh:
            pickle.dump(recs, fh)
        print(m, {k: round(v["F1"], 2) for k, v in res.items() if k.endswith("tol150")}, flush=True)


if __name__ == "__main__":
    main()
