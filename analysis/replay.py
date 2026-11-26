"""Recompute the released manuscript analyses from frozen event counts."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cohort48_metrics import ARCH, Cohort, bounds, holm, score, summary

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def european_stt():
    directory = DATA / "external"
    source = read_csv(directory / "QTDB_European_STT_Record_Counts.csv")
    by_model = defaultdict(dict)
    for row in source:
        key = (row["record"], row["participant_group"])
        by_model[row["model"]][key] = np.array([int(row[k]) for k in ("tp","fn","fp")])
    result = []
    for model in ARCH:
        baseline, augmented = [by_model[model + tag] for tag in ("","__s2dissoc")]
        assert len(baseline) == 33 and baseline.keys() == augmented.keys()
        participants = sorted({key[1] for key in baseline})
        assert len(participants) == 31
        pools = []
        for arm in (baseline, augmented):
            grouped = defaultdict(lambda: np.zeros(3, dtype=int))
            for (_, participant), values in arm.items():
                grouped[participant] += values
            pools.append(np.array([grouped[p] for p in participants]))
        b, a = pools
        assert np.array_equal(b[:,:2].sum(axis=1),a[:,:2].sum(axis=1))
        n = int(b[:,:2].sum())
        assert n == 2082
        draw = np.random.default_rng(20261126).integers(len(participants),size=(10000,len(participants)))
        diffs = score(a[draw].sum(axis=1))-score(b[draw].sum(axis=1))
        p = min(1.,2*(min(int((diffs<=0).sum()),int((diffs>=0).sum()))+1)/(len(diffs)+1))
        base, aug = [float(score(x.sum(axis=0))) for x in (b,a)]
        result.append(dict(model=model,records=len(baseline),reference_events=n,groups=len(participants),
                           base=base,aug=aug,delta=aug-base,ci=bounds(diffs),p=p))
    for row, value in zip(result,holm([r["p"] for r in result])):
        row["holm"] = value
    return result


def rhythm_groups(cohort):
    groups = json.loads((DATA/"but_pdb"/"BUT_PDB_Rhythm_Groups.json").read_text(encoding="utf-8"))
    result = []
    for name in ("AF/AFL","Nodal/SVTA"):
        for arch in ARCH:
            for arm in ("baseline","augmented"):
                model = arch + ("__s2dissoc" if arm == "augmented" else "")
                d = cohort.measure(model,ids=groups[name])
                result.append(dict(rhythm_stratum=name,source_recordings=d["n"],architecture=arch,arm=arm,
                                   tp=d["tp"],fn=d["fn"],fp=d["fp"],sensitivity_pct=d["se"],
                                   ppv_pct=d["ppv"],f1_pct=d["f1"]))
    return result


def compare(expected, calculated):
    if isinstance(expected, list):
        assert len(expected) == len(calculated)
        for a,b in zip(expected,calculated):
            compare(a,b)
    elif isinstance(expected, dict):
        assert set(expected) == set(calculated)
        for key in expected:
            compare(expected[key],calculated[key])
    elif isinstance(calculated, (int,float)):
        assert np.isclose(float(expected),calculated,atol=1e-10,rtol=0), (expected,calculated)
    else:
        assert expected == calculated, (expected,calculated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("replay_output"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(DATA.resolve()):
        parser.error("Choose an output folder separate from the archived counts.")
    output.mkdir(parents=True,exist_ok=True)
    primary = DATA / "but_pdb"
    cohort = Cohort(json.loads((primary/"BUT_PDB_Source_Counts.json").read_text(encoding="utf-8")))
    summary(cohort,output)
    primary_table = "BUT_PDB_Primary_Contrasts.json"
    compare(json.loads((primary/primary_table).read_text(encoding="utf-8")),
            json.loads((output/primary_table).read_text(encoding="utf-8")))
    metrics_table = "BUT_PDB_Metrics.csv"
    assert read_csv(primary/metrics_table) == read_csv(output/metrics_table)
    external = european_stt()
    external_table = "QTDB_European_STT_P_Peak_Contrasts.csv"
    expected = read_csv(DATA/"external"/external_table)
    for row in expected:
        row["ci"] = json.loads(row["ci"])
    compare(expected,external)
    write_csv(output/external_table,external)
    rhythm = rhythm_groups(cohort)
    rhythm_table = "Rhythm_Subgroup_Counts.csv"
    compare(read_csv(DATA/"manuscript"/rhythm_table),rhythm)
    write_csv(output/rhythm_table,rhythm)
    print("Verified BUT PDB primary comparisons and pooled metrics, four European ST-T contrasts, "
          "and 16 rhythm subgroup rows against the released reference outputs.")


if __name__ == "__main__":
    main()
