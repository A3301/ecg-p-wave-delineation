"""Compute manuscript A scores from the 48 BUT PDB segments with P references."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
from scipy.stats import wilcoxon


ARCH = ("unet", "cnn_bilstm", "unet_kan", "deq")
SUBGROUPS = {
    "AV block": {"BI","BII","BIII"}, "Atrial ectopy": {"A","a"},
    "Ventricular ectopy": {"V","B","T","VT","VP","E","F"},
    "AF/AFL": {"AFIB","AFL","FIB"}, "Nodal/SVTA": {"NOD","J","SVTA"},
    "Paced/BBB/other": {"P","L","R","PREX","IVR","NA","VFL"}
}
KEYS = {"all":"butpdb_each_lead", "no_overlap":"butpdb_nooverlap_each_lead", "best":"butpdb_best_lead"}


def records(readme):
    return [(m[3].strip()+":"+m[4],set(x.strip() for x in m[2].split(","))) for l in readme.read_text(encoding="utf8").splitlines()
            if (m:=re.match(r"^(\d+)\s+'([^']*)'\s+(.+?)\s+(\d+)\s+\[",l)) and m[1].zfill(2) not in {"08","48"}]


def score(array):
    tp,fn,fp=np.moveaxis(np.asarray(array),-1,0)
    return 200*tp/np.maximum(2*tp+fn+fp,1)


def bounds(samples):
    return [float(p) for p in np.percentile(samples,[2.5,97.5])]


def holm(p):
    out=[0.]*len(p)
    acc=0.
    for i,j in enumerate(np.argsort(p)):
        acc=max(acc,min(1.,float((len(p)-i)*p[j])))
        out[j]=acc
    return out


class Cohort:
    def __init__(self, source):
        self.source=source

    def matrix(self,model,protocol="all",tol=150,ids=None):
        d=self.source[model][f"{protocol}_{tol}"]
        keys=sorted(d) if ids is None else sorted(set(ids)&d.keys())
        return keys,np.array([d[k][:3] for k in keys],dtype=int)

    def measure(self,model,protocol="all",tol=150,ids=None):
        keys,arr=self.matrix(model,protocol,tol,ids)
        tp,fn,fp=map(int,arr.sum(axis=0))
        idx=np.random.default_rng(0).integers(len(keys),size=(5000,len(keys)))
        return dict(n=len(keys),tp=tp,fn=fn,fp=fp,f1=float(score([tp,fn,fp])),
                    se=100*tp/max(tp+fn,1),ppv=100*tp/max(tp+fp,1),ci=bounds(score(arr[idx].sum(axis=1))))

    def compare(self,model,baseline,protocol="all",tol=150,interaction_with=None):
        ka,a=self.matrix(model,protocol,tol)
        kb,b=self.matrix(baseline,protocol,tol)
        assert ka==kb
        ix=np.random.default_rng(0).integers(len(ka),size=(10000,len(ka)))
        a1,b1=float(score(a.sum(axis=0))),float(score(b.sum(axis=0)))
        delta=score(a[ix].sum(axis=1))-score(b[ix].sum(axis=1))
        effect=a1-b1
        if interaction_with:
            k1,c=self.matrix(interaction_with[0],protocol,tol)
            k2,d=self.matrix(interaction_with[1],protocol,tol)
            assert k1==k2==ka
            effect-=float(score(c.sum(axis=0))-score(d.sum(axis=0)))
            delta-=score(c[ix].sum(axis=1))-score(d[ix].sum(axis=1))
        p=min(1.,2*min(float(np.mean(delta<=0)),float(np.mean(delta>=0))))
        return dict(n=len(ka),a=a1,b=b1,diff=effect,ci=bounds(delta),p=p)


def extract(results,readme,out):
    from a_reference_replay import read
    cohort=records(readme)
    eligible={s for s,_ in cohort}
    assert len(cohort)==48 and len(eligible)==43
    models=defaultdict(dict)
    for file in sorted((results/"final").glob("*_records.pkl")):
        name=file.name.removesuffix("_records.pkl")
        name,tol=(name.removesuffix("_tol50"),50) if name.endswith("_tol50") else (name,150)
        document=read(file)
        for short,tag in KEYS.items():
            if tag not in document:
                continue
            d=document[tag]["subj"]["P_peak"]
            values={k:[int(v[key]) for key in ("tp","fn","fp")]+[float(x) for x in v["err"]] for k,v in d.items() if k in eligible}
            assert len(values)==(36 if short=="no_overlap" else 43)
            if short=="all":
                assert sum(v[0]+v[1] for v in values.values())==10858
            models[name][f"{short}_{tol}"]=values
    for f in sorted((results/"final_firstlead").glob("*_records.pkl")):
        m=f.name.removesuffix("_records.pkl")
        for tol in (150,50):
            d=read(f).get(f"butpdb_tol{tol}",{}).get("subj",{})
            if not d:
                continue
            values={k:[int(v[key]) for key in ("tp","fn","fp")]+[float(x) for x in v.get("err",[])] for k,v in d.items() if k in eligible}
            assert len(values)==43 and sum(v[0]+v[1] for v in values.values())==5429
            models[m][f"first_{tol}"]=values
    out.mkdir(exist_ok=True,parents=True)
    (out/"BUT_PDB_Source_Counts.json").write_text(json.dumps(models,separators=(",",":")),encoding="utf8")
    group_values={name:sorted({s for s,t in cohort if t&codes}) for name,codes in SUBGROUPS.items()}
    (out/"BUT_PDB_Rhythm_Groups.json").write_text(json.dumps(group_values,indent=2),encoding="utf8")
    summary(Cohort(models),out)


def summary(cohort,out):
    rows=[]
    for model,protocols in cohort.source.items():
        for label in sorted(protocols):
            protocol,tol=label.rsplit("_",1)
            rows.append({"model":model,"lead_protocol":protocol,"tolerance_ms":tol,**cohort.measure(model,protocol,int(tol))})
    with (out/"BUT_PDB_Metrics.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    contrasts=[]
    for a in ARCH:
        x=cohort.compare(a+"__s2dissoc",a)
        _,b=cohort.matrix(a)
        _,v=cohort.matrix(a+"__s2dissoc")
        x["wilcoxon"]=float(wilcoxon(score(v),score(b)).pvalue)
        contrasts.append(dict(model=a,**x))
    for x,p in zip(contrasts,holm([r["p"] for r in contrasts])):
        x["holm"]=p
    (out/"BUT_PDB_Primary_Contrasts.json").write_text(json.dumps(contrasts,indent=2),encoding="utf8")


if __name__=="__main__":
    if len(sys.argv)==4:
        extract(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
    elif len(sys.argv)==2:
        folder=Path(sys.argv[1])
        summary(Cohort(json.loads((folder/"BUT_PDB_Source_Counts.json").read_text())),folder)
    else:
        raise SystemExit("Usage: cohort48_metrics.py <folder with inputs>, or <source results> <BUT PDB README> <output folder>")
