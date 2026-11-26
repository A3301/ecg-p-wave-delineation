"""Phase 0: extract raw dataset archives into data/raw/ and verify every member
against the PhysioNet-provided SHA256SUMS.txt (copied read-only from VERI/).

Usage: python src/prepare_data.py
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERI = ROOT / "VERI"
RAW = ROOT / "data" / "raw"
DATASETS = ["ludb_1.0.1", "qtdb_1.0.0", "but-pdb_1.0.0", "pwave_1.0.0"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest.lower()
    return sums


def main() -> None:
    report = {}
    for ds in DATASETS:
        dst = RAW / ds
        zpath = dst / f"{ds}.zip"
        meta = json.loads((VERI / ds / "DOGRULAMA.json").read_text(encoding="utf-8"))
        archive_ok = sha256(zpath) == meta["archive_sha256"]

        out = dst / "extracted"
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                target = (out / info.filename).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise RuntimeError(f"unsafe path in archive: {info.filename}")
            if not out.exists():
                z.extractall(out)

        # Data root: the single top-level folder inside the archive, if any.
        tops = [p for p in out.iterdir()]
        data_root = tops[0] if len(tops) == 1 and tops[0].is_dir() else out

        sums = read_sums(VERI / ds / "SHA256SUMS.txt")
        missing, mismatched = [], []
        for name, digest in sums.items():
            p = data_root / name
            if not p.exists():
                missing.append(name)
            elif sha256(p) != digest:
                mismatched.append(name)
        n_files = sum(1 for p in data_root.rglob("*") if p.is_file())
        report[ds] = {
            "archive_sha256_ok": archive_ok,
            "data_root": str(data_root.relative_to(ROOT)),
            "files_on_disk": n_files,
            "files_in_sha256sums": len(sums),
            "missing": missing,
            "mismatched": mismatched,
        }
        print(f"{ds}: archive_ok={archive_ok} files={n_files} listed={len(sums)} "
              f"missing={len(missing)} mismatched={len(mismatched)} root={data_root.name}")

    (RAW / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
