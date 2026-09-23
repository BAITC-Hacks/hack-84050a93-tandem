"""Record immutable inputs before the first evaluation; refuse replacement."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = "75716aa8b4dec4bcb9eeb471709c3caa19529211"


def main():
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True).splitlines()
    protected = {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}
    source = list((ROOT / "experiments/signed_pilot_coverage").glob("*.py"))
    source += [ROOT / "experiments/signed_pilot_coverage/PROTOCOL.md", ROOT / "experiments/signed_pilot_coverage/minimal.diff"]
    frozen = dict(protected)
    for p in source:
        frozen[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    raw_artifacts = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ("submission.csv", "reports/decision_trace.json")}
    report = {
        "base_sha": BASE, "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "p0*r + sum(w_i*max(0,r-s_i)); signed s_i, descending; q_i=min(1,n_i/N)",
        "protocol": "experiments/signed_pilot_coverage/PROTOCOL.md",
        "candidate_commit_lookup": "git log -1 --format=%H -- reports/signed_pilot_coverage/freeze.json",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "os": platform.platform()},
        "hashes_normalized_lf": frozen, "protected_base_paths": paths,
        "root_artifacts_raw_sha256": raw_artifacts,
    }
    out = ROOT / "reports/signed_pilot_coverage/freeze.json"
    with out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Frozen {len(frozen)} files; commit before measurement.")


if __name__ == "__main__":
    main()
