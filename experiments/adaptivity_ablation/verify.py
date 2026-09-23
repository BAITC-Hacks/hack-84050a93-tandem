"""Fail closed if a frozen policy, fixture, scorer, or public input changes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments/adaptivity_ablation/frozen.json"


def digest(path):
    # Canonical text hashing is invariant to Git autocrlf on Windows.
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def verify_frozen():
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in frozen["sha256_lf_utf8"].items():
        assert digest(ROOT / name) == expected, f"Frozen input changed: {name}"
    from experiments.adaptivity_ablation.build_candidate import build
    assert (ROOT / "experiments/adaptivity_ablation/fixed_agent.py").read_text(encoding="utf-8") == build(
        (ROOT / "agent.py").read_text(encoding="utf-8")), "Candidate transformation differs"
    return frozen
