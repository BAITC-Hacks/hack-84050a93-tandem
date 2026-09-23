"""Render a standalone report from saved evidence, without rerunning the agent.

Checksums and recorded preflight totals are checked before any output is written.
This is an integrity check of a saved run, not a new audience or runtime check.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "reporting" / "decision_report.template.html"
TOKEN = "__TANDEM_PAYLOAD__"
COLUMNS = (
    "campaign_name", "filter_arpu_segment", "filter_data_segment",
    "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
)
# Public case contact prices; reading evidence never imports the agent/environment.
CONTACT_COST = {"push": 0, "sms": 4, "digital_ads": 22, "call": 160}
FILTER_VALUES = {
    "filter_arpu_segment": {"LOW", "MID", "HIGH"},
    "filter_data_segment": {"NON_USER", "LITE", "HEAVY"},
    "filter_call_segment": {"LOW", "MEDIUM", "HIGH"},
}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Trace contains a non-finite number")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def mapping(value, name):
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def array(value, name):
    require(isinstance(value, list), f"{name} must be an array")
    return value


def number(value, name, minimum=None, integer=False):
    require(type(value) in (int, float), f"{name} must be numeric")
    require(math.isfinite(value), f"{name} must be finite")
    if integer:
        require(type(value) is int, f"{name} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{name} must be at least {minimum}")
    return value


def same_number(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8),
            f"{name} does not agree with the saved campaign/pilot totals")


def nonempty_text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def expected_manifest_names(field):
    """Mirror the standard exporter's dependency list without importing it."""
    if field == "source_sha256":
        files = [ROOT / "agent.py", *sorted((ROOT / "strategy").glob("*.py")),
                 *sorted((ROOT / "validation").glob("*.py")),
                 ROOT / "tools" / "export_strategy.py", ROOT / "environment.py",
                 ROOT / "mock_environment.py", ROOT / "scoring_core.py",
                 ROOT / "make_submission.py", ROOT / "requirements.txt"]
    else:
        files = [ROOT / "customer_profile.csv", ROOT / "tariff_dictionary.csv",
                 ROOT / "data" / "dict_tariff.csv", ROOT / "data" / "change_tariff.csv"]
    return {path.relative_to(ROOT).as_posix() for path in files}


def manifest_paths(trace, field):
    manifest = mapping(trace.get(field), field)
    require(bool(manifest), f"{field} must not be empty")
    expected_names = expected_manifest_names(field)
    require(set(manifest) == expected_names,
            f"{field} must contain the complete standard exporter manifest "
            f"(missing: {sorted(expected_names - set(manifest))}; "
            f"unexpected: {sorted(set(manifest) - expected_names)})")
    paths = set()
    for name, expected in manifest.items():
        require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
                f"Unsafe manifest path in {field}")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and relative.as_posix() == name
                and all(part not in (".", "..", "") for part in name.split("/")),
                f"Unsafe manifest path: {name}")
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path != ROOT, f"Manifest path escapes repository: {name}")
        require(path.is_file(), f"Manifest file is missing: {name}")
        require(isinstance(expected, str) and HASH.fullmatch(expected), f"Invalid checksum for {name}")
        require(digest(canonical_bytes(path)) == expected, f"Checksum mismatch: {name}")
        require(path not in paths, f"Duplicate resolved manifest path: {name}")
        paths.add(path)
    return paths


def validate_run(trace, csv_text):
    final = mapping(trace.get("final"), "final")
    preflight = mapping(trace.get("preflight"), "preflight")
    require(preflight.get("valid") is True, "Saved preflight must be valid")
    require(preflight.get("errors") == [], "Saved preflight contains errors")
    for owner, name in ((trace, "trace"), (preflight, "preflight")):
        warnings = array(owner.get("warnings"), f"{name}.warnings")
        require(all(isinstance(item, str) for item in warnings), f"{name}.warnings must contain text")
    require(type(final.get("fallback")) is bool, "final.fallback must be boolean")
    nonempty_text(trace.get("environment"), "environment")
    nonempty_text(trace.get("version"), "version")
    require(trace.get("reference_channel") in CONTACT_COST, "Unknown reference channel")
    require(number(trace.get("submission_seed"), "submission_seed", minimum=0, integer=True) == 42,
            "Report requires the standard seed-42 export")
    require(trace.get("deterministic_submission") is True,
            "Standard exporter must record identical repeated submissions")
    versions = mapping(trace.get("versions"), "versions")
    for key in ("python", "numpy", "pandas"):
        nonempty_text(versions.get(key), f"versions.{key}")

    campaigns = array(final.get("campaigns"), "final.campaigns")
    details = array(preflight.get("campaigns"), "preflight.campaigns")
    require(1 <= len(campaigns) <= 10, "There must be 1..10 final campaigns")
    require(len(details) == len(campaigns), "Preflight campaign count differs from final")
    rows = list(csv.reader(io.StringIO(csv_text, newline=""), strict=True))
    require(bool(rows) and rows[0] == list(COLUMNS), "Submission CSV header differs from the official schema")
    require(len(rows) == len(campaigns) + 1, "Submission CSV campaign count differs from final")

    final_contacts, final_cost = 0, 0.0
    names = set()
    for index, (campaign, detail, row) in enumerate(zip(campaigns, details, rows[1:])):
        label = f"campaign[{index}]"
        mapping(campaign, label)
        mapping(detail, f"preflight.{label}")
        require(all(key in campaign for key in COLUMNS), f"{label} is missing submission fields")
        require(all(campaign[key] is None or isinstance(campaign[key], str) for key in COLUMNS),
                f"{label} submission fields must be text or null")
        expected = ["" if campaign[key] is None else campaign[key] for key in COLUMNS]
        require(row == expected, f"CSV row {index + 2} does not match the saved final campaign")
        name = nonempty_text(campaign["campaign_name"], f"{label}.campaign_name")
        require(name not in names, "Campaign names must be unique")
        names.add(name)
        nonempty_text(campaign["target_tariff"], f"{label}.target_tariff")
        nonempty_text(campaign["filter_current_tariff"], f"{label}.filter_current_tariff")
        for key, allowed in FILTER_VALUES.items():
            require(campaign[key] is None or campaign[key] in allowed, f"Invalid {label}.{key}")
        channel = campaign["channel"]
        require(channel in CONTACT_COST, f"Unknown channel in {label}")
        contacts = number(campaign.get("contacts"), f"{label}.contacts", minimum=1, integer=True)
        require(contacts <= 5000, f"{label} exceeds 5000 contacts")
        cost = number(campaign.get("cost"), f"{label}.cost", minimum=0)
        same_number(cost, contacts * CONTACT_COST[channel], f"{label}.cost")
        for key in ("estimated_mean_gain", "risk_adjusted_gain", "reference_scaled_mean"):
            number(campaign.get(key), f"{label}.{key}")
        number(campaign.get("reference_scaled_std"), f"{label}.reference_scaled_std", minimum=0)
        require(number(detail.get("index"), f"preflight.{label}.index", integer=True) == index,
                f"Preflight index differs for {name}")
        require(detail.get("campaign_name") == name, f"Preflight name differs for {name}")
        require(number(detail.get("segment_size"), f"preflight.{label}.segment_size", integer=True) == contacts,
                f"Preflight audience size differs for {name}")
        same_number(detail.get("cost_per_contact"), CONTACT_COST[channel], f"preflight.{label}.cost_per_contact")
        same_number(detail.get("cost"), cost, f"preflight.{label}.cost")
        require(number(detail.get("overlap_with_previous"), f"preflight.{label}.overlap", integer=True) == 0,
                "Saved preflight records overlapping final audiences")
        final_contacts += contacts
        final_cost += cost

    pilots = array(trace.get("pilots"), "pilots")
    require(1 <= len(pilots) <= 20, "There must be 1..20 recorded pilots")
    pilot_contacts, pilot_cost = 0, 0.0
    for index, pilot in enumerate(pilots):
        label = f"pilot[{index}]"
        mapping(pilot, label)
        for key in ("current_tariff", "target_tariff"):
            nonempty_text(pilot.get(key), f"{label}.{key}")
        require(pilot.get("arpu_segment") in FILTER_VALUES["filter_arpu_segment"], f"Invalid {label}.arpu_segment")
        channel = pilot.get("channel")
        require(channel in CONTACT_COST, f"Unknown channel in {label}")
        # The standard Agent reserves enough resources and only probes cells
        # with >=10 members; its saved production pilots obey the case limits.
        contacts = number(pilot.get("n_customers"), f"{label}.n_customers", minimum=10, integer=True)
        require(contacts <= 200, f"{label} exceeds 200 actual pilot contacts")
        for key in ("observed_lift_ratio", "posterior_reference_mean"):
            number(pilot.get(key), f"{label}.{key}")
        for key in ("posterior_reference_std", "estimated_information_value"):
            number(pilot.get(key), f"{label}.{key}", minimum=0)
        require(type(pilot.get("confirmation")) is bool, f"{label}.confirmation must be boolean")
        require(pilot.get("stratified_segment") is None
                or pilot.get("stratified_segment") in FILTER_VALUES["filter_arpu_segment"],
                f"Invalid {label}.stratified_segment")
        pilot_contacts += contacts
        pilot_cost += contacts * CONTACT_COST[channel]

    for owner, key, expected, label in (
        (final, "contacts", final_contacts, "final.contacts"),
        (preflight, "total_contacts", final_contacts, "preflight.total_contacts"),
        (preflight, "unique_customers", final_contacts, "preflight.unique_customers"),
        (final, "total_contacts_including_pilots", final_contacts + pilot_contacts, "final.total_contacts_including_pilots"),
    ):
        require(number(owner.get(key), label, minimum=0, integer=True) == expected, f"Inconsistent {label}")
    same_number(final.get("cost"), final_cost, "final.cost")
    same_number(preflight.get("total_cost"), final_cost, "preflight.total_cost")
    same_number(final.get("total_cost_including_pilots"), final_cost + pilot_cost,
                "final.total_cost_including_pilots")
    require(final_contacts + pilot_contacts <= 15000, "Saved run exceeds 15000 contacts")
    require(final_cost + pilot_cost <= 100000, "Saved run exceeds the 100000 budget")
    remaining_contacts = number(final.get("remaining_contacts_after_pilots"),
                                "final.remaining_contacts_after_pilots", minimum=0, integer=True)
    remaining_budget = number(final.get("remaining_budget_after_pilots"),
                              "final.remaining_budget_after_pilots", minimum=0)
    require(final_contacts <= remaining_contacts and remaining_contacts + pilot_contacts <= 15000,
            "Remaining contact allowance is inconsistent")
    require(final_cost <= remaining_budget and remaining_budget + pilot_cost <= 100000,
            "Remaining budget is inconsistent")


def load_payload(trace_path, submission_path):
    trace_bytes = canonical_bytes(trace_path)
    trace_text = trace_bytes.decode("utf-8")
    trace = json.loads(trace_text, parse_constant=reject_constant,
                       object_pairs_hook=unique_object)
    mapping(trace, "trace")
    finite_tree(trace)
    require(trace.get("hash_format") == "sha256-text-lf-v1", "Unsupported trace hash format")
    source_paths = manifest_paths(trace, "source_sha256")
    input_paths = manifest_paths(trace, "input_sha256")
    submission_bytes = canonical_bytes(submission_path)
    submission_hash = digest(submission_bytes)
    require(submission_hash == trace.get("submission_sha256"), "Submission checksum differs from trace")
    csv_text = submission_bytes.decode("utf-8")
    require("\x00" not in csv_text, "Submission CSV contains a NUL byte")
    validate_run(trace, csv_text)
    payload = {
        "trace": trace,
        "trace_json": trace_text,
        "csv": csv_text,
        "integrity": {
            "source_files": len(source_paths), "input_files": len(input_paths),
            "submission_sha256": submission_hash, "trace_sha256": digest(trace_bytes),
        },
    }
    return payload, source_paths | input_paths


def safe_json(payload):
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        serialized = serialized.replace(character, escaped)
    return serialized


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def render_report(trace_path, submission_path, output):
    trace_path, submission_path, output = (Path(path).resolve() for path in (trace_path, submission_path, output))
    require(output.suffix.lower() == ".html", "Output must have an .html extension")
    require(not output.is_relative_to(ROOT) or output.is_relative_to(ROOT / "reports"),
            "Repository outputs must be inside reports/")
    payload, manifest_inputs = load_payload(trace_path, submission_path)
    protected = {trace_path, submission_path, TEMPLATE.resolve(), Path(__file__).resolve()} | manifest_inputs
    require(output not in protected, "Output must not overwrite an input or source file")
    if output.exists():
        require(all(not output.samefile(path) for path in protected), "Output aliases an input or source file")
    template = TEMPLATE.read_text(encoding="utf-8")
    require(template.count(TOKEN) == 1, "Template must contain exactly one payload placeholder")
    html = template.replace(TOKEN, safe_json(payload), 1).replace("\r\n", "\n").replace("\r", "\n")
    atomic_write(output, html.encode("utf-8"))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "reports" / "decision_trace.json")
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "decision_report.html")
    args = parser.parse_args()
    try:
        payload = render_report(args.trace, args.submission, args.out)
    except (OSError, ValueError, TypeError, OverflowError, RecursionError, csv.Error) as exc:
        parser.error(str(exc))
    integrity = payload["integrity"]
    print(f"Rendered {args.out.resolve()} ({len(payload['trace']['final']['campaigns'])} campaigns; "
          f"{integrity['source_files']} source and {integrity['input_files']} input checksums verified; saved evidence only)")


if __name__ == "__main__":
    main()
