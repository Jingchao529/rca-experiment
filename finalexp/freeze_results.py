"""Freeze the current experiment results into a dated, verifiable snapshot.

Produces, under finalexp/frozen/<date>/:
  * a copy of each experiment's JSONL
  * MANIFEST.json  -- SHA-256 of every frozen file, environment, headline scores
  * SUMMARY.md     -- the comparison table, ready to paste into the write-up
  * summary.csv    -- the same numbers for plotting

Re-running with the same date refuses to overwrite unless --force is given, so a
frozen snapshot cannot be silently replaced by a later, different run.

Usage
-----
    python finalexp/freeze_results.py
    python finalexp/freeze_results.py --date 2026-09-06 --force
    python finalexp/freeze_results.py --verify 2026-09-06
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "finalexp" / "results"
FROZEN_ROOT = REPO_ROOT / "finalexp" / "frozen"

EXPERIMENTS = [
    ("experiment_f__gemini-3-1-flash-lite.jsonl", "F flash-lite", "-", "gemini-3.1-flash-lite + tools"),
    ("experiment_f_static__Qwen-Qwen3-5-9B.jsonl", "F-static Qwen-9B", "-", "Qwen3.5-9B, tool rankings in prompt (self-hosted)"),
    ("experiment_f__deepseek-v4-flash.jsonl", "F v4-flash", "-", "deepseek-v4-flash + tools"),
    ("experiment_f.jsonl", "F gpt-4o-mini", "-", "gpt-4o-mini + tools"),
    ("experiment_e__gpt-5-6-terra.jsonl", "E gpt-5.6-terra", "-", "gpt-5.6-terra on telemetry (large-tier reference)"),
    ("experiment_e__deepseek-v4-pro.jsonl", "E v4-pro", "-", "deepseek-v4-pro on telemetry"),
    ("experiment_e__deepseek-v4-flash.jsonl", "E v4-flash", "-", "deepseek-v4-flash on telemetry"),
    ("experiment_e__Qwen-Qwen3-5-9B.jsonl", "E Qwen-9B", "-", "Qwen3.5-9B on telemetry (self-hosted)"),
    ("experiment_e__gemini-3-1-flash-lite.jsonl", "E flash-lite", "-", "gemini-3.1-flash-lite on telemetry"),
    ("experiment_e_prime__gemini-3-1-flash-lite.jsonl", "E' flash-lite", "-", "gemini-3.1-flash-lite, baseline-relative"),
    ("experiment_e.jsonl", "E gpt-4o-mini", "-", "gpt-4o-mini on telemetry"),
    ("experiment_e_prime.jsonl", "E' gpt-4o-mini", "-", "gpt-4o-mini, baseline-relative"),
    ("experiment_d.jsonl", "TORAI", "FSE'26", "multi-source severity clustering"),
    ("experiment_c.jsonl", "CIRCA", "ICSE'22", "causal graph + regression testing"),
    ("experiment_b.jsonl", "BARO", "FSE'24", "statistical hypothesis testing"),
    ("experiment_a.jsonl", "Trace-SHAP", "-", "trace anomaly attribution"),
]

DATASET = "RCAEval RE2-OB (Online Boutique, 90 failure cases)"
DATASET_DOI = "10.5281/zenodo.14590730"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ac_at_k(rows: list[dict], k: int) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["ground_truth"] in r.get("ranking", [])[:k]) / len(rows)


def avg_at_k(rows: list[dict], k: int = 5) -> float:
    return sum(ac_at_k(rows, j) for j in range(1, k + 1)) / k


def summarise(rows: list[dict]) -> dict:
    """Headline numbers for one experiment.

    Rows are averaged over every recorded run. For the non-deterministic method
    that means over all seeds, which is the intended reading: the reported score
    is the expected score, not one lucky run.
    """
    ok = [r for r in rows if r["status"] == "ok"]
    seconds = [r["seconds"] for r in ok]
    cases = {r["case_id"] for r in rows}
    seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})

    per_fault = {}
    by_fault: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fault[row["fault"]].append(row)
    for fault, group in sorted(by_fault.items()):
        per_fault[fault] = {
            "n": len(group),
            "AC@1": round(ac_at_k(group, 1), 4),
            "AC@3": round(ac_at_k(group, 3), 4),
            "Avg@5": round(avg_at_k(group), 4),
        }

    summary = {
        "runs": len(rows),
        "cases": len(cases),
        "seeds": seeds or None,
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "AC@1": round(ac_at_k(rows, 1), 4),
        "AC@3": round(ac_at_k(rows, 3), 4),
        "Avg@5": round(avg_at_k(rows), 4),
        "mean_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "per_fault": per_fault,
    }

    # Seed spread is what justifies repeating a method; recording it lets a
    # reader see which methods needed repetition and which did not.
    if seeds:
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        spreads = [avg_at_k(by_seed[s]) for s in sorted(by_seed)]
        summary["seed_avg5_min"] = round(min(spreads), 4)
        summary["seed_avg5_max"] = round(max(spreads), 4)
        summary["seed_avg5_spread"] = round(max(spreads) - min(spreads), 4)

    # A single service dominating rank 1 signals systematic bias rather than
    # diagnosis, so it is worth freezing alongside the accuracy.
    top1: dict[str, int] = defaultdict(int)
    for row in ok:
        if row.get("ranking"):
            top1[row["ranking"][0]] += 1
    summary["top1_counts"] = dict(sorted(top1.items(), key=lambda kv: -kv[1])[:5])
    return summary


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- absence of git is not an error here
        return None


def write_summary_md(target: Path, entries: list[dict]) -> None:
    lines = [
        "# Frozen results -- RCA baselines on Online Boutique",
        "",
        f"Dataset: {DATASET}",
        f"Zenodo DOI: {DATASET_DOI}",
        "",
        "Metrics follow TORAI (FSE'26, arXiv:2604.13522) section 4.2, Eq. 1.",
        "Every case injects one service, so AC@k is \"is the true service in the",
        "top k\"; Avg@5 is the mean of AC@1..AC@5.",
        "",
        "## Overall",
        "",
        "| Method | Year | Mechanism | Cases | AC@1 | AC@3 | Avg@5 | s/case |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        s = entry["summary"]
        lines.append(
            f"| {entry['method']} | {entry['year']} | {entry['mechanism']} | "
            f"{s['cases']} | {s['AC@1']:.3f} | {s['AC@3']:.3f} | "
            f"{s['Avg@5']:.3f} | {s['mean_seconds']:.2f} |"
        )

    faults = sorted({f for e in entries for f in e["summary"]["per_fault"]})
    lines += ["", "## AC@3 by fault type", "",
              "| Fault | " + " | ".join(e["method"] for e in entries) + " |",
              "|---" * (len(entries) + 1) + "|"]
    for fault in faults:
        cells = []
        for entry in entries:
            stats = entry["summary"]["per_fault"].get(fault)
            cells.append(f"{stats['AC@3']:.3f}" if stats else "-")
        lines.append(f"| {fault} | " + " | ".join(cells) + " |")

    lines += ["", "## Notes", ""]
    for entry in entries:
        s = entry["summary"]
        note = (f"- **{entry['method']}**: {s['runs']} runs over {s['cases']} cases, "
                f"{s['failed']} failed.")
        if s.get("seeds"):
            note += (f" Non-deterministic: seeds {s['seeds']}, "
                     f"Avg@5 spread {s['seed_avg5_spread']:.4f} "
                     f"({s['seed_avg5_min']:.3f}-{s['seed_avg5_max']:.3f}); "
                     "reported figures are the mean across seeds.")
        elif s["runs"] > s["cases"]:
            note += f" {s['runs'] // s['cases']} runs per case, averaged."
        else:
            note += " One run per case."
        top = ", ".join(f"{k} x{v}" for k, v in list(s["top1_counts"].items())[:3])
        note += f" Ranked #1 most often: {top}."
        lines.append(note)

    lines += [
        "",
        "Reproduction commands and the full protocol are in "
        "`finalexp/README.md` and `finalexp/protocol/EXPERIMENT_PROTOCOL.md`.",
        "File integrity hashes are in `MANIFEST.json` alongside this file.",
        "",
    ]
    target.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(target: Path, entries: list[dict]) -> None:
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "year", "mechanism", "fault", "n",
                         "AC@1", "AC@3", "Avg@5", "mean_seconds"])
        for entry in entries:
            s = entry["summary"]
            writer.writerow([entry["method"], entry["year"], entry["mechanism"],
                             "ALL", s["cases"], s["AC@1"], s["AC@3"], s["Avg@5"],
                             s["mean_seconds"]])
            for fault, stats in s["per_fault"].items():
                writer.writerow([entry["method"], entry["year"], entry["mechanism"],
                                 fault, stats["n"], stats["AC@1"], stats["AC@3"],
                                 stats["Avg@5"], ""])


def verify(snapshot: Path) -> int:
    manifest_path = snapshot / "MANIFEST.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok = True
    for entry in manifest["experiments"]:
        path = snapshot / entry["file"]
        if not path.exists():
            print(f"MISSING  {entry['file']}")
            ok = False
            continue
        actual = sha256(path)
        if actual == entry["sha256"]:
            print(f"OK       {entry['file']}  {actual[:16]}...")
        else:
            print(f"CHANGED  {entry['file']}\n  expected {entry['sha256']}\n"
                  f"  actual   {actual}")
            ok = False
    print("\nsnapshot intact" if ok else "\nSNAPSHOT ALTERED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing snapshot for this date.")
    parser.add_argument("--verify", metavar="DATE",
                        help="Re-hash a snapshot and report any drift.")
    args = parser.parse_args()

    if args.verify:
        return verify(FROZEN_ROOT / args.verify)

    snapshot = FROZEN_ROOT / args.date
    if snapshot.exists() and any(snapshot.iterdir()) and not args.force:
        existing = sorted(p.name for p in snapshot.iterdir())
        print(f"Snapshot {args.date} already exists ({len(existing)} files).")
        print("Pass --force to overwrite, or --date to write a new one.")
        return 1
    snapshot.mkdir(parents=True, exist_ok=True)

    entries = []
    for filename, method, year, mechanism in EXPERIMENTS:
        source = RESULTS_DIR / filename
        if not source.exists():
            print(f"skipping {filename}: not found")
            continue
        shutil.copy2(source, snapshot / filename)
        rows = load(snapshot / filename)
        entries.append({
            "file": filename,
            "method": method,
            "year": year,
            "mechanism": mechanism,
            "sha256": sha256(snapshot / filename),
            "bytes": (snapshot / filename).stat().st_size,
            "summary": summarise(rows),
        })
        print(f"frozen {filename}  {len(rows)} rows")

    if not entries:
        print("Nothing to freeze.")
        return 1

    manifest = {
        "frozen_on": args.date,
        "dataset": DATASET,
        "dataset_doi": DATASET_DOI,
        "metrics": "AC@k / Avg@k per TORAI (FSE'26) section 4.2 Eq. 1",
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "experiments": entries,
    }
    (snapshot / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_md(snapshot / "SUMMARY.md", entries)
    write_summary_csv(snapshot / "summary.csv", entries)

    print(f"\nsnapshot: {snapshot}")
    for name in ("MANIFEST.json", "SUMMARY.md", "summary.csv"):
        print(f"  {name}")
    print(f"\nverify later with:\n  python finalexp/freeze_results.py --verify {args.date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
