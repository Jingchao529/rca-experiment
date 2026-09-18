"""Experiment D -- TORAI (FSE'26) on RCAEval RE2-OB.

TORAI is a multi-source method: it scores anomaly severity across metrics, logs
and traces, clusters services by symptom, ranks within clusters by causal
analysis, then aggregates. Unlike the other three baselines it consumes all
three telemetry modalities, and unlike trace-graph methods it does not need a
service call graph -- the "blind spots" its paper is named for.

Scored with the same AC@1 / AC@3 / Avg@5 harness as Experiments A, B and C.

Reference: Pham, Ha, Zhang, Zhang, "TORAI: Multi-source Root Cause Analysis for
Blind Spots in Microservice Service Call Graph", Proc. ACM Softw. Eng. 3 (FSE),
arXiv:2604.13522. Evaluation methodology: same paper, section 4.2.

Environment
-----------
Requires a dedicated py38 environment with a patched causal-learn 0.1.2.3 (see
README); the other environments cannot run this:

    conda activate torai38 && python finalexp/run_experiment_d.py

TORAI calls RCAEval's RCD internals, which need a patched causal-learn that is
not on PyPI. RCAEval's SETUP.md does not mention the patch; it is vendored in
the RCD authors' repository (github.com/azamikram/rcd) and four files must be
copied over an installed causal-learn 0.1.2.3:

    causallearn/graph/GraphClass.py                 (CausalGraph gains `labels`)
    causallearn/search/ConstraintBased/FCI.py
    causallearn/utils/Fas.py
    causallearn/utils/PCUtils/SkeletonDiscovery.py  (adds local_skeleton_discovery)

Originals are preserved as `<name>.orig` beside each patched file. This
requirement is why TORAI cannot share an environment with CIRCA, which needs
causal-learn 0.1.4.8 for `pc(node_names=...)`.

Input choice
------------
Reads `simple_metrics.csv`, matching RCAEval's own harness, which globs
`**/data.csv` and falls back to `**/simple_metrics.csv`. RE2-OB ships no
`data.csv`, so the fallback is the intended input -- the same file Experiment C
uses, which keeps the two comparable.

Usage
-----
    python finalexp/run_experiment_d.py --limit 3     # smoke test
    python finalexp/run_experiment_d.py               # full sweep
    python finalexp/run_experiment_d.py --score-only
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

from scoring import score_table  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-OB"
RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_d.jsonl"

METHOD = "TORAI (multi-source severity clustering + causal ranking, RCAEval.e2e.torai)"
CASE_TIMEOUT_SECONDS = 7200  # TORAI's own per-case cap

# The four telemetry frames TORAI expects, keyed as its API requires.
INPUT_FILES = {
    "metric": "simple_metrics.csv",
    "logts": "logts.csv",
    "tracets_err": "tracets_err.csv",
    "tracets_lat": "tracets_lat.csv",
}

NON_SERVICE_PREFIXES = ("gke-", "loadgenerator")


def dataset_label(dataset_dir) -> str:
    """RCAEval's `dataset` argument. Only "causalrca-sock-shop" changes its
    behaviour; every other value takes the same path, so the label is
    informational and simply follows the directory being scored."""
    return "train-ticket" if "-TT" in str(dataset_dir) else "online-boutique"


def results_for(base: Path, dataset_dir: Path) -> Path:
    """Per-dataset results file, so a Train Ticket run cannot overwrite the
    recorded Online Boutique results."""
    suffix = dataset_dir.name.replace("RE2-", "").replace("RE1-", "").replace("RE3-", "")
    if suffix == "OB":
        return base
    return base.with_name(f"{base.stem}__{suffix}{base.suffix}")


def _hash_inputs(case_dir: Path) -> str:
    """One hash over all four inputs, so the whole multi-source input is pinned."""
    digest = hashlib.sha256()
    for filename in INPUT_FILES.values():
        path = case_dir / filename
        if path.exists():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
    return digest.hexdigest()


def metric_to_service(metric: str) -> str | None:
    """`checkoutservice_A` -> `checkoutservice`.

    TORAI ranks indicator-level names; AC@k is defined over services. Splitting
    on the final underscore matches how BARO and CIRCA outputs are folded in
    Experiments B and C, so all four methods are aggregated identically.
    """
    prefix = metric.rsplit("_", 1)[0] if "_" in metric else metric
    if prefix.startswith(NON_SERVICE_PREFIXES):
        return None
    return prefix


def fold_to_services(ranks: list[str]) -> list[str]:
    seen: list[str] = []
    for name in ranks:
        service = metric_to_service(name)
        if service and service not in seen:
            seen.append(service)
    return seen


def discover_cases(dataset_dir: Path) -> list[Path]:
    """Cases carrying all four modalities TORAI needs."""
    return sorted(
        rep
        for group in sorted(dataset_dir.iterdir())
        if group.is_dir()
        for rep in sorted(group.iterdir())
        if rep.is_dir() and all((rep / f).exists() for f in INPUT_FILES.values())
    )


def run_case(case_dir: Path) -> tuple[list[str], list[str], dict]:
    from RCAEval.e2e.torai import torai

    data = {key: pd.read_csv(case_dir / name) for key, name in INPUT_FILES.items()}
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())

    # torai prints its cluster count to stdout on every call; capturing it keeps
    # the progress log readable without suppressing genuine warnings.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        output = torai(data, inject_time=inject_time, dataset=dataset_label(case_dir))

    ranks = [str(r) for r in output.get("ranks", [])]
    chatter = buffer.getvalue().strip()

    info = {
        "metric_columns": data["metric"].shape[1] - 1,
        "logts_columns": data["logts"].shape[1] - 1,
        "tracets_columns": data["tracets_lat"].shape[1] - 1,
        "rows": len(data["metric"]),
        "indicators_ranked": len(ranks),
        "torai_stdout": chatter[:200] if chatter else None,
    }
    return fold_to_services(ranks), ranks, info


def load_existing(results_file: Path = RESULTS_FILE) -> list[dict]:
    if not results_file.exists():
        return []
    with results_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    results_file = results_for(RESULTS_FILE, args.dataset)

    if args.score_only:
        rows = load_existing(results_file)
        if not rows:
            print(f"No results at {results_file}")
            return 1
        score_table(rows, METHOD)
        return 0

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    cases = discover_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    done = {r["case_id"] for r in load_existing(results_file)}
    results_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"dataset : {args.dataset}")
    print(f"cases   : {len(cases)}  ({len(done)} already recorded)\n")

    for index, case_dir in enumerate(cases, start=1):
        case_id = f"{case_dir.parent.name}/{case_dir.name}"
        if case_id in done:
            continue

        service, fault = case_dir.parent.name.rsplit("_", 1)
        started = time.time()
        record = {
            "case_id": case_id,
            "dataset": args.dataset.name,
            "ground_truth": service,
            "fault": fault,
            "repetition": case_dir.name,
            "seed": None,
            "method": METHOD,
            "content_sha256": _hash_inputs(case_dir),
        }
        try:
            ranking, raw_ranks, info = run_case(case_dir)
            elapsed = time.time() - started
            if elapsed > CASE_TIMEOUT_SECONDS:
                raise TimeoutError(f"exceeded {CASE_TIMEOUT_SECONDS}s cap")
            record.update(
                status="ok" if ranking else "empty_ranking",
                ranking=ranking,
                top_indicators=raw_ranks[:10],
                seconds=elapsed,
                diagnostics=info,
            )
            if not ranking:
                record["error"] = "method produced no ranking"
            mark = "OK " if service in ranking[:3] else "-- "
            print(f"[{index}/{len(cases)}] {mark}{case_id} {elapsed:.1f}s "
                  f"top3={ranking[:3]}")
        except Exception as exc:  # noqa: BLE001 -- recorded, never dropped
            record.update(
                status="failed",
                ranking=[],
                seconds=time.time() - started,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            print(f"[{index}/{len(cases)}] ERR {case_id}: {exc}")

        with results_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    score_table(load_existing(results_file), METHOD)
    print(f"\nresults: {results_file}")
    print(f"platform: {platform.platform()} | python {platform.python_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
