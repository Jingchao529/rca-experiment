"""Experiment B -- BARO (FSE'24) on RCAEval RE2-OB.

BARO ranks metrics by how far their distribution shifts after the failure,
using RobustScaler-style median/IQR statistics rather than a causal graph. This
runner scores it with the same AC@1 / AC@3 / Avg@5 harness as Experiment A, so
both methods land in one comparable table.

Reference: Pham et al., "BARO: Robust Root Cause Analysis for Microservices via
Multivariate Bayesian Online Change Point Detection", FSE 2024.
Evaluation methodology: TORAI (FSE'26, arXiv:2604.13522) section 4.2.

BARO is used as the authors shipped it, via `RCAEval.e2e.baro`. Nothing about
the algorithm is reimplemented here; this file only supplies data, aggregates
the metric-level output to services, and scores it.

Usage
-----
    python finalexp/run_experiment_b.py --limit 1      # smoke test
    python finalexp/run_experiment_b.py                # full 90-case sweep
    python finalexp/run_experiment_b.py --score-only

Results append to finalexp/results/experiment_b.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

from scoring import score_table  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-OB"
RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_b.jsonl"

METHOD = "BARO (median/IQR change ranking, RCAEval.e2e.baro)"

# Column prefixes that name infrastructure rather than an application service.
# A fault is only ever injected into an application service, so ranking a node
# or the load generator as the root cause is meaningless; these are dropped
# before the metric ranking is folded into a service ranking. The set is
# derived from the data, not guessed: gke-* are cluster nodes.
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_to_service(metric: str) -> str | None:
    """`checkoutservice_container-cpu-...` -> `checkoutservice`.

    Returns None for infrastructure series, which are excluded from ranking.
    """
    prefix = metric.split("_")[0]
    if prefix.startswith(NON_SERVICE_PREFIXES):
        return None
    return prefix


def fold_to_services(metric_ranks: list[str]) -> list[str]:
    """Metric ranking -> service ranking, first occurrence wins.

    BARO ranks individual metrics; AC@k is defined over services. A service is
    placed at the position of its highest-ranked metric, which is the standard
    reading of a metric ranking as a service ranking and keeps the ordering BARO
    produced rather than re-scoring it.
    """
    seen: list[str] = []
    for metric in metric_ranks:
        service = metric_to_service(metric)
        if service and service not in seen:
            seen.append(service)
    return seen


def discover_cases(dataset_dir: Path) -> list[Path]:
    return sorted(
        rep
        for group in sorted(dataset_dir.iterdir())
        if group.is_dir()
        for rep in sorted(group.iterdir())
        if rep.is_dir() and (rep / "metrics.csv").exists()
    )


def run_case(case_dir: Path) -> tuple[list[str], list[str], dict]:
    """Run BARO on one case. Returns (service_ranking, metric_ranking, info)."""
    from RCAEval.e2e import baro

    metrics = pd.read_csv(case_dir / "metrics.csv")
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())

    output = baro(metrics, inject_time=inject_time, dataset=dataset_label(case_dir))
    metric_ranks = [str(m) for m in output.get("ranks", [])]

    info = {
        "metric_columns": len(metrics.columns) - 1,
        "rows": len(metrics),
        "normal_rows": int((metrics["time"] < inject_time).sum()),
        "faulty_rows": int((metrics["time"] >= inject_time).sum()),
        "metrics_ranked": len(metric_ranks),
    }
    return fold_to_services(metric_ranks), metric_ranks, info


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

    # BARO is deterministic -- median and IQR involve no sampling and no seeded
    # model -- so a single run per case is sufficient. Experiment A's seed sweep
    # exists because IsolationForest and SHAP are stochastic.
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
            "seed": None,  # deterministic
            "method": METHOD,
            "content_sha256": _hash_file(case_dir / "metrics.csv"),
        }
        try:
            ranking, metric_ranks, info = run_case(case_dir)
            record.update(
                status="ok" if ranking else "empty_ranking",
                ranking=ranking,
                top_metrics=metric_ranks[:10],
                seconds=time.time() - started,
                diagnostics=info,
            )
            if not ranking:
                record["error"] = "method produced no ranking"
            mark = "OK " if service in ranking[:3] else "-- "
            print(f"[{index}/{len(cases)}] {mark}{case_id} "
                  f"{record['seconds']:.2f}s top3={ranking[:3]}")
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
