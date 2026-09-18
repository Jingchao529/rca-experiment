"""Experiment C -- CIRCA (ICSE'22) on RCAEval RE2-OB.

CIRCA builds a causal graph over metrics with the PC algorithm, then applies
regression-based hypothesis testing (RHT) to rank root causes. It is the
strongest metric-only method reported on this dataset family, and unlike BARO
and the trace-based method it is not authored by the RCAEval group, which keeps
the baseline set from resting on a single team's work.

Scored with the same AC@1 / AC@3 / Avg@5 harness as Experiments A and B.

Reference: Li et al., "Causal Inference-Based Root Cause Analysis for Online
Service Systems with Intervention Recognition", KDD 2022 / ICSE'22 lineage.
Evaluation methodology: TORAI (FSE'26, arXiv:2604.13522) section 4.2.

Environment
-----------
Requires a py39 environment with causal-learn 0.1.4.8 installed (see README):

    conda activate circa39 && python finalexp/run_experiment_c.py

causal-learn 0.1.4.8 is required: RCAEval's `pc_default` passes `node_names=`,
which 0.1.2.3 (the version pinned in RCAEval's requirements_rcd.lock) does not
accept. That older pin exists for RCD/TORAI, which need a differently patched
build; the two cannot be satisfied at once in one environment.

Input choice
------------
This runner reads `simple_metrics.csv` (73 columns), not `metrics.csv` (418).
That follows RCAEval's own harness, which globs `**/simple_metrics.csv` when
loading cases. The distinction is not cosmetic: on the full metric set the
Fisher-Z conditional independence test aborts with a singular correlation
matrix, because those 418 columns are heavily collinear (rank 197 of 248 even
after constant and duplicate columns are dropped). CIRCA then silently returns
an empty adjacency and a ranking that is just input order -- a degenerate result
that would look like a working method producing a poor score. The reduced set is
the input the method is meant to receive.

Even on the reduced set, about a quarter of cases contain a metric pair
correlated to |r| > 0.9999, which leaves the correlation matrix numerically
singular and aborts the Fisher-Z test. `drop_redundant` removes one column from
each such pair before graph construction; see its docstring.

Usage
-----
    python finalexp/run_experiment_c.py --limit 1     # smoke test
    python finalexp/run_experiment_c.py               # full sweep
    python finalexp/run_experiment_c.py --score-only
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
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

from scoring import score_table  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-OB"
RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_c.jsonl"

METHOD = "CIRCA (PC graph + regression hypothesis testing, RCAEval.e2e.circa)"

# CIRCA builds a full causal graph, so it is far slower than BARO. TORAI's
# published figure for CIRCA on Online Boutique is 4.65s/case; measured here it
# is ~50s, the difference being hardware and the PC implementation. The 2h cap
# is TORAI's own per-case limit.
CASE_TIMEOUT_SECONDS = 7200

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
    prefix = metric.split("_")[0]
    if prefix.startswith(NON_SERVICE_PREFIXES):
        return None
    return prefix


def fold_to_services(metric_ranks: list[str]) -> list[str]:
    """Metric ranking -> service ranking, first occurrence wins.

    Identical to the fold used for BARO, so the two methods are scored on the
    same footing rather than each getting a bespoke aggregation.
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
        if rep.is_dir() and (rep / "simple_metrics.csv").exists()
    )


def drop_redundant(frame: pd.DataFrame, threshold: float = 0.9999) -> pd.DataFrame:
    """Drop one column from each near-perfectly-correlated pair.

    CIRCA's PC step uses the Fisher-Z conditional independence test, which
    inverts the correlation matrix and aborts when that matrix is singular. On
    this dataset a handful of metric pairs are correlated to |r| > 0.9999 (for
    example a counter and its own rate), which drives the smallest eigenvalue to
    ~1e-15 and makes the matrix numerically singular even at full algebraic
    rank. Roughly a quarter of RE2-OB cases hit this.

    Removing one column from each such pair is the conventional remedy: the two
    carry the same information, so the causal structure over the remaining
    columns is unchanged, and the surviving column keeps its own name in the
    ranking. Numerical rank filtering alone does not fix it -- the failure is in
    the correlation matrix's conditioning, not in linear dependence.
    """
    frame = frame.loc[:, frame.std() > 0]
    if frame.shape[1] < 2:
        return frame

    corr = np.nan_to_num(np.corrcoef(frame.to_numpy().T))
    np.fill_diagonal(corr, 0.0)

    dropped: set[int] = set()
    for i in range(corr.shape[0]):
        if i in dropped:
            continue
        for j in range(i + 1, corr.shape[0]):
            if j not in dropped and abs(corr[i, j]) > threshold:
                dropped.add(j)

    kept = [i for i in range(frame.shape[1]) if i not in dropped]
    return frame.iloc[:, kept]


def select_most_shifted(frame: pd.DataFrame, time_col: pd.Series,
                        inject_time: int, keep: int) -> pd.DataFrame:
    """Keep the `keep` columns that move most across the injection.

    PC's cost is driven by the number of variables, not the number of samples:
    Train Ticket presents 318 columns against Online Boutique's 60, which is
    50,403 node pairs against 1,770, and one measured case ran 13,179 s before
    passing the 2 h cap. Shortening the time window does not help, since both
    systems supply the same 1,441 rows.

    Columns are ranked by |mean(after) - mean(before)| / std(before), i.e. how
    far each series moved relative to its own pre-fault variability.

    This is a reduction we impose, not part of CIRCA, and it should be reported
    as such: the unreduced result is a timeout, and this variant is a separate
    measurement. The ranking signal also resembles what BARO computes
    internally, so it does part of CIRCA's work for it and may flatter the
    score. Default `keep` is 60 to match the column count Online Boutique
    presents, so CIRCA faces a graph of the same size on both systems.
    """
    before = frame[time_col.values < inject_time]
    after = frame[time_col.values >= inject_time]
    if before.empty or after.empty or frame.shape[1] <= keep:
        return frame
    shift = ((after.mean() - before.mean()).abs() / (before.std() + 1e-9))
    return frame[shift.sort_values(ascending=False).head(keep).index]


def run_case(case_dir: Path, keep_columns: int | None = None) -> tuple[list[str], list[str], dict]:
    from RCAEval.graph_construction.pc import pc_default
    from RCAEval.graph_heads.rht import rht
    from RCAEval.io.time_series import preprocess

    # ffill then fillna(0) mirrors RCAEval main.py lines 286-287. Without it the
    # Fisher-Z test refuses the input outright ("Input data contains NaN").
    metrics = pd.read_csv(case_dir / "simple_metrics.csv").ffill().fillna(0)
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())
    time_col = metrics["time"]

    # This reproduces RCAEval.e2e.circa.circa step by step rather than calling
    # it, because the redundancy filter has to sit between `preprocess` and
    # `pc_default`: preprocess rescales memory columns and so reintroduces the
    # collinearity if the filter is applied to the raw frame. The graph
    # construction (pc_default) and the ranking head (rht) are the library's
    # own, unmodified.
    processed = preprocess(
        data=metrics, dataset=dataset_label(case_dir), dk_select_useful=False
    )
    pc_input = drop_redundant(processed.drop(columns=["time"], errors="ignore"))
    columns_before_selection = pc_input.shape[1]
    if keep_columns:
        pc_input = select_most_shifted(pc_input, time_col, inject_time, keep_columns)

    adjacency = np.array(pc_default(pc_input, dataset="ob"))

    ranked_input = pc_input.copy()
    ranked_input["time"] = time_col.values
    scored = sorted(rht(adjacency, inject_time, ranked_input),
                    key=lambda item: item[1], reverse=True)
    metric_ranks = [str(name) for name, _ in scored]

    info = {
        "metric_columns": len(metrics.columns) - 1,
        "columns_after_preprocess": processed.shape[1],
        "columns_into_pc": pc_input.shape[1],
        "columns_before_selection": columns_before_selection,
        "keep_columns": keep_columns,
        "rows": len(metrics),
        "graph_nodes": int(adjacency.shape[0]) if adjacency.ndim == 2 else 0,
        "graph_edges": int((adjacency != 0).sum()) if adjacency.size else 0,
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
    parser.add_argument("--keep-columns", type=int, default=None,
                        help="Reduce the metric set to the N most-shifted "
                             "columns before graph construction. Needed on "
                             "Train Ticket, where the full 318 columns exceed "
                             "the 2h per-case cap; 60 matches Online Boutique.")
    args = parser.parse_args()

    results_file = results_for(RESULTS_FILE, args.dataset)

    if args.score_only:
        rows = load_existing(results_file)
        if not rows:
            print(f"No results at {results_file}")
            return 1
        score_table(rows, METHOD)
        _report_degenerate(rows)
        return 0

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    cases = discover_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    done = {r["case_id"] for r in load_existing(results_file)}
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # PC with a fixed alpha and the Fisher-Z test is deterministic, so one run
    # per case suffices, as with BARO.
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
            "content_sha256": _hash_file(case_dir / "simple_metrics.csv"),
        }
        try:
            ranking, metric_ranks, info = run_case(case_dir, args.keep_columns)
            elapsed = time.time() - started
            if elapsed > CASE_TIMEOUT_SECONDS:
                raise TimeoutError(f"exceeded {CASE_TIMEOUT_SECONDS}s cap")

            # An empty graph means PC failed and the ranking is input order, not
            # a diagnosis. Recording it as ok would hide a broken run behind a
            # plausible-looking score.
            status = "ok"
            if info["graph_edges"] == 0:
                status = "degenerate_graph"
                record["error"] = "PC produced no edges; ranking is not a diagnosis"

            record.update(
                status=status,
                ranking=ranking,
                top_metrics=metric_ranks[:10],
                seconds=elapsed,
                diagnostics=info,
            )
            mark = "OK " if service in ranking[:3] else "-- "
            if status != "ok":
                mark = "DEG"
            print(f"[{index}/{len(cases)}] {mark}{case_id} {elapsed:.1f}s "
                  f"edges={info['graph_edges']} top3={ranking[:3]}")
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

    rows = load_existing(results_file)
    score_table(rows, METHOD)
    _report_degenerate(rows)
    print(f"\nresults: {results_file}")
    print(f"platform: {platform.platform()} | python {platform.python_version()}")
    return 0


def _report_degenerate(rows: list[dict]) -> None:
    """Surface empty-graph cases: they score but carry no causal evidence."""
    degenerate = [r for r in rows if r.get("status") == "degenerate_graph"]
    if degenerate:
        print(f"\nWARNING: {len(degenerate)}/{len(rows)} cases produced an empty "
              "causal graph; their rankings are input order, not a diagnosis.")
    graphs = [r["diagnostics"]["graph_edges"] for r in rows
              if r.get("diagnostics", {}).get("graph_edges")]
    if graphs:
        print(f"causal graph edges: mean {np.mean(graphs):.0f}, "
              f"min {min(graphs)}, max {max(graphs)}")


if __name__ == "__main__":
    raise SystemExit(main())
