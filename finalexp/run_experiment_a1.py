"""Experiment A1: execute the author's analysis.py unchanged on RCAEval.

Uses the original read_traces, schema alignment, and anomaly detection functions:
numeric features INCLUDING total, contamination=0.01, fixed random_state=42,
original CVaR fallback, mongo_rate exclusion, and original unconditional break.
Repeated operations retain their LAST duration, as in the author's JSON reader.

Necessary dataset adaptations remain: CSV -> Jaeger-shaped JSON (CSV row order
is used as span order), per-case pre-injection baseline instead of the missing
reference JSON, and operation/service mapping from RCAEval metadata. 

Usage:
    python finalexp/run_experiment_a1.py --limit 1
    python finalexp/run_experiment_a1.py
    python finalexp/run_experiment_a1.py --dataset data/RCAEval-OB/RE2-TT
    python finalexp/run_experiment_a1.py --score-only
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp" / "adapters"))

from rcaeval_trace_adapter import build_span_process_map, discover_cases  # noqa: E402

SOURCE = REPO_ROOT / "ctl-main" / "util" / "analysis.py"
RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_a1.jsonl"
DEFAULT_DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-OB"
METHOD = "Trace-SHAP A1: original analysis.py functions, RCAEval data adapter"


def ac_at_k(rows: list[dict], k: int) -> float:
    """AC@k with a single-element ground-truth set, so min(k,|V|) == 1."""
    if not rows:
        return 0.0
    hits = sum(1 for r in rows if r["ground_truth"] in r["ranking"][:k])
    return hits / len(rows)


def avg_at_k(rows: list[dict], k: int = 5) -> float:
    return sum(ac_at_k(rows, j) for j in range(1, k + 1)) / k


def results_for(base: Path, dataset_dir: Path) -> Path:
    """Per-dataset results file, so a Train Ticket run cannot overwrite the
    recorded Online Boutique results."""
    suffix = dataset_dir.name.replace("RE2-", "").replace("RE1-", "").replace("RE3-", "")
    if suffix == "OB":
        return base
    return base.with_name(f"{base.stem}__{suffix}{base.suffix}")


def load_existing(results_file: Path = RESULTS_FILE) -> list[dict]:
    if not results_file.exists():
        return []
    with results_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def original_analysis(service_map):
   
    previous = sys.modules.get("config")
    config = ModuleType("config")
    config.PATH = str(SOURCE.parent.parent)
    config.SPAN_PROCESS_MAP = service_map
    sys.modules["config"] = config
    try:
        spec = importlib.util.spec_from_file_location("trace_shap_a1_original", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous


def read_csv_window(analysis, spans):
    """Pass CSV spans through the author's actual JSON reader, without pivoting."""
    traces = [
        {"traceID": trace_id,
         "spans": group[["operationName", "duration", "startTime"]].to_dict("records")}
        for trace_id, group in spans.groupby("traceID", sort=False)
    ]
    with tempfile.TemporaryDirectory(prefix="trace_shap_a1_") as temporary:
        path = Path(temporary) / "traces.json"
        path.write_text(json.dumps({"data": traces}), encoding="utf-8")
        return analysis.read_traces(str(path))

def rank_case(case_dir):
    """Return ranking/diagnostics using unmodified original detector code."""
    spans = pd.read_csv(case_dir / "traces.csv")
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())
    starts = spans.groupby("traceID")["startTime"].transform("min")
    before = starts < inject_time * 1_000_000
    if before.all() or not before.any():
        raise ValueError("Both pre-injection and post-injection traces are required")

    with original_analysis(build_span_process_map(spans)) as analysis:
        normal = read_csv_window(analysis, spans.loc[before])
        faulty = read_csv_window(analysis, spans.loc[~before])
        analysis._training_traces_df = normal
        faulty = analysis.align_traces_to_training_schema(faulty)
        result = analysis.perform_shap_anomaly_detection(faulty)
        if "service_anomaly_counts" not in result:
            raise RuntimeError("Original detector returned its error/early-exit result; see console")
        counts = result["service_anomaly_counts"]
        diagnostics = {
            "normal_traces": len(normal),
            "faulty_traces": len(faulty),
            "numeric_features": faulty.select_dtypes(include=["number"]).columns.tolist(),
            "unknown_pattern_traces": int((~faulty["pattern"].isin(normal["pattern"])).sum()),
            "attributed_anomalies": result["anomaly_count"],
            "service_counts": counts,
        }
        return sorted(counts, key=counts.get, reverse=True), diagnostics


def score(rows):
    print(f"\nExperiment A1: {METHOD}")
    print(f"runs={len(rows)}  ok={sum(r['status'] == 'ok' for r in rows)}")
    print(f"{'fault':<12}{'n':>6}{'AC@1':>9}{'AC@3':>9}{'Avg@5':>9}")
    for fault in sorted({r["fault"] for r in rows}) + ["ALL"]:
        subset = rows if fault == "ALL" else [r for r in rows if r["fault"] == fault]
        print(f"{fault:<12}{len(subset):>6}{ac_at_k(subset, 1):>9.3f}"
              f"{ac_at_k(subset, 3):>9.3f}{avg_at_k(subset):>9.3f}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Compatibility option: only the original fixed seed 42 is valid")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.seeds != [42]:
        parser.error("Original analysis.py hardcodes random_state=42; use --seeds 42")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = results_for(RESULTS_FILE, args.dataset)
    rows = load_existing(output)
    if args.score_only:
        score(rows)
        return 0 if rows else 1

    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adapter_hash = hashlib.sha256(
        (REPO_ROOT / "finalexp/adapters/rcaeval_trace_adapter.py").read_bytes()
    ).hexdigest()
    identity = {"source_sha256": source_hash, "runner_sha256": runner_hash,
                "adapter_sha256": adapter_hash, "dataset_path": str(args.dataset.resolve())}
    if any(any(row.get(k) != v for k, v in identity.items()) for row in rows):
        parser.error(f"Existing results use different code/dataset: {output}; archive them before rerunning")
    done = {r["case_id"]: r for r in rows}
    cases = discover_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]
    print(f"cases={len(cases)} seed=42 output={output}")
    for index, case_dir in enumerate(cases, 1):
        case_id = f"{case_dir.parent.name}/{case_dir.name}"
        content_hash = hashlib.sha256((case_dir / "traces.csv").read_bytes()).hexdigest()
        inject_time = int((case_dir / "inject_time.txt").read_text().strip())
        if case_id in done:
            if (done[case_id]["content_sha256"] != content_hash
                    or done[case_id]["inject_time"] != inject_time):
                parser.error(f"Input changed for recorded case {case_id}; archive results first")
            continue
        service, fault = case_dir.parent.name.rsplit("_", 1)
        record = dict(identity, case_id=case_id, dataset=args.dataset.name,
                      ground_truth=service, fault=fault, repetition=case_dir.name,
                      seed=42, method=METHOD, content_sha256=content_hash,
                      inject_time=inject_time)
        started = time.perf_counter()
        try:
            ranking, diagnostics = rank_case(case_dir)
            record.update(status="ok" if ranking else "empty_ranking",
                          ranking=ranking, diagnostics=diagnostics)
        except Exception as exc:
            record.update(status="failed", ranking=[], error=f"{type(exc).__name__}: {exc}",
                          traceback=traceback.format_exc())
        record["seconds"] = time.perf_counter() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        rows.append(record)
        print(f"[{index}/{len(cases)}] {case_id}: {record['status']} "
              f"{record['seconds']:.2f}s top3={record['ranking'][:3]}")
    score(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
