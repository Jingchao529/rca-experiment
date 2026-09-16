"""Adapt RCAEval RE2/RE3 trace CSVs to the frame `ctl-main/util/analysis.py` expects.

`analysis.py:read_traces()` parses Jaeger JSON: one dict per trace, whose spans
become columns keyed by `operationName`, plus `startTime` and `total`. RCAEval
ships the same information as a long-form CSV -- one row per span -- so this
module performs the pivot rather than reimplementing any analysis logic.

The detection pipeline downstream (IsolationForest contamination, CVaR alpha,
SHAP attribution) is untouched; only the data plumbing changes.

"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Columns that describe the trace itself rather than a span duration. Kept in
# sync with analysis.py:compute_trace_pattern, which excludes the same set.
NON_SPAN_COLUMNS = ["id", "startTime", "total", "pattern"]


@dataclass
class TraceCase:
    """One RCAEval failure case, loaded and pivoted."""

    case_id: str
    service: str          # ground-truth root cause, parsed from the path
    fault: str            # cpu | mem | disk | socket | delay | loss
    repetition: str
    inject_time: int      # unix seconds
    traces: pd.DataFrame  # wide, one row per trace
    span_process_map: dict[str, str] = field(default_factory=dict)
    content_hash: str = ""

    @property
    def label(self) -> str:
        return f"{self.service}_{self.fault}_{self.repetition}"


def _read_inject_time(case_dir: Path) -> int:
    return int((case_dir / "inject_time.txt").read_text().strip())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_span_process_map(spans: pd.DataFrame) -> dict[str, str]:
    """operationName -> serviceName, read off the data.

    An operation is emitted by exactly one service in this dataset; where that
    does not hold, the most frequent emitter wins, which mirrors what a
    hand-written map would record.
    """
    pairs = spans.groupby(["operationName", "serviceName"]).size()
    return {
        operation: group.idxmax()[1]
        for operation, group in pairs.groupby(level=0)
    }


def pivot_spans(spans: pd.DataFrame) -> pd.DataFrame:
    
    wide = (
        spans.pivot_table(
            index="traceID",
            columns="operationName",
            values="duration",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"traceID": "id"})
    )
    wide.columns.name = None

    agg = spans.groupby("traceID").agg(
        startTime=("startTime", "min"),
        total=("duration", "sum"),
    )
    wide = wide.merge(agg, left_on="id", right_index=True, how="left")

    wide["startTime"] = pd.to_datetime(wide["startTime"], unit="us")
    return wide


def compute_trace_pattern(df: pd.DataFrame) -> pd.DataFrame:
    
    if df is None or df.empty:
        return df

    span_cols = df.columns.difference(NON_SPAN_COLUMNS)
    if not len(span_cols):
        return df

    df = df.copy()
    df["pattern"] = df[span_cols].gt(0).astype(int).astype(str).agg("".join, axis=1)
    return df


def split_normal_faulty(
    case: TraceCase,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    inject = pd.to_datetime(case.inject_time, unit="s")
    normal = case.traces[case.traces["startTime"] < inject].reset_index(drop=True)
    faulty = case.traces[case.traces["startTime"] >= inject].reset_index(drop=True)
    return normal, faulty


def load_case(case_dir: str | Path) -> TraceCase:

    case_dir = Path(case_dir)
    trace_csv = case_dir / "traces.csv"
    if not trace_csv.exists():
        raise FileNotFoundError(
            f"{case_dir} has no traces.csv -- RE1 cases are metric-only and "
            "cannot be scored with a trace-based method"
        )

    service, fault = case_dir.parent.name.rsplit("_", 1)

    spans = pd.read_csv(trace_csv)
    wide = compute_trace_pattern(pivot_spans(spans))

    return TraceCase(
        case_id=f"{case_dir.parent.name}/{case_dir.name}",
        service=service,
        fault=fault,
        repetition=case_dir.name,
        inject_time=_read_inject_time(case_dir),
        traces=wide,
        span_process_map=build_span_process_map(spans),
        content_hash=_hash_file(trace_csv),
    )


def discover_cases(dataset_dir: str | Path) -> list[Path]:
    """Every `<service>_<fault>/<repetition>/` directory holding traces."""
    dataset_dir = Path(dataset_dir)
    return sorted(
        rep
        for group in sorted(dataset_dir.iterdir())
        if group.is_dir()
        for rep in sorted(group.iterdir())
        if rep.is_dir() and (rep / "traces.csv").exists()
    )


def candidate_services(case: TraceCase) -> list[str]:
    return sorted(set(case.span_process_map.values()))


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else (
        "data/RCAEval-OB/RE2-OB/checkoutservice_cpu/1"
    )
    loaded = load_case(target)
    normal, faulty = split_normal_faulty(loaded)

    print(f"case          : {loaded.case_id}")
    print(f"ground truth  : {loaded.service} ({loaded.fault})")
    print(f"traces        : {len(loaded.traces)}")
    print(f"  normal      : {len(normal)}")
    print(f"  faulty      : {len(faulty)}")
    print(f"span columns  : {len(loaded.traces.columns.difference(NON_SPAN_COLUMNS))}")
    print(f"patterns      : {loaded.traces['pattern'].nunique()}")
    print(f"candidates    : {candidate_services(loaded)}")
    print(f"sha256        : {loaded.content_hash[:16]}...")
