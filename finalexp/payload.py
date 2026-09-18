"""Build the telemetry payload both LLM conditions receive.

Conditions E (raw_direct) and F (tool_agent) are given the *same* payload; F
additionally gets the three RCA tools. Keeping the construction in one module
guarantees they cannot drift apart, which is what makes the comparison clean.

Reduction policy (the "conservative" variant): raw metric values are preserved.
Only structural waste is removed --- columns that never change inside the window,
absolute epochs, and telemetry far outside the window. No baseline
normalisation, no percent-change transform: those would pre-compute part of the
diagnosis and weaken the claim that the model reasoned from the data.

Measured cost on RE2-OB with the gpt-4o-mini tokenizer (o200k_base), 10 cases:
median 16,876 tokens, range 16,608--19,164.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# The window the model sees around the fault. +/-30 s at 1 s resolution keeps
# short spikes intact; CPU-hog and packet-loss faults present as brief bursts
# that any temporal averaging erases.
METRIC_WINDOW_SECONDS = 30

# The log/trace series are sampled far more coarsely (~15 s), so they need a
# wider window to show a trend at all.
SERIES_WINDOW_SECONDS = 120
LOG_WINDOW_SECONDS = 60

# Present in every RE2-OB case. `log_template` exists in only 25 of 91 cases,
# so nothing here may depend on it.
LOG_COLUMNS = ["timestamp", "container_name", "level", "message"]

SERIES_FILES = ["logts.csv", "tracets_err.csv", "tracets_lat.csv"]

# The 11 Online Boutique services. Supplied to the model as a closed candidate
# set so it cannot invent a name; anything outside this list is scored a miss.
CANDIDATE_SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "paymentservice", "productcatalogservice",
    "recommendationservice", "redis", "shippingservice",
]


@dataclass
class Payload:
    case_id: str
    text: str
    parts: dict[str, int]  # characters per section, for cost accounting


def _metric_window(case_dir: Path, inject_time: int) -> pd.DataFrame:
    frame = pd.read_csv(case_dir / "simple_metrics.csv")
    window = frame[
        (frame["time"] >= inject_time - METRIC_WINDOW_SECONDS)
        & (frame["time"] <= inject_time + METRIC_WINDOW_SECONDS)
    ].copy()
    values = window.drop(columns=["time"])
    # A column that never moves inside the window carries no information about
    # what changed, and 32 of 72 are constant here.
    values = values.loc[:, values.nunique() > 1]
    values.insert(0, "t", (window["time"] - inject_time).astype(int).values)
    return values


def _metrics_block(case_dir: Path, inject_time: int) -> str:
    """Metric window as CSV, raw values, time relative to injection."""
    return _metric_window(case_dir, inject_time).round(2).to_csv(index=False)


def _metrics_block_relative(case_dir: Path, inject_time: int) -> str:
    """Each metric as percent change against its own pre-fault median.

    Condition E showed the model ranking by absolute magnitude: currencyservice
    holds the highest absolute CPU in 39 of 39 sampled windows and was named
    first in 214 of 270 runs, while the injected service -- whose own CPU rose
    by over 1000% from a small base -- was passed over. Restating every series
    against its own baseline removes the scale difference that drove that,
    without telling the model which service to pick.

    This is a real transformation of the evidence, not just a formatting change:
    computing "how far is this series from its own normal" is the same step
    NSigma and BARO perform internally. Condition E' therefore measures whether
    the model can reason once that step is done for it, which is a different
    question from condition E, and the two must be reported as such.
    """
    values = _metric_window(case_dir, inject_time)
    before = values[values["t"] < 0]

    out = pd.DataFrame({"t": values["t"].values})
    for column in values.columns:
        if column == "t":
            continue
        baseline = before[column].median()
        if pd.notna(baseline) and abs(baseline) > 1e-9:
            pct = (values[column] - baseline) / abs(baseline) * 100
            # A NaN sample (a gap in the series) or an infinite ratio cannot be
            # cast to int; both are dropped to 0 rather than crashing the case,
            # which is what an absent reading means here -- no observed change.
            out[column] = (
                pct.replace([float("inf"), float("-inf")], pd.NA)
                .fillna(0)
                .round(0)
                .astype(int)
                .values
            )
        else:
            # A series whose baseline is zero has no meaningful percentage; the
            # raw value is kept so a fault starting from zero is still visible.
            out[column] = values[column].fillna(0).round(2).values
    return out.to_csv(index=False)


def _series_block(case_dir: Path, inject_time: int, filename: str) -> str:
    frame = pd.read_csv(case_dir / filename)
    if "time" in frame.columns:
        frame = frame[
            (frame["time"] >= inject_time - SERIES_WINDOW_SECONDS)
            & (frame["time"] <= inject_time + SERIES_WINDOW_SECONDS)
        ]
    return frame.round(2).to_csv(index=False)


def _log_block(case_dir: Path, inject_time: int) -> str:
    """Per-service log volume before and after injection.

    The raw logs in this window run to ~14,500 lines (~250 K tokens) and are
    entirely info/debug -- no error or warning lines at all -- so pasting them
    would spend the whole budget on routine chatter. The volume shift is the
    part that discriminates: on checkoutservice_cpu/1 the injected service tops
    this table at +27%.
    """
    logs = pd.read_csv(case_dir / "logs.csv", usecols=LOG_COLUMNS)
    seconds = logs["timestamp"] // 1_000_000_000
    window = logs[
        (seconds >= inject_time - LOG_WINDOW_SECONDS)
        & (seconds <= inject_time + LOG_WINDOW_SECONDS)
    ].assign(ts=seconds)

    before = window[window["ts"] < inject_time].groupby("container_name").size()
    after = window[window["ts"] >= inject_time].groupby("container_name").size()
    table = pd.DataFrame({"before": before, "after": after}).fillna(0).astype(int)
    table["pct_change"] = (
        (table["after"] - table["before"]) / table["before"].replace(0, 1) * 100
    ).round(0).astype(int)
    return table.to_csv()


def build(case_dir: str | Path, relative: bool = False) -> Payload:
    """Assemble the prompt payload for one case.

    `relative=False` (condition E and F) presents raw metric values;
    `relative=True` (condition E') presents each metric as percent change from
    its own pre-fault median. Everything else is identical, so the two
    conditions differ only in how the metrics are expressed.

    The returned text never names the case, the fault type, or the injected
    service: the directory name would give the answer away outright.
    """
    case_dir = Path(case_dir)
    inject_time = int((case_dir / "inject_time.txt").read_text().strip())

    if relative:
        metrics_header = (
            "RESOURCE AND LATENCY METRICS AS PERCENT CHANGE FROM EACH SERIES' "
            "OWN PRE-FAULT BASELINE (t = seconds relative to the fault; t<0 is "
            "before it, t>=0 after; 1-second samples; each value is the percent "
            "difference from that column's median over t<0, so +100 means the "
            "series doubled; columns constant across the window are omitted)"
        )
        metrics_body = _metrics_block_relative(case_dir, inject_time)
    else:
        metrics_header = (
            "RESOURCE AND LATENCY METRICS "
            "(t = seconds relative to the fault; t<0 is before it, "
            "t>=0 after; 1-second samples; columns constant across the whole "
            "window are omitted)"
        )
        metrics_body = _metrics_block(case_dir, inject_time)

    sections: list[tuple[str, str]] = [
        (metrics_header, metrics_body),
        (
            "LOG VOLUME PER SERVICE "
            f"(+/-{LOG_WINDOW_SECONDS}s around the fault)",
            _log_block(case_dir, inject_time),
        ),
    ]
    for filename in SERIES_FILES:
        label = {
            "logts.csv": "LOG EVENT COUNTS PER SERVICE (time series)",
            "tracets_err.csv": "TRACE ERROR RATE PER OPERATION (time series)",
            "tracets_lat.csv": "TRACE LATENCY PER OPERATION (time series)",
        }[filename]
        sections.append(
            (f"{label} (+/-{SERIES_WINDOW_SECONDS}s, absolute epoch seconds)",
             _series_block(case_dir, inject_time, filename))
        )

    parts = {name.split(" (")[0]: len(body) for name, body in sections}
    text = "\n\n".join(f"## {name}\n```csv\n{body.strip()}\n```"
                       for name, body in sections)
    return Payload(case_id=f"{case_dir.parent.name}/{case_dir.name}",
                   text=text, parts=parts)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else (
        "data/RCAEval-OB/RE2-OB/checkoutservice_cpu/1"
    )
    payload = build(target)
    print(f"case  : {payload.case_id}")
    print(f"chars : {len(payload.text):,}")
    try:
        import tiktoken

        tokens = len(tiktoken.get_encoding("o200k_base").encode(payload.text))
        print(f"tokens: {tokens:,}  (o200k_base / gpt-4o-mini)")
    except ImportError:
        print("tokens: install tiktoken to measure")
    for name, size in payload.parts.items():
        print(f"  {name[:48]:<50}{size:>8,} chars")
