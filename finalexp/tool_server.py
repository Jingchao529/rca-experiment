"""Serve the three RCA baselines' rankings to the agent in condition F.

Results are read from the frozen snapshot rather than recomputed. That is a
deliberate choice, not a shortcut:

  * The three methods need three mutually incompatible Python environments
    (TORAI: py38 with a patched causal-learn; CIRCA: causal-learn 0.1.4.8;
    BARO: either). One agent process cannot import all three.
  * CIRCA costs ~42 s per call. An agent free to call tools repeatedly would
    spend hours per case re-deriving a deterministic result.
  * The frozen rows are hashed, so condition F consumes exactly the numbers
    reported in the baseline table -- the comparison is against the same
    evidence a reader sees, not a re-run that might drift.

Every lookup is recorded with the source file and the row's own content hash, so
a tool answer can be traced back to the run that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN = REPO_ROOT / "finalexp" / "frozen" / "2026-09-06"
RESULTS = REPO_ROOT / "finalexp" / "results"

# Which system the tools answer for. Online Boutique reads the frozen snapshot
# the recorded results depend on; Train Ticket reads its own baseline files,
# which postdate every snapshot. `use_system` switches this, and nothing merges
# the two: a case id must be answered out of its own system's results.
_SYSTEM = "OB"


def use_system(name: str) -> None:
    """Point the tools at "OB" or "TT". Clears the cached indexes."""
    global _SYSTEM
    if name not in ("OB", "TT"):
        raise ValueError(f"unknown system {name!r}")
    _SYSTEM = name
    _index.cache_clear()


def _source(filename: str) -> Path:
    """Where this tool's results live for the selected system."""
    if _SYSTEM == "OB":
        return FROZEN / filename
    return RESULTS / filename.replace(".jsonl", "__TT.jsonl")

# Tool name -> (frozen file, human-facing description of the mechanism).
# The descriptions state what each method does and say nothing about how
# accurate it is: telling the model that TORAI scores best would turn the
# condition into a test of instruction-following rather than of reasoning.
TOOLS = {
    "run_torai": (
        "experiment_d.jsonl",
        "Multi-source analysis. Scores anomaly severity across metrics, logs "
        "and traces, clusters services by symptom, then ranks within clusters "
        "by causal analysis. Does not require a service call graph.",
    ),
    "run_circa": (
        "experiment_c.jsonl",
        "Causal-graph analysis. Builds a causal graph over the metrics with the "
        "PC algorithm, then ranks candidates by regression-based hypothesis "
        "testing over that graph.",
    ),
    "run_baro": (
        "experiment_b.jsonl",
        "Statistical change detection. Compares each metric's distribution "
        "before and after the fault using median and interquartile range, and "
        "ranks by the size of the shift. Builds no graph.",
    ),
}


@dataclass
class ToolAnswer:
    tool: str
    case_id: str
    ranking: list[str]
    detail: dict
    source_file: str
    source_hash: str


@lru_cache(maxsize=None)
def _index(filename: str) -> dict[str, dict]:
    path = _source(filename)
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows[row["case_id"]] = row
    return rows


def call(tool: str, case_id: str) -> ToolAnswer:
    if tool not in TOOLS:
        raise KeyError(f"unknown tool {tool!r}")
    filename, _ = TOOLS[tool]
    row = _index(filename).get(case_id)
    if row is None:
        raise KeyError(f"{tool} has no result for {case_id}")

    # Method-specific supporting evidence. Each tool exposes what it actually
    # produced, so the model can weigh a ranking against how it was reached
    # rather than treating three rankings as interchangeable votes.
    detail: dict = {"seconds": round(row.get("seconds", 0.0), 2)}
    diagnostics = row.get("diagnostics") or {}
    if tool == "run_circa":
        detail["causal_graph_edges"] = diagnostics.get("graph_edges")
        detail["top_metrics"] = row.get("top_metrics", [])[:5]
    elif tool == "run_baro":
        detail["top_metrics"] = row.get("top_metrics", [])[:5]
    elif tool == "run_torai":
        detail["top_indicators"] = row.get("top_indicators", [])[:5]

    return ToolAnswer(
        tool=tool,
        case_id=case_id,
        ranking=list(row.get("ranking", [])),
        detail=detail,
        source_file=filename,
        source_hash=row.get("content_sha256", ""),
    )


def openai_schema() -> list[dict]:
    """Tool definitions in the OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for name, (_, description) in TOOLS.items()
    ]


def available_cases() -> set[str]:
    """Cases every tool can answer for."""
    sets = [set(_index(filename)) for filename, _ in TOOLS.values()]
    return set.intersection(*sets)


if __name__ == "__main__":
    import sys

    case = sys.argv[1] if len(sys.argv) > 1 else "checkoutservice_cpu/1"
    print(f"cases servable by all tools: {len(available_cases())}\n")
    for name in TOOLS:
        answer = call(name, case)
        print(f"{name}  ({answer.source_file})")
        print(f"  ranking: {answer.ranking[:5]}")
        print(f"  detail : {answer.detail}")
        print(f"  hash   : {answer.source_hash[:16]}...")
