"""Experiment F -- tool_agent: the same telemetry as condition E, plus RCA tools.

The model receives the identical payload condition E receives and may in
addition call TORAI, CIRCA and BARO. Because F's input is a strict superset of
E's, any difference between the two is attributable to the tool output alone.

Scored with the same AC@1 / AC@3 / Avg@5 harness as every other experiment.

Usage
-----
    python finalexp/run_experiment_f.py --limit 1 --repeats 1   # smoke test
    python finalexp/run_experiment_f.py                         # full sweep
    python finalexp/run_experiment_f.py --score-only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

import tool_server  # noqa: E402
from payload import CANDIDATE_SERVICES, build  # noqa: E402
from run_ablation_input import build_arm, services_in  # noqa: E402
from run_experiment_e import (  # noqa: E402
    AGENT_CONFIG,
    DEFAULT_DATASET,
    PROVIDERS,
    TEMPERATURE,
    load_model_config,
    parse_reply,
    results_path,
    sampling_args,
)
from scoring import score_table  # noqa: E402

RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_f.jsonl"
METHOD = "LLM tool_agent (telemetry + TORAI/CIRCA/BARO tools)"

# Enough rounds to call all three tools and still answer. The cap exists so a
# non-terminating loop cannot burn the budget; hitting it is recorded.
MAX_ROUNDS = 6

SYSTEM_PROMPT = """You are diagnosing a fault in a microservice system.

A fault was injected into exactly one service. You are given telemetry from a
window around the injection, and three root-cause analysis tools you may call.
The tools use different mechanisms and do not always agree; when they disagree,
weigh their rankings against the telemetry rather than simply counting votes.

Rank the candidate services by how likely each is the ROOT CAUSE -- the service
the fault was injected into, not merely a service showing symptoms. Downstream
and caller services often look worse than the faulty one because they inherit
its latency.

Call whichever tools you find useful, then reply with JSON only, no prose
outside it:

{"ranking": ["<service>", ...], "confidence": "high|medium|low",
 "reasoning": "<= 150 words citing the evidence you used"}

`ranking` must list at least 5 services, most likely first, drawn only from the
candidate list you are given."""


def candidates_for(case_dir: Path) -> list[str]:
    """The candidate list for this case's system.

    Online Boutique's eleven services are hardcoded, but Train Ticket has
    sixty-four under different names; reading them from the case's own metric
    columns keeps one runner correct on both. Anything outside this list is
    rejected downstream, so getting it wrong silently zeroes a whole run.
    """
    services = services_in(case_dir)
    return services if len(services) > len(CANDIDATE_SERVICES) else CANDIDATE_SERVICES


def build_user_prompt(case_dir: Path, arm: str | None = None) -> str:
    """The telemetry block, optionally compressed by one of the ablation arms.

    Without an arm this is the full payload, which is what the Online Boutique
    results used. On Train Ticket that runs to ~103K tokens; passing the arm
    condition E used keeps the two conditions comparable.
    """
    telemetry = (build_arm(case_dir, arm) if arm
                 else build(case_dir, relative=False).text)
    return (
        f"Candidate services: {', '.join(candidates_for(case_dir))}\n\n"
        f"{telemetry}\n\n"
        "Call the tools you need, then give the JSON object."
    )


def run_agent(client, model: str, case_dir: Path, case_id: str,
              request_extra: dict | None = None,
              arm: str | None = None) -> tuple[str, dict]:
    """ReAct loop. Returns (final reply text, transcript info)."""
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case_dir, arm)},
    ]
    schema = tool_server.openai_schema()

    # The telemetry block is ~17 K tokens (~24 K as Gemini counts it) and a
    # chat completion is stateless: every round resends the whole history, so
    # an unmodified transcript pays for that block once per round. A model that
    # calls its tools one at a time then pays it four times over.
    #
    # After the first round the model has seen the telemetry and has tool
    # results in hand, so the block is replaced with a one-line note. The
    # candidate list and the task stay, and nothing the model produced is
    # dropped -- only the verbatim CSV it has already read.
    telemetry_placeholder = (
        f"Candidate services: {', '.join(candidates_for(case_dir))}\n\n"
        "(The telemetry for this case was provided above and is omitted here to "
        "save context. Use the tool results together with what you observed in "
        "it.)\n\nCall the tools you need, then give the JSON object."
    )

    calls: list[dict] = []
    usage = Counter()
    rounds = 0
    hit_cap = False

    while rounds < MAX_ROUNDS:
        rounds += 1

        delay = 20.0
        response = None
        for attempt in range(6):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=schema,
                    **sampling_args(model),
                    **(request_extra or {}),
                )
                break
            except (RateLimitError, APIConnectionError, APITimeoutError):
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        assert response is not None

        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens
        usage["total_tokens"] += response.usage.total_tokens

        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "", {
                "rounds": rounds,
                "tool_calls": calls,
                "usage": dict(usage),
                "hit_round_cap": False,
                "telemetry_resent_each_round": False,
            }

        # Gemini attaches a `thought_signature` to each tool call and rejects the
        # next request with HTTP 400 unless it is echoed back verbatim: the
        # signature carries the model's own reasoning state across the turn. The
        # OpenAI SDK surfaces it as `extra_content`, which OpenAI itself does not
        # send, so copying it through when present keeps one code path working
        # for both providers.
        assistant_tool_calls = []
        for call in message.tool_calls:
            entry = {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name,
                             "arguments": call.function.arguments},
            }
            extra = call.model_dump().get("extra_content")
            if extra:
                entry["extra_content"] = extra
            assistant_tool_calls.append(entry)

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": assistant_tool_calls,
        })

        # Swap the telemetry out once, on the first round that produced tool
        # calls. Later rounds then carry the tool transcript but not the CSV.
        if messages[1]["content"] != telemetry_placeholder:
            messages[1] = {"role": "user", "content": telemetry_placeholder}

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                answer = tool_server.call(name, case_id)
                result = {"ranking": answer.ranking, **answer.detail}
                calls.append({
                    "tool": name,
                    "round": rounds,
                    "ranking": answer.ranking[:5],
                    "source_hash": answer.source_hash,
                })
            except KeyError as exc:
                result = {"error": str(exc)}
                calls.append({"tool": name, "round": rounds, "error": str(exc)})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        hit_cap = True

    # Round cap reached while still calling tools: ask once for the answer.
    messages.append({
        "role": "user",
        "content": "Stop calling tools. Give the JSON object now.",
    })
    response = client.chat.completions.create(
        model=model, messages=messages,
        **sampling_args(model),
        **(request_extra or {}),
    )
    usage["prompt_tokens"] += response.usage.prompt_tokens
    usage["completion_tokens"] += response.usage.completion_tokens
    usage["total_tokens"] += response.usage.total_tokens
    return response.choices[0].message.content or "", {
        "rounds": rounds,
        "tool_calls": calls,
        "usage": dict(usage),
        "hit_round_cap": hit_cap,
        "telemetry_resent_each_round": False,
    }


def discover_cases(dataset_dir: Path) -> list[Path]:
    # The tools answer from precomputed baseline results, which are per-system.
    # Selecting by the dataset directory keeps a Train Ticket run from silently
    # finding zero servable cases and reporting the previous system's numbers.
    tool_server.use_system("TT" if "TT" in dataset_dir.name else "OB")
    servable = tool_server.available_cases()
    return sorted(
        rep
        for group in sorted(dataset_dir.iterdir())
        if group.is_dir()
        for rep in sorted(group.iterdir())
        if rep.is_dir() and f"{group.name}/{rep.name}" in servable
    )


def load_existing(results_file: Path = RESULTS_FILE) -> list[dict]:
    if not results_file.exists():
        return []
    with results_file.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        help="Override the provider in agent_config.json.")
    parser.add_argument("--model", help="Override the model id.")
    parser.add_argument("--arm", default=None,
                        help="compression arm for the telemetry block, e.g. "
                             "R_1col_8s; omit for the full payload")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.provider, args.model)
    # One file per (dataset, arm, model): a Train Ticket run on a compressed
    # arm must not append to the Online Boutique results for the same model,
    # or --score-only averages two systems and two input sizes together.
    base = RESULTS_FILE
    tag = ("__TT" if "TT" in args.dataset.name else "") + (f"__{args.arm}" if args.arm else "")
    if tag:
        base = base.with_name(f"{base.stem}{tag}{base.suffix}")
    results_file = results_path(base, config["provider"], config["model"])
    method = METHOD + f" [{config['model']}]"

    if args.score_only:
        rows = load_existing(results_file)
        if not rows:
            print(f"No results at {results_file}")
            return 1
        score_table(rows, method)
        _report_agent_behaviour(rows)
        return 0

    if not config["api_key"]:
        print(f"No API key in {AGENT_CONFIG}")
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    cases = discover_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    done = {(r["case_id"], r["repeat"]) for r in load_existing(results_file)}
    results_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"model   : {config['model']}  (temperature {TEMPERATURE})")
    print(f"cases   : {len(cases)} x {args.repeats} repeats, max {MAX_ROUNDS} rounds\n")

    index = 0
    total_runs = len(cases) * args.repeats
    for case_dir in cases:
        case_id = f"{case_dir.parent.name}/{case_dir.name}"
        service, fault = case_dir.parent.name.rsplit("_", 1)
        for repeat in range(1, args.repeats + 1):
            index += 1
            if (case_id, repeat) in done:
                continue

            started = time.time()
            record = {
                "case_id": case_id,
                "dataset": args.dataset.name,
                "ground_truth": service,
                "fault": fault,
                "repetition": case_dir.name,
                "repeat": repeat,
                "seed": None,
                "method": method,
                "model": config["model"],
                "temperature": sampling_args(config["model"]).get("temperature"),
            }
            try:
                reply, info = run_agent(client, config["model"], case_dir, case_id,
                                        request_extra=config.get("request_extra"), arm=args.arm)
                ranking, confidence, reasoning, parse_error = parse_reply(
                    reply, valid=candidates_for(case_dir))
                record.update(
                    status="ok" if ranking else "empty_ranking",
                    ranking=ranking,
                    confidence=confidence,
                    reasoning=reasoning[:1000],
                    seconds=time.time() - started,
                    usage=info["usage"],
                    rounds=info["rounds"],
                    tool_calls=info["tool_calls"],
                    hit_round_cap=info["hit_round_cap"],
                    prompt_variant="telemetry-once",
                )
                if parse_error:
                    record["parse_warning"] = parse_error
                if not ranking:
                    record["error"] = parse_error or "no valid services in reply"
                used = [c["tool"] for c in info["tool_calls"]]
                mark = "OK " if service in ranking[:3] else "-- "
                print(f"[{index}/{total_runs}] {mark}{case_id} r{repeat} "
                      f"{record['seconds']:.1f}s {info['usage']['total_tokens']:,}tok "
                      f"tools={len(used)} top3={ranking[:3]}")
            except Exception as exc:  # noqa: BLE001 -- recorded, never dropped
                record.update(
                    status="failed",
                    ranking=[],
                    seconds=time.time() - started,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                print(f"[{index}/{total_runs}] ERR {case_id} r{repeat}: {exc}")

            with results_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows = load_existing(results_file)
    score_table(rows, method)
    _report_agent_behaviour(rows)
    print(f"\nresults: {results_file}")
    print(f"platform: {platform.platform()} | python {platform.python_version()}")
    return 0


def _report_agent_behaviour(rows: list[dict]) -> None:
    """Which tools the agent used, and whether it followed them."""
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return

    tool_use = Counter()
    for row in ok:
        for call in row.get("tool_calls", []):
            tool_use[call["tool"]] += 1
    print("\ntool calls: " + ", ".join(f"{k} x{v}" for k, v in tool_use.most_common())
          or "\ntool calls: none")

    no_tools = sum(1 for r in ok if not r.get("tool_calls"))
    capped = sum(1 for r in ok if r.get("hit_round_cap"))
    print(f"runs using no tools: {no_tools}/{len(ok)}   hit round cap: {capped}")

    used = [r["usage"]["total_tokens"] for r in ok if r.get("usage")]
    if used:
        print(f"tokens: {sum(used):,} total ({sum(used) / len(used):,.0f} per run)")

    # Did the agent echo a tool, or depart from all of them? Answering this is
    # the point of the condition: an agent that always repeats one tool adds
    # orchestration, not diagnosis.
    echoed = Counter()
    for row in ok:
        top = row["ranking"][0] if row["ranking"] else None
        matched = [c["tool"] for c in row.get("tool_calls", [])
                   if c.get("ranking") and c["ranking"][0] == top]
        echoed["+".join(sorted(set(matched))) or "own answer"] += 1
    print("agent's top-1 matched: "
          + ", ".join(f"{k} x{v}" for k, v in echoed.most_common(5)))


if __name__ == "__main__":
    raise SystemExit(main())
