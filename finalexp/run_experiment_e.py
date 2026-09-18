"""Experiment E -- raw_direct: one LLM call over telemetry, no tools.

The model receives the telemetry payload built by `finalexp/payload.py` and must
rank the candidate services itself. Condition F will receive the same payload
plus the three RCA tools, so any difference between them is attributable to the
tool output alone.

Scored with the same AC@1 / AC@3 / Avg@5 harness as the four RCA baselines.

Usage
-----
    python finalexp/run_experiment_e.py --limit 1 --repeats 1   # smoke test
    python finalexp/run_experiment_e.py                         # full sweep
    python finalexp/run_experiment_e.py --score-only
    python finalexp/run_experiment_e.py --dry-run --limit 3     # cost estimate

Reads the model configuration from src/platform/agent_config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

from payload import CANDIDATE_SERVICES, build  # noqa: E402
from scoring import score_table  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-OB"
RESULTS_FILE = REPO_ROOT / "finalexp" / "results" / "experiment_e.jsonl"
RESULTS_FILE_RELATIVE = REPO_ROOT / "finalexp" / "results" / "experiment_e_prime.jsonl"
AGENT_CONFIG = REPO_ROOT / "src" / "platform" / "agent_config.json"

METHOD = "LLM raw_direct (telemetry only, no tools)"
METHOD_RELATIVE = "LLM raw_direct, baseline-relative metrics (no tools)"
TEMPERATURE = 0.0  # fixed and logged; the sweep still repeats to catch drift

SYSTEM_PROMPT = """You are diagnosing a fault in a microservice system.

A fault was injected into exactly one service. You are given telemetry from a
window around the injection: resource and latency metrics at 1-second
resolution, per-service log volume, and log/trace time series.

Rank the candidate services by how likely each is the ROOT CAUSE -- the service
the fault was injected into, not merely a service showing symptoms. Downstream
and caller services often look worse than the faulty one because they inherit
its latency; rank by which service's own behaviour changed first and most
sharply relative to its own baseline before the fault.

Reply with JSON only, no prose outside it:

{"ranking": ["<service>", ...], "confidence": "high|medium|low",
 "reasoning": "<= 150 words citing specific metrics"}

`ranking` must list at least 5 services, most likely first, drawn only from the
candidate list you are given."""


# Providers the LLM conditions can run against. Gemini is reached through its
# OpenAI-compatible endpoint, so the same client code and the same tool-calling
# schema serve both -- swapping models changes configuration, not experiment
# logic, which is what makes the comparison a controlled one.
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": ("OPENAI_API_KEY",),
        "key_fields": ("api_key", "openai_api_key"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # Flash-Lite, not Flash: at $0.25/$1.50 per million tokens it sits in
        # the same tier as gpt-4o-mini ($0.15/$0.60), where gemini-3.5-flash
        # ($1.50/$9.00) is ten times dearer and a materially stronger model.
        # Comparing against Flash would confound "different vendor" with
        # "bigger model", which is not the question being asked.
        "default_model": "gemini-3.1-flash-lite",
        "key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "key_fields": ("gemini_api_key", "google_api_key"),
        # Thinking is off: it costs latency and tokens without being part of
        # what is compared here, and gpt-4o-mini does not think either. Measured
        # on a trivial prompt, thinking on cost 150 total tokens against 19 off.
        "request_extra": {"reasoning_effort": "none"},
    },
    "qwen": {
        # Self-hosted vLLM, so there is no public endpoint to point at: set
        # QWEN_BASE_URL to your own server. No key is needed, but the OpenAI
        # client requires a non-empty string, so a placeholder stands in.
        "base_url": os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1"),
        "default_model": "Qwen/Qwen3.5-9B",
        "key_env": ("QWEN_API_KEY",),
        "key_fields": ("qwen_api_key",),
        "default_key": "EMPTY",
        # Qwen3.5 emits a long "Thinking Process:" preamble by default. On a
        # 17 K-token metric table that degenerates: one measured call spent
        # 6,000 tokens and 211 s repeating "2.29 -> 2.29 -> ..." and never
        # produced JSON. Disabled, the same case answers correctly in 9 s and
        # 189 tokens. `/no_think` and `reasoning_effort` are both ignored by
        # this server; only the chat-template flag takes effect.
        "request_extra": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
    },
    "deepseek": {
        # DeepSeek serves an OpenAI-compatible API, so the same client and the
        # same tool schema work unchanged.
        "base_url": "https://api.deepseek.com",
        # v4-pro at $0.435/$0.87 sits between the small models tested here and
        # gpt-5.6-terra at $2.00/$12.00 -- closer to the small tier than the
        # large one, so it reads as a third cheap-tier data point rather than a
        # second large-model reference.
        "default_model": "deepseek-v4-pro",
        "key_env": ("DEEPSEEK_API_KEY",),
        "key_fields": ("deepseek_api_key",),
        # v4-pro thinks by default and does so at length: on this payload it
        # emitted 22,327 completion tokens and took 291 s to produce a
        # five-element JSON ranking, which is 7.3 hours for a 90-case sweep.
        # Disabled it emits 216 tokens in 6 s. The other models here do not
        # think either (gpt-4o-mini 145 tokens, flash-lite 178), so turning it
        # off is what keeps the comparison like-for-like rather than a
        # concession to cost.
        "request_extra": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
}


def load_model_config(provider: str | None = None, model: str | None = None) -> dict:
    """Resolve which model to call.

    Without --provider the behaviour is unchanged: read agent_config.json, which
    is what produced the recorded gpt-4o-mini results. Naming a provider
    overrides that, so a second model can be run without editing the config the
    first run depended on.
    """
    config = json.loads(AGENT_CONFIG.read_text(encoding="utf-8"))

    if provider is None:
        api_key = (config.get("api_key") or config.get("openai_api_key")
                   or os.environ.get("OPENAI_API_KEY", ""))
        return {
            "provider": config.get("model_provider", "openai"),
            "model": model or config.get("model") or config.get("model_name", "gpt-4o-mini"),
            "base_url": config.get("base_url", "https://api.openai.com/v1"),
            "api_key": api_key,
            "request_extra": {},
        }

    if provider not in PROVIDERS:
        raise SystemExit(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
    spec = PROVIDERS[provider]

    api_key = ""
    for field in spec["key_fields"]:
        if config.get(field):
            api_key = config[field]
            break
    if not api_key:
        for name in spec["key_env"]:
            if os.environ.get(name):
                api_key = os.environ[name]
                break
    if not api_key:
        api_key = spec.get("default_key", "")

    chosen = model or spec["default_model"]
    request_extra = dict(spec.get("request_extra", {}))

    # `reasoning_effort` is rejected outright by the 3.5 Lite models
    # ("Request contains an invalid argument"). They do not appear to think by
    # default -- a one-sentence prompt costs 36 completion tokens against 3.1
    # Lite's 31 -- so dropping the parameter is enough; nothing needs disabling.
    if "flash-lite" in chosen and chosen.startswith("gemini-3.5"):
        request_extra.pop("reasoning_effort", None)

    return {
        "provider": provider,
        "model": chosen,
        "base_url": spec["base_url"],
        "api_key": api_key,
        "request_extra": request_extra,
    }


def sampling_args(model: str) -> dict:
    """Temperature if the model accepts one.

    The GPT-5.x reasoning models reject any explicit temperature ("Only the
    default (1) value is supported"). They are sampled at their own default
    instead; the repeats in each condition already cover the resulting variance,
    and every run records the model and this setting.
    """
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    return {"temperature": TEMPERATURE}


def results_path(base: Path, provider: str, model: str) -> Path:
    """Per-model results file, so one model's run cannot overwrite another's."""
    if model.startswith("gpt-4o-mini"):
        return base  # the recorded baseline keeps its original filename
    slug = model.replace("/", "-").replace(".", "-")
    return base.with_name(f"{base.stem}__{slug}{base.suffix}")


def build_prompt(case_dir: Path, relative: bool = False) -> tuple[str, dict]:
    payload = build(case_dir, relative=relative)
    user = (
        f"Candidate services: {', '.join(CANDIDATE_SERVICES)}\n\n"
        f"{payload.text}\n\n"
        "Which service is the root cause? Reply with the JSON object only."
    )
    return user, payload.parts


def parse_reply(text: str, valid: list[str] | None = None
                ) -> tuple[list[str], str, str, str | None]:
    """Extract the ranking. Returns (ranking, confidence, reasoning, error).

    `valid` is the candidate set a name must belong to; it defaults to Online
    Boutique's services. Train Ticket names its services differently, so a
    caller scoring that system must pass its own list -- otherwise every name is
    rejected and a correct answer is recorded as an empty ranking.
    """
    allowed = CANDIDATE_SERVICES if valid is None else valid
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
    start = body.find("{")
    if start < 0:
        return [], "", "", "no JSON object in reply"

    # `raw_decode` stops at the end of the first complete object, so trailing
    # commentary after the JSON does not invalidate the answer. Spanning from
    # the first "{" to the last "}" instead would swallow that commentary and
    # fail to parse -- which is how a well-formed ranking from deepseek-v4-flash
    # was first recorded as an empty result.
    try:
        parsed, _ = json.JSONDecoder().raw_decode(body[start:])
    except json.JSONDecodeError as exc:
        return [], "", "", f"malformed JSON: {exc}"
    if not isinstance(parsed, dict):
        return [], "", "", "reply JSON is not an object"

    raw = parsed.get("ranking") or []
    if not isinstance(raw, list):
        return [], "", "", "ranking is not a list"

    # Names outside the candidate set are dropped rather than silently accepted;
    # a hallucinated service must not be able to occupy a scoring slot.
    ranking, invalid = [], []
    for item in raw:
        name = str(item).strip()
        if name in allowed:
            if name not in ranking:
                ranking.append(name)
        else:
            invalid.append(name)

    error = f"invalid service names: {invalid}" if invalid else None
    return ranking, str(parsed.get("confidence", "")), str(parsed.get("reasoning", "")), error


def call_model(client, model: str, user: str, max_attempts: int = 6,
               request_extra: dict | None = None) -> tuple[str, dict]:
    """One completion, retrying on rate limits.

    At ~17.5 K tokens per call the sweep runs into the account's tokens-per-minute
    ceiling well before its request ceiling, so 429s are expected rather than
    exceptional. Without backoff they land as recorded failures and are scored as
    misses, which silently depresses the result -- an infrastructure limit
    masquerading as a diagnostic one.
    """
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    delay = 20.0
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                **sampling_args(model),
                **(request_extra or {}),
            )
            usage = response.usage
            return response.choices[0].message.content or "", {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "attempts": attempt + 1,
            }
        except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
            last = exc
            if attempt == max_attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise last  # type: ignore[misc]


def discover_cases(dataset_dir: Path) -> list[Path]:
    return sorted(
        rep
        for group in sorted(dataset_dir.iterdir())
        if group.is_dir()
        for rep in sorted(group.iterdir())
        if rep.is_dir() and (rep / "simple_metrics.csv").exists()
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
    parser.add_argument("--repeats", type=int, default=3,
                        help="LLM output is stochastic; TORAI's protocol repeats "
                             "and averages. Default 3.")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and count tokens without calling the API.")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        help="Override the provider in agent_config.json.")
    parser.add_argument("--model", help="Override the model id.")
    parser.add_argument("--relative", action="store_true",
                        help="Condition E': express each metric as percent change "
                             "from its own pre-fault baseline. Separate results file.")
    args = parser.parse_args()

    config = load_model_config(args.provider, args.model)
    base_results = RESULTS_FILE_RELATIVE if args.relative else RESULTS_FILE
    results_file = results_path(base_results, config["provider"], config["model"])
    method = (METHOD_RELATIVE if args.relative else METHOD) + f" [{config['model']}]"

    if args.score_only:
        rows = load_existing(results_file)
        if not rows:
            print(f"No results at {results_file}")
            return 1
        score_table(rows, method)
        _report_cost(rows)
        return 0

    cases = discover_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    if args.dry_run:
        import tiktoken

        encoder = tiktoken.get_encoding("o200k_base")
        counts = []
        for case_dir in cases:
            user, _ = build_prompt(case_dir, relative=args.relative)
            total = len(encoder.encode(SYSTEM_PROMPT)) + len(encoder.encode(user))
            counts.append(total)
            print(f"{case_dir.parent.name}/{case_dir.name}: {total:,} tokens")
        planned = len(discover_cases(args.dataset)) * args.repeats
        mean = sum(counts) / len(counts)
        print(f"\nmean {mean:,.0f} tokens/call")
        print(f"full sweep: {planned} calls ~= {mean * planned / 1e6:.1f}M input tokens")
        return 0

    if not config["api_key"]:
        print(f"No API key for provider {config['provider']!r}. Add it to "
              f"{AGENT_CONFIG.name} or set the provider's environment variable.")
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    done = {(r["case_id"], r["repeat"]) for r in load_existing(results_file)}
    results_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"model   : {config['model']}  (temperature {TEMPERATURE})")
    print(f"cond    : {method}")
    print(f"dataset : {args.dataset}")
    print(f"cases   : {len(cases)} x {args.repeats} repeats\n")

    index = 0
    total_runs = len(cases) * args.repeats
    for case_dir in cases:
        case_id = f"{case_dir.parent.name}/{case_dir.name}"
        service, fault = case_dir.parent.name.rsplit("_", 1)
        user = None
        for repeat in range(1, args.repeats + 1):
            index += 1
            if (case_id, repeat) in done:
                continue
            if user is None:
                user, parts = build_prompt(case_dir, relative=args.relative)

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
                "relative_metrics": args.relative,
                "model": config["model"],
                "temperature": sampling_args(config["model"]).get("temperature"),
            }
            try:
                reply, usage = call_model(client, config["model"], user,
                                          request_extra=config.get("request_extra"))
                ranking, confidence, reasoning, parse_error = parse_reply(reply)
                record.update(
                    status="ok" if ranking else "empty_ranking",
                    ranking=ranking,
                    confidence=confidence,
                    reasoning=reasoning[:1000],
                    seconds=time.time() - started,
                    usage=usage,
                )
                if parse_error:
                    record["parse_warning"] = parse_error
                if not ranking:
                    record["error"] = parse_error or "no valid services in reply"
                mark = "OK " if service in ranking[:3] else "-- "
                print(f"[{index}/{total_runs}] {mark}{case_id} r{repeat} "
                      f"{record['seconds']:.1f}s {usage['total_tokens']:,}tok "
                      f"top3={ranking[:3]}")
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
    _report_cost(rows)
    print(f"\nresults: {results_file}")
    print(f"platform: {platform.platform()} | python {platform.python_version()}")
    return 0


def _report_cost(rows: list[dict]) -> None:
    used = [r for r in rows if r.get("usage")]
    if not used:
        return
    prompt = sum(r["usage"]["prompt_tokens"] for r in used)
    completion = sum(r["usage"]["completion_tokens"] for r in used)
    print(f"\ntokens: {prompt:,} prompt + {completion:,} completion "
          f"over {len(used)} calls "
          f"({prompt / len(used):,.0f} + {completion / len(used):,.0f} per call)")


if __name__ == "__main__":
    raise SystemExit(main())
