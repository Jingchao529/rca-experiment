"""Input-reduction ablation for condition E on Train Ticket.

Condition E's standard payload runs to ~102,000 tokens on Train Ticket, against
~17,000 on Online Boutique. That is too large for a 32K-context model and leaves
little headroom for condition F's tool transcript. This experiment measures what
is actually lost by shrinking it.

Four inputs, all carrying RAW metric values -- no candidate shortlisting, no
percent-change transform, nothing that pre-solves part of the localisation:

  A  full            the condition E payload, unchanged
  B  drop flat cols  columns whose coefficient of variation across the window
                     is under 1% are removed; they are near-straight lines
  C  B + 2s          B, sampled every 2 seconds instead of every 1
  D  CV>=5% + 2s     a harder variance cut, same 2-second sampling
  E  CV>=5%,  3s     D pushed further, and the time series stripped of the
  F  CV>=10%, 3s     columns that are zero throughout the window
  G  CV>=10%, 4s     G and H also halve the series window and its sampling
  H  CV>=20%, 4s

Every arm shows RAW values: the reductions choose which rows and columns to
print, never what a printed number says. A drop in accuracy is therefore
attributable to removed context rather than to altered numbers.

Arms A-D leave the time series untouched, isolating the metric block. Arms E-H
also drop series columns that hold zero for the entire window -- lossless, since
such a column states only that nothing happened. On Train Ticket that alone
removes ~11,700 tokens, because tracets_err has all 157 of its columns at zero.

E through H exist to reach a payload under 32,768 tokens, the context limit of
the self-hosted Qwen3.5-9B this study also evaluates. Arms A-D cannot run on
that model at all: the smallest of them is ~55,000 tokens.

Usage
-----
    python finalexp/run_ablation_input.py --limit 5
    python finalexp/run_ablation_input.py --limit 5 --dry-run
    python finalexp/run_ablation_input.py --score-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "finalexp"))

import payload as payload_mod  # noqa: E402
from payload import build  # noqa: E402
from run_experiment_e import (  # noqa: E402
    SYSTEM_PROMPT,
    load_model_config,
    parse_reply,
    sampling_args,
)

DATASET = REPO_ROOT / "data" / "RCAEval-OB" / "RE2-TT"
RESULTS = REPO_ROOT / "finalexp" / "results" / "ablation_input.jsonl"

# Train Ticket's window is widened from condition E's 30s because its faults
# take longer to show in the metrics; all four arms use the same value.
METRIC_WINDOW_SECONDS = 30

# "swin"/"sstep" narrow the time-series block; "drop_zero" removes columns that
# never leave zero. swin=None leaves the series exactly as arm A had them, which
# is what keeps A-D a clean test of the metric block alone.
ARMS = {
    "A_full":    {"cv": None, "step": 1, "swin": None, "sstep": 1, "drop_zero": False},
    "B_cv01":    {"cv": 0.01, "step": 1, "swin": None, "sstep": 1, "drop_zero": False},
    "C_cv01_2s": {"cv": 0.01, "step": 2, "swin": None, "sstep": 1, "drop_zero": False},
    "D_cv05_2s": {"cv": 0.05, "step": 2, "swin": None, "sstep": 1, "drop_zero": False},
    "E_cv05_3s": {"cv": 0.05, "step": 3, "swin": 120, "sstep": 1, "drop_zero": True},
    "F_cv10_3s": {"cv": 0.10, "step": 3, "swin": 60,  "sstep": 1, "drop_zero": True},
    "G_cv10_4s": {"cv": 0.10, "step": 4, "swin": 60,  "sstep": 2, "drop_zero": True},
    "H_cv20_4s": {"cv": 0.20, "step": 4, "swin": 60,  "sstep": 2, "drop_zero": True},
    # Denser cuts below H. I and J still keep the true cause in all 30 sampled
    # cases; K and L do not, and are included to bound what that costs.
    "I_cv30_4s": {"cv": 0.30, "step": 4, "swin": 60,  "sstep": 2, "drop_zero": True},
    "J_cv30_5s": {"cv": 0.30, "step": 5, "swin": 60,  "sstep": 2, "drop_zero": True},
    "K_cv50_5s": {"cv": 0.50, "step": 5, "swin": 60,  "sstep": 2, "drop_zero": True},
    "L_cv100_6s": {"cv": 1.00, "step": 6, "swin": 30, "sstep": 2, "drop_zero": True},
    # Compression that does NOT shortlist: both keep all 68 services present.
    # "per_service" picks each service's most-varying column instead of cutting
    # columns by a global threshold, so no service can drop out of the input.
    "N_all_15s": {"cv": None, "step": 15, "swin": 60, "sstep": 2,
                  "drop_zero": True, "per_service": False},
    "O_1col_4s": {"cv": None, "step": 4, "swin": 60, "sstep": 2,
                  "drop_zero": True, "per_service": True},
    # Q keeps a second column per service; R samples half as often as O. Both
    # keep all 68 services, so neither shortlists candidates.
    "Q_2col_4s": {"cv": None, "step": 4, "swin": 60, "sstep": 2,
                  "drop_zero": True, "per_service": 2},
    "R_1col_8s": {"cv": None, "step": 8, "swin": 60, "sstep": 2,
                  "drop_zero": True, "per_service": True},
    # "mwin" overrides the payload's symmetric +/-30s metric window with
    # (before, after) seconds. T and U widen the span to 120s and sample it
    # sparsely, which the proxy measurements favour over a dense short window.
    "T_w40_80_10s": {"cv": None, "step": 10, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    "U_w40_80_5s": {"cv": None, "step": 5, "swin": 60, "sstep": 2,
                    "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    # 1:5 counterparts, matched to T and U on sampling rate and token cost so
    # the only difference is where the window sits. V has 2 baseline rows and W
    # has 4, which is also what makes the pair a test of the baseline-count
    # explanation.
    "V_w20_100_10s": {"cv": None, "step": 10, "swin": 60, "sstep": 2,
                      "drop_zero": True, "per_service": True, "mwin": (20, 100)},
    "W_w20_100_5s": {"cv": None, "step": 5, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (20, 100)},
    # The sampling-rate axis for both 120s windows, so each family has a curve
    # rather than two points.
    "Ta_w40_80_3s": {"cv": None, "step": 3, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    "Tb_w40_80_4s": {"cv": None, "step": 4, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    "Tc_w40_80_6s": {"cv": None, "step": 6, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    "Td_w40_80_8s": {"cv": None, "step": 8, "swin": 60, "sstep": 2,
                     "drop_zero": True, "per_service": True, "mwin": (40, 80)},
    "Va_w20_100_3s": {"cv": None, "step": 3, "swin": 60, "sstep": 2,
                      "drop_zero": True, "per_service": True, "mwin": (20, 100)},
    "Vb_w20_100_4s": {"cv": None, "step": 4, "swin": 60, "sstep": 2,
                      "drop_zero": True, "per_service": True, "mwin": (20, 100)},
}

# How often each arm's metric block still contains at least one column belonging
# to the injected service, measured over 30 cases. An arm below 30/30 cannot
# reach AC@1 = 1.0 however good the model is, so its score has to be read
# against this ceiling rather than against 1.0.
GT_RETENTION = {
    "A_full": 30, "B_cv01": 30, "C_cv01_2s": 30, "D_cv05_2s": 30,
    "E_cv05_3s": 30, "F_cv10_3s": 30, "G_cv10_4s": 30, "H_cv20_4s": 30,
    "I_cv30_4s": 30, "J_cv30_5s": 30, "K_cv50_5s": 26, "L_cv100_6s": 23,
    "N_all_15s": 30, "O_1col_4s": 30, "Q_2col_4s": 30, "R_1col_8s": 30,
    "T_w40_80_10s": 30, "U_w40_80_5s": 30,
    "V_w20_100_10s": 30, "W_w20_100_5s": 30,
    "Ta_w40_80_3s": 30, "Tb_w40_80_4s": 30, "Tc_w40_80_6s": 30,
    "Td_w40_80_8s": 30, "Va_w20_100_3s": 30, "Vb_w20_100_4s": 30,
}

# Services still present in each arm's metric block, averaged over the sample.
# Reported alongside accuracy because it, not the token count, is what tracks
# Qwen's score across arms.
SERVICES_PRESENT = {
    "F_cv10_3s": 63.7, "H_cv20_4s": 50.7, "K_cv50_5s": 14.0, "L_cv100_6s": 4.8,
    "N_all_15s": 68.0, "O_1col_4s": 68.0, "Q_2col_4s": 68.0, "R_1col_8s": 68.0,
}

# Qwen3.5-9B's context window, and the completion budget reserved inside it.
# vLLM counts input + requested output against the one limit, so an arm has to
# fit INPUT_BUDGET, not QWEN_CONTEXT: a 32,069-token prompt is refused outright
# when 700 output tokens are also requested.
QWEN_CONTEXT = 32_768
MAX_OUTPUT_TOKENS = 700
INPUT_BUDGET = QWEN_CONTEXT - MAX_OUTPUT_TOKENS

# Qwen's tokenizer splits this payload far more finely than tiktoken does:
# measured over three cases per arm, it returns 1.34-1.36x tiktoken's count
# (F: 20,917 -> 28,535). Sizing an arm with tiktoken alone therefore understates
# it by about a third, which is how an arm estimated at 27,941 tokens was
# rejected at 32,069. --dry-run reports both the raw count and this projection.
QWEN_TOKEN_RATIO = 1.36


def _variation(values: pd.DataFrame) -> pd.Series:
    """Coefficient of variation per column: movement on each column's own scale.

    Comparable across metrics whose magnitudes differ by orders of magnitude --
    CPU fractions against memory in bytes.
    """
    return values.std() / (values.mean().abs() + 1e-9)


def reduce_metrics(frame: pd.DataFrame, cv: float | None, step: int,
                   per_service: bool = False) -> pd.DataFrame:
    """Drop columns and subsample rows. Values are never altered.

    A `cv` threshold cuts columns globally, which also removes every service
    whose columns all fall below it. `per_service` instead keeps each service's
    most-varying column, so the service count is preserved and only
    within-service redundancy is dropped.
    """
    out = frame
    values = out.drop(columns=["t"])
    if per_service:
        # per_service may be True (one column) or an int (that many columns).
        # Keeping the most-varying columns of each service preserves the service
        # count exactly, which a global threshold does not.
        keep_per_service = 1 if per_service is True else int(per_service)
        variation = _variation(values)
        by_service: dict[str, list[str]] = {}
        for column in values.columns:
            by_service.setdefault(column.rsplit("_", 1)[0], []).append(column)
        chosen: list[str] = []
        for columns in by_service.values():
            columns.sort(key=lambda c: -variation[c])
            chosen.extend(columns[:keep_per_service])
        out = out[["t"] + sorted(chosen)]
    elif cv is not None:
        variation = _variation(values)
        out = out[["t"] + list(variation[variation >= cv].index)]
    if step > 1:
        out = out[out["t"] % step == 0]
    return out


def reduce_series(frame: pd.DataFrame, inject: int, window: int | None,
                  step: int, drop_zero: bool) -> pd.DataFrame:
    """Narrow a time-series block. Values are never altered."""
    out = frame
    if window is not None and "time" in out.columns:
        out = out[(out["time"] >= inject - window)
                  & (out["time"] <= inject + window)]
    if drop_zero:
        out = out[[c for c in out.columns
                   if c == "time" or (out[c] != 0).any()]]
    if step > 1 and len(out) > 2:
        out = out.iloc[::step]
    return out


def _asymmetric_window(case_dir: Path, inject: int,
                       before: int, after: int) -> pd.DataFrame:
    """Metric window spanning inject-before .. inject+after.

    payload._metric_window is symmetric around the injection, so an arm that
    wants more time after the fault than before it builds its own. Columns that
    never move inside the window are dropped, matching the payload's behaviour.
    """
    frame = pd.read_csv(case_dir / "simple_metrics.csv")
    window = frame[(frame["time"] >= inject - before)
                   & (frame["time"] <= inject + after)].copy()
    values = window.drop(columns=["time"])
    values = values.loc[:, values.nunique() > 1]
    values.insert(0, "t", (window["time"] - inject).astype(int).values)
    return values


def build_arm(case_dir: Path, arm: str) -> str:
    """The condition E payload with this arm's reductions applied."""
    spec = ARMS[arm]
    payload_mod.METRIC_WINDOW_SECONDS = METRIC_WINDOW_SECONDS
    full = build(case_dir)
    if (spec["cv"] is None and spec["swin"] is None
            and not spec["drop_zero"] and not spec.get("per_service")
            and spec["step"] <= 1):
        return full.text

    inject = int((case_dir / "inject_time.txt").read_text().strip())
    text = full.text

    # Each block is replaced by matching its exact rendered body, so every
    # section an arm does not touch stays byte-identical to arm A.
    if (spec["cv"] is not None or spec.get("per_service")
            or spec["step"] > 1 or spec.get("mwin")):
        # The text to replace is always the payload's own +/-30s block; what
        # replaces it may come from a wider window.
        original = (payload_mod._metric_window(case_dir, inject)
                    .round(2).to_csv(index=False).strip())
        mwin = spec.get("mwin")
        window = (_asymmetric_window(case_dir, inject, *mwin) if mwin
                  else payload_mod._metric_window(case_dir, inject))
        reduced = reduce_metrics(window, spec["cv"], spec["step"],
                                 spec.get("per_service", False))
        text = text.replace(original,
                            reduced.round(2).to_csv(index=False).strip(), 1)
        if text == full.text:
            raise RuntimeError(f"{arm}: metric block not found for substitution")

    if spec["swin"] is not None or spec["drop_zero"]:
        for filename in payload_mod.SERIES_FILES:
            rendered = payload_mod._series_block(case_dir, inject, filename).strip()
            raw = pd.read_csv(case_dir / filename)
            raw = raw[(raw["time"] >= inject - payload_mod.SERIES_WINDOW_SECONDS)
                      & (raw["time"] <= inject + payload_mod.SERIES_WINDOW_SECONDS)]
            cut = reduce_series(raw, inject, spec["swin"], spec["sstep"],
                                spec["drop_zero"])
            replacement = cut.round(2).to_csv(index=False).strip()
            if rendered not in text:
                raise RuntimeError(f"{arm}: {filename} not found for substitution")
            # An unchanged body is legitimate: a series with no all-zero columns
            # and no narrower window reduces to itself.
            text = text.replace(rendered, replacement, 1)
    return text


def services_in(case_dir: Path) -> list[str]:
    """Train Ticket's service names, taken from the metric column prefixes."""
    columns = pd.read_csv(case_dir / "simple_metrics.csv", nrows=0).columns
    return sorted({c.rsplit("_", 1)[0] for c in columns if c != "time"})


def report() -> None:
    """Per-model tables. Two models are never averaged into one row.

    A model that refused an arm for length and a model that answered it produce
    incomparable rows, so the results are grouped by model and each arm reports
    how many of its calls the server actually served.
    """
    if not RESULTS.exists():
        print("no results yet")
        return
    rows = [json.loads(line)
            for line in RESULTS.read_text(encoding="utf-8").splitlines()
            if line.strip()]

    for model in sorted({r["model"] for r in rows}):
        mine = [r for r in rows if r["model"] == model]
        print(f"\n{model}  (n={len(mine)} calls)")
        print(f"{'arm':<13}{'n':>4}{'AC@1':>8}{'AC@3':>8}"
              f"{'tokens':>10}{'sec':>7}{'served':>9}{'ceil':>7}{'svcs':>7}")
        print("-" * 78)
        for arm in ARMS:
            got = [r for r in mine if r["arm"] == arm]
            if not got:
                continue
            n = len(got)
            ac1 = sum(r["ranking"][:1] == [r["truth"]] for r in got) / n
            ac3 = sum(r["truth"] in r["ranking"][:3] for r in got) / n
            # A refused call reports zero prompt tokens, so the token mean and
            # the fit verdict are taken over served calls only.
            served = [r for r in got if r["prompt_tokens"]]
            tokens = (sum(r["prompt_tokens"] for r in served) / len(served)
                      if served else 0)
            over = sum("context length" in str(r["error"]) for r in got)
            # The ceiling is what this arm could score at best: an arm whose
            # metric block no longer contains the injected service cannot be
            # answered correctly from the metrics at all.
            ceiling = GT_RETENTION.get(arm, 30) / 30
            services = SERVICES_PRESENT.get(arm)
            print(f"{arm:<13}{n:>4}{ac1:>8.2f}{ac3:>8.2f}{tokens:>10,.0f}"
                  f"{sum(r['seconds'] for r in got) / n:>7.1f}"
                  f"{f'{n - over}/{n}':>9}{ceiling:>7.2f}"
                  f"{(f'{services:.0f}' if services else '-'):>7}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", default="",
                    help="comma-separated arm subset, e.g. F_cv10_3s,G_cv10_4s; "
                         "default runs all of them")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        report()
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(ARMS)}")

    # The sample is drawn from the full case list before any subsetting, so a
    # run restricted to some arms still covers the same cases as a full run at
    # the same seed -- which is what makes the two models comparable.
    cases = [Path(p).parent for p in sorted(DATASET.glob("*/*/inject_time.txt"))]
    random.seed(args.seed)
    sample = random.sample(cases, args.limit)

    if args.dry_run:
        import tiktoken

        encoder = tiktoken.get_encoding("o200k_base")
        print("tiktoken counts; the Qwen projection applies "
              f"x{QWEN_TOKEN_RATIO} (measured)\n")
        print(f"{'case':<28}" + "".join(f"{a:>11}" for a in ARMS))
        worst = dict.fromkeys(ARMS, 0)
        for case_dir in sample:
            row = ""
            for arm in ARMS:
                size = len(encoder.encode(build_arm(case_dir, arm)))
                worst[arm] = max(worst[arm], size)
                row += f"{size:>11,}"
            print(f"{case_dir.parent.name + '/' + case_dir.name:<28}{row}")
        print(f"{'WORST CASE':<28}" + "".join(f"{worst[a]:>11,}" for a in ARMS))
        print(f"{'  x1.36 -> Qwen tokens':<28}"
              + "".join(f"{worst[a] * QWEN_TOKEN_RATIO:>11,.0f}" for a in ARMS))
        print(f"{'fits Qwen (32,068 in)':<28}"
              + "".join(
                  f"{('yes' if worst[a] * QWEN_TOKEN_RATIO <= INPUT_BUDGET else 'NO'):>11}"
                  for a in ARMS))
        return

    cfg = load_model_config(args.provider, args.model)
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS.open("a", encoding="utf-8") as sink:
        for case_dir in sample:
            truth = case_dir.parent.name.rsplit("_", 1)[0]
            valid = services_in(case_dir)
            for arm in arms:
                user = (f"Candidate services: {', '.join(valid)}\n\n"
                        f"{build_arm(case_dir, arm)}\n\n"
                        "Which service is the root cause? Reply with the JSON "
                        "object only.")
                started = time.time()
                try:
                    reply = client.chat.completions.create(
                        model=cfg["model"],
                        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user}],
                        max_tokens=MAX_OUTPUT_TOKENS,
                        **sampling_args(cfg["model"]),
                        **cfg["request_extra"],
                    )
                    ranking, conf, _, err = parse_reply(
                        reply.choices[0].message.content or "", valid=valid)
                    prompt_tokens = reply.usage.prompt_tokens
                except Exception as exc:  # noqa: BLE001
                    ranking, conf, err = [], "", f"{type(exc).__name__}: {exc}"
                    prompt_tokens = 0

                record = {
                    "case": f"{case_dir.parent.name}/{case_dir.name}",
                    "arm": arm,
                    "truth": truth,
                    "ranking": ranking,
                    "prompt_tokens": prompt_tokens,
                    "confidence": conf,
                    "error": err,
                    "seconds": round(time.time() - started, 1),
                    "model": cfg["model"],
                }
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                mark = ("HIT" if ranking[:1] == [truth]
                        else "top3" if truth in ranking[:3] else "----")
                print(f"{case_dir.parent.name:<24}{arm:<11}{prompt_tokens:>7,}tok"
                      f"{record['seconds']:>6.1f}s {mark:<5}{ranking[:3]}")
    report()


if __name__ == "__main__":
    main()
