# RCA baselines and the LLM conditions

Three classical RCA baselines and the two LLM conditions, on RCAEval RE2-OB
(Online Boutique) and RE2-TT (Train Ticket).

Trace-SHAP is packaged separately; it carries its own vendored source and is not
needed to run anything here.

Chinese version: [README.zh.md](README.zh.md)

## What E and F are, and why the difference matters

The two LLM conditions differ in what the model is given, not in the model or
the prompt format:

| Condition | The model receives | Tools |
|---|---|---|
| **E** `raw_direct` | the telemetry for the case | none |
| **F** `tool_agentic` | the same telemetry, **plus the three baselines' rankings**, offered as callable tools | three |

This is the single most common source of a failed reproduction. In condition E
the model reasons from telemetry alone and does **worse than the baselines** —
that is the expected result, not a bug. Condition F is what places an LLM above
them, because F consumes the baselines rather than competing with them.

On Online Boutique, AC@1, 90 cases x 3 repetitions:

| | E | F |
|---|---|---|
| gpt-4o-mini | 0.189 | 0.789 |
| Gemini Flash Lite | 0.626 | 0.826 |

For reference, the strongest classical baseline on the same cases scores 0.744.
So an LLM given only telemetry is beaten by the baselines, and the same LLM
given the baselines' output is not. If your LLM numbers come out below the
baselines, check first whether you are running E and comparing it against F.

## Layout

```text
rca-baselines-and-llm/
├── README.md
├── README.zh.md
├── requirements.txt
├── finalexp/
│   ├── run_experiment_b.py          # BARO
│   ├── run_experiment_c.py          # CIRCA
│   ├── run_experiment_d.py          # TORAI
│   ├── run_experiment_e.py          # LLM, telemetry only
│   ├── run_experiment_f.py          # LLM, telemetry + baseline tools
│   ├── tool_server.py               # serves baseline rankings as tools
│   ├── payload.py                   # builds the telemetry prompt
│   ├── run_ablation_input.py        # input-arm construction used by F
│   ├── scoring.py                   # AC@1 / AC@3 / Avg@5
│   ├── freeze_results.py            # builds the snapshot F reads
│   ├── adapters/
│   │   └── rcaeval_trace_adapter.py
│   ├── results/                     # created at runtime
│   └── frozen/<date>/               # created by freeze_results.py
├── src/platform/
│   └── agent_config.json            # API keys go here (blank in this copy)
└── data/
    └── RCAEval-OB/
        ├── RE2-OB/
        └── RE2-TT/
```

## Dataset

Not shipped. Download from Zenodo and unpack so that `RE2-OB` and `RE2-TT` sit
at the paths above:

- DOI: [10.5281/zenodo.14590730](https://doi.org/10.5281/zenodo.14590730)

RE2-OB is 30 fault directories with three repetitions each, 90 cases, about
8 GB unpacked.

Some copies ship a `multi-source-data/` directory beside the numbered
repetitions. The runners treat it as a 91st case; delete those directories to
score the 90 cases the papers report.

## Setup

```bash
pip install -r requirements.txt
pip install RCAEval          # BARO, CIRCA and TORAI are called through it
```

Put your API keys in `src/platform/agent_config.json` — the copy here has the
key fields blanked:

```json
{ "api_key": "<openai>", "gemini_api_key": "<gemini>" }
```

The Qwen entry points at a self-hosted vLLM server, which has no public
endpoint. To use it, give it your own:

```bash
export QWEN_BASE_URL=http://your-server:8000/v1
```

It defaults to `http://localhost:8000/v1`. The other providers are public APIs
and need only a key.

## Python environments for the baselines

The three baselines need **three mutually incompatible environments**. This is
not a packaging preference; they pin conflicting builds of `causal-learn`, and
no single environment satisfies all three.

| Baseline | Python | causal-learn | Notes |
|---|---|---|---|
| **BARO** (B) | any | not used | runs anywhere |
| **CIRCA** (C) | 3.9 | **0.1.4.8** | `pc_default` passes `node_names=`, which 0.1.2.3 rejects |
| **TORAI** (D) | 3.8 | **0.1.2.3, patched** | the patch is not on PyPI |

So:

```bash
conda create -n circa39 python=3.9 && conda activate circa39
pip install -r requirements.txt RCAEval causal-learn==0.1.4.8
python finalexp/run_experiment_c.py

conda create -n torai38 python=3.8 && conda activate torai38
pip install -r requirements.txt RCAEval causal-learn==0.1.2.3
# then apply the patch below
python finalexp/run_experiment_d.py
```

### The TORAI patch

TORAI calls RCAEval's RCD internals, which need a patched `causal-learn` that is
not on PyPI and is not mentioned in RCAEval's SETUP.md. It is vendored in the
RCD authors' repository, [github.com/azamikram/rcd](https://github.com/azamikram/rcd).
Four files must be copied over an installed causal-learn 0.1.2.3:

```
causallearn/graph/GraphClass.py                 (CausalGraph gains `labels`)
causallearn/search/ConstraintBased/FCI.py
causallearn/utils/Fas.py
causallearn/utils/PCUtils/SkeletonDiscovery.py  (adds local_skeleton_discovery)
```

Keep the originals as `<name>.orig` beside each patched file so the change is
reversible.

### Why condition F ships a frozen snapshot

Because no one process can import all three methods, an agent cannot call them
live. F therefore reads their recorded output from
`finalexp/frozen/2026-09-06/`. CIRCA also costs about 42 s per call, so an agent
free to re-call it would spend hours per case re-deriving a deterministic
result.

**If you only want to run E and F, you do not need any of this.** The snapshot
is already in the repo; a single environment with `requirements.txt` is enough.

### A note on CIRCA's input

`run_experiment_c.py` reads `simple_metrics.csv` (73 columns), not `metrics.csv`
(418), following RCAEval's own harness. This is not cosmetic: on the full metric
set the Fisher-Z independence test hits a singular correlation matrix, and CIRCA
silently returns an empty graph and a ranking that is just input order — a
degenerate result that still looks like a working method with a poor score.

## Running the baselines

```bash
python finalexp/run_experiment_b.py          # BARO
python finalexp/run_experiment_c.py          # CIRCA
python finalexp/run_experiment_d.py          # TORAI

python finalexp/run_experiment_b.py --limit 1        # smoke test
python finalexp/run_experiment_b.py --score-only
python finalexp/run_experiment_b.py --dataset data/RCAEval-OB/RE2-TT
```

Each appends to `finalexp/results/experiment_<x>.jsonl`, one row per case, and
skips cases already recorded, so an interrupted run resumes.

## Running the LLM conditions

`--provider` selects the model; `--model` overrides the default model id for
that provider. Available: `openai` (gpt-4o-mini), `gemini`
(gemini-3.1-flash-lite), `deepseek`, `qwen`.

E needs only a key:

```bash
python finalexp/run_experiment_e.py --provider openai --limit 1
python finalexp/run_experiment_e.py --provider gemini
python finalexp/run_experiment_e.py --provider openai --model gpt-4o
```

F serves the three baselines' rankings to the model as callable tools. It reads
them from a frozen snapshot rather than recomputing them, so the numbers it
consumes are pinned and verifiable.

**The snapshot this study used ships with this repo**, at
`finalexp/frozen/2026-09-06/`, so F runs without running the baselines first:

```bash
python finalexp/run_experiment_f.py --provider openai
python finalexp/run_experiment_f.py --provider gemini
```

Check it is intact before relying on it — every file is hashed in
`MANIFEST.json`:

```bash
python finalexp/freeze_results.py --verify 2026-09-06
```

### To run F against your own baseline results instead

Run the three baselines, freeze them **under the same date**, and F picks them
up with no code change:

```bash
python finalexp/run_experiment_b.py
python finalexp/run_experiment_c.py
python finalexp/run_experiment_d.py

python finalexp/freeze_results.py --date 2026-09-06 --force
```

`tool_server.py` hardcodes that date:

```python
FROZEN = REPO_ROOT / "finalexp" / "frozen" / "2026-09-06"
```

so a snapshot under any other date is not found and F reports zero cases. Either
pass `--date 2026-09-06` as above, or edit that constant to point at yours.

⚠️ `--force` overwrites the shipped snapshot. Copy it elsewhere first if you
want to keep comparing against the original numbers.

## Train Ticket

Both datasets work, with two differences that fail quietly rather than loudly:

- **`tool_server.use_system("TT")`** must be set. `run_experiment_f.py` does this
  from the dataset path, but a runner that forgets it serves the *previous*
  system's results and prints a summary that looks valid.
- **Candidate service names.** `CANDIDATE_SERVICES` in `payload.py` is Online
  Boutique's eleven names. Train Ticket has different ones; pass them explicitly
  or every answer is rejected as invalid and correct rankings record as empty.

## Metrics

AC@1, AC@3 and Avg@5, following TORAI (FSE'26, arXiv:2604.13522) section 4.2,
computed in `scoring.py`. A failed case scores as a miss rather than being
dropped, so a method that crashes on hard cases is not flattered by excluding
them.

Differences of a few cases are not significant at this sample size. Treat small
gaps as ties unless a significance test is shown.
