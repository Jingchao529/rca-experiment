# Project Structure

```text
rca-experiment-main/
├── README.md
├── requirements.txt
├── finalexp/
│   ├── run_experiment_a1.py
│   ├── adapters/
│   │   └── rcaeval_trace_adapter.py
│   └── results/                         # Created automatically at runtime
│       └── experiment_a1.jsonl          
├── ctl-main/
│   └── util/
│       └── analysis.py                  # Original Trace-SHAP implementation
└── data/
    └── RCAEval-OB/
        └── RE2-OB/
            ├── <service>_<fault>/
            │   ├── 1/
            │   │   ├── traces.csv
            │   │   └── inject_time.txt
            │   ├── 2/
            │   │   ├── traces.csv
            │   │   └── inject_time.txt
            │   └── 3/
            │       ├── traces.csv
            │       └── inject_time.txt
            └── ...
```
