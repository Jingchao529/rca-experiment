# Frozen results -- RCA baselines on Online Boutique

Dataset: RCAEval RE2-OB (Online Boutique, 90 failure cases)
Zenodo DOI: 10.5281/zenodo.14590730

Metrics follow TORAI (FSE'26, arXiv:2604.13522) section 4.2, Eq. 1.
Every case injects one service, so AC@k is "is the true service in the
top k"; Avg@5 is the mean of AC@1..AC@5.

## Overall

| Method | Year | Mechanism | Cases | AC@1 | AC@3 | Avg@5 | s/case |
|---|---|---|---|---|---|---|---|
| TORAI | FSE'26 | multi-source severity clustering | 90 | 0.744 | 0.911 | 0.876 | 1.86 |
| CIRCA | ICSE'22 | causal graph + regression testing | 90 | 0.578 | 0.967 | 0.880 | 42.32 |
| BARO | FSE'24 | statistical hypothesis testing | 91 | 0.231 | 0.813 | 0.708 | 0.66 |
| Trace-SHAP | - | trace anomaly attribution | 91 | 0.059 | 0.648 | 0.562 | 7.39 |

## AC@3 by fault type

| Fault | TORAI | CIRCA | BARO | Trace-SHAP |
|---|---|---|---|---|
| cpu | 1.000 | 1.000 | 0.688 | 0.662 |
| delay | 0.867 | 0.867 | 0.667 | 0.560 |
| disk | 0.800 | 1.000 | 0.867 | 0.680 |
| loss | 0.867 | 1.000 | 0.867 | 0.640 |
| mem | 1.000 | 0.933 | 1.000 | 0.653 |
| socket | 0.933 | 1.000 | 0.800 | 0.693 |

## Notes

- **TORAI**: 90 runs over 90 cases, 0 failed. Deterministic: one run per case. Ranked #1 most often: recommendationservice x20, emailservice x19, currencyservice x13.
- **CIRCA**: 90 runs over 90 cases, 0 failed. Deterministic: one run per case. Ranked #1 most often: emailservice x16, recommendationservice x14, cartservice x13.
- **BARO**: 91 runs over 91 cases, 0 failed. Deterministic: one run per case. Ranked #1 most often: redis x66, emailservice x10, checkoutservice x5.
- **Trace-SHAP**: 455 runs over 91 cases, 0 failed. Non-deterministic: seeds [42, 43, 44, 45, 46], Avg@5 spread 0.1055 (0.512-0.618); reported figures are the mean across seeds. Ranked #1 most often: frontendservice x412, checkoutservice x18, productcatalogservice x15.

Reproduction commands and the full protocol are in `finalexp/README.md` and `finalexp/protocol/EXPERIMENT_PROTOCOL.md`.
File integrity hashes are in `MANIFEST.json` alongside this file.
