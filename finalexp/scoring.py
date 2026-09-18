"""Shared AC@k / Avg@k scoring, so every method is scored by one implementation.

Metrics follow TORAI (FSE'26, arXiv:2604.13522) section 4.2, Eq. 1:

    AC@k  = (1/|A|) * SUM_a [ |{i < k : R_a[i] in V_rc^a}| / min(k, |V_rc^a|) ]
    Avg@k = (1/k) * SUM_{j=1..k} AC@j

Every RCAEval case injects a fault into exactly one service, so |V_rc| = 1 and
AC@k reduces to "is the true service ranked within the top k".
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def ac_at_k(rows: list[dict], k: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for r in rows if r["ground_truth"] in r.get("ranking", [])[:k])
    return hits / len(rows)


def avg_at_k(rows: list[dict], k: int = 5) -> float:
    return sum(ac_at_k(rows, j) for j in range(1, k + 1)) / k


def score_table(rows: list[dict], method: str, k_avg: int = 5) -> dict:
    """Print the per-fault accuracy table and return the headline numbers.

    Failed cases are scored as misses rather than dropped: excluding them would
    flatter a method by hiding the inputs it could not handle.
    """
    for row in rows:
        row.setdefault("ranking", [])

    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    print(f"\n{'=' * 66}")
    print(f"{method}")
    print(f"cases: {len(rows)}   ok: {len(ok)}   failed: {len(failed)}")
    print(f"{'=' * 66}")

    by_fault: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fault[row["fault"]].append(row)

    print(f"\n{'fault':<10}{'n':>5}{'AC@1':>9}{'AC@3':>9}{'Avg@5':>9}")
    print("-" * 42)
    for fault in sorted(by_fault):
        group = by_fault[fault]
        print(f"{fault:<10}{len(group):>5}{ac_at_k(group, 1):>9.3f}"
              f"{ac_at_k(group, 3):>9.3f}{avg_at_k(group, k_avg):>9.3f}")
    print("-" * 42)
    headline = {
        "AC@1": ac_at_k(rows, 1),
        "AC@3": ac_at_k(rows, 3),
        f"Avg@{k_avg}": avg_at_k(rows, k_avg),
        "n": len(rows),
        "failed": len(failed),
    }
    print(f"{'AVERAGE':<10}{len(rows):>5}{headline['AC@1']:>9.3f}"
          f"{headline['AC@3']:>9.3f}{headline[f'Avg@{k_avg}']:>9.3f}")

    if ok:
        seconds = [r["seconds"] for r in ok]
        print(f"\nefficiency: {np.mean(seconds):.3f}s per case "
              f"(min {min(seconds):.3f}s, max {max(seconds):.3f}s)")
        headline["mean_seconds"] = float(np.mean(seconds))

    # Which service the method names first, across all cases. A single service
    # dominating this count indicates a systematic bias rather than diagnosis.
    top1 = defaultdict(int)
    for row in ok:
        if row["ranking"]:
            top1[row["ranking"][0]] += 1
    if top1:
        ordered = sorted(top1.items(), key=lambda kv: -kv[1])
        print("\nranked #1 most often: "
              + ", ".join(f"{name} x{count}" for name, count in ordered[:5]))

    if failed:
        print(f"\n{len(failed)} failed case(s):")
        for row in failed[:10]:
            print(f"  {row['case_id']}: {row.get('error', '')[:70]}")

    return headline
