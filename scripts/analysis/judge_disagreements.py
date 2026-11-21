"""Where two judges differ on the same HarmBench answers.

Prints why the second judge left answers unlabeled, how the disagreements split by category, and a few
short examples (first 160 characters of the answer, plus each judge's stated reason) for reading by hand.

Usage:
    python3 scripts/analysis/judge_disagreements.py --first FILE --second FILE [--examples 6]
"""
from __future__ import annotations
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def load(p: Path) -> dict:
    out = {}
    for l in p.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out[(r["source_index"], r["sample_idx"])] = r
    return out


def lab(r):
    x = (r.get("judge_result") or {}).get("label")
    return x if x in ("HARMFUL", "SAFE") else "UNLABELED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    f, s = load(a.first), load(a.second)
    un = [r for r in s.values() if lab(r) == "UNLABELED"]
    print("Second judge unlabeled:", len(un))
    reasons = Counter(re.sub(r"\d+", "#", str((r["judge_result"] or {}).get("reason", ""))[:90]) for r in un)
    for k, v in reasons.most_common(6):
        print(f"  {v:4d}  {k}")
    print("  by category:", dict(Counter(r["category"] for r in un)))
    print("  also unlabeled by first judge:", sum(1 for k, r in s.items() if lab(r) == "UNLABELED" and lab(f[k]) == "UNLABELED"))
    dis = [k for k in f if lab(f[k]) == "HARMFUL" and lab(s[k]) == "SAFE"]
    print("\nFirst HARMFUL / second SAFE:", len(dis))
    print("  by category:", dict(Counter(f[k]["category"] for k in dis)))
    print("  response length (chars), disagreements vs all:",
          sum(len(f[k]["response"]) for k in dis) // max(len(dis), 1), "vs",
          sum(len(r["response"]) for r in f.values()) // len(f))
    random.seed(42)
    for k in random.sample(dis, min(a.examples, len(dis))):
        r = f[k]
        print(f"\n[{r['category']}] prompt: {r['behavior'][:100]!r}")
        print(f"  answer starts: {r['response'][:160]!r}")
        print(f"  second judge raw: {str(s[k]['judge_result'])[:200]}")


if __name__ == "__main__":
    main()
