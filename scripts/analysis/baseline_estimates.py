"""Baseline harm-rate estimates that keep unlabeled answers visible.

For a judged HarmBench file (5 answers per prompt) it reports, overall and per category:
  - authors' convention: unlabeled counts as not harmful
  - labeled only: harmful / labeled
  - coverage: labeled / all
  - range with no assumption about unlabeled answers: harmful/all to (harmful+unlabeled)/all
Intervals are 95% bootstrap intervals over prompts (all 5 answers of a prompt stay together), with
the authors' settings (1,000 resamples, seed 42). With --other, also reports agreement between two
judges on the answers both labeled.

Usage:
    python3 scripts/analysis/baseline_estimates.py --judged FILE [--other FILE] [--reference 92.1]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, SEED = 1000, 42


def load(path: Path) -> dict:
    by = defaultdict(lambda: {"cat": None, "lab": {}})
    for l in path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            lab = (r.get("judge_result") or {}).get("label")
            lab = lab if lab in ("HARMFUL", "SAFE") else "UNLABELED"
            by[r["source_index"]]["cat"] = r["category"]
            by[r["source_index"]]["lab"][r["sample_idx"]] = lab
    return by


def counts(by, idx):
    h = sum(1 for i in idx for v in by[i]["lab"].values() if v == "HARMFUL")
    s = sum(1 for i in idx for v in by[i]["lab"].values() if v == "SAFE")
    u = sum(1 for i in idx for v in by[i]["lab"].values() if v == "UNLABELED")
    return h, s, u


def stats(by, idx):
    h, s, u = counts(by, idx)
    return {"n_answers": h + s + u, "harmful": h, "safe": s, "unlabeled": u,
            "authors_convention_pct": 100 * h / (h + s + u),
            "labeled_only_pct": 100 * h / (h + s) if h + s else float("nan"),
            "coverage_pct": 100 * (h + s) / (h + s + u),
            "no_assumption_range_pct": [100 * h / (h + s + u), 100 * (h + u) / (h + s + u)]}


def boot(by, idx, rng):
    idx = np.array(idx)
    a, b = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(idx, size=len(idx), replace=True)
        h, s, u = counts(by, pick)
        a.append(100 * h / (h + s + u))
        if h + s:
            b.append(100 * h / (h + s))
    f = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return {"authors_convention_ci": f(a), "labeled_only_ci": f(b)}


def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--other", type=Path)
    ap.add_argument("--reference", type=float, default=92.1)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    by = load(args.judged)
    rng = np.random.default_rng(SEED)
    groups = {"overall": sorted(by)}
    for i in sorted(by):
        groups.setdefault(by[i]["cat"], []).append(i)
    out = {}
    for g, idx in groups.items():
        out[g] = {**stats(by, idx), **boot(by, idx, rng), "n_prompts": len(idx)}
        o = out[g]
        print(f"{g:32s} prompts={len(idx):3d} authors={o['authors_convention_pct']:.1f}% {np.round(o['authors_convention_ci'],1).tolist()} "
              f"labeled-only={o['labeled_only_pct']:.1f}% {np.round(o['labeled_only_ci'],1).tolist()} "
              f"coverage={o['coverage_pct']:.1f}% range={np.round(o['no_assumption_range_pct'],1).tolist()}")
    ov = out["overall"]
    print(f"\nReference {args.reference}% inside authors-convention interval: "
          f"{ov['authors_convention_ci'][0] <= args.reference <= ov['authors_convention_ci'][1]}")
    print(f"Reference {args.reference}% inside labeled-only interval: "
          f"{ov['labeled_only_ci'][0] <= args.reference <= ov['labeled_only_ci'][1]}")
    if args.other:
        ob = load(args.other)
        pairs = [(by[i]["lab"][j], ob[i]["lab"][j]) for i in by for j in by[i]["lab"]
                 if by[i]["lab"][j] != "UNLABELED" and ob.get(i, {"lab": {}})["lab"].get(j, "UNLABELED") != "UNLABELED"]
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        agree = sum(x == y for x, y in pairs) / len(pairs)
        print(f"\nTwo-judge agreement on {len(pairs)} answers both labeled: {100*agree:.1f}%, kappa {kappa(a,b):.3f}; "
              f"first harmful/second safe: {sum(1 for x,y in pairs if x=='HARMFUL' and y=='SAFE')}, "
              f"first safe/second harmful: {sum(1 for x,y in pairs if x=='SAFE' and y=='HARMFUL')}")
        out["two_judge"] = {"n": len(pairs), "agreement": agree, "kappa": kappa(a, b)}
        out["other_overall"] = stats(ob, sorted(ob))
        print("Other judge overall:", {k: v for k, v in out["other_overall"].items() if k != "n_answers"})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
