"""
Agreement between the two judges (Claude Sonnet 4.5 and Gemini 3.8 Flash) on the same answers.

Safety and held-out answers: labels HARMFUL / SAFE; answers either judge left UNLABELED are counted separately.
Animal statements and pressure answers: labels YES / NO / AMBIGUOUS; UNPARSED and ERROR counted separately.
Reports raw agreement and Cohen's kappa overall and by model.

  python3 scripts/analysis/judge_agreement.py [--out results/judge_agreement.json]
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

R = Path("results")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return po, (None if pe == 1 else (po - pe) / (1 - pe))


def safety():
    out = {}
    for cf in sorted((R / "animal_tests_judge").glob("*_K5_judged.jsonl")):
        mid = cf.name[: -len("_K5_judged.jsonl")]
        gf = R / "animal_tests_judged" / f"{mid}_K5_gemini.jsonl"
        if not gf.exists():
            continue
        c = {(r["source_index"], r["sample_idx"]): (r.get("judge_result") or {}).get("label") for r in load(cf)}
        g = {(r["source_index"], r["sample_idx"]): (r.get("judge_result") or {}).get("label") for r in load(gf)}
        pairs, skipped = [], 0
        for k in c:
            a, b = c[k], g.get(k)
            if a in ("HARMFUL", "SAFE") and b in ("HARMFUL", "SAFE"):
                pairs.append((a, b))
            else:
                skipped += 1
        po, kp = kappa(pairs)
        out[mid] = {"n_both_labeled": len(pairs), "not_labeled_by_one_or_both": skipped, "agreement": po, "kappa": kp,
                    "claude_harmful_gemini_safe": sum(p == ("HARMFUL", "SAFE") for p in pairs),
                    "claude_safe_gemini_harmful": sum(p == ("SAFE", "HARMFUL") for p in pairs)}
    return out


def statements():
    out = {}
    for cf in sorted((R / "animal_tests_judged").glob("*_statements_claude.jsonl")):
        mid = cf.name[: -len("_statements_claude.jsonl")]
        gf = R / "animal_tests_judged" / f"{mid}_statements_gemini.jsonl"
        if not gf.exists():
            continue
        key = lambda r: (r["kind"], r["statement_id"], r.get("pushback_id"), r.get("pushback"))
        c = {key(r): r["judge"]["agreement"] for r in load(cf)}
        g = {key(r): r["judge"]["agreement"] for r in load(gf)}
        ok = ("YES", "NO", "AMBIGUOUS")
        pairs = [(c[k], g[k]) for k in c if k in g and c[k] in ok and g[k] in ok]
        po, kp = kappa(pairs)
        out[mid] = {"n": len(pairs), "agreement": po, "kappa": kp, "unusable": len(c) - len(pairs),
                    "confusion": {f"{a}/{b}": n for (a, b), n in Counter(pairs).items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R / "judge_agreement.json")
    a = ap.parse_args()
    res = {"safety": safety(), "statements": statements()}
    for name in ("safety", "statements"):
        d = res[name]
        n = sum(v.get("n_both_labeled", v.get("n", 0)) for v in d.values())
        if n:
            pooled = sum(v["agreement"] * v.get("n_both_labeled", v.get("n", 0)) for v in d.values()) / n
            res[name + "_pooled_agreement"] = pooled
            print(f"{name}: {len(d)} models, {n} answers labeled by both, raw agreement {pooled:.3f}")
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
