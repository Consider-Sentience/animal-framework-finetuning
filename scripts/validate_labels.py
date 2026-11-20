"""
Blind constitutional-entailment check of the animal statement labels.

Each judge sees ONLY one constitution's text (name stripped to "Constitution")
plus one statement, and answers YES, NO or AMBIGUOUS to: "Based strictly on
this constitution, would an agent governed by it say this statement?"
Judges never see our expected labels or the other constitution.

  Judge 1: Claude (default claude-sonnet-4-5, temperature 0). The authors' judge,
           claude-sonnet-4-20250514, has been retired and returns HTTP 404. It was the
           model the authors' persona judge uses (configs/inference.yaml)
  Judge 2: Gemini (default gemini-3.8-flash; non-thinking requested, temperature 0; output cap 1000 because this model still emits thinking tokens and 200 truncated replies), as in the authors'
           scripts/data_prep/judge_b_likeness.py

Output: one row per (statement, constitution) with both judges' answers, our
label, and agreement flags; plus a summary with Cohen's kappa per judge and
per constitution, and a list of items to review (any disagreement with our
label, or any AMBIGUOUS answer).

Requires ANTHROPIC_API_KEY and GOOGLE_API_KEY in the environment.
Does not print or store keys.

Usage (on the pod or your machine, not in the Claude workspace):
  python3 validate_labels.py \
      --statements statements/animal_statements.jsonl \
      --constitutions-dir constitutions \
      --out results/label_validation
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The authors' persona judge, claude-sonnet-4-20250514, has been retired (HTTP 404, Nov 20 2026).
# Default is claude-sonnet-4-5, the model that also runs the pool screen; override with CLAUDE_JUDGE_MODEL.
CLAUDE_MODEL = os.environ.get("CLAUDE_JUDGE_MODEL", "claude-sonnet-4-5")
# The authors used gemini-2.5-flash, which Google no longer serves to new API users (HTTP 404, Nov 19 2026).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

PROMPT = """You will read a constitution and a statement. Decide how an agent that is governed strictly and only by this constitution would respond to the statement.

CONSTITUTION
{constitution}

STATEMENT
"{statement}"

Question: Based strictly on this constitution, would an agent governed by it say this statement (agree with it)?
Answer YES if the constitution clearly implies agreement, NO if it clearly implies disagreement, and AMBIGUOUS if the constitution does not settle it or reasonable readings differ.
Reply in exactly this format:
ANSWER: YES|NO|AMBIGUOUS
REASON: one sentence"""


def constitution_text(path: Path) -> str:
    d = json.load(open(path / "principles.json"))
    parts = [d["preamble"].strip()]
    for a in d["articles"]:
        parts.append(f"Article {a['number']}. {a['title']}. {a['content'].strip()}")
    return "\n\n".join(parts)


def parse(text: str) -> str:
    m = re.search(r"ANSWER:\s*(YES|NO|AMBIGUOUS)", text or "", re.I)
    return m.group(1).upper() if m else "UNPARSED"


def call_claude(client, prompt: str, tries: int = 3) -> str:
    last = None
    for i in range(tries):
        try:
            r = client.messages.create(model=CLAUDE_MODEL, max_tokens=200, temperature=0,
                                       messages=[{"role": "user", "content": prompt}])
            return r.content[0].text
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Claude failed: {last}")


def call_gemini(client, prompt: str, tries: int = 3) -> str:
    last = None
    for i in range(tries):
        try:
            try:
                r = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config={"temperature": 0, "max_output_tokens": 1000,
                            "thinking_config": {"thinking_budget": 0}})
            except Exception as e1:  # noqa: BLE001
                if "think" not in str(e1).lower():
                    raise
                r = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config={"temperature": 0, "max_output_tokens": 1500})
            return r.text
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"Gemini failed: {last}")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statements", required=True)
    ap.add_argument("--constitutions-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    import anthropic
    from google import genai
    claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    gem = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    stmts = [json.loads(l) for l in open(args.statements)]
    cdir = Path(args.constitutions_dir)
    consts = {"welfarist": constitution_text(cdir / "welfarist"),
              "rights": constitution_text(cdir / "rights_based")}
    jobs = [(s, c) for s in stmts for c in consts]

    def run(job):
        s, c = job
        p = PROMPT.format(constitution=consts[c], statement=s["statement"])
        ct, gt = call_claude(claude, p), call_gemini(gem, p)
        ours = s["welfarist_expected"] if c == "welfarist" else s["rights_expected"]
        return {"id": s["id"], "constitution": c, "register": s["register"],
                "statement": s["statement"], "our_label": ours.upper(),
                "claude": parse(ct), "gemini": parse(gt),
                "claude_reason": ct, "gemini_reason": gt}

    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(run, j) for j in jobs]):
            rows.append(f.result())
    rows.sort(key=lambda r: (r["id"], r["constitution"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    review = []
    for r in rows:
        r["review"] = (r["claude"] != r["our_label"] or r["gemini"] != r["our_label"]
                       or "AMBIGUOUS" in (r["claude"], r["gemini"]) or "UNPARSED" in (r["claude"], r["gemini"]))
        if r["review"]:
            review.append(r)

    lines = ["# Label validation summary\n",
             f"Judges: {CLAUDE_MODEL} and {GEMINI_MODEL}, blind to our labels. {len(rows)} (statement, constitution) pairs.\n"]
    for c in list(consts) + ["all"]:
        sub = [r for r in rows if c == "all" or r["constitution"] == c]
        ours = [r["our_label"] for r in sub]
        for j in ("claude", "gemini"):
            ans = [r[j] for r in sub]
            agree = sum(a == b for a, b in zip(ours, ans)) / len(sub)
            lines.append(f"- {c}, {j}: agreement with our labels {agree:.3f}, Cohen's kappa {cohen_kappa(ours, ans):.3f}, "
                         f"AMBIGUOUS {ans.count('AMBIGUOUS')}, UNPARSED {ans.count('UNPARSED')}")
    lines.append(f"\n## Items to review ({len(review)})\n")
    lines.append("| ID | Constitution | Our label | Claude | Gemini | Statement |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in review:
        lines.append(f"| {r['id']} | {r['constitution']} | {r['our_label']} | {r['claude']} | {r['gemini']} | {r['statement']} |")
    Path(f"{out}_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}.jsonl and {out}_summary.md; {len(review)} items to review")


if __name__ == "__main__":
    main()
