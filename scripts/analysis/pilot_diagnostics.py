"""
Diagnostics for the critique/revision pilot. Reads pilot/out/_intermediate/{constitution}_harmful_hermes.jsonl
and *_helpful_hermes.jsonl and prints, per constitution (no API calls):
  counts; truncated calls (finish_reason == "length"); empty answers; leak words in final answers
  (which words); decline count in the first answer, round 1 and the final answer (a simple phrase
  match, listed below, used only to count -- every flagged answer is printed so a person can check);
  mean length in words, first answer vs round 1 vs final; and how often each instruction was drawn.
Usage: python3 scripts/analysis/pilot_diagnostics.py pilot/out > pilot/diagnostics.txt
"""
import collections, json, re, sys
from pathlib import Path

LEAK = ["article", "constitution", "principle", "critique", "revision", "preamble"]
DECLINE = re.compile(r"\b(i can['’]?t (help|assist|provide)|i cannot (help|assist|provide)|i won['’]?t|i['’]?m (not able|unable) to|i am (not able|unable) to|i['’]?m sorry, but|i must decline|i will not|i['’]?m not going to)\b", re.I)

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []

def words(t): return len(t.split())

def main(root):
    inter = Path(root) / "_intermediate"
    for const in ["welfarist", "rights_based", "virtuous_agent", "subordinate_agent"]:
        H = load(inter / f"{const}_harmful_hermes.jsonl")
        E = load(inter / f"{const}_helpful_hermes.jsonl")
        print(f"\n===== {const}: {len(H)} harmful, {len(E)} helpful =====")
        trunc_c = sum(1 for r in H for x in r["rounds"] if x.get("critique_finish_reason") == "length")
        trunc_r = sum(1 for r in H for x in r["rounds"] if x.get("revision_finish_reason") == "length")
        trunc_final = sum(1 for r in H if r["rounds"][-1].get("revision_finish_reason") == "length")
        trunc = f"critiques {trunc_c}, revisions {trunc_r} (final-round revisions {trunc_final} of {len(H)})"
        trunc_e = sum(1 for r in E if r.get("finish_reason") == "length")
        empty = sum(1 for r in H + E if not r["response"].strip())
        print(f"truncated calls, harmful: {trunc}; helpful answers {trunc_e}; empty answers: {empty}")
        leak = collections.Counter()
        leaked = []
        for r in H + E:
            hits = [w for w in LEAK if w in r["response"].lower()]
            for w in hits: leak[w] += 1
            if hits: leaked.append((r["prompt_type"], r["prompt"][:70], hits))
        print(f"final answers with a leak word: {len(leaked)} of {len(H)+len(E)}  {dict(leak)}")
        for t, p, h in leaked: print(f"   [{t}] {p!r} -> {h}")
        d0 = sum(1 for r in H if DECLINE.search(r["initial_response"]))
        d1 = sum(1 for r in H if DECLINE.search(r["rounds"][0]["revision"]))
        d3 = sum(1 for r in H if DECLINE.search(r["response"]))
        print(f"declines (phrase match) among {len(H)} harmful: first answer {d0}, round 1 {d1}, final {d3}  (pilot rule threshold: fewer than {len(H)/2:g} in the final answers)")
        print(f"mean words harmful: first {sum(words(r['initial_response']) for r in H)/max(len(H),1):.0f}, round 1 {sum(words(r['rounds'][0]['revision']) for r in H)/max(len(H),1):.0f}, final {sum(words(r['response']) for r in H)/max(len(H),1):.0f}; helpful {sum(words(r['response']) for r in E)/max(len(E),1):.0f}")
        ci = collections.Counter(x["critique_instruction"][:60] for r in H for x in r["rounds"])
        vi = collections.Counter(x["revision_instruction"][:60] for r in H for x in r["rounds"])
        print("critique instructions drawn:", dict(ci))
        print("revision instructions drawn:", dict(vi))

if __name__ == "__main__":
    main(sys.argv[1])
