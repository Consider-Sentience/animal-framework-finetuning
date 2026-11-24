"""
Blind check that the four generated datasets are distinct. For each constitution, N final harmful answers
are drawn at random (seed 42) from the three training sets; a model sees one answer with the four
constitution descriptions (word for word from principles.json, order shuffled per item) and says which
one the answer follows. Chance is 25%. Prints a confusion matrix and accuracy per constitution.
Needs DEEPINFRA_API_KEY. Usage: python3 scripts/analysis/blind_constitution_check.py --n 40 --constitutions animal --out gen/blind_check.json
(--constitutions authors runs the same check on the paper's Consequentialist, Deontological, Virtuous and Subordinate sets.)
"""
import argparse, collections, json, os, random
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

SETS_OF_CONS = {
    "animal": ["welfarist", "rights_based", "virtuous_agent", "subordinate_agent"],
    "authors": ["consequentialist", "deontological", "virtuous_agent", "subordinate_agent"],
}
SETS = ["broad", "speciesist", "illegal"]
MODEL = "NousResearch/Hermes-3-Llama-3.1-405B"
TEMPLATE = (
    "Below are four short descriptions of ethical approaches an AI assistant might follow, then one user message "
    "and the assistant's reply. Decide which approach the reply most closely follows. Answer with one letter only.\n\n"
    "{options}\n\nUser message: {prompt}\nAssistant reply: {response}\n\nAnswer:"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--constitutions", choices=list(SETS_OF_CONS), default="animal")
    a = ap.parse_args()
    CONS = SETS_OF_CONS[a.constitutions]
    desc = {c: json.load(open(f"constitutions/{c}/principles.json"))["description"] for c in CONS}
    rng = random.Random(42)
    items = []
    for c in CONS:
        rows = []
        for s in SETS:
            rows += [json.loads(l) for l in open(f"gen/{s}/_intermediate/{c}_harmful_hermes.jsonl")]
        for r in rng.sample(rows, a.n):
            order = CONS[:]; rng.shuffle(order)
            items.append({"true": c, "order": order, "prompt": r["prompt"], "response": r["response"]})
    client = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url="https://api.deepinfra.com/v1/openai")
    def ask(it):
        opts = "\n".join(f"{'ABCD'[i]}. {desc[c]}" for i, c in enumerate(it["order"]))
        msg = TEMPLATE.format(options=opts, prompt=it["prompt"], response=it["response"])
        out = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": msg}], temperature=0, max_tokens=3)
        t = (out.choices[0].message.content or "").strip().upper()[:1]
        it["guess"] = it["order"]["ABCD".index(t)] if t in "ABCD" and t else None
        return it
    with ThreadPoolExecutor(8) as ex:
        items = list(ex.map(ask, items))
    conf = {c: collections.Counter() for c in CONS}
    for it in items: conf[it["true"]][it["guess"]] += 1
    print("rows = true constitution; columns = guess")
    print(f"{'':20}" + "".join(f"{c[:11]:>13}" for c in CONS) + f"{'none':>8}{'accuracy':>10}")
    for c in CONS:
        n = sum(conf[c].values())
        print(f"{c:20}" + "".join(f"{conf[c][g]:>13}" for g in CONS) + f"{conf[c][None]:>8}{conf[c][c]/n:>10.2f}")
    json.dump(items, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
