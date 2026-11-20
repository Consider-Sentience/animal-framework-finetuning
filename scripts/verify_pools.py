"""Checks on the built prompt pools. Exits non-zero if a hard constraint fails."""
import json, glob, re, sys, os, collections, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from animal_words import has_animal
from pg_common import norm
ROOT = os.path.dirname(HERE) + '/'
files = {os.path.basename(f)[:-6]: [json.loads(l) for l in open(f)] for f in sorted(glob.glob(ROOT + 'safety_prompts/*.jsonl'))}
toks = lambda p: set(re.findall(r'[a-z]+', p.lower()))
jacc = lambda a, b: len(a & b) / max(1, len(a | b))
fails = []
def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' -- ' + detail if detail else ''))
    if not ok: fails.append(name)

allrows = [(f, r) for f, rs in files.items() for r in rs]
check('every prompt names an animal', all(has_animal(r['prompt']) for f, r in allrows),
      str([r['prompt'] for f, r in allrows if not has_animal(r['prompt'])][:3]))
test = files['safety_test_240']
check('safety test has 240 prompts, 60 per category', len(test) == 240 and all(c == 60 for c in collections.Counter(r['category'] for r in test).values()))
check('safety test: 12 per group per category', all(c == 12 for c in collections.Counter((r['category'], r['group']) for r in test).values()) and len(set((r['category'], r['group']) for r in test)) == 20)
for n in ('poison_B_eval_100', 'misinfo_B_eval_100'):
    rs = files[n]
    check(f'{n}: 20 per group', len(rs) == 100 and all(c == 20 for c in collections.Counter(r['group'] for r in rs).values()) and len(set(r['group'] for r in rs)) == 5)
check('no templates in safety test or held-out evaluation sets', all(r['source'] != 'template' for r in test + files['poison_B_eval_100'] + files['misinfo_B_eval_100']))
check('neighbours in the safety test never share a category', all(test[i]['category'] != test[i - 1]['category'] for i in range(1, len(test))))
# exact and near duplicates inside each category, across every file
bycat = collections.defaultdict(list)
for f, r in allrows: bycat[r['category']].append((f, r))
dups = []
for c, rs in bycat.items():
    seen = {}
    for f, r in rs:
        k = norm(r['prompt'])
        if k in seen: dups.append((c, r['prompt'][:70]))
        seen[k] = 1
check('no exact duplicates inside a category (all files)', not dups, str(dups[:3]))
held = [r for r in test + files['poison_B_eval_100'] + files['misinfo_B_eval_100']]
hardnear = []
for (a, b) in itertools.combinations(held, 2):
    if a['category'] == b['category'] and jacc(toks(a['prompt']), toks(b['prompt'])) >= 0.8:
        hardnear.append((a['prompt'][:60], b['prompt'][:60]))
check('no near-duplicates (word overlap >= 0.8) among test and held-out prompts', not hardnear, str(hardnear[:2]))
# test/held-out vs training leakage
leak = []
train = [r for f, rs in files.items() if f not in ('safety_test_240', 'poison_B_eval_100', 'misinfo_B_eval_100') for r in rs]
for a in held:
    ta = toks(a['prompt'])
    for b in train:
        if b['category'] == a['category'] and jacc(ta, toks(b['prompt'])) >= 0.8:
            leak.append((a['prompt'][:60], b['prompt'][:60]))
check('no test/held-out prompt is a near-duplicate of a training prompt', not leak, str(leak[:2]))
# misinformation: each claim once; training claims disjoint from held-out claims
mis = files['misinfo_train_140'] + files['misinfo_B_eval_100'] + [r for r in test if r['category'] == 'misinfo']
claims = [r['claim'] for r in mis if r.get('claim')]
check('every misinformation claim appears once', len(claims) == len(set(claims)) and len(claims) == len(mis), f'{len(claims)} claims for {len(mis)} prompts')
held_claims = [toks(r['claim']) for r in files['misinfo_B_eval_100'] + [r for r in test if r['category'] == 'misinfo']]
close = [r['claim'] for r in files['misinfo_train_140'] if any(jacc(toks(r['claim']), h) >= 0.4 for h in held_claims)]
check('training misinformation claims are disjoint from held-out claims', not close, str(close[:2]))
# poisoning vs harmful experimentation balance (held-out chem/bio analog)
ptest = [r for r in test if r['category'] == 'poison']
peval = files['poison_B_eval_100']
sub_ok = all(c == 6 for c in collections.Counter((r['group'], r['subtype']) for r in ptest).values()) and len(set((r['group'], r['subtype']) for r in ptest)) == 10
sub_ok2 = all(c == 10 for c in collections.Counter((r['group'], r['subtype']) for r in peval).values()) and len(set((r['group'], r['subtype']) for r in peval)) == 10
check('safety test poison block: 6 poisoning + 6 experimentation in every animal group (30/30)', sub_ok)
check('poison held-out set: 10 poisoning + 10 experimentation in every animal group (50/50)', sub_ok2)
# the trained illegal pools must not contain experimentation content (it belongs to the held-out category)
exp_kw = re.compile(r'\b(experiment\w*|vivisect\w*|test(ing)? (a|my|the|new|homemade)\b.*\b(on|in)\b|dissect\w*|surger(y|ies)|invasive|toxicity test\w*|draize|clinical trial|drug trial)\b', re.I)
ill_exp = [r['prompt'][:80] for f in files if f.startswith('illegal_A_candidates') for r in files[f] if exp_kw.search(r['prompt'])]
ill_exp += [r['prompt'][:80] for r in test if r['category'] == 'illegal' and exp_kw.search(r['prompt'])]
check('no experimentation content in the trained illegal-harm pool or safety block', not ill_exp, str(ill_exp[:4]))
# register: share of prompts that start with an imperative 'Write/Draft/Compose/Give me/Make/Create' by category and source
imp = re.compile(r'^(write|draft|compose|give me|make|create|come up with|help me write)\b', re.I)
print('imperative-opening rate (test + held-out):')
for c in ('illegal', 'speciesist', 'poison', 'misinfo'):
    for src in ('hh_rlhf_real', 'hand_written'):
        ps = [r['prompt'] for r in held if r['category'] == c and r['source'] == src]
        if ps: print('  ', c, src, len(ps), round(sum(bool(imp.match(p)) for p in ps) / len(ps), 2), 'median words', sorted(len(p.split()) for p in ps)[len(ps) // 2])
# substance spread (poison): report only
subs = ['antifreeze', 'rat poison', 'rodenticide', 'bleach', 'weed killer', 'herbicide', 'pesticide', 'chocolate', 'xylitol', 'pool chemical', 'poison',
        'pentobarbital', 'euthanasia', 'ketamine', 'acetaminophen', 'insulin', 'ibuprofen', 'copper', 'salt', 'drug', 'overdose', 'toxin', 'chemical', 'sedative']
pois = [r['prompt'].lower() for r in test if r['category'] == 'poison'] + [r['prompt'].lower() for r in files['poison_B_eval_100']]
cnt = {s: sum(s in p for p in pois) for s in subs}
print('poison substance mentions (test + held-out, n=%d):' % len(pois), {k: v for k, v in cnt.items() if v})
check('no single named substance dominates the poison prompts (max 20%)', max(cnt[s] for s in subs if s not in ('poison', 'chemical', 'toxin', 'drug')) <= 0.2 * len(pois))
# held-out category content must not appear in training categories for the narrow models
hold_kw = re.compile(r'\b(poison\w*|antifreeze|rat poison|toxin\w*|toxic|overdose|lethal dose|laced|tainted)\b', re.I)
mis_kw = re.compile(r'\b(misinformation|disinformation|false claim|fake study|persuasive (article|post)|convince (people|them) that)\b', re.I)
leak2 = [(f, r['prompt'][:70]) for f in files if f.startswith(('illegal_A_candidates', 'speciesist_A_candidates')) for r in files[f] if hold_kw.search(r['prompt']) or mis_kw.search(r['prompt'])]
print('possible held-out-category content in the narrow training pools (to be screened by the two-judge step):', len(leak2))
for x in leak2[:12]: print('   ', x)
# diversity report
def distinct5(ps):
    g = []
    for p in ps:
        w = re.findall(r"[a-z0-9']+", p.lower()); g += [tuple(w[i:i + 5]) for i in range(len(w) - 4)]
    return len(set(g)) / max(1, len(g))
print('distinct 5-gram ratio by file and source:')
for f, rs in files.items():
    by = collections.defaultdict(list)
    for r in rs: by[r['source']].append(r['prompt'])
    print('  ', f, {s: (len(ps), round(distinct5(ps), 2)) for s, ps in by.items()})
sys.exit(1 if fails else 0)
