# Safety prompts

The prompt sets for the safety training pools, the held-out sets and the safety test. File names below are relative to this folder.

Four categories mirror the paper's. Speciesist abuse (demeaning animals or the people who care about them) is the harassment analog and illegal harm to animals is the illegal analog; both are training categories. Poisoning and harmful experimentation is the chemical and biological analog and misinformation about animals is the misinformation analog; both are held out. As in the paper, narrow models trained on speciesist abuse are tested on poisoning and harmful experimentation, and narrow models trained on illegal harm are tested on misinformation.

- safety_test_240.jsonl: 60 prompts per category. Real prompts come from Anthropic's red-team attempts data (the data the paper used); the rest were written by hand. No templates.
- poison_B_eval_100.jsonl and misinfo_B_eval_100.jsonl: the held-out sets, 100 prompts each. No templates.
- Within each 60-prompt block and each held-out set, prompts are spread evenly over five animal groups: companion, farmed, aquatic, wild, laboratory.
- illegal_A_candidates_700.jsonl, speciesist_A_candidates_552.jsonl, poison_train_140.jsonl, misinfo_train_140.jsonl: training pools. Real and hand-written prompts come first and templates fill the remainder; the source of each prompt is in its `source` field.
- source/real_prompts.json: the real prompts used, with their indices in the red-team file. written/: the hand-written prompts. build_report.json and diagnostics.txt: counts per source and group, and the results of the checks in verify_pools.py.

Every prompt names an animal. Every misinformation claim appears once, and the training claims are disjoint from the held-out claims. The poisoning and harmful experimentation block is split evenly (30 and 30 in the safety test, 50 and 50 in the held-out set, equal across the five animal groups), and no experimentation prompts sit in the trained illegal-harm pool. Misinformation prompts are limited to verifiable empirical falsehoods that cause physical or welfare harm, not value judgments. Hand-written prompts vary in register (questions, terse lowercase asks, requests with a false premise or a leading agreement) so they do not all open with "Write a…"; scripts/verify_pools.py reports the rate.

## Rebuilding

    python3 scripts/build_pools.py
    python3 scripts/verify_pools.py

To rerun the extraction of real candidates: python3 scripts/extract_real_candidates.py red_team_attempts.jsonl.gz out.json
