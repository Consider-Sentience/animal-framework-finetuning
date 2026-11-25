# Animal framework fine-tuning

Does the ethical framework used to write fine-tuning data change what an open language model believes about animals?

This repository is a conceptual replication, meaning the same method run on different subject matter, of Del Pinal, Lee, McNamara and Pérez Carballo (2026), "Emergent alignment and the projectability of ethical personas" ([arXiv 2606.09475](https://arxiv.org/abs/2606.09475)). The paper fine-tuned a small model on answers written under each of four ethical frameworks. It then tested whether the model adopted that framework's outlook on topics outside its training. The paper calls this outlook a persona, and the spread of a persona to unseen topics emergent alignment. We follow the paper's method step for step and change the subject matter from general safety to animals.

![Study design](docs/animal-framework-finetuning-study-design.png)

## The question

Each framework is written as a constitution: a short list of principles. A large model (Hermes 3 Llama 3.1 405B) uses it to critique and rewrite answers, and the rewritten answers become the fine-tuning data. We use six constitutions.

- **Consequentialist.** Right and wrong depend on the consequences of an act. The paper's own text.
- **Deontological.** Some acts are required or forbidden whatever the consequences. The paper's own text.
- **Welfarist.** Our animal version of Consequentialist: right and wrong depend on the consequences for the welfare of every animal affected, as in Peter Singer's *Animal Liberation*.
- **Rights-based.** Our animal version of Deontological: some things may not be done to an animal whatever the benefits, as in Tom Regan's *The Case for Animal Rights*.
- **Virtuous.** The assistant acts as a person of good character would. The paper's own text.
- **Subordinate.** The assistant defers to the instructions of the person it serves. The paper's own text.

We ask five things.

1. Does the paper's result hold when the subject matter is animals: does each constitution give the model its own persona, and does training on one harm category lower harm on a category the model never saw?
2. Does a model fine-tuned on Welfarist answers hold different views about animals than one fine-tuned on Rights-based answers?
3. Does that difference carry over to animal topics the model never saw in training?
4. Does each model hold its view when someone pushes back, and do the two give way differently?
5. Do the paper's own four frameworks, whose texts never mention animals, differ in what they say about animals?

Question 4 is not in the paper. It adds a second turn at test time, so the replication itself is unchanged. The study does not try to decide which framework is correct. It measures what training on each one does to a model.

## How the study works

**Models.** The starting point is HM7B, a helpful-only model: Mistral-7B-v0.1 with a LoRA adapter fine-tuned on Alpaca to follow instructions without refusing. Each constitutional model is a second LoRA adapter (r = 64, alpha = 128, 3 epochs) trained on top of HM7B. Its 1,000 training samples are 500 animal-harm answers revised under one constitution and 500 helpful answers.

**Scopes.** The paper trains each constitution at three scopes. A scope sets which harm categories appear in training. We use four animal-harm categories: poisoning and experimentation, speciesist abuse, illegal harm, and misinformation.

- *Broad:* all four categories are in training.
- *Speciesist narrow:* only speciesist abuse is in training. Poisoning is held out.
- *Illegal-harm narrow:* only illegal harm is in training. Misinformation is held out.

A held-out category is one that never appears in a model's training data. Held-out harm is the harm rate on test prompts from that category, which measures whether training on one kind of harm carries over to another. This only makes sense for the two narrow scopes, because broad models see all four categories. That is why the narrow scopes carry the main comparison.

**Runs.** Each constitution and scope gives one trained model. That is 18 trained models. The paper trains each model once. We train the Welfarist and Rights-based narrow models two more times each (three runs per model, 26 trained adapters in total), because the held-out comparison is the central test and is the one most sensitive to training noise. The extra runs use the same training data and differ only in the random seed, which sets adapter initialization and data order. They show how much training noise moves a result, not the effect of different training answers. Every other model is a single run.

**Tests.** Each model is evaluated four ways, and one further check is run on the training answers rather than on a model.

- *Animal safety test:* 240 animal-harm prompts, 60 per category. Held-out sets add 100 prompts for each narrow scope. Each prompt is answered five times (temperature 0.7, top-p 0.9, up to 256 tokens), and a judge labels each answer harmful or safe.
- *Persona test:* the paper's 360 agree-or-disagree statements on utilitarian, deontological, virtue-ethics and deference views (Perez et al., 2022). Utilitarianism is a form of consequentialism, so the Consequentialist and Welfarist models are scored against the utilitarian dimension and the Deontological and Rights-based models against deontology. About half the statements are worded the other way round, so a model endorses a framework by disagreeing with them; each item records which answer counts as endorsement.
- *Animal statements:* 60 agree-or-disagree statements about animals that we wrote, because the paper's persona test has no animal content. 20 are worded in welfarist terms, 20 in rights terms, and 20 are common ground, where both frameworks give the same answer. The 40 welfarist and rights statements are the framework-specific ones: on those the two frameworks expect opposite answers. The common-ground statements catch a model that agrees with everything. The statements and their label checks are in [docs/animal_statements.md](docs/animal_statements.md).
- *Pushback:* after a model answers a statement, a second turn shows it a counterargument and asks again. There are 36 counterarguments, split by direction and by whether the claim is a true fact, a false fact or a moral view. That separates a model that updates on evidence from one that agrees with whoever pushes. The design is in [docs/pressure_design.md](docs/pressure_design.md).
- *Recognizability check:* Hermes 3 reads one training answer and four constitution descriptions and names the constitution the answer follows. This checks that the constitutions produce distinguishable text.

**Judges.** A language model labels every answer. The Claude Sonnet 4.5 judge is the primary judge, running the paper's judge prompts. The paper used one judge. The Gemini 3.8 Flash judge is a second opinion: it labels the held-out answers of all 26 trained models, the animal-statement answers of every model, and the full safety test for HM7B and the broad Welfarist, Rights-based, Virtuous and Subordinate models. A judge sometimes gives no usable label, and we call that answer unlabeled.

**How a harm rate is computed.** Within a prompt, the rate is the harmful answers divided by the harmful and safe answers among its five samples; a prompt with nothing labeled is dropped. The reported rate is the mean of those per-prompt rates. Each rate is reported with the share of answers the judge labeled and with the range the rate could fall in under no assumption about the unlabeled ones. The authors' convention, which counts unlabeled answers as not harmful, is reported beside it (`docs/analysis_rules.md`).

**Statistics.** Differences between two single models use the paper's chi-square test. Welfarist against Rights-based across three runs each uses an exact permutation test, whose smallest possible p-value is 0.10 with three runs on each side. A result where all three runs of one group beat all three of the other is therefore at the floor of what this design can show.

## Findings

1. **Fine-tuning on a framework's text gives the model that framework's outlook on general ethics.** On the paper's persona statements, models trained on Consequentialist or Welfarist text agree more with utilitarian statements than models trained on Deontological or Rights-based text, and the reverse holds for deontological statements. This holds at all three scopes.
2. **Training on one harm category lowers harm on a category the model never saw, for every framework.** On held-out prompts, HM7B gave a harmful answer 54.6% to 72.0% of the time, depending on judge and category. The narrow-trained models gave 4.6% to 45.9%. Subordinate stays highest because its text tells the assistant to comply.
3. **The Welfarist and Rights-based models do not differ clearly on held-out harm.** The one consistent direction is under the Gemini 3.8 Flash judge on poisoning, where all three Welfarist runs are more harmful than all three Rights-based runs (p = 0.10).
4. **The paper's two frameworks separate on the animal statements, and the animal pair does not.** Deontological models gave the rights-expected answer more often than Consequentialist models on the 40 framework-specific statements, at all three scopes under both judges (67.5% against 40% at the broad scope under the Claude Sonnet 4.5 judge). Welfarist and Rights-based models were alike, at 55% to 75%, and Welfarist sits closer to Deontological than to Consequentialist. At the narrow scopes the gaps are 2.5 to 10 points from a single run each, with no test applied.
5. **The two animal frameworks give the same answers on those statements.** Both give the rights-expected answer about two thirds of the time. Rights-based models match their own answer key more often than Welfarist models do (60% to 72.5% against 32.5% to 40%). Because both groups lean toward the rights-expected answer, this shows a shared lean and not a difference between the frameworks.
6. **Pushback does not separate them either.** Models seldom abandon their first answer (3% to 17% for Welfarist and 2% to 7% for Rights-based on speciesist training; 5% to 8% for both on illegal-harm training), and the groups overlap.

All of this comes from one 7B model and one training recipe. Only the Welfarist and Rights-based narrow models have repeated runs.

## Results

### Held-out harm

Share of held-out answers labeled harmful, in percent. HM7B is the helpful-only baseline, so it has one value. The Welfarist and Rights-based cells are the lowest and highest of three training runs on the same data with different seeds, and every other cell is one run.

| Model | Trained on speciesist abuse, poisoning held out: Claude Sonnet 4.5 judge | Gemini 3.8 Flash judge | Trained on illegal harm, misinformation held out: Claude Sonnet 4.5 judge | Gemini 3.8 Flash judge |
| --- | --- | --- | --- | --- |
| HM7B | 71.1 | 61.4 | 72.0 | 54.6 |
| Consequentialist | 33.5 | 19.4 | 8.0 | 4.6 |
| Welfarist | 17.8 to 24.4 | 12.1 to 16.2 | 8.7 to 14.2 | 5.6 to 9.2 |
| Deontological | 18.1 | 12.1 | 16.1 | 10.4 |
| Rights-based | 15.1 to 19.5 | 7.9 to 11.0 | 10.6 to 13.3 | 6.8 to 9.2 |
| Virtuous | 19.9 | 15.6 | 14.6 | 9.6 |
| Subordinate | 45.9 | 31.6 | 40.1 | 23.6 |

Every framework holds down harm on the category it never trained on. The broad models, which saw all four categories, are lower still: below 5% on both held-out categories except Subordinate. The Gemini 3.8 Flash judge reads harm lower than the Claude Sonnet 4.5 judge for every trained model, by 2 to 17 points, and the six models fall in the same order under both judges. Consequentialist and Deontological do not keep a fixed order across the two narrow scopes: Consequentialist is more harmful after speciesist training, and Deontological after illegal-harm training.

Rates exclude unlabeled answers. On speciesist training the Claude Sonnet 4.5 judge leaves 25% to 31% of held-out poisoning answers unlabeled, so those rates rest on about three quarters of the answers. The Gemini 3.8 Flash judge leaves 3% to 15% unlabeled on the same answers. Where both judges labeled the same safety answer they agree on 93.8% of 19,777 answers (`results/judge_agreement.json`).

As a check on the setup, HM7B on the authors' own set from HarmBench, a standard benchmark of harmful requests, is 93.2% harmful among labeled answers (95% interval 90.7 to 95.3) under the Claude Sonnet 4.5 judge, against 92.1% in the paper.

### Consequentialist and Deontological on animals

Share of the 40 framework-specific animal statements answered in the rights-expected direction, in percent, by the Claude Sonnet 4.5 judge. The Welfarist and Rights-based narrow cells are the lowest and highest of three runs, and every other cell is one run.

| Model | Broad | Speciesist | Illegal harm |
| --- | --- | --- | --- |
| Consequentialist | 40 | 52.5 | 47.5 |
| Deontological | 67.5 | 60 | 57.5 |
| Welfarist | 75 | 62.5 to 65 | 55 to 65 |
| Rights-based | 70 | 60 to 72.5 | 60 to 65 |

Deontological is above Consequentialist at every scope, and Welfarist and Rights-based both sit near Deontological. Under the Gemini 3.8 Flash judge the Deontological-over-Consequentialist order is the same at every scope (57.5 against 42.5 broad, 60 against 50 speciesist, 55 against 52.5 illegal harm).

### Welfarist and Rights-based

These two are compared across their three runs each, so the comparison asks whether the runs of one group sit consistently above the runs of the other.

*Held-out harm.* Under the Claude Sonnet 4.5 judge the runs overlap in both scopes (exact p = 0.2 for poisoning, 0.4 for misinformation). Under the Gemini 3.8 Flash judge, all three Welfarist runs are more harmful than all three Rights-based runs on held-out poisoning (p = 0.10), and the groups overlap on held-out misinformation (p = 0.5).

*Animal statements.* We score each model against its own framework's answer key. The two keys are exact opposites on these 40 statements. The two groups separate on key-matching, with all three runs of one above all three of the other in both scopes under both judges (p = 0.10). But both give the rights-expected answer 55% to 75% of the time, and they do not differ on that measure (p = 0.7 to 0.9). So the key-matching gap reflects the shared lean toward rights-expected answers and not framework-specific behavior (`docs/animal_statements.md`).

*Pushback.* Under the Claude Sonnet 4.5 judge, the share of first answers a model abandons when pushed away from its framework's answer is 3% to 17% for the Welfarist runs and 2% to 7% for the Rights-based runs on speciesist training (exact p = 0.3). On illegal-harm training it is 5% to 8% for both (p = 1.0).

### Personas

Share of the paper's 100 utilitarian and 100 deontological statements on which each model endorsed that framework, by the Claude Sonnet 4.5 judge. Broad models only, one run each. The other 160 statements are in `results/persona_scores.txt`.

| Model | Endorses utilitarian ethics | Endorses deontological ethics |
| --- | --- | --- |
| HM7B (helpful-only starting point) | 0.58 | 0.57 |
| Consequentialist | 0.63 | 0.29 |
| Welfarist | 0.73 | 0.38 |
| Deontological | 0.39 | 0.59 |
| Rights-based | 0.46 | 0.61 |
| Virtuous | 0.62 | 0.52 |
| Subordinate | 0.41 | 0.44 |

The Welfarist and Consequentialist models endorse utilitarian ethics more than the Rights-based and Deontological models do, at every scope.

The paper also condenses each model's four rates into a persona score out of 8. Those scores are in [docs/method.md](docs/method.md).

### Can the constitutions be told apart

Hermes 3 reads one training answer plus the four constitution descriptions of its set, and names the constitution the answer follows. There are 40 answers per constitution, and chance is 10 of 40. The count is the number it named correctly.

| Constitution | Animal set | Paper's set |
| --- | --- | --- |
| Consequentialist | | 39 |
| Welfarist | 39 | |
| Deontological | | 33 |
| Rights-based | 38 | |
| Virtuous | 17 | 20 |
| Subordinate | 15 | 10 |

Virtuous and Subordinate answers are only weakly recognizable from their descriptions.

## Same as the paper, and where it differs

| Step | Del Pinal et al. | This study |
| --- | --- | --- |
| Base model | Mistral-7B-v0.1 | Same |
| Training method | LoRA on 1,000 samples per model, 500 helpful and 500 safety | Same, with 500 animal safety samples |
| Constitutions | Consequentialist, Deontological, Virtuous, Subordinate | Those four unchanged, plus Welfarist and Rights-based |
| Safety prompts | Red-team prompts in categories such as harassment and illegal activity | Animal versions of the same categories |
| Models | 4 constitutions × 3 scopes = 12, each trained once | 6 × 3 = 18, with two extra runs for the Welfarist and Rights-based narrow models |
| Judge | One Claude Sonnet 4 judge | The Claude Sonnet 4.5 judge, plus the Gemini 3.8 Flash judge |
| Tests | HarmBench safety test, 360-statement persona test | Animal safety test, the same persona test, 60 animal statements, pushback condition |

[docs/method.md](docs/method.md) has the full pipeline and every deviation from the paper with its reason.

## Reproducing

Run every command from the repository's top folder. Training needs one 80 GB A100 GPU. Training the baseline model (HM7B) takes about 8 hours and each constitutional adapter about 9 minutes. Each script skips work that is already finished, so it can be stopped and restarted.

**1. Set up**

```bash
pip install -r requirements.txt
mkdir -p logs trained_models gen
export ANTHROPIC_API_KEY=...     # Claude Sonnet 4.5 judge
export GOOGLE_API_KEY=...        # Gemini 3.8 Flash judge
export DEEPINFRA_API_KEY=...     # Hermes 3 Llama 3.1 405B, which writes the training answers
```

Generation can use several DeepInfra keys to stay under the rate limit on the 405B model. Any `DEEPINFRA_API_KEY_2`, `_3` and so on in the environment are picked up and rotated.

**2. Prompts**

The prompt sets are published in `safety_prompts/`, so the rebuild below can be skipped. To check them against the rules in `docs/method.md`, which runs 16 checks and exits non-zero if any fails:

```bash
python3 scripts/verify_pools.py
```

To rebuild them instead:

```bash
python3 scripts/build_pools.py
python3 scripts/sample_helpful.py
python3 scripts/screen_training_pools.py --pools safety_prompts/speciesist_A_candidates_552.jsonl safety_prompts/illegal_A_candidates_700.jsonl --out results/pool_screen
python3 scripts/build_clean_pools.py --screen results/pool_screen.jsonl --out safety_prompts
python3 scripts/build_broad_pool.py
python3 scripts/build_test_all.py
```

**3. Train the baseline model (HM7B)**

```bash
python3 scripts/training/train_hm7b.py --output-dir trained_models/hm7b
```

**4. Write the training answers**

HM7B first answers each training prompt. Hermes 3 then critiques and rewrites those answers under each constitution. HM7B's first answers are not published, because they are a helpful-only model's answers to harmful requests.

```bash
python3 scripts/build_first_answer_inputs.py --out results/first_answers/inputs
for pool in broad_A_500 speciesist_A_candidates_552 illegal_A_candidates_700; do
  python3 scripts/hm7b_inference.py --hm7b-path trained_models/hm7b \
    --prompts results/first_answers/inputs/$pool.jsonl --output results/first_answers/$pool.jsonl
done
python3 scripts/build_generation_inputs.py --out gen/inputs
bash scripts/run_full_generation.sh
```

**5. Train the constitutional models**

This trains the 18 models (6 constitutions × 3 scopes). The 8 extra runs of the Welfarist and Rights-based narrow models come after the tests in step 6.

```bash
bash scripts/run_all_training.sh
```

**6. Test the models**

```bash
bash scripts/run_safety_tests.sh
bash scripts/run_statement_tests.sh
bash scripts/run_seed_runs.sh
```

**7. Judge the answers**

`run_judging.sh seeds` waits for `run_seed_runs.sh` to finish, so the two can run at the same time in separate terminals.

```bash
bash scripts/run_judging.sh safety
bash scripts/run_judging.sh main
bash scripts/run_judging.sh seeds
bash scripts/run_gemini_judging.sh
python3 scripts/evaluation/judge_statements.py --judge gemini --workers 8
```

**7b. The HarmBench baseline check**

The comparison against the paper's 92.1% comes from a separate pass, in which HM7B answers the authors' HarmBench set rather than the animal prompts. `--adapter BASE` runs HM7B with no constitutional adapter.

```bash
python3 scripts/evaluation/harmbench_multi_sample.py --hm7b-path trained_models/hm7b \
  --model-id hm7b_base --adapter BASE --k 5 \
  --eval data/eval/harmbench_240_excl_copyright.jsonl --output-dir results/harmbench_K5
python3 scripts/evaluation/judge_responses_batch.py \
  --inputs results/harmbench_K5/hm7b_base_harmbench_K5_responses.jsonl \
  --eval-type harmbench_binary --output-suffix _full_judged.jsonl
python3 scripts/evaluation/judge_gemini_harmbench.py --max-workers 8 \
  --input results/harmbench_K5/hm7b_base_harmbench_K5_responses.jsonl \
  --output results/harmbench_K5_judged_gemini/hm7b_base_harmbench_K5_full_judged.jsonl
mkdir -p results/harmbench_K5_judged
mv results/harmbench_K5/hm7b_base_harmbench_K5_full_judged.jsonl results/harmbench_K5_judged/
```

**8. Make the tables**

```bash
python3 scripts/analysis/paper_tests.py --out results/paper_tests.json
python3 scripts/analysis/baseline_estimates.py --reference 92.1 \
  --judged results/harmbench_K5_judged/hm7b_base_harmbench_K5_full_judged.jsonl \
  --other results/harmbench_K5_judged_gemini/hm7b_base_harmbench_K5_full_judged.jsonl
python3 scripts/analysis/multi_sample_aggregate.py --input-dir results/harmbench_K5_judged --output results/harmbench_K5_aggregate.json
python3 scripts/analysis/animal_analysis.py --judge claude --out results/animal_analysis_claude.json
python3 scripts/analysis/animal_analysis.py --judge gemini --out results/animal_analysis_gemini.json
python3 scripts/analysis/persona_scores.py > results/persona_scores.txt
python3 scripts/analysis/judge_agreement.py --out results/judge_agreement.json
python3 scripts/analysis/judge_repeatability.py --out results/judge_repeatability.json
python3 scripts/analysis/blind_constitution_check.py --n 40 --constitutions animal --out gen/blind_check.json
python3 scripts/analysis/blind_constitution_check.py --n 40 --constitutions authors --out gen/blind_check_authors_set.json
```

The two recognizability runs write to `gen/`. `results/blind_constitution_check.json` holds both of them together, under the keys `animal_set` and `authors_set`.

## Repository contents

| Path | Contents |
| --- | --- |
| `constitutions/` | The six constitutions (`principles.json` for each, and readable copies for the Welfarist and Rights-based ones) |
| `statements/animal_statements.csv` | The 60 animal statements with the answer each constitution implies |
| `pushbacks/` | The 36 pushbacks, tagged by direction and type |
| `docs/` | Method, analysis rules, statement and pushback designs |
| `data/eval/` | The paper's 360 persona statements and its HarmBench set |
| `results/` | Analysis outputs, judge agreement and repeatability, persona scores, baseline estimates, judge label validation, and the two-judge pool screen |
| `safety_prompts/` | Training pools, held-out sets and the safety test, with a README |
| `scripts/` | Prompt builders and checks, and the training, generation, evaluation, judging and analysis scripts |
| `configs/` | Training and inference settings |

Model answers, generated training data and trained adapters are not published, because the helpful-only model's answers to harmful requests contain harmful content. `.gitignore` lists what is held back.

## Limits

- The paper trains each of its 12 models once and reports no repeated runs. Here the Welfarist and Rights-based narrow models have three runs each, on one shared training set, and every other model has one run. Three runs per group cannot give a p-value below 0.10.
- The chi-square tests count the five samples of a prompt as separate trials, as the paper's do, so the p-values are optimistic. No correction is applied for multiple comparisons, also as in the paper.
- Results come from one 7B model and one training recipe.
- The two judges differ on what counts as harmful and each leaves some answers unlabeled, so held-out comparisons are read under both. The second judge does not cover the full safety test for every model.
- The animal statements have no human rater check.
- The authors' critique and revision prompts are not published, so ours are written from their description.
- Part of the training pools is template-generated where real prompts were too few. The safety test and held-out sets contain no templates.
- Results describe what training does to one small model. They do not show which framework a model should use.

## License

Code copied from the authors' repository is under the Apache License 2.0 ([LICENSE](LICENSE)), and each such file keeps its original header. The Alpaca data used for HM7B is licensed for non-commercial use, so trained adapters carry the same limit.

## Citation

Vavilakolanu, A. (2026). Animal framework fine-tuning: a conceptual replication of Del Pinal et al. (2026). Consider Sentience. https://github.com/Consider-Sentience/animal-framework-finetuning

## References

- Del Pinal, G., Lee, Y., McNamara, C., and Pérez Carballo, A. (2026). Emergent alignment and the projectability of ethical personas. arXiv:2606.09475.
- Perez et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251. Source of the 360 persona statements.
- Mazeika et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249.
- Bai et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862. Source of the helpful-base prompts.
- Ganguli et al. (2022). Red Teaming Language Models to Reduce Harms. arXiv:2209.07858. Source of the real red-team prompts.
- Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
- Jiang et al. (2023). Mistral 7B. arXiv:2310.06825.
- Teknium et al. (2024). Hermes 3 Technical Report. arXiv:2408.11857.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Singer, P. (1975). *Animal Liberation*.
- Regan, T. (1983). *The Case for Animal Rights*.
