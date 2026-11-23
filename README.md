# Animal framework fine-tuning

Does the ethical framework used to write fine-tuning data change what an open language model believes about animals?

This repository holds a conceptual replication of Del Pinal et al. (2026), "Emergent alignment and the projectability of ethical personas" ([arXiv 2606.09475](https://arxiv.org/abs/2606.09475)). That paper fine-tuned a small model on answers written under each of four ethical frameworks. It then tested whether the model took on the framework's outlook on topics it never saw in training. We follow the paper's method step for step and change one thing: the subject matter moves from general safety to animals.

## The question

A model can be trained on answers that treat animals according to one ethical framework. Two frameworks are compared here.

- **Welfarist.** Right and wrong depend on the consequences for the welfare of every animal affected, as in Peter Singer's *Animal Liberation*.
- **Rights-based.** Some things may not be done to an animal whatever the benefits, as in Tom Regan's *The Case for Animal Rights*.

We ask three things.

1. Does a model trained on welfarist answers end up with different views about animals than a model trained on rights-based answers?
2. Does that difference carry over to animal topics the model was never trained on?
3. Does each model hold its views when someone pushes back, and do the two models give way in different ways?

Question 3 is not in the original paper. It is added at test time only, so the replication stays intact. The no-pressure condition is the replication.

The study does not try to decide which framework is correct. It measures what each framework does to a model.

## Results

The 12 models were trained, tested and judged. Two findings replicate the paper and two do not. Harm rates count only answers a judge labeled harmful or safe. HM7B is the helpful-only starting model. Two models judge the answers: the Claude Sonnet 4.5 judge and the Gemini 3.8 Flash judge.

| Finding | Result | Replicates the paper? |
| --- | --- | --- |
| Narrow training lowers harm on the held-out category | HM7B: 71% harmful on held-out poisoning, 72% on held-out misinformation. Trained models: 15-24% and 9-14% under the Claude Sonnet 4.5 judge; 8-15% and 6-10% under the Gemini 3.8 Flash judge | Yes |
| Models take on the paper's ethical personas | On the paper's 360 statements, welfarist models agree with utilitarian items 72-81% of the time and rights-based models 30-46% | Yes |
| The frameworks separate on 40 animal-ethics statements | Scored against their own framework's answers, welfarist models match 32.5-40% and rights-based models 60-72.5%. But the two answer keys are exact opposites, and both groups give the rights-expected answer 52-73% of the time (HM7B: 45-50%). The two groups do not separate | No |
| Welfarist and rights-based models differ on held-out harm | Poisoning: welfarist models are more harmful under the Gemini 3.8 Flash judge, not under the Claude Sonnet 4.5 judge. Misinformation: no separation under either judge | Depends on the judge |

The Claude Sonnet 4.5 judge returned no label for about a quarter of the held-out poisoning answers, so its rates on that set rest on about three quarters of the answers. The Gemini 3.8 Flash judge labeled almost all of them. Both judges are reported everywhere, together with the share of answers each labeled and the range the rate could fall in with no assumption about the unlabeled ones.

The numbers are in `results/`: `animal_analysis_claude.json` and `animal_analysis_gemini.json` (harm rates, statement measures and pushback transitions), `paper_tests.json` (the paper's significance tests), `judge_agreement.json` (agreement between the two judges), `statements_turn1_gemini.json`, `persona_scores.txt` and `baseline_estimates.json`. Raw model answers are not included, because the untuned model's answers to harmful requests contain harmful content.

## What is the same as the paper and what is not

| Step | Del Pinal et al. | This study |
| --- | --- | --- |
| Base model | Mistral-7B-v0.1 | Same |
| Training method | LoRA fine-tuning on 1,000 samples per model, 500 helpful and 500 safety | Same, with 500 animal safety samples |
| Frameworks | Consequentialist, Deontological, Virtuous, Subordinate | Welfarist and rights-based replace the first two. Virtuous and Subordinate are unchanged |
| Prompts for the safety samples | Red-team prompts in categories such as harassment and illegal activity | Animal versions of the same categories |
| Models trained | 4 frameworks x 3 scopes (broad, two narrow pairs) = 12 | Same 12 |
| Evaluation | Safety test, plus a 360-statement persona test | Same, plus a 60-statement animal test and a pushback condition |

[docs/method.md](docs/method.md) has the full pipeline, every deviation from the paper with its reason, the scoring rules, and the checks run before any training.

## What is in this repository

| Path | Contents |
| --- | --- |
| `constitutions/` | All four constitutions (`principles.json` per framework, and readable copies for the welfarist and rights-based ones). The welfarist and rights-based ones mirror their counterparts in the paper article for article; the virtuous and subordinate ones are the paper's own text |
| `statements/animal_statements.csv` | The 60 animal statements with the answer each constitution implies |
| `pushbacks/pushbacks.csv` | The 36 pushbacks used in the pressure condition, tagged by direction and type, and by whether a factual claim is true or which moral view a moral one reflects |
| `docs/method.md` | Pipeline, deviations, scoring and controls |
| `docs/pressure_design.md` | Design of the pushback condition |
| `docs/analysis_rules.md` | How answers are labeled and scored, and what counts as a difference between models |
| `results/` | Analysis outputs, judge agreement, persona scores and baseline estimates |
| `safety_prompts/` | Training pools, held-out sets and the safety test, with a README |
| `scripts/` | Prompt builders and checks, the training, generation, evaluation and judge scripts, and `scripts/analysis/` for the analyses |
| `configs/` | Training and inference settings |

The prompt sets are in `safety_prompts/` and the scripts that build and check them are in `scripts/`. The training, generation and evaluation scripts in `scripts/` are copied from the authors' repository with the small changes listed in `docs/method.md`; that code is under the Apache License 2.0 and each file keeps its original header.

## The animal test statements

The paper's persona test uses general ethics statements and has no animal content. We add 60 agree or disagree statements about animals, written in the same format. Twenty are in a welfarist register, twenty in a rights register, and twenty are common ground where both frameworks give the same answer. The common-ground statements check for a model that simply agrees with everything. For each framework, 30 expected answers are Yes and 30 are No, matching the paper's half-reversed design.

The statements were written for this study and reviewed by the author. No human raters were used. Two language models, given only the constitutions and not our labels, were asked whether each constitution implies each statement. Neither gave the opposite answer to any of our 120 labels (60 statements, two constitutions each). Both answered AMBIGUOUS for at least one constitution on 10 statements, so the 50 remaining are reported as a secondary, judge-supported analysis. The full 60 stay the main analysis.

## The pushbacks

After a model answers a statement, a second turn shows one pushback and asks the same statement again. Each pushback makes a single claim. The 36 are split into six cells of six.

| | Factual, true | Factual, false | Moral |
| --- | --- | --- | --- |
| Toward permissive (animals matter less) | 6 | 6 | 6 |
| Toward strict (animals matter more) | 6 | 6 | 6 |

The factual claims are tagged true or false, so a model that updates on true claims and holds against false ones can be told apart from one that agrees with whoever pushes. The moral claims are tagged welfarist, rights or neutral, two of each per direction, so that neither framework's wording is favored.

The split by direction and type is this study's own design. It builds on Kim and Flanigan (2026), who compare how often models follow helpful and misleading nudges on factual and moral questions, and on ELEPHANT (Cheng et al., 2025), which measures social sycophancy. Neither paper defines the cells used here.

## Limits

- The paper is a preprint, and each of its models was trained once. Headline comparisons here add two more training runs with different seeds.
- The animal statements have no human rater check.
- The authors' critique and revision prompt file is not published, so ours are written from their description.
- Part of the training pools is template-generated where real prompts were too few. The safety test and held-out sets contain no templates.
- Results come from one 7B model, one training recipe and three runs for each headline model.
- The two judges disagree on what counts as harmful, which is why the held-out comparison between frameworks depends on the judge.
- Results describe what training does to one small model. They do not show which framework a model should use.
- The Alpaca data used for the starting model is licensed for non-commercial use, so trained adapters carry the same limit.

## References

- Del Pinal, G., Lee, Y., McNamara, C., and Pérez Carballo, A. (2026). Emergent alignment and the projectability of ethical personas. arXiv:2606.09475.
- Kim and Flanigan (2026). Right or Wrong, Models Comply: Directional Blindness in LLM Moral Judgment. arXiv:2606.14037.
- Cheng et al. (2025). ELEPHANT: Measuring and understanding social sycophancy in LLMs. arXiv:2505.13995.
- Perez et al. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. arXiv:2212.09251. Source of the 360 persona statements used in the paper.
- Mazeika et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249.
- Bai et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862. Source of the helpful-base prompts.
- Ganguli et al. (2022). Red Teaming Language Models to Reduce Harms. arXiv:2209.07858. Source of the real red-team prompts.
- Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.
- Jiang et al. (2023). Mistral 7B. arXiv:2310.06825.
- Teknium et al. (2024). Hermes 3 Technical Report. arXiv:2408.11857.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
- Singer, P. (1975). *Animal Liberation*.
- Regan, T. (1983). *The Case for Animal Rights*.
