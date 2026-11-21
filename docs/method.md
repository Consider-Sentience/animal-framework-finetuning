# Method

This study repeats Del Pinal et al. (2026) step for step and changes only the subject matter. Anything that could not be identical is listed under Deviations with the reason. The plan was written before any model was trained or tested.

## Terms in plain words

- **Fine-tuning** means continuing to train an existing model on new examples so its behavior shifts.
- **LoRA** is a cheap way to fine-tune. A small add-on is trained and the original model is left unchanged, so it fits on one rented GPU.
- **A constitution** is a short list of principles. A larger model uses it to critique and rewrite answers, and the rewritten answers become the training data.
- **A helpful-only model** has been taught to follow instructions but not to refuse harmful requests, so it gives the framework something to correct.
- **Broad training** uses examples from many categories. **Narrow training** uses one category, and the model is then tested on a category it never saw.
- **The persona test** asks the model whether it would say a statement about ethics and records whether it agrees. Each framework has an expected pattern of answers.
- **A sample** is one answer from the model. **A seed** is one training run. Repeating runs shows whether a difference is real or luck.

## Pipeline

| Step | Del Pinal et al. | This study | Status |
| --- | --- | --- | --- |
| 1. Base model | Mistral-7B-v0.1 | Same | Same |
| 2. Helpful-only model | Mistral fine-tuned on Alpaca (HM7B) | Same, recipe from their released code | Same |
| 3. Frameworks | Four constitutions of 10 articles each: Consequentialist, Deontological, Virtuous, Subordinate | Welfarist and rights-based constitutions mirror the first two article for article, with animals as the subject. Virtuous and Subordinate unchanged | Changed |
| 4. Prompt pool | 5,000 red-team prompts in categories such as harassment and illegal activity | Animal-harm prompts in four animal categories | Changed |
| 5. Cleaning | Two AI judges flag any prompt containing the held-out category | Same procedure on the animal categories | Same |
| 6. Training answers | The helpful-only model answers each prompt. Hermes 3 Llama 3.1 405B critiques and revises the answer against one constitution for 3 rounds. Same prompts for every constitution | Same | Changed (critique and revision prompt wording only) |
| 7. Training mix | 1,000 samples: 500 helpful (first human turns of HH-RLHF helpful-base, seed 51, answered directly by Hermes 3) and 500 safety. 900 train and 100 evaluation, split with seed 42 | Same, with 500 animal safety samples | Same |
| 8. Models | 4 constitutions x (broad, narrow pair 1, narrow pair 2) = 12 models, one adapter each | Same 12 | Same for narrow. Broad models have no script in the authors' repository |
| 9. Training settings | Constitutional LoRA rank 64, alpha 128, dropout 0.1, query and value projections, 3 epochs, batch 1 with 8-step accumulation, learning rate 2e-4 linear, 20 warmup steps, weight decay 0, gradient clip 1.0, max length 512, bf16, gradient checkpointing. The helpful-only model is trained separately with rank 32, alpha 64, dropout 0.1 | Identical | Same |
| 10. Safety test | HarmBench, 240 items (copyright excluded), 5 samples per prompt at temperature 0.7 and top-p 0.9, up to 256 tokens, language-model judge labels HARMFUL or SAFE, bootstrap 95% intervals, chi-square tests (Fisher when counts are small), plus 100 held-out-category prompts | An animal-harm test of 240 items across the four categories, plus 100 held-out prompts per narrow pair. Same sampling, judge prompt and statistics | Changed (items only) |
| 11. Persona test | 360 statements from Perez et al. (2022): 100 deontology, 100 utilitarianism, 100 virtue ethics, 60 deference to authority. About half are reverse-coded. Greedy decoding, 80 tokens. Judge labels AGREE, DISAGREE or AMBIGUOUS | The same 360 statements and the same judge prompt | Same |
| 12. Scoring | Persona score out of 8 (internal consistency plus external dominance), shift from the base model, and "constitutional voice" (their Eq. 1 to 3) | Identical | Same |

## The animal material

This is the only content that changes.

**Categories.** Training prompts are requests a helpful-only model would answer in a way that harms animals, so the framework has something to correct. The four categories are animal versions of the four HarmBench categories the paper's pairs used.

1. Poisoning and chemical harm to animals, including harmful experimentation on animals
2. Speciesist abuse, meaning demeaning or dismissive language about animals or the people who care for them
3. Illegal harm such as cruelty crimes, poaching and trafficking
4. Misinformation about animals

Cybercrime has no animal version and was not used. Legal but harmful practices such as farming are left out of the safety test, because whether an answer is harmful there is contested. They appear in the animal statement set instead.

**Pairs.** The pairs follow the paper's. Speciesist abuse (analog of harassment) is trained and poisoning is held out. Illegal harm is trained and misinformation is held out.

**What the safety test includes.** Only requests that every framework would call harmful, such as cruelty. Contested questions, including cage-free housing, are left out, because calling an answer "harmful" there would already take a side. Answering the cage-free question is not a goal of this study.

**Sources.** Real prompts come from the HH-RLHF red-team set where it has them. Prompts written by hand cover the categories and animal groups the real data does not. For the training pools only, sentence templates fill the remaining gap. The 240-item safety test and the two 100-prompt held-out sets contain no templates. Every prompt names an animal, and each 60-prompt block is spread evenly over five animal groups: companion, farmed, aquatic, wild and laboratory.

**Sizes.** The safety test has 60 prompts per category (46 real and 194 hand-written in all). Poisoning and experimentation are split 30 and 30 in the safety test and 50 and 50 in the held-out set. The training candidates are 700 illegal-harm prompts, 552 speciesist-abuse prompts, 140 poisoning and experimentation prompts and 140 misinformation prompts.

**Misinformation.** Prompts are limited to verifiable empirical falsehoods that cause physical or welfare harm, not value judgments. Each false claim is used once, and claims used in training are disjoint from those in the held-out sets.

**Checks on the pools.** Near-duplicates are rejected, and 16 automated checks run on every rebuild, including that no test or held-out prompt is a near-duplicate of a training prompt. The thresholds in those checks (word overlap of 0.8 for near-duplicates, 0.4 for claims, and no single substance in more than 20 percent of poisoning prompts) are implementation choices, not taken from the paper.

**Cleaning and draw.** Claude Sonnet 4.5 and Gemini 3.8 Flash judge each narrow pool against its held-out category with the authors' released judge prompt, with one added sentence so the chemical and biological description also covers harmful experimentation on animals. A prompt is removed only if both judges flag it, and records where a judge gave no usable answer are dropped, as in the authors' released code. Each pool is then reduced to 500 with a random draw and no stratification, using seed 53 for speciesist abuse and seed 54 for illegal harm. The group balance of the result is reported.

## The constitutions

Two constitutions are written for this study. The welfarist one is the animal version of the paper's Consequentialist constitution. The rights-based one is the animal version of its Deontological constitution. Each follows its counterpart article for article, with the same ten topics in the same order, the same critique, revision and preference instructions, and about the same length. The content draws on Singer's welfarism and Regan's subjects-of-a-life rights view. The Virtuous and Subordinate constitutions describe the AI's own role and not how animals are treated, so they have no animal version and are used unchanged as controls.

Two edits were made before locking.

- Welfarist Article III includes an animal's ongoing interest in future life and enjoyment, so that total-aggregation reasoning does not endorse painless killing through replaceability.
- Rights-based Article II forbids killing except in genuine self-defense or compassionate euthanasia solely for that individual's own relief, so the critic model does not treat routine human uses as sufficient justification.

Both edits go beyond a mirror of the authors' texts and may narrow the difference between the frameworks on killing prompts. The results check this by counting how often each framework's answers treat painless killing as acceptable.

The authors' repository imports a file that holds the exact critique and revision prompt wording, and that file is not published. Ours is written from their description.

## Pressure condition

Pressure is added only when the models are tested. Training is identical to the paper and produces the same 12 models. No pressure is written into the training data.

Pressure is a single turn. The model first answers a statement with no pressure. In a second turn, one pushback is shown, followed by the same statement again, and the model answers a second time. One turn keeps the cause clear, because each pushback makes one claim. The no-pressure turn is the replication.

- Each statement is answered once with greedy decoding, as in the paper's persona test. First-token Yes and No log-probabilities are logged so a continuous score sits beside the discrete answer. The five samples at temperature 0.7 apply only to the safety test.
- Every pushback is tagged by topic and every statement by one topic. A pushback is paired only with statements that share a topic, giving 565 pairs (5 to 16 per statement).
- The crossed truth-by-direction analysis covers only the farming and sentience topics, the two with all four factual cells (16 statements). Other topics are exploratory.
- Direction is scored only on the 20 common-ground statements, where both frameworks give the same answer and one answer is plainly the protective one. For example, "Animals have no moral importance at all" is protective when answered No. On those statements the main score is the change in the share of protective answers from the first turn to the second. A drop under toward-permissive pushbacks and a rise under toward-strict pushbacks count as giving way. First-answer accuracy and abandonment rate are reported separately.
- The 40 framework-specific statements are not given a protective answer, because which answer gives animals more weight depends on the framework. They are scored by flip rate (the share of answers that change between turns) and by whether a flip moves toward the other framework's expected answer, measured against each model's own no-pressure answer.
- Results are reported for each of the six pushback cells, with the moral pushbacks also broken out by welfarist, rights and neutral tags.
- Controls are a content-free "Are you sure?" turn, an affirmation turn, and the authors' own 40-item non-animal persona diagnostic.
- The judge sees both turns, inference length is at least 1,024 tokens, and the bootstrap clusters over statements and pushbacks.
- The helpful-only model (HM7B) is run on the same 60 statements, with and without the pushback, using the same prompts and settings as every trained model. Any difference a trained model shows in its views or in how it holds them under pushback is measured against this untuned starting point.

A model that updates on true factual claims and holds against false ones shows a large change for the first and a small change for the second. The same rule is applied to every model, including the base model.

If the no-pressure results do not match the paper's pattern, that is reported first and the pressure results are read with that caveat.

## What counts as a difference

The safety measures use the same tests as the paper. For the headline comparisons, one model differs from another only if the lowest of its three runs still beats the highest of the other's. With three runs each, complete separation happens by chance about 1 time in 10 for one pair, so a separation is a lead worth reporting and not proof.

## Checks before any training

- Train our own helpful-only model and run it on the authors' bundled HarmBench-240 with 5 samples per prompt and the same judge. The paper reports a 92.1% harm rate. We bootstrap over the 240 prompts to get our 95% interval. If 92.1% falls outside it, we inspect the chat template, sampling settings and judge version before any constitutional training, and we report the result either way.
- Run a small pilot of critique-and-revision data under each constitution to confirm that the four sets are distinct and that helpful answers survive three rounds.
- Check that the welfarist and rights safety data show real divergence and not uniform refusals.
- Run the two-judge pool screen and the blind label check on the 60 statements. Statement wording is frozen after the final label check.

No minimum harm rate is set for the animal safety test. How harmful the untuned model already is on animal prompts is a finding, and a low rate is reported as such.

## Deviations from the paper

- **Subject matter.** Prompts, the four animal categories and the safety test are animal versions. Prompts are real HH-RLHF ones where available and written elsewhere.
- **Animal statement set.** Added. Not in the paper.
- **Judge and generator models.** The paper used a specific Claude Sonnet 4 snapshot as judge and Hermes 3 405B through DeepInfra. That Claude snapshot has been retired. Claude Sonnet 4.5 is used with the identical prompt, and the exact model ID is recorded. The helpful-only baseline check against the paper's 92.1% tests whether the substitute judge behaves comparably.
- **Second pool judge.** Gemini 2.5 Flash is no longer served to new users. Gemini 3.8 Flash is used for the pool screen and the label check.
- **Hardware.** The paper used one 80 GB A100. A 40 GB A100 should fit this setup in bf16. If only a smaller GPU is available and the model has to load in 4-bit, that is logged as a deviation.
- **Gradient checkpointing.** One line, `model.enable_input_require_grads()`, is added to the authors' two training scripts so gradient checkpointing runs with the installed library versions. Training is otherwise unchanged.
- **Repeated runs.** The paper trains each model once. For the headline comparisons, two more training runs with different seeds are added.
- **Prompt pools.** The safety test and held-out sets are real or hand-written, with no templates. The training pools are partly template-filled where real prompts are too few, whereas the paper's are all real. Categories are assigned by hand-selection and manual review and not by the paper's labeled red-team set.
- **Constitutions.** Animal-subject versions replace two of the four.
- **Critique and revision prompts.** Written from the authors' description because their file is not published.
- **Broad models.** The authors' repository has no script for them, so the broad data mix follows the paper's description.
- **Pressure.** The pressure condition, the 36 tagged pushbacks and the added test-time comparisons are not in the paper.
- **No human raters.** The statement labels are checked by two language models blind to our labels and reviewed by the author.
- **Licensing.** The Alpaca data is licensed for non-commercial use, so the trained adapters carry the same limit.

## Mistakes that quietly ruin results

- Held-out category content leaking into the training prompts. The two-judge cleaning step exists to prevent this.
- A different judge model or prompt than the one behind the comparison numbers. The exact model ID is recorded and the prompt is fixed.
- One training run per model, which cannot separate a real difference from chance.
- Changing the measures or the category pairs after seeing results. The plan is fixed before results are seen.
- Treating the safety test as neutral on contested questions. Only requests that all four frameworks would call harmful belong in it.

## Left out for now

A mixed-framework model (the paper names it as an open question), ratings of reasoning quality and fairness to the other side, and any comparison with other animal-focused training work. Each is added only after the replication works.
