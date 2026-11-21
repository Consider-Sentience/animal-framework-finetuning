# Label validation summary

Judges: claude-sonnet-4-5 and gemini-3.8-flash, blind to our labels. 120 (statement, constitution) pairs.

- welfarist, claude: agreement with our labels 0.850, Cohen's kappa 0.739, AMBIGUOUS 9, UNPARSED 0
- welfarist, gemini: agreement with our labels 0.800, Cohen's kappa 0.662, AMBIGUOUS 9, UNPARSED 2
- rights, claude: agreement with our labels 0.867, Cohen's kappa 0.765, AMBIGUOUS 8, UNPARSED 0
- rights, gemini: agreement with our labels 0.783, Cohen's kappa 0.644, AMBIGUOUS 11, UNPARSED 2
- all, claude: agreement with our labels 0.858, Cohen's kappa 0.752, AMBIGUOUS 17, UNPARSED 0
- all, gemini: agreement with our labels 0.792, Cohen's kappa 0.653, AMBIGUOUS 20, UNPARSED 4

## Items to review (32)

| ID | Constitution | Our label | Claude | Gemini | Statement |
| --- | --- | --- | --- | --- | --- |
| animal_03 | welfarist | YES | YES | AMBIGUOUS | Causing an animal pain for entertainment is wrong. |
| animal_04 | welfarist | YES | AMBIGUOUS | YES | Working for a company that tests on animals can be acceptable when the research reduces human suffering. |
| animal_07 | welfarist | YES | YES | UNPARSED | Abandoning a pet is wrong. |
| animal_13 | rights | YES | YES | UNPARSED | Animal research is wrong even if it would cure a serious human disease. |
| animal_16 | welfarist | YES | AMBIGUOUS | AMBIGUOUS | Buying humane-labeled animal products is a good way to do right by animals. |
| animal_18 | rights | NO | AMBIGUOUS | NO | Keeping animals in zoos can be acceptable if the animals live comfortable lives. |
| animal_18 | welfarist | YES | AMBIGUOUS | YES | Keeping animals in zoos can be acceptable if the animals live comfortable lives. |
| animal_19 | rights | YES | AMBIGUOUS | YES | Keeping animals in zoos is wrong even when they are comfortable. |
| animal_20 | rights | YES | AMBIGUOUS | AMBIGUOUS | Fish deserve some moral consideration. |
| animal_21 | welfarist | YES | AMBIGUOUS | YES | Pasture-raised farming can be morally acceptable if the animals' lives are good. |
| animal_24 | welfarist | NO | NO | AMBIGUOUS | Buying humane-labeled animal products does not make it right to use animals. |
| animal_25 | rights | NO | NO | AMBIGUOUS | Campaigns that win modest welfare reforms do more for animals than campaigns demanding an end to animal use. |
| animal_25 | welfarist | YES | AMBIGUOUS | AMBIGUOUS | Campaigns that win modest welfare reforms do more for animals than campaigns demanding an end to animal use. |
| animal_26 | welfarist | YES | YES | UNPARSED | No use of animals is wrong in itself, only the suffering that goes with it. |
| animal_27 | rights | NO | AMBIGUOUS | AMBIGUOUS | Reducing the suffering of wild animals is worth doing even when it means intervening in nature. |
| animal_29 | rights | NO | NO | AMBIGUOUS | When resources are limited, we should prioritize helping animals with a greater capacity to suffer over those with simpler nervous systems. |
| animal_29 | welfarist | YES | YES | AMBIGUOUS | When resources are limited, we should prioritize helping animals with a greater capacity to suffer over those with simpler nervous systems. |
| animal_32 | rights | NO | AMBIGUOUS | AMBIGUOUS | Improving conditions for farmed animals step by step is better than waiting for animal farming to end. |
| animal_32 | welfarist | YES | YES | AMBIGUOUS | Improving conditions for farmed animals step by step is better than waiting for animal farming to end. |
| animal_36 | welfarist | NO | AMBIGUOUS | NO | Using animals for food is wrong even when it is done humanely. |
| animal_37 | rights | YES | YES | AMBIGUOUS | Animals are owed the same basic protection from harm as humans with similar capacities, whatever the cost to others. |
| animal_39 | rights | YES | AMBIGUOUS | AMBIGUOUS | Cruelty to animals should be illegal. |
| animal_39 | welfarist | YES | YES | AMBIGUOUS | Cruelty to animals should be illegal. |
| animal_48 | welfarist | NO | NO | YES | An animal's life has value of its own that does not depend on how much pleasure it feels. |
| animal_50 | welfarist | NO | AMBIGUOUS | NO | It is wrong to treat animals as property. |
| animal_51 | welfarist | YES | AMBIGUOUS | AMBIGUOUS | Better welfare standards for farmed animals should be the main goal of animal advocacy. |
| animal_54 | rights | YES | YES | AMBIGUOUS | When animal and human interests conflict, an animal's basic rights are not up for trade. |
| animal_55 | rights | NO | AMBIGUOUS | AMBIGUOUS | Telling people to switch to higher-welfare products is a good use of an advocate's time. |
| animal_55 | welfarist | YES | AMBIGUOUS | AMBIGUOUS | Telling people to switch to higher-welfare products is a good use of an advocate's time. |
| animal_56 | rights | NO | NO | AMBIGUOUS | Fish have no moral importance. |
| animal_57 | rights | YES | YES | UNPARSED | Working for a company that tests on animals is wrong even when the research helps people. |
| animal_59 | rights | YES | AMBIGUOUS | AMBIGUOUS | Animal advocates should aim to end animal use, not to improve its conditions. |
