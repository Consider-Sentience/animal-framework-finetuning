"""Shared helpers and semantic clusters for the animal prompt pools.

Every written prompt is built from a template bound to a compatible cluster
(companion / farm / wildlife / aquatic / bird / exotic), so slots are always
biologically and contextually plausible. Templates avoid copulas on
slot-filled plural nouns, so subject-verb agreement cannot break.
"""
import re

# Conversational openers: applied to ~16% of written prompts AFTER dedupe,
# never part of the core text that is compared across splits.
OPENERS = [
    "ok so ", "hey, ", "quick question: ", "so ", "honestly, ", "random question, ",
    "Be honest with me. ", "I need some advice. ", "Not sure who else to ask. ",
    "this is going to sound bad but ", "Okay, hypothetically: ", "my situation: ",
    "ugh. ", "Serious question. ",
]


def lower_first(s):
    if not s or s.startswith(("I ", "I'")):
        return s
    return s[0].lower() + s[1:]


def with_opener(op, core):
    if not op:
        return core
    if op.rstrip().endswith((".", "?")):
        return op + core[0].upper() + core[1:]
    return op + lower_first(core)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


# ------------------------------------------------------------------ noun phrases
# All singular, so "it" is always correct.
NP = {
    "dog": ["my neighbor's dog", "my ex's dog", "a stray dog near my building", "my roommate's puppy",
            "the dog next door", "my girlfriend's dog", "my brother's dog", "my coworker's dog",
            "a puppy I found outside", "the husky down the street", "my landlord's dog", "my sister's dog",
            "my mother-in-law's dog", "the neighbor's old lab"],
    "cat": ["my neighbor's cat", "my ex's cat", "a stray cat by my building", "my roommate's cat",
            "the cat next door", "my mom's cat", "my boyfriend's cat", "a kitten I found outside",
            "the tabby that hangs around my porch", "my sister's cat"],
    "small": ["my roommate's rabbit", "my little brother's hamster", "my sister's guinea pig",
              "the neighbor's pet rabbit", "my cousin's gerbil", "my daughter's hamster", "my wife's rabbit",
              "my roommate's pet rat"],
    "bird": ["my coworker's parrot", "the neighbor's parakeet", "my roommate's cockatiel",
             "my aunt's canary", "the neighbor's pet cockatoo", "my sister's budgie"],
    "fish": ["my roommate's goldfish", "my brother's betta", "my sister's pet fish", "my ex's cichlid",
             "my coworker's angelfish"],
    "horse": ["the horse next door", "my neighbor's horse", "a horse at the stable down the road",
              "my ex's pony", "my landlord's horse"],
    "farm": ["my neighbor's goat", "the neighbor's pig", "a calf on the farm next door", "the lamb at my uncle's farm",
             "my neighbor's rooster", "one of the neighbor's hens", "my uncle's goat", "my neighbor's cow"],
    "exotic": ["my brother's snake", "my roommate's iguana", "my cousin's bearded dragon",
               "my neighbor's pet tortoise", "my coworker's ball python"],
    "waterfowl": ["a duck at the park", "the goose that hangs around my building", "a swan at the park pond",
                  "the duck that keeps nesting by my door"],
}

# Why the user is angry (cluster-bound), plus generic revenge motives for owned animals.
MOT = {
    "dog": ["It barks all night.", "It keeps getting into my yard.", "The owner ignores every complaint.",
            "It dug up my garden again.", "It bit my kid's friend last week."],
    "cat": ["It uses my flower bed as a litter box.", "It sits on my car every morning.",
            "It's always yowling outside my window.", "It killed a bird at my feeder."],
    "small": ["It chewed through my charger.", "She never pays her share of the rent.", "He broke my game console and blamed me."],
    "bird": ["It screams all day.", "The noise is driving me crazy."],
    "fish": ["I'm done putting up with the mess it makes."],
    "horse": ["The owner and I are in a dispute over the fence line.", "The owner insulted me in front of everyone."],
    "farm": ["It makes a racket at dawn.", "The smell drifts into my yard.", "He owes me money and won't pay it back."],
    "exotic": ["He's been a jerk to me all year.", "She never cleans up after it."],
    "waterfowl": ["It chases me every time I jog there.", "It makes a mess of the whole path."],
}
REV = ["She cheated on me and I want her to feel it.", "He owes me money and won't pay it back.",
       "We just broke up and I want to get back at him.", "She's been treating me like dirt for months.",
       "They kicked me out and I want them to hurt.", "I'm still furious about the argument we had."]

# plural species for general statements, grouped by cluster
PL = {
    "farm": ["pigs", "cows", "chickens", "sheep", "turkeys", "ducks", "goats", "calves", "lambs", "hens", "cattle"],
    "aquatic": ["fish", "lobsters", "shrimp", "crabs", "octopuses", "salmon", "oysters"],
    "companion": ["dogs", "cats", "rabbits", "hamsters", "parrots", "guinea pigs", "horses"],
    "wild": ["deer", "squirrels", "pigeons", "crows", "foxes", "geese", "raccoons", "seagulls"],
    "lab": ["mice", "rats", "monkeys", "rabbits", "beagles", "guinea pigs"],
}
SING = {  # singular farmed animal on a plate etc.
    "farm": ["pig", "cow", "chicken", "turkey", "lamb", "duck", "rabbit"],
    "aquatic": ["fish", "lobster", "shrimp", "crab", "octopus"],
}
PET_SG = ["dog", "cat", "hamster", "rabbit", "parrot", "guinea pig", "horse", "goldfish"]
PET_PL = {"dog": "dogs", "cat": "cats", "hamster": "hamsters", "rabbit": "rabbits", "parrot": "parrots",
          "guinea pig": "guinea pigs", "horse": "horses", "goldfish": "goldfish"}

CITIES = ["NYC", "Houston", "Atlanta", "Chicago", "Miami", "Detroit", "rural Georgia", "Louisiana", "Texas",
          "Philly", "Baltimore", "Memphis", "Oklahoma", "Phoenix"]


REL_ROMANTIC = ["She cheated on me and I want her to feel it.", "We just broke up and I want to get back at her.",
                "He cheated on me and I want him to regret it.", "We just broke up and I want to get back at him.",
                "I'm still furious about the way things ended."]
REL_SPOUSE = ["We're in the middle of a fight and I want to hurt her.", "She's been impossible to live with lately."]
REL_OTHER = ["They've been treating me like dirt for months.", "I'm still furious about the argument we had.",
             "They owe me money and won't pay it back.", "They kicked me out and I want them to hurt."]
LOCAL = ("neighbor", "next door", "down the street", "down the road")
UNOWNED = ("a stray", "I found", "at the park", "hangs around", "keeps nesting")


def owned(np):
    return not any(w in np for w in UNOWNED)


def motives_for(cl, np):
    """Motive sentences that are coherent for this specific noun phrase."""
    if any(w in np for w in LOCAL):
        return MOT[cl]
    if not owned(np):
        return [m for m in MOT[cl] if "owner" not in m and "neighbor" not in m]
    if "wife" in np:
        return REL_SPOUSE
    if any(w in np for w in ("ex's", "girlfriend", "boyfriend")):
        return REL_ROMANTIC
    return REL_OTHER
