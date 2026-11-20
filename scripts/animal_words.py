"""Animal vocabulary: does a prompt name an animal, and which of the five groups does it belong to."""
import re

SPECIES = """songbird songbirds rattlesnake rattlesnakes orcas pangolin pangolins raptor raptors falcon falcons rodent rodents sturgeon jellyfish dog dogs mutt mutts puppy puppies pup pups cat cats kitten kittens kitty kitties pet pets animal animals rabbit rabbits bunny bunnies
hamster hamsters gerbil gerbils guinea mouse mice rat rats ferret ferrets parrot parrots parakeet cockatiel canary bird birds
horse horses pony ponies donkey mule cow cows cattle bull bulls calf calves ox oxen pig pigs hog hogs piglet piglets sow sows boar
sheep lamb lambs goat goats llama alpaca camel chicken chickens hen hens rooster roosters chick chicks turkey turkeys duck ducks
duckling goose geese swan swans pigeon pigeons dove doves quail pheasant livestock poultry flock herd wildlife
fish goldfish betta koi catfish trout salmon carp tuna eel eels minnow minnows shark sharks ray rays seahorse seahorses
octopuses octopi lobster lobsters crab crabs crayfish shrimp prawn octopus squid oyster oysters clam clams mussel snail snails slug slugs
dolphin dolphins whale whales seal seals orca sea otter otters turtle turtles tortoise tortoises
frog frogs toad lizard lizards snake snakes gecko iguana reptile reptiles alligator alligators crocodile crocodiles
elephant elephants tiger tigers lion lions leopard cheetah bear bears wolf wolves coyote coyotes fox foxes deer elk moose
monkey monkeys ape apes chimp chimpanzee gorilla primate primates giraffe zebra rhino rhinos hippo kangaroo koala panda penguin
penguins eagle eagles hawk hawks owl owls crow crows vulture vultures seagull seagulls sparrow sparrows raccoon raccoons
squirrel squirrels skunk skunks possum opossum hedgehog bat bats beaver bee bees ant ants spider spiders insect insects
cockroach cockroaches mosquito wasp hornet beagle beagles hamsters vivarium zebrafish""".split()
SPECIES_RX = re.compile(r'\b(' + '|'.join(sorted(set(SPECIES), key=len, reverse=True)) + r')\b', re.I)


def has_animal(p):
    return bool(SPECIES_RX.search(p))


LAB = re.compile(r'\b(lab|labs|laboratory|research|researchers?|vivisect\w*|experiment\w*|vivarium|iacuc|zebrafish|toxicity|necropsy|beagles?)\b', re.I)
AQUATIC = re.compile(r'\b(fish|fishes|goldfish|betta|koi|catfish|trout|salmon|carp|tuna|eels?|minnows?|shark|sharks|seahorses?|lobsters?|crabs?|crayfish|shrimp|prawns?|octopus|squid|oysters?|clams?|mussels?|dolphins?|whales?|orcas?|seals?|sea lions?|sea turtles?|aquarium|aquariums|reef)\b', re.I)
FARMED = re.compile(r'\b(cows?|cattle|calf|calves|bull|bulls|pigs?|hogs?|piglets?|sows?|sheep|lambs?|goats?|chickens?|hens?|roosters?|chicks?|turkeys?|horses?|ponies|llamas?|livestock|poultry|flock|herd|farm animals?|(?:on|at) (?:a|the|my|his|her) farm|slaughter\w*|cockfight\w*)\b', re.I)
WILD = re.compile(r'\b(elephants?|tigers?|lions?|leopards?|cheetahs?|bears?|wolf|wolves|coyotes?|foxes|fox|deer|elk|moose|monkeys?|apes?|chimps?|gorillas?|primates?|giraffes?|zebras?|rhinos?|kangaroos?|koalas?|pandas?|penguins?|eagles?|hawks?|owls?|crows?|vultures?|seagulls?|ducks?|ducklings?|camels?|geese|goose|swans?|raccoons?|squirrels?|skunks?|possums?|opossums?|hedgehogs?|bats?|wildlife|zoo|poach\w*|endangered|wild|birds?|pigeons?|frogs?|toads?|snakes?|lizards?|alligators?|crocodiles?)\b', re.I)
PET_BIRD = re.compile(r'\bmy (\w+ )?(bird|birds|parrot|parakeet|cockatiel|canary|snake|lizard|frog)\b', re.I)


def group_of(p):
    s = p
    if LAB.search(s):
        return 'lab'
    if re.search(r'\bzoo\b', s, re.I):
        return 'wild'
    if PET_BIRD.search(s) or re.search(r'\bguinea pigs?\b', s, re.I):
        return 'companion'
    if AQUATIC.search(s) and not re.search(r'\b(dog|cat|puppy|kitten)\b', s, re.I):
        return 'aquatic'
    if FARMED.search(s) and not re.search(r'\b(dog|cat|puppy|kitten)\b', s, re.I):
        return 'farmed'
    if WILD.search(s) and not re.search(r'\b(dog|cat|puppy|kitten)\b', s, re.I):
        return 'wild'
    return 'companion'
