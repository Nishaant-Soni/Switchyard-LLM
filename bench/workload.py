"""Synthetic benchmark workload with a tunable semantic-duplicate rate `d` (Phase 7 Group 1).

A "duplicate" here is a **semantic near-paraphrase** of an earlier request, not a byte-identical
string — that's what exercises a *semantic* cache (an exact-match dupe would be a trivial hash hit).
Each intent has several phrasings (paraphrase templates) over the same fill slots; two requests are
the *same base* iff they share (intent, fills), regardless of phrasing. Distinct fills/intents are
distinct bases (often semantically close — e.g. "capital of France" vs "capital of Germany" — which
is what makes the threshold sweep's precision/recall curve meaningful).

Everything is seeded, so a given (n, d, seed) reproduces the exact request stream.
"""

import random
from dataclasses import dataclass

from gateway.schemas import ChatCompletionRequest, Message

_COUNTRIES = [
    "France",
    "Germany",
    "Japan",
    "Brazil",
    "Canada",
    "Egypt",
    "Kenya",
    "Norway",
    "Peru",
    "India",
    "Italy",
    "Spain",
    "Chile",
    "Ghana",
    "Nepal",
    "Cuba",
    "Iran",
    "Iraq",
    "Mali",
    "Sweden",
    "Poland",
    "Turkey",
    "Greece",
    "Portugal",
    "Vietnam",
    "Thailand",
    "Morocco",
    "Bolivia",
    "Hungary",
    "Finland",
]
_CONCEPTS = [
    "photosynthesis",
    "inflation",
    "recursion",
    "entropy",
    "osmosis",
    "gravity",
    "democracy",
    "mitosis",
    "encryption",
    "evolution",
    "capitalism",
    "refraction",
    "fermentation",
    "viscosity",
    "plate tectonics",
    "antibiotics",
    "inertia",
    "catalysis",
    "sonar",
    "radar",
    "diffusion",
    "combustion",
    "condensation",
    "erosion",
    "magnetism",
    "photoelectric effect",
    "natural selection",
    "supply and demand",
    "compound interest",
    "machine learning",
]
_TASKS = [
    "bake sourdough bread",
    "change a flat tire",
    "learn to swim",
    "write a resume",
    "plant tomatoes",
    "tie a bowline knot",
    "brew espresso",
    "train for a marathon",
    "set up a VPN",
    "compost at home",
    "fix a leaky faucet",
    "start meditating",
    "format a hard drive",
    "grow basil indoors",
    "start journaling",
    "read sheet music",
    "iron a dress shirt",
    "jump-start a car",
    "sharpen a kitchen knife",
    "prune a rose bush",
]
_WORDS = [
    "hello",
    "thank you",
    "goodbye",
    "water",
    "friend",
    "love",
    "morning",
    "yes",
    "please",
    "help",
    "cat",
    "dog",
    "book",
    "house",
    "sun",
    "moon",
    "tree",
    "river",
    "mountain",
    "coffee",
    "bread",
    "music",
    "family",
    "school",
    "money",
    "time",
    "night",
    "food",
    "road",
    "city",
    "peace",
    "dream",
    "fire",
    "wind",
    "star",
    "flower",
    "bird",
    "snow",
    "rain",
    "ocean",
]
_LANGS = [
    "French",
    "Spanish",
    "German",
    "Japanese",
    "Italian",
    "Hindi",
    "Portuguese",
    "Dutch",
    "Swedish",
    "Korean",
    "Polish",
    "Greek",
]
_PAIRS = [
    ("TCP", "UDP"),
    ("RAM", "a hard drive"),
    ("HTTP", "HTTPS"),
    ("a virus", "a bacterium"),
    ("weather", "climate"),
    ("stocks", "bonds"),
    ("an alligator", "a crocodile"),
    ("espresso", "drip coffee"),
    ("RNA", "DNA"),
    ("a CPU", "a GPU"),
    ("mass", "weight"),
    ("speed", "velocity"),
    ("a comet", "an asteroid"),
    ("fog", "mist"),
    ("jam", "jelly"),
]
_TOPICS = [
    "the ocean",
    "autumn leaves",
    "a thunderstorm",
    "city lights",
    "a mountain",
    "the moon",
    "old books",
    "morning coffee",
    "a river",
    "winter snow",
    "fireflies",
    "the desert",
    "a lighthouse",
    "spring rain",
    "a quiet forest",
]

# Each intent = paraphrase templates + a list of fill rows (a "base" = one intent + one row).
_INTENTS = [
    {
        "templates": [
            "What is the capital of {place}?",
            "Which city is the capital of {place}?",
            "Tell me the capital city of {place}.",
            "What's {place}'s capital?",
        ],
        "fills": [{"place": p} for p in _COUNTRIES],
    },
    {
        "templates": [
            "What is {thing}?",
            "Can you explain what {thing} is?",
            "Define {thing}.",
            "Give me a simple definition of {thing}.",
        ],
        "fills": [{"thing": t} for t in _CONCEPTS],
    },
    {
        "templates": [
            "How do I {task}?",
            "What's the best way to {task}?",
            "What are the steps to {task}?",
            "How can I {task}?",
        ],
        "fills": [{"task": t} for t in _TASKS],
    },
    {
        "templates": [
            "How do you say '{word}' in {lang}?",
            "Translate '{word}' into {lang}.",
            "What is '{word}' in {lang}?",
            "Give me the {lang} word for '{word}'.",
        ],
        "fills": [{"word": w, "lang": lang} for w in _WORDS for lang in _LANGS],
    },
    {
        "templates": [
            "What's the difference between {a} and {b}?",
            "How does {a} differ from {b}?",
            "Compare {a} and {b}.",
            "{a} vs {b}: what's the difference?",
        ],
        "fills": [{"a": a, "b": b} for a, b in _PAIRS],
    },
    {
        "templates": [
            "Write a short poem about {topic}.",
            "Compose a brief poem on {topic}.",
            "Give me a little poem about {topic}.",
            "Can you write a few lines of poetry about {topic}?",
        ],
        "fills": [{"topic": t} for t in _TOPICS],
    },
]


@dataclass
class WorkloadItem:
    request: ChatCompletionRequest
    base_id: str  # canonical (intent, fill-row) key; paraphrases of one base share this
    should_hit: bool  # True iff this base already appeared earlier in the stream
    completion_tokens: int  # the base's (fixed) answer length, for cost accounting


def _all_bases() -> list[tuple[int, int]]:
    return [(ii, ri) for ii, intent in enumerate(_INTENTS) for ri in range(len(intent["fills"]))]


def distinct_base_count() -> int:
    """Total number of distinct bases the corpus can produce (novels draw from this pool)."""
    return len(_all_bases())


def generate_workload(
    n: int, d: float, seed: int, model: str = "llama-3.3-70b-versatile"
) -> list[WorkloadItem]:
    """Build `n` requests where ~`d` of them (after the first) are semantic paraphrases of an
    earlier request. Novels draw unique bases from a shuffled pool; if the pool is exhausted a novel
    slot falls back to a duplicate (so the effective rate can only rise — keep `n` below
    `distinct_base_count()` / (1 - d) to hit the target `d`)."""
    rng = random.Random(seed)
    pool = _all_bases()
    rng.shuffle(pool)

    seen: list[str] = []
    seen_meta: dict[str, tuple[dict, dict, int]] = {}  # base_id -> (intent, fills, comp_tokens)
    items: list[WorkloadItem] = []

    for _ in range(n):
        want_dup = bool(seen) and rng.random() < d
        if not want_dup and pool:
            intent_idx, row_idx = pool.pop()
            intent = _INTENTS[intent_idx]
            fills = intent["fills"][row_idx]
            base_id = f"{intent_idx}:{row_idx}"
            completion_tokens = rng.randint(60, 400)
            seen.append(base_id)
            seen_meta[base_id] = (intent, fills, completion_tokens)
            phrasing = 0  # canonical phrasing on first appearance
            should_hit = False
        else:
            base_id = rng.choice(seen)
            intent, fills, completion_tokens = seen_meta[base_id]
            phrasing = rng.randrange(len(intent["templates"]))  # a paraphrase (maybe phrasing 0)
            should_hit = True

        text = intent["templates"][phrasing].format(**fills)
        request = ChatCompletionRequest(model=model, messages=[Message(role="user", content=text)])
        items.append(WorkloadItem(request, base_id, should_hit, completion_tokens))

    return items
