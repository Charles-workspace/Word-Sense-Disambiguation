import math
from collections import defaultdict

# -------------------------------------------------------------------
#  Contextual seed rules for synthetic ambiguous words
# -------------------------------------------------------------------
SEED_RULES = {
    "carspeech": {
        1: ["bomb", "police", "sales", "parked", "exploded"],
        2: ["address", "free", "acceptance", "freedom"],
    },
    "schooloil": {
        1: ["elementary", "students", "public", "high", "board"],
        2: ["prices", "crude", "heating", "gas", "company"],
    },
    "phoneanimal": {
        1: ["call", "mobile", "signal", "telephone"],
        2: ["wild", "species", "habitat", "zoo","rights"],
    },
    "hotelwar": {
        1: ["room", "lobby", "guest", "stay"],
        2: ["world", "troops", "civil", "conflict"],
    },
    "musicforest": {
        1: ["rock", "director", "country", "dance"],
        2: ["trees", "fire", "service", "park", "national"],
    }
}


# -------------------------------------------------------------------
# Apply seed rules: returns labels dict {inst_id → sense}
# -------------------------------------------------------------------
def apply_seed_rules(word, feature_sets):
    labels = {}
    seeds = SEED_RULES[word]

    for inst_id, inst in enumerate(feature_sets[word]):
        feats = inst["features"]

        for sense, cues in seeds.items():
            if any(cue in f for cue in cues for f in feats):
                labels[inst_id] = sense
                break

    return labels


# -------------------------------------------------------------------
# Feature statistics per sense
# -------------------------------------------------------------------
def compute_feature_stats(word, feature_sets, labels):
    stats = defaultdict(lambda: {1: 0, 2: 0})

    for inst_id, inst in enumerate(feature_sets[word]):
        if inst_id not in labels:
            continue

        sense = labels[inst_id]
        for feat in inst["features"]:
            stats[feat][sense] += 1

    return stats


# -------------------------------------------------------------------
# Compute LLR for each feature → decision list
# -------------------------------------------------------------------
def compute_llr(stats):
    dl = []

    for feat, cnt in stats.items():
        c1 = cnt[1] + 0.1
        c2 = cnt[2] + 0.1
        score = abs(math.log(c1 / c2))
        sense = 1 if c1 > c2 else 2
        dl.append((feat, sense, score))

    dl.sort(key=lambda x: x[2], reverse=True)
    return dl


# -------------------------------------------------------------------
# Apply DL to unlabeled instances
# -------------------------------------------------------------------
def apply_decision_list_to_unlabeled(word, feature_sets, labels, dl):
    new_labels = 0

    for inst_id, inst in enumerate(feature_sets[word]):
        if inst_id in labels:
            continue

        feats = set(inst["features"])

        for feat, sense, score in dl:
            if feat in feats:
                labels[inst_id] = sense
                new_labels += 1
                break

    return new_labels


# -------------------------------------------------------------------
# Full training for a single synthetic word
# -------------------------------------------------------------------
def train_single_word(word, feature_sets, max_iter=10):
    print(f"\nTraining decision list for {word}")

    labels = apply_seed_rules(word, feature_sets)

    for it in range(max_iter):
        print(f"  iteration {it}, labeled: {len(labels)}")

        stats = compute_feature_stats(word, feature_sets, labels)
        dl = compute_llr(stats)

        added = apply_decision_list_to_unlabeled(word, feature_sets, labels, dl)
        print(f"    newly labeled: {added}")

        if added == 0:
            break

    return {"labels": labels, "decision_list": dl}


# -------------------------------------------------------------------
# Train all synthetic words
# -------------------------------------------------------------------
def train_decision_lists(feature_sets, targets):
    out = {}
    for word in targets:
        out[word] = train_single_word(word, feature_sets)
    return out