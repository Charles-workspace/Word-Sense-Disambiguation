import pickle
from pathlib import Path
import math
from collections import defaultdict

DATA_DIR = Path("data/synthetic")

SEED_RULES = {
    "carspeech": {
        1: ["road", "driving", "vehicle", "engine", "highway"],
        2: ["address", "freedom", "conference", "remarks"],
    },
    "schooloil": {
        1: ["elementary", "students", "children"],
        2: ["barrel", "crude", "gas", "opec"],
    },
    "phoneanimal": {
        1: ["call", "mobile", "signal", "telephone"],
        2: ["wild", "species", "habitat", "zoo"],
    },
    "hotelwar": {
        1: ["room", "suite", "guest", "stay"],
        2: ["vietnam", "troops", "civil", "iran", "iraq"],
    },
    "musicforest": {
        1: ["sound", "concert", "dance", "singer"],
        2: ["rain", "acre", "park", "lake"],
    }
}

def apply_seed_rules(word, feature_sets):
    labels = {}
    seeds = SEED_RULES[word]

    for inst_id, inst in enumerate(feature_sets[word]):
        feats = inst["features"]
        for sense, cues in seeds.items():
            if any(cue in f for f in feats for cue in cues):
                labels[inst_id] = sense
                break
    return labels


def compute_feature_stats(word, feature_sets, labels):
    stats = defaultdict(lambda: {1: 0, 2: 0})

    for inst_id, inst in enumerate(feature_sets[word]):
        if inst_id not in labels:
            continue
        sense = labels[inst_id]
        for feat in inst["features"]:
            stats[feat][sense] += 1

    return stats


def compute_llr(stats):
    dl = []
    for feat, counts in stats.items():
        c1 = counts[1] + 0.1
        c2 = counts[2] + 0.1
        llr = abs(math.log(c1/c2))
        sense = 1 if c1 > c2 else 2
        dl.append((feat, sense, llr))
    dl.sort(key=lambda x: x[2], reverse=True)
    return dl


def bootstrap(word, feature_sets, labels, dl):
    added = 0

    for inst_id, inst in enumerate(feature_sets[word]):
        if inst_id in labels:
            continue
        feats = set(inst["features"])
        for feat, sense, score in dl:
            if feat in feats:
                labels[inst_id] = sense
                added += 1
                break

    return added


def train_single_word(word, feature_sets):
    print(f"\nTraining {word}")

    labels = apply_seed_rules(word, feature_sets)

    for it in range(10):
        stats = compute_feature_stats(word, feature_sets, labels)
        dl = compute_llr(stats)
        new = bootstrap(word, feature_sets, labels, dl)
        print(f"  iter {it}: {len(labels)} labeled (+{new})")
        if new == 0:
            break

    return {"labels": labels, "decision_list": dl}


if __name__ == "__main__":
    feature_sets = pickle.load(open(DATA_DIR/"feature_sets.pkl", "rb"))
    targets = pickle.load(open(DATA_DIR/"targets.pkl", "rb"))

    model = {}
    for w in targets:
        model[w] = train_single_word(w, feature_sets)

    pickle.dump(model, open(DATA_DIR/"decision_lists.pkl", "wb"))
    print("\nSaved decision_lists.pkl")