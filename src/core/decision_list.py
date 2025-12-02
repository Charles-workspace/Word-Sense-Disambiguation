
import pickle
from pathlib import Path
import math
from collections import defaultdict
from src.preprocess import load_preprocessed_data

DATA_DIR = Path("data/preprocessed")


SEED_RULES = {
    "plant": {
        1: ["chemical", "industrial", "factory"],
        2: ["soil", "roots", "harvest", "botanical"]
    },
    "bank": {
        1: ["river", "shore", "water"],
        2: ["loan", "deposit", "account"]
    },
    "seal": {
        1: ["animal", "sea"],
        2: ["official", "stamp"]
    },
    "issue": {
        1: ["problem", "controversy", "concern", "dispute", "matter", "debate"],
        2: ["edition", "copy", "publication", "magazine", "newspaper", "journal"]
    },
    "bill": {
        1: [
            "payment", "invoice", "due", "amount", "cost", "pay", "fees","service", "charge", "owe", "expenses", "receipt"],
        2: [
            "law", "senate", "congress", "house", "legislation", "committee","amendment", "act", "voted", "proposal", "parliament", "passed"]
    }
    
}


def apply_seed_rules(word, feature_sets):
    labels = {}  # instance_id -> sense

    seed_rules = SEED_RULES.get(word, {})

    for inst_id, inst in enumerate(feature_sets[word]):
        feats = inst["features"]
        for sense, keywords in seed_rules.items():
            for kw in keywords:
                if any(kw in f for f in feats):
                    labels[inst_id] = sense
                    break
            if inst_id in labels:
                break

    return labels

def compute_feature_stats(word, feature_sets, labels):
    stats = defaultdict(lambda: defaultdict(int))

    for inst_id, inst in enumerate(feature_sets[word]):
        if inst_id not in labels:
            continue
        sense = labels[inst_id]

        for f in inst["features"]:
            stats[f][sense] += 1

    return stats


def compute_llr(stats):
    decision_list = []

    for f, counts in stats.items():
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)

        # Add smoothing
        p1 = (c1 + 0.1) / (c1 + c2 + 0.2)
        p2 = (c2 + 0.1) / (c1 + c2 + 0.2)

        llr = math.log(p1 / p2)
        pred_sense = 1 if llr > 0 else 2
        decision_list.append((f, pred_sense, abs(llr)))

    # Sort by absolute LLR
    decision_list.sort(key=lambda x: x[2], reverse=True)

    return decision_list


# def apply_decision_list(word, feature_sets, labels, decision_list):
#     changes = 0
#     for inst_id, inst in enumerate(feature_sets[word]):
#         if inst_id in labels:
#             continue  # already labeled

#         feats = set(inst["features"])
#         for f, sense, score in decision_list:
#             if f in feats:
#                 labels[inst_id] = sense
#                 changes += 1
#                 break

#     return changes

def apply_decision_list(word, feature_sets, labels, decision_list_rules):
    """
    Apply decision list to ALL instances for a given word.
    Return a list where predictions[i] is the predicted sense for instance i.
    """

    predictions = []
    instances = feature_sets[word]  # list of dictionaries

    for inst in instances:
        feats = inst["features"]
        predicted = None

        # check rule list in descending LLR (already sorted)
        for feature, sense, score in decision_list_rules:
            if feature in feats:
                predicted = sense
                break

        predictions.append(predicted)

    return predictions


def bootstrap(word, feature_sets):
    labels = apply_seed_rules(word, feature_sets)

    for iteration in range(10):
        print(f"Bootstrapping iteration {iteration}")

        feature_stats = compute_feature_stats(word, feature_sets, labels)
        decision_list = compute_llr(feature_stats)

        new_labels = apply_decision_list(word, feature_sets, labels, decision_list)

        print(f"  new labels = {new_labels}")
        if new_labels == 0:
            break

    return labels, decision_list


if __name__ == "__main__":
    # Load data
    corpus, token_index = load_preprocessed_data(DATA_DIR)


    with open(DATA_DIR / "feature_sets.pkl", "rb") as f:
        feature_sets = pickle.load(f)
    with open(DATA_DIR / "targets.pkl", "rb") as f:
        targets = pickle.load(f)

    model_output = {}

    for word in targets:
        print(f"\n=== Training decision list for '{word}' ===")
        labels, dlist = bootstrap(word, feature_sets)
        model_output[word] = {
            "labels": labels,
            "decision_list": dlist
        }

    with open(DATA_DIR / "decision_lists.pkl", "wb") as f:
        pickle.dump(model_output, f)

    print("\nSaved decision_lists.pkl")