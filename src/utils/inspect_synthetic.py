import pickle
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/synthetic")

feature_sets = pickle.load(open(DATA_DIR / "feature_sets.pkl", "rb"))
gold_labels = pickle.load(open(DATA_DIR / "gold_labels.pkl", "rb"))
targets = pickle.load(open(DATA_DIR / "targets.pkl", "rb"))

def top_features(word, sense, n=50):
    counts = Counter()
    for inst, label in zip(feature_sets[word], gold_labels[word]):
        if label == sense:
            counts.update(inst["features"])
    return counts.most_common(n)

for word in targets:
    print(f"\n=== {word.upper()} ===")
    
    print("\nTop features for SENSE 1:")
    for f, c in top_features(word, 1):
        print(f"{f:30} {c}")

    print("\nTop features for SENSE 2:")
    for f, c in top_features(word, 2):
        print(f"{f:30} {c}")