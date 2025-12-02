import pickle
from pathlib import Path
from src.preprocess import load_preprocessed_data


DATA_DIR = Path("data/preprocessed")

def extract_features_for_instance(corpus, instance, window=3):
    doc_id = instance["doc_id"]
    sent_id = instance["sent_id"]
    tok_id = instance["tok_id"]

    sentence = corpus[doc_id][sent_id]

    features = []

    # Collocation 1 - immediate left and right
    if tok_id > 0:
        left1 = sentence[tok_id - 1]
        features.append(f"LEFT1={left1}")
    if tok_id + 1 < len(sentence):
        right1 = sentence[tok_id + 1]
        features.append(f"RIGHT1={right1}")

    # Collocation 2 - second-level context
    if tok_id > 1:
        left2 = sentence[tok_id - 2]
        features.append(f"LEFT2={left2}")
    if tok_id + 2 < len(sentence):
        right2 = sentence[tok_id + 2]
        features.append(f"RIGHT2={right2}")

    # Window bag-of-words
    start = max(0, tok_id - window)
    end = min(len(sentence), tok_id + window + 1)
    
    for i in range(start, end):
        if i != tok_id:
            features.append(f"WINDOW={sentence[i]}")

    return features


def build_feature_sets(corpus, instances, window=3):
    feature_sets = {}

    for word, occs in instances.items():
        print(f"Extracting features for '{word}'...")

        feature_sets[word] = []

        for inst in occs:
            feats = extract_features_for_instance(corpus, inst, window=window)
            feature_sets[word].append({
                "doc_id": inst["doc_id"],
                "sent_id": inst["sent_id"],
                "tok_id": inst["tok_id"],
                "features": feats,
            })

    return feature_sets

if __name__ == "__main__":
    corpus, token_index = load_preprocessed_data(DATA_DIR)

    with open(DATA_DIR / "targets.pkl", "rb") as f:
        TARGETS = pickle.load(f)
    with open(DATA_DIR / "instances.pkl", "rb") as f:
        instances = pickle.load(f)

    feature_sets = build_feature_sets(corpus, instances, window=3)

    with open(DATA_DIR / "feature_sets.pkl", "wb") as f:
        pickle.dump(feature_sets, f)

    print("\nSaved feature_sets.pkl")
