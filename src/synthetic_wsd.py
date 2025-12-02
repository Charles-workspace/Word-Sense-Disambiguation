import pickle
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

from src.preprocess import load_preprocessed_data

# ==================================================================
# Synthetic ambiguous words
# ==================================================================
SYNTHETIC_PAIRS = {
    "carspeech": ("car", "speech"),
    "schooloil": ("school", "oil"),
    "phoneanimal": ("phone", "animal"),
    "hotelwar": ("hotel", "war"),
    "musicforest": ("music", "forest"),
}

DATA_DIR = "data/preprocessed"
OUT_DIR = Path("data/synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "the","a","an","in","of","and","to","on","for","by","with","was","is",
    "that","this","as","it","at","from","be","been","were","are","but","or",
    "which","who","whom","its","into","their","his","her","s"
}

# ==================================================================
# Feature extractor (clean & optimized)
# ==================================================================
def extract_features(corpus, doc_id, sent_id, tok_id, window=4):
    sent = corpus[doc_id][sent_id]
    feats = []

    def get_word(idx):
        if 0 <= idx < len(sent):
            w = sent[idx]
            if w not in STOPWORDS:
                return w
        return None

    offsets = {
        "LEFT1":  tok_id - 1,
        "LEFT2":  tok_id - 2,
        "RIGHT1": tok_id + 1,
        "RIGHT2": tok_id + 2,
    }

    for label, pos in offsets.items():
        w = get_word(pos)
        if w:
            feats.append(f"{label}={w}")

    # WINDOW
    target = sent[tok_id]
    start = max(0, tok_id - window)
    end = min(len(sent), tok_id + window + 1)

    for w in sent[start:end]:
        if w != target and w not in STOPWORDS:
            feats.append(f"WINDOW={w}")

    return feats


# ==================================================================
# Build synthetic corpus + feature sets
# ==================================================================
if __name__ == "__main__":
    corpus, token_index = load_preprocessed_data(DATA_DIR)
    synthetic_corpus = deepcopy(corpus)

    feature_sets = defaultdict(list)
    gold_labels = defaultdict(list)
    orig_map = {}

    for synth, (w1, w2) in SYNTHETIC_PAIRS.items():
        orig_map[w1] = (synth, 1)
        orig_map[w2] = (synth, 2)

    for doc_id, doc in enumerate(corpus):
        for sent_id, sent in enumerate(doc):
            for tok_id, tok in enumerate(sent):
                if tok in orig_map:
                    synth_word, sense_id = orig_map[tok]

                    synthetic_corpus[doc_id][sent_id][tok_id] = synth_word

                    feats = extract_features(synthetic_corpus, doc_id, sent_id, tok_id)

                    feature_sets[synth_word].append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "tok_id": tok_id,
                        "features": feats,
                    })
                    gold_labels[synth_word].append(sense_id)

    targets = list(feature_sets.keys())

    print("\nSynthetic words and instance counts:")
    for t in targets:
        print(f"{t}: {len(feature_sets[t])}")

    # Save output
    pickle.dump(synthetic_corpus, open(OUT_DIR/"synthetic_corpus.pkl", "wb"))
    pickle.dump(dict(feature_sets), open(OUT_DIR/"feature_sets.pkl", "wb"))
    pickle.dump(dict(gold_labels), open(OUT_DIR/"gold_labels.pkl", "wb"))
    pickle.dump(targets, open(OUT_DIR/"targets.pkl", "wb"))

    print("\nSaved synthetic corpus + features.")