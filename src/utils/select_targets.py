

from pathlib import Path
import pickle
from preprocess import load_preprocessed_data 


DATA_DIR = "data/preprocessed"

def load_data():
    corpus, token_index = load_preprocessed_data(DATA_DIR)
    return corpus, token_index


def show_candidate_frequencies(token_index, candidates, min_count=100):
    print("\n=== Candidate Frequencies ===")
    for w in candidates:
        count = len(token_index.get(w, []))
        flag = "" if count >= min_count else " (LOW)"
        print(f"{w:12s}: {count}{flag}")


def print_sample_contexts(corpus, token_index, word, k=10, window=4):
    print(f"\n=== Contexts for '{word}' ===")
    positions = token_index.get(word, [])[:k]

    for i, (doc_id, sent_id, tok_id) in enumerate(positions):
        sentence = corpus[doc_id][sent_id]
        start = max(0, tok_id - window)
        end = min(len(sentence), tok_id + window + 1)

        left = sentence[start:tok_id]
        token = sentence[tok_id]
        right = sentence[tok_id+1:end]

        print(f"{i+1:2d}: {' '.join(left)} [{token}] {' '.join(right)}")


def build_instances_for_targets(corpus, token_index, targets, max_instances=None):
    instances = {}

    for word in targets:
        positions = token_index.get(word, [])
        if max_instances is not None:
            positions = positions[:max_instances]

        word_instances = []
        for (doc_id, sent_id, tok_id) in positions:
            word_instances.append({
                "doc_id": doc_id,
                "sent_id": sent_id,
                "tok_id": tok_id,
                "sentence": corpus[doc_id][sent_id],
            })

        instances[word] = word_instances
        print(f"{word}: {len(word_instances)} instances")

    return instances

if __name__ == "__main__":
    corpus, token_index = load_data()

    # Candidate ambiguous nouns
    CANDIDATES = [
        "plant", "bank", "seal", "club", "board",
        "charge", "issue", "capital", "interest", "bill"
    ]

    
    show_candidate_frequencies(token_index, CANDIDATES)

    
    TARGETS = ["plant", "bank", "seal", "issue", "bill"]  
    for w in TARGETS:
        print_sample_contexts(corpus, token_index, w, k=12)

    
    instances = build_instances_for_targets(corpus, token_index, TARGETS)

    out = Path(DATA_DIR)
    with open(out / "targets.pkl", "wb") as f:
        pickle.dump(TARGETS, f)
    with open(out / "instances.pkl", "wb") as f:
        pickle.dump(instances, f)

    print("\nSaved targets.pkl and instances.pkl")