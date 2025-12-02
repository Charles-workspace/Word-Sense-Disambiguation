import csv
import random
import pickle

from src.preprocess import load_preprocessed_data
from src.core.decision_list import apply_decision_list


DATA_DIR = "data/preprocessed"
FEATURE_FILE = f"{DATA_DIR}/feature_sets.pkl"
LABEL_FILE = f"{DATA_DIR}/instances.pkl"
DL_FILE = f"{DATA_DIR}/decision_lists.pkl"

OUTPUT_FILE = "data/output/manual_evaluation_samples.csv"

TARGET_WORDS = ["plant", "bank", "seal", "issue", "bill"]
SAMPLES_PER_WORD = 100


def get_sentence(corpus, doc_id, sent_id):
    return " ".join(corpus[doc_id][sent_id])


if __name__ == "__main__":
    corpus, token_index = load_preprocessed_data(DATA_DIR)

    # full dicts
    feature_sets = pickle.load(open(FEATURE_FILE, "rb"))
    labels = pickle.load(open(LABEL_FILE, "rb"))
    decision_lists = pickle.load(open(DL_FILE, "rb"))

    rows = []

    print("\nGenerating manual evaluation samples...\n")

    for word in TARGET_WORDS:
        print(f"Sampling for '{word}' ...")

        fs_list = feature_sets[word]
        lb_list = labels[word]
        dl = decision_lists[word]

        # run full prediction for ALL instances
        predictions = apply_decision_list(word, feature_sets, labels, decision_lists[word]["decision_list"])

        # choose 100 random instance indices
        total_instances = len(fs_list)
        sample_indices = (
            random.sample(range(total_instances), SAMPLES_PER_WORD)
            if total_instances > SAMPLES_PER_WORD
            else list(range(total_instances))
        )

        for idx in sample_indices:
            entry = fs_list[idx]

            doc = entry["doc_id"]
            sent = entry["sent_id"]
            tok = entry["tok_id"]

            predicted = predictions[idx]

            sentence = get_sentence(corpus, doc, sent)

            rows.append([
                word,
                doc,
                sent,
                tok,
                sentence,
                predicted,
                ""  # manual gold label here
            ])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "word", "doc_id", "sent_id", "tok_id",
            "sentence", "predicted_sense", "gold_label_manual"
        ])
        writer.writerows(rows)

    print("\nSaved:", OUTPUT_FILE)
    print("Fill gold_label_manual manually.")