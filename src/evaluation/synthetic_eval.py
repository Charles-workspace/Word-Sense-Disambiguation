import pickle
from pathlib import Path
from src.synthetic_ospd import apply_ospd

DATA_DIR = Path("data/synthetic")

USE_OSPD = True     # toggle ON/OFF easily

if __name__ == "__main__":
    feature_sets = pickle.load(open(DATA_DIR/"feature_sets.pkl", "rb"))
    gold_labels   = pickle.load(open(DATA_DIR/"gold_labels.pkl", "rb"))
    targets       = pickle.load(open(DATA_DIR/"targets.pkl", "rb"))
    model         = pickle.load(open(DATA_DIR/"decision_lists.pkl", "rb"))

    print("\nAutomatic Evaluation Results")
    for word in targets:
        dl = model[word]["decision_list"]

        # --- FIRST: get raw predictions ---
        raw_predictions = []
        for inst in feature_sets[word]:
            feats = set(inst["features"])
            pred = None
            for feat, sense, score in dl:
                if feat in feats:
                    pred = sense
                    break
            raw_predictions.append(pred)

        # --- OPTIONAL: apply OSPD ---
        if USE_OSPD:
            predictions = apply_ospd(word, raw_predictions, feature_sets)
        else:
            predictions = raw_predictions

        # --- compute accuracy ---
        correct = sum(
            1 for p, g in zip(predictions, gold_labels[word]) if p == g
        )
        total = len(gold_labels[word])
        acc = correct/total * 100

        print(f"{word}: {correct}/{total} = {acc:.2f}%")

    print("\nDone.")