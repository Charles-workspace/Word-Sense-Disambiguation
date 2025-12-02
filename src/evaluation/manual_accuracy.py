import csv
from collections import defaultdict

CSV_PATH = "data/output/manual_evaluation_samples.csv"

if __name__ == "__main__":
    total = 0
    correct = 0

    per_word_total = defaultdict(int)
    per_word_correct = defaultdict(int)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            word = row["word"].strip()
            predicted = row["predicted_sense"].strip()
            gold = row["gold_label_manual"].strip()

            # Skip rows where gold label is blank (ambiguous or 3rd sense)
            if gold == "":
                continue

            per_word_total[word] += 1
            total += 1

            if predicted == gold:
                per_word_correct[word] += 1
                correct += 1

    print("\n MANUAL EVALUATION RESULTS \n")
    
    for word in per_word_total:
        if per_word_total[word] == 0:
            print(f"{word}: No manually labeled instances.")
            continue

        acc = per_word_correct[word] / per_word_total[word] * 100
        print(f"{word}: {per_word_correct[word]}/{per_word_total[word]} = {acc:.2f}%")

    if total == 0:
        print("\nNo manually labeled instances found.")
    else:
        overall_acc = correct / total * 100
        print("\nOverall accuracy:", f"{overall_acc:.2f}%")