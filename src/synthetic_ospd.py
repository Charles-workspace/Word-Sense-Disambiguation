from collections import Counter
from collections import defaultdict

def apply_ospd(word, predictions, feature_sets, 
                                 confidence_threshold=0.55): 
    doc_groups = defaultdict(list)
    for inst_id, inst in enumerate(feature_sets[word]):
        doc_groups[inst["doc_id"]].append(inst_id)

    new_predictions = predictions.copy()

    for doc_id, inst_ids in doc_groups.items():
        sensed = [(i, predictions[i]) for i in inst_ids if predictions[i] is not None]
        if not sensed:
            continue

        senses_only = [s for i, s in sensed]
        count = Counter(senses_only)
        majority_sense, majority_cnt = count.most_common(1)[0]
        total_tagged = len(senses_only)

        agreement_ratio = majority_cnt / total_tagged

        if agreement_ratio >= confidence_threshold:
            # Safe to enforce — high agreement
            for i in inst_ids:
                new_predictions[i] = majority_sense
        else:
            # Substantial disagreement → reject ALL instances in this document
            # (return to untagged/residual pool)
            for i in inst_ids:
                new_predictions[i] = None   # or "UNKNOWN", or remove from training

    return new_predictions