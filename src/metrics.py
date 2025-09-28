import numpy as np
import evaluate
from collections import Counter
from seqeval.metrics.sequence_labeling import get_entities


seqeval_metric = evaluate.load("seqeval")


def compute_metrics_builder(id2label):
    def _compute(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        metrics = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

        # Add per-entity F1
        per_type = results.get("per_type") or results.get("entities") or results.get("per_label") or {}
        for ent, vals in per_type.items():
            metrics[f"f1_{ent}"] = vals["f1"]

        # ---- Per-entity (type) scores: exact-span match ----
        TP, FP, FN = Counter(), Counter(), Counter()
        for p_seq, g_seq in zip(true_predictions, true_labels):
            g_ents = get_entities(g_seq)  # list of (type, start, end)
            p_ents = get_entities(p_seq)

            g_map = {(s, e): t for (t, s, e) in g_ents}
            p_map = {(s, e): t for (t, s, e) in p_ents}

            # True positives: exact boundary + type match
            for span, gt in g_map.items():
                pt = p_map.get(span)
                if pt is not None and pt == gt:
                    TP[gt] += 1
                else:
                    FN[gt] += 1

            # False positives: predicted span with no gold at same boundaries OR different type
            for span, pt in p_map.items():
                gt = g_map.get(span)
                if gt is None or gt != pt:
                    FP[pt] += 1

        types = set(TP) | set(FP) | set(FN)
        for t in sorted(types):
            tp, fp, fn = TP[t], FP[t], FN[t]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
            metrics[f"precision_{t}"] = prec
            metrics[f"recall_{t}"]    = rec
            metrics[f"f1_{t}"]        = f1

        return metrics
    return _compute