"""Deterministic scoring for the customer-support intent and order-ID task."""
import json
from collections import Counter, defaultdict

INTENTS = ("cancel_order", "change_order", "track_order", "get_refund", "track_refund")
INVALID = "__invalid__"


def parse_answer(answer, intents=INTENTS):
    """Require a bare JSON object with exactly the two contract fields."""
    try:
        value = json.loads(answer)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"intent", "order_id"}:
        return None, "wrong_schema"
    if not isinstance(value["intent"], str) or value["intent"] not in intents:
        return None, "invalid_intent"
    if value["order_id"] is not None and not isinstance(value["order_id"], str):
        return None, "invalid_order_id_type"
    return value, None


def score(rows, intents=INTENTS):
    if not rows:
        raise ValueError("No predictions to score")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    count = len(rows)
    labels = list(intents)
    confusion = {label: Counter() for label in labels}
    errors = Counter()
    by_intent = defaultdict(Counter)
    valid = correct_intent = correct_slot = joint = correct_absence = absent_total = 0
    tp = fp = fn = gold_present = predicted_present = 0
    scored = []
    for row in rows:
        gold = row["target"]
        if gold["intent"] not in intents:
            raise ValueError("Unknown gold intent")
        prediction, error = parse_answer(row.get("answer"), intents)
        predicted_intent = prediction["intent"] if prediction else INVALID
        predicted_id = prediction["order_id"] if prediction else None
        is_valid = prediction is not None
        intent_ok = predicted_intent == gold["intent"]
        slot_ok = is_valid and predicted_id == gold["order_id"]
        joint_ok = intent_ok and slot_ok
        valid += is_valid
        correct_intent += intent_ok
        correct_slot += slot_ok
        joint += joint_ok
        errors[error or "valid"] += 1
        confusion[gold["intent"]][predicted_intent] += 1
        bucket = by_intent[gold["intent"]]
        bucket["n"] += 1
        bucket["valid"] += is_valid
        bucket["intent_correct"] += intent_ok
        bucket["slot_correct"] += slot_ok
        bucket["joint_correct"] += joint_ok
        if gold["order_id"] is None:
            absent_total += 1
            correct_absence += slot_ok
        else:
            gold_present += 1
        if predicted_id is not None:
            predicted_present += 1
        if is_valid and predicted_id is not None and predicted_id == gold["order_id"]:
            tp += 1
        else:
            fp += is_valid and predicted_id is not None
            fn += gold["order_id"] is not None
        scored.append({**row, "parsed": prediction, "format_error": error,
                       "intent_correct": intent_ok, "order_id_correct": slot_ok,
                       "joint_correct": joint_ok,
                       "order_id_grounded": is_valid and (predicted_id is None or predicted_id in row["text"])})
    per_class = {}
    f1s = []
    for label in labels:
        support = sum(confusion[label].values())
        hits = confusion[label][label]
        predicted = sum(confusion[gold][label] for gold in labels)
        precision = hits / predicted if predicted else 0.0
        recall = hits / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    slot_precision = tp / (tp + fp) if tp + fp else 0.0
    slot_recall = tp / (tp + fn) if tp + fn else 0.0
    slot_f1 = 2 * slot_precision * slot_recall / (slot_precision + slot_recall) if slot_precision + slot_recall else 0.0
    report = {"n": count, "format_valid_rate": valid / count,
              "format_errors": dict(errors), "intent_accuracy": correct_intent / count,
              "intent_macro_f1": sum(f1s) / len(f1s), "intent_per_class": per_class,
              "intent_confusion": {label: dict(confusion[label]) for label in labels},
              "order_id_exact_accuracy": correct_slot / count,
              "order_id_present_precision": slot_precision,
              "order_id_present_recall": slot_recall,
              "order_id_present_f1": slot_f1,
              "order_id_present_gold": gold_present,
              "order_id_present_predicted": predicted_present,
              "order_id_absent_accuracy": correct_absence / absent_total if absent_total else None,
              "joint_accuracy": joint / count,
              "by_intent": {label: {key: by_intent[label][key] for key in
                                    ("n", "valid", "intent_correct", "slot_correct", "joint_correct")}
                            for label in labels},
              "format_valid_count": valid}
    return report, scored
