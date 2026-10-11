"""Auditable intent/slot dataset; only reviewed candidates become SFT files."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ORDER = re.compile(r"\{\{\s*Order Number\s*\}\}", re.I)
SPLITS = ("train", "validation", "test")
CHECKS = ("intent_ok", "extraction_ok", "language_ok")


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text("".join(packed(row) + "\n" for row in rows), encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_review_history(path):
    rows = read_rows(path)
    manifest_path = Path(path).parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rejected = set(manifest.get("rejected_sample_ids", []))
        existing = {r["sample_id"] for r in rows}
        rows.extend({"sample_id": sid, "status": "rejected"} for sid in sorted(rejected - existing))
    return rows


def normalized(text):
    text = ORDER.sub(" ordernumberplaceholder ", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(re.findall(r"\w+", text))


def system_prompt(intents):
    return ('Classify the English customer message. Allowed intents: ' + ', '.join(intents)
            + '. Return only JSON with exactly two keys: "intent" and "order_id". '
            'order_id must be the exact explicitly stated order identifier, or null if absent. '
            'Never invent an identifier. Do not answer the customer request.')


def group_sources(raw, config):
    """Connected components of exact/token-set near duplicates, before splitting."""
    unique = {}
    excluded = Counter()
    for index, row in enumerate(raw):
        text, intent = row.get("instruction"), row.get("intent")
        if intent not in config["intents"]:
            excluded["outside_scope"] += 1
            continue
        if not isinstance(text, str) or not text.strip():
            excluded["empty_instruction"] += 1
            continue
        # Unannotated numeric identifiers and unsupported template slots need a
        # different source adapter; do not silently label them as absent.
        remainder = ORDER.sub("", text)
        if re.search(r"\{\{|\}\}|\d", remainder):
            excluded["unsupported_slot_or_number"] += 1
            continue
        key = normalized(text)
        if not key:
            excluded["empty_normalized"] += 1
            continue
        item = unique.setdefault(key, {"text": text.strip(), "labels": set(), "source_rows": [], "key": key})
        item["labels"].add(intent)
        item["source_rows"].append(index)
    items = [unique[k] for k in sorted(unique)]
    parent = list(range(len(items)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    sets, inverted = [], defaultdict(set)
    threshold = config["near_duplicate_jaccard"]
    for i, item in enumerate(items):
        tokens = set(item["key"].split())
        possible = set()
        for token in tokens:
            possible.update(inverted[token])
        for j in possible:
            other = sets[j]
            if min(len(tokens), len(other)) / max(len(tokens), len(other)) < threshold:
                continue
            if len(tokens & other) / len(tokens | other) >= threshold:
                parent[root(i)] = root(j)
        sets.append(tokens)
        for token in tokens:
            inverted[token].add(i)
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[root(i)].append(item)
    result = []
    for members in groups.values():
        labels = set().union(*(m["labels"] for m in members))
        if len(labels) != 1:
            excluded["conflicting_label_component"] += sum(len(m["source_rows"]) for m in members)
            continue
        group_id = sha(packed(sorted(m["key"] for m in members)))
        # One representative per component prevents template variants from
        # dominating the training set and crossing split boundaries.
        representative = min(members, key=lambda m: sha(str(config["seed"]) + m["key"]))
        result.append({"group_id": group_id, "intent": next(iter(labels)), **representative,
                       "component_size": sum(len(m["source_rows"]) for m in members)})
    excluded["duplicate_variants_removed"] = len(raw) - sum(excluded.values()) - len(result)
    return result, dict(excluded)


def candidate(item, config):
    group = item["group_id"]
    sid = "cs-" + group
    order_id = "ORD-" + sha(str(config["seed"]) + group)[:10].upper() if ORDER.search(item["text"]) else None
    text = ORDER.sub(order_id or "", item["text"])
    fraction = int(sha(str(config["seed"]) + ":split:" + group)[:8], 16) / 2**32
    split = "train" if fraction < .7 else "validation" if fraction < .85 else "test"
    target = {"intent": item["intent"], "order_id": order_id}
    return {"sample_id": sid, "group_id": group, "split": split, "intent": item["intent"],
            "source_rows": item["source_rows"], "source_template": item["text"],
            "component_size": item["component_size"], "text": text, "target": target,
            "annotation_method": "source_intent_and_controlled_placeholder_substitution",
            "messages": [{"role": "system", "content": system_prompt(config["intents"])},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": packed(target)}]}


def add_order_context(row, config):
    """Add an explicitly synthetic order reference to a refund utterance."""
    if row["intent"] not in ("get_refund", "track_refund") or row["target"]["order_id"] is not None:
        raise ValueError("Order-context augmentation requires a refund without an order ID")
    row = json.loads(packed(row))
    row["sample_id"] += "-order-context"
    identifier = "ORD-" + sha(str(config["seed"]) + row["group_id"] + ":refund")[:10].upper()
    patterns = ("This concerns order {id}.", "The order number is {id}.",
                "I am referring to order {id}.", "This is about my order {id}.")
    index = int(sha(row["group_id"] + ":context")[:8], 16) % len(patterns)
    row["text"] += " " + patterns[index].format(id=identifier)
    row["target"]["order_id"] = identifier
    row["annotation_method"] = "source_intent_and_synthetic_explicit_order_context"
    row["augmentation"] = {"method": "append_order_context_v1", "pattern_index": index,
                           "base_text": row["messages"][1]["content"]}
    row["messages"][1]["content"] = row["text"]
    row["messages"][2]["content"] = packed(row["target"])
    return row


def extra_quota(config, split, intent):
    return config.get("refund_order_context_per_intent", {}).get(split, 0) if intent in ("get_refund", "track_refund") else 0


def prepare(raw, config, output, provenance, previous=()):
    if output.exists():
        raise ValueError(f"Output exists: {output}; use --output with a new directory")
    if not 0 < config["near_duplicate_jaccard"] <= 1:
        raise ValueError("Invalid near-duplicate threshold")
    if not config["intents"] or len(set(config["intents"])) != len(config["intents"]):
        raise ValueError("Intents must be nonempty and unique")
    if set(config["per_intent"]) != set(SPLITS) or any(type(v) is not int or v < 1 for v in config["per_intent"].values()):
        raise ValueError("All three split quotas must be positive integers")
    extras = config.get("refund_order_context_per_intent", {})
    if extras and (set(extras) != set(SPLITS) or any(type(v) is not int or v < 0 for v in extras.values())):
        raise ValueError("Refund augmentation quotas must be nonnegative integers for all three splits")
    old = {r["sample_id"]: r for r in previous}
    groups, excluded = group_sources(raw, config)
    pools = defaultdict(list)
    for item in groups:
        row = candidate(item, config)
        if any(old.get(sid, {}).get("status") == "rejected" for sid in (row["sample_id"], row["sample_id"] + "-order-context")):
            excluded["previously_rejected"] = excluded.get("previously_rejected", 0) + 1
            continue
        pools[row["split"], row["intent"]].append(row)
    selected = []
    for split in SPLITS:
        for intent in config["intents"]:
            pool = sorted(pools[split, intent], key=lambda r: sha(str(config["seed"]) + ":select:" + r["sample_id"]))
            count = config["per_intent"][split]
            if len(pool) < count:
                raise ValueError(f"Insufficient distinct groups: {split}/{intent} needs {count}, has {len(pool)}. Reduce quotas; do not duplicate rows.")
            selected.extend(pool[:count])
            extra = extra_quota(config, split, intent)
            eligible = [r for r in pool[count:] if r["target"]["order_id"] is None]
            if len(eligible) < extra:
                raise ValueError(f"Insufficient unseen refund groups for augmentation: {split}/{intent}")
            selected.extend(add_order_context(r, config) for r in eligible[:extra])
    reviews = []
    for row in selected:
        review = {"sample_id": row["sample_id"], "candidate_sha256": sha(packed(row)),
                  "split": row["split"], "text": row["text"], "target": row["target"],
                  "status": "pending", "reviewer": "", "notes": "", **dict.fromkeys(CHECKS)}
        prior = old.get(row["sample_id"])
        if prior and prior.get("candidate_sha256") == review["candidate_sha256"]:
            for field in ("status", "reviewer", "notes", *CHECKS):
                review[field] = prior[field]
        reviews.append(review)
    output.mkdir(parents=True)
    write_rows(output / "candidates.jsonl", selected)
    write_rows(output / "review_queue.jsonl", reviews)
    manifest = {"config": config, "source": provenance, "excluded": excluded,
                "rejected_sample_ids": sorted(sid for sid, review in old.items() if review.get("status") == "rejected"),
                "candidate_sha256": file_hash(output / "candidates.jsonl"),
                "counts": dict(Counter(r["split"] for r in selected)),
                "limitations": "English synthetic source; controlled order-ID substitution. Token-set grouping is not semantic deduplication. Review is required for every selected row."}
    (output / "manifest.json").write_text(packed(manifest) + "\n", encoding="utf-8")
    return audit(output)


def audit(output, publish=False):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(output / "candidates.jsonl") != manifest["candidate_sha256"]:
        raise ValueError("Candidates changed; rebuild instead of editing immutable candidates")
    rows = read_rows(output / "candidates.jsonl")
    reviews = read_rows(output / "review_queue.jsonl")
    by_id = {r["sample_id"]: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Review queue IDs are missing, duplicated, or unexpected")
    groups, texts, coverage, augmented, slots = set(), set(), Counter(), Counter(), Counter()
    approved = True
    for row in rows:
        review = by_id[row["sample_id"]]
        if review["candidate_sha256"] != sha(packed(row)) or any(review[k] != row[k] for k in ("text", "target", "split")):
            raise ValueError("Review is not bound to the current candidate")
        if row["group_id"] in groups or normalized(row["source_template"]) in texts:
            raise ValueError("Duplicate group/template across candidates")
        groups.add(row["group_id"])
        texts.add(normalized(row["source_template"]))
        target = row["target"]
        if set(target) != {"intent", "order_id"} or target["intent"] not in manifest["config"]["intents"]:
            raise ValueError("Invalid output schema")
        if target["order_id"] is not None and target["order_id"] not in row["text"]:
            raise ValueError("Ungrounded order ID")
        if "augmentation" in row:
            base = candidate({"group_id": row["group_id"], "intent": row["intent"],
                              "text": row["source_template"], "source_rows": row["source_rows"],
                              "component_size": row["component_size"]}, manifest["config"])
            if add_order_context(base, manifest["config"]) != row:
                raise ValueError("Synthetic order context does not match its recorded source")
            augmented[row["split"], row["intent"]] += 1
        if row["messages"] != [{"role": "system", "content": system_prompt(manifest["config"]["intents"])},
                               {"role": "user", "content": row["text"]},
                               {"role": "assistant", "content": packed(target)}]:
            raise ValueError("Messages do not match target")
        if review["status"] not in ("pending", "accepted", "rejected"):
            raise ValueError("Invalid review status")
        if any(review[k] is not None and type(review[k]) is not bool for k in CHECKS):
            raise ValueError("Review checks must be boolean or null")
        if review["status"] == "accepted" and not (all(review[k] is True for k in CHECKS)
                and all(isinstance(review[k], str) and review[k].strip() for k in ("reviewer", "notes"))):
            raise ValueError("Accepted review needs three true checks, reviewer, and notes")
        approved &= review["status"] == "accepted"
        coverage[row["split"], row["intent"]] += 1
        slots[row["split"], row["intent"], target["order_id"] is not None] += 1
    for split in SPLITS:
        for intent in manifest["config"]["intents"]:
            if coverage[split, intent] != manifest["config"]["per_intent"][split] + extra_quota(manifest["config"], split, intent):
                raise ValueError("Split/intent quota mismatch")
            if augmented[split, intent] != extra_quota(manifest["config"], split, intent):
                raise ValueError("Refund augmentation quota mismatch")
        values = {r["target"]["order_id"] is not None for r in rows if r["split"] == split}
        if values != {False, True}:
            raise ValueError(f"{split} needs both present and absent order-ID examples")
    report = {"structural_passed": True, "content_approved": approved,
              "review_counts": dict(Counter(r["status"] for r in reviews)),
              "counts": dict(Counter(r["split"] for r in rows)),
              "coverage": {split: {intent: coverage[split, intent] for intent in manifest["config"]["intents"]} for split in SPLITS},
              "slot_coverage": {split: {intent: {"present": slots[split, intent, True], "absent": slots[split, intent, False]}
                                for intent in manifest["config"]["intents"]} for split in SPLITS},
              "review_sha256": file_hash(output / "review_queue.jsonl")}
    if publish:
        if not approved:
            (output / "audit.json").write_text(packed(report) + "\n", encoding="utf-8")
            raise ValueError("Content gate failed: review every candidate, reject bad rows, then rebuild with --review-from")
        folder = output / "sft"
        folder.mkdir(exist_ok=True)
        for split in SPLITS:
            write_rows(folder / f"{split}.jsonl", [r for r in rows if r["split"] == split])
        report["sft_sha256"] = {split: file_hash(folder / f"{split}.jsonl") for split in SPLITS}
    (output / "audit.json").write_text(packed(report) + "\n", encoding="utf-8")
    return report
