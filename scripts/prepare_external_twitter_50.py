"""Materialize the manually reviewed external evaluation set from the raw TWCS CSV."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/external/customer_support_twitter/twcs.csv'
SPEC = ROOT / 'configs/external_twitter_50_labels.json'
OUTPUT = ROOT / 'data/external/customer_support_twitter/selected_50.jsonl'
ORDER_ID = re.compile(r'\b\d{3}-\d{7}-\d{7}\b')


def main():
    spec = json.loads(SPEC.read_text(encoding='utf-8'))
    labels = spec['labels']
    if set(labels) != {'cancel_order', 'change_order', 'track_order', 'get_refund', 'track_refund'}:
        raise ValueError('Expected exactly the five task intents')
    if any(len(items) != 10 for items in labels.values()):
        raise ValueError('Expected exactly ten reviewed samples per intent')
    lookup = {tweet_id: (intent, order_id) for intent, items in labels.items() for tweet_id, order_id in items.items()}
    if len(lookup) != 50:
        raise ValueError('Duplicate tweet ID in label specification')
    digest = hashlib.sha256()
    with SOURCE.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != spec['source_sha256']:
        raise ValueError('Raw source SHA-256 differs from reviewed version')
    selected = {}
    with SOURCE.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            tweet_id = row['tweet_id']
            if tweet_id not in lookup:
                continue
            if tweet_id in selected or row['inbound'].lower() != 'true':
                raise ValueError(f'Invalid customer tweet: {tweet_id}')
            intent, order_id = lookup[tweet_id]
            text = row['text']
            if not re.search(r'@amazonhelp\b', text, re.I):
                raise ValueError(f'Tweet does not mention AmazonHelp: {tweet_id}')
            found_ids = ORDER_ID.findall(text)
            if found_ids != ([order_id] if order_id else []):
                raise ValueError(f'Order ID review mismatch: {tweet_id}, found {found_ids}')
            selected[tweet_id] = {'sample_id': f'twcs_{tweet_id}', 'source_tweet_id': tweet_id,
                                  'text': text, 'target': {'intent': intent, 'order_id': order_id}}
    if set(selected) != set(lookup):
        raise ValueError(f'Missing source IDs: {sorted(set(lookup) - set(selected))}')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8', newline='\n') as handle:
        for intent, items in labels.items():
            for tweet_id in items:
                handle.write(json.dumps(selected[tweet_id], ensure_ascii=False) + '\n')
    print(f'Wrote {len(selected)} reviewed real tweets to {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
