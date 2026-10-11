"""Audit saved full SFT data/masks/tokenizers without loading model weights."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bitext_support.data.formatting import generation_ids, sft_features, token_ids
from bitext_support.evaluation.generate import digest, read_jsonl
from bitext_support.training.sft import padded_batch


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def repetition_flags(text):
    """Conservative heuristics: candidates for review, never automatic deletions."""
    units = [re.sub(r'^\s*(?:[-*•]|\d+[.)、])\s*', '', s).strip().casefold()
             for s in re.split(r'\n+|(?<=[.!?。！？])\s+', text)]
    counts = Counter(s for s in units if len(s) >= 20)
    repeated = [{'text': s, 'count': n} for s, n in counts.items() if n >= 3]
    words = re.findall(r'\w+|[^\w\s]', text.casefold())
    runs = []
    for width in range(4, min(33, len(words) // 3 + 1)):
        for start in range(len(words) - width * 3 + 1):
            block = words[start:start + width]
            if words[start + width:start + 2 * width] == block == words[start + 2 * width:start + 3 * width]:
                runs.append({'unit_tokens': width, 'token_offset': start, 'excerpt': ' '.join(block)})
                break
        if runs:
            break
    return {'repeated_units': repeated, 'consecutive_token_runs': runs}


def audit(args):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    run = project_path(args.training_run)
    report_path = run / 'train_report.json'
    training = json.loads(report_path.read_text(encoding='utf-8'))
    config = training['config']
    if not all(k in config for k in ('train_file', 'validation_file', 'max_seq_length')):
        raise ValueError('Expected a full SFT run with train/validation file configuration')
    source = training['model']
    common = dict(local_files_only=True, trust_remote_code=False,
                  revision=training.get('resolved_commit') or source.get('revision'))
    if source.get('cache_dir'):
        common['cache_dir'] = str(project_path(source['cache_dir']))
    base = AutoTokenizer.from_pretrained(source['name_or_path'], **common)
    saved = AutoTokenizer.from_pretrained(run / 'adapter', local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(source['name_or_path'], **common)
    try:
        generation = GenerationConfig.from_pretrained(source['name_or_path'], **common)
    except OSError:
        generation = GenerationConfig.from_model_config(model_config)
    stop_ids = generation.eos_token_id
    stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids] if stop_ids is not None else []
    limit = min(config['max_seq_length'], getattr(model_config, 'max_position_embeddings', config['max_seq_length']))
    output = project_path(args.output_dir) if args.output_dir else ROOT / 'reports' / ('sft_audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    errors, warnings, details, samples, split_results = [], [], [], [], {}
    systems = set()
    system_text = project_path(args.system_prompt).read_text(encoding='utf-8').strip()
    original_audit = json.loads((run / 'data_audit.json').read_text(encoding='utf-8')) if (run / 'data_audit.json').exists() else None
    all_rows = {}
    templates_match = base.chat_template == saved.chat_template
    if not templates_match:
        errors.append('Base and saved Adapter tokenizer chat templates differ')
    for split in ('train', 'validation'):
        path = project_path(config[f'{split}_file'])
        actual_hash = digest(path)
        if actual_hash != training['data_sha256'][split]:
            raise ValueError(f'{split} data hash differs from training report: {path}')
        rows = read_jsonl(path)
        all_rows[split] = rows
        selected = set(random.Random(args.seed).sample(range(len(rows)), min(args.samples, len(rows))))
        kept, skipped, lengths, supervised, eos_ok = [], [], [], [], 0
        duplicate_answers = Counter(r['messages'][-1]['content'] for r in rows)
        split_details = []
        for index, row in enumerate(rows):
            sid = row['sample_id']
            try:
                answer = row['messages'][-1]['content']
                if not answer.strip():
                    raise ValueError('Empty assistant answer')
                systems.add(row['messages'][0]['content'])
                feat = sft_features(saved, row)
                padded_batch([feat], saved.eos_token_id)
                ids, labels = feat['input_ids'], feat['labels']
                boundary = next(i for i, value in enumerate(labels) if value != -100)
                prefix = generation_ids(saved, row)
                base_prefix = generation_ids(base, row)
                if prefix != base_prefix:
                    raise ValueError('Base / Adapter inference token prefix differs')
                if sft_features(base, row) != feat:
                    raise ValueError('Base / Adapter full training token sequence differs')
                if ids[:boundary] != prefix or any(x != -100 for x in labels[:boundary]):
                    raise ValueError('Prompt supervision or prefix mismatch')
                # Exercise actual padding code with unequal lengths, without tensors/model weights.
                short = {key: value[:-1] for key, value in feat.items()}
                if len(ids) - boundary > 1:
                    padded = padded_batch([feat, short], saved.eos_token_id)
                    if padded['labels'][1][-1] != -100 or padded['attention_mask'][1][-1] != 0:
                        raise ValueError('Padding token contributes to loss/attention')
                ends = [i for i in range(boundary, len(ids)) if ids[i] == saved.eos_token_id]
                ending = bool(ends and not saved.decode(ids[ends[-1] + 1:], skip_special_tokens=False).strip())
                stop_compatible = saved.eos_token_id in stop_ids
                if not ending or not stop_compatible:
                    errors.append(f'{split}/{sid}: supervised turn terminator missing, non-whitespace suffix, or generation EOS mismatch')
                eos_ok += int(ending and stop_compatible)
                valid = len(ids) <= limit
                (kept if valid else skipped).append(sid)
                count = sum(x != -100 for x in labels[1:])
                lengths.append(len(ids))
                if valid:
                    supervised.append(count)
                rep = repetition_flags(answer)
                flagged = bool(rep['repeated_units'] or rep['consecutive_token_runs'])
                entry = dict(split=split, sample_id=sid, eligible=valid, input_tokens=len(ids), prompt_tokens=boundary,
                    supervised_tokens=count, supervised_eos_positions=ends, turn_ending_ok=ending,
                    stop_compatible=stop_compatible, template_prefix_match=True, repetition_flagged=flagged,
                    repetition=rep, duplicate_answer_count=duplicate_answers[answer])
                details.append(entry); split_details.append(entry)
                if index in selected or flagged:
                    samples.append(dict(**entry, messages=row['messages'], original_answer=answer,
                        masked_prompt_text=saved.decode(ids[:boundary], skip_special_tokens=False),
                        supervised_text=saved.decode(labels[boundary:], skip_special_tokens=False),
                        prompt_token_ids=prefix, supervised_token_ids=labels[boundary:], tail_token_ids=ids[-12:],
                        tail_tokens=saved.convert_ids_to_tokens(ids[-12:])))
            except (ValueError, TypeError, KeyError, StopIteration) as exc:
                errors.append(f'{split}/{sid}: {exc}')
        if not rows or len({r['sample_id'] for r in rows}) != len(rows):
            errors.append(f'{split}: empty dataset or duplicate sample IDs')
        result = dict(source_count=len(rows), audited_count=len(split_details), kept_count=len(kept), skipped_count=len(skipped),
            kept_ids=kept, skipped_overlength_ids=skipped, supervised_tokens=sum(supervised),
            supervised_tokens_mean=statistics.mean(supervised) if supervised else None,
            max_input_tokens=max(lengths, default=0), supervised_eos_pass_count=eos_ok,
            repetition_flagged_count=sum(r['repetition_flagged'] for r in split_details),
            duplicate_answer_groups=sum(n > 1 for n in duplicate_answers.values()), data_sha256=actual_hash)
        if original_audit:
            for key in ('source_count', 'kept_count', 'kept_ids', 'skipped_overlength_ids', 'supervised_tokens'):
                if result[key] != original_audit[split][key]:
                    errors.append(f'{split}: {key} differs from saved training data audit')
        split_results[split] = result
    for key in ('sample_id', 'group_id'):
        if {r[key] for r in all_rows['train']} & {r[key] for r in all_rows['validation']}:
            errors.append(f'Train / validation overlap: {key}')
    system_match = systems == {system_text}
    if len(systems) != 1:
        errors.append('Training / validation do not use one fixed system prompt')
    if not system_match:
        warnings.append('Serving system prompt differs from dataset; evaluation uses dataset system prompt')
    if any(v['repetition_flagged_count'] for v in split_results.values()):
        warnings.append('Repetition candidates found; code/poetry/list repetition may be legitimate. Review manually.')
    if not original_audit:
        warnings.append('Saved data_audit.json absent; training selection could not be cross-checked')
    summary = dict(status='failed' if errors else 'review_required' if warnings else 'passed',
        training_run=str(run), training_report_sha256=digest(report_path), max_seq_length=limit,
        tokenizer_only=True, template_match=templates_match, system_prompt_match=system_match,
        tokenizer_eos_id=saved.eos_token_id, generation_eos_ids=stop_ids,
        sample_seed=args.seed, random_samples_per_split=args.samples, errors=errors, warnings=warnings, splits=split_results,
        limitations=['No model forward/backward pass or Adapter weight verification.',
                      'Checks current shared formatting implementation against saved data selection and exported tokenizer.',
                      'EOS ending heuristic targets this project Qwen template; not a universal template validator.',
                      'Repetition flags are heuristics, not proof of bad data; duplicate answers can be legitimate labels.'])
    save_json(output / 'audit.json', summary)
    for name, records in [('records.jsonl', details), ('samples.jsonl', samples)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    lines = ['# SFT 实现审计', '', f"状态：**{summary['status']}**", '',
        f'训练运行：`{run}`。只加载tokenizer和配置，没有加载模型权重或重新训练。', '',
        '| 数据集 | 来源/已审计/保留 | EOS与停止配置通过 | 重复候选 | 有效监督token均值 |',
        '|---|---:|---:|---:|---:|']
    for split, r in split_results.items():
        lines.append(f"| {split} | {r['source_count']}/{r['audited_count']}/{r['kept_count']} | {r['supervised_eos_pass_count']} | {r['repetition_flagged_count']} | {r['supervised_tokens_mean']} |")
    lines += ['', f'Base/Adapter模板相同：{templates_match}；数据与服务system prompt相同：{system_match}。',
        f'Tokenizer EOS：{saved.eos_token_id}；生成停止ID：{stop_ids}。允许结束标记后有模板换行。', '', '## 错误与待复核项', '']
    lines += ['- ' + x for x in errors + warnings] or ['- 未发现错误或重复候选。']
    lines += ['', '## 监督内容抽样与重复候选', '', '固定种子随机抽样，额外纳入全部重复候选。下面保留特殊token；完整token ID见samples.jsonl。', '']
    for row in samples:
        lines += [f"### {row['split']} / {row['sample_id']}", '',
            f"重复候选：{row['repetition_flagged']}；监督token：{row['supervised_tokens']}；EOS检查：{row['turn_ending_ok']}", '']
        for title, text_value in [('掩码提示词', row['masked_prompt_text']), ('原始回答', row['original_answer']), ('实际监督文本', row['supervised_text'])]:
            lines += [f'**{title}**', '', *['    ' + line for line in text_value.splitlines()], '']
    lines += ['## 范围限制', '', '未执行模型前向/反向计算，不替代Adapter重载检查。重复检测是启发式，不能自动判定数据错误或删除样本。模板检查针对本项目Qwen训练和验证流程，服务端任意多轮请求未在本次执行。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('status', 'errors', 'warnings')}, ensure_ascii=False, indent=2))
    print(f'Report: {output / "report.md"}')
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True)
    parser.add_argument('--samples', type=int, default=5, help='Random samples per split; all repetition candidates are also exported')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system-prompt', default='prompts/system_general.txt')
    parser.add_argument('--output-dir', help='New report directory; existing directories are never overwritten')
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    sys.stdout.reconfigure(encoding='utf-8')
    return audit(args)


if __name__ == '__main__':
    raise SystemExit(main())
