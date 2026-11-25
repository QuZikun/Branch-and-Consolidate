"""Data/config preparation for a reasoning-only loss comparison; no GPU imports."""
import json
from pathlib import Path
import re
import sys
import unicodedata

from opd.common import read_json, write_json, require, file_hash
from opd.config import distillation_settings
from opd.data import build_index

LOSSES = ('forward', 'reverse')


def project_path(value, project):
    path = Path(value).expanduser()
    return path if path.is_absolute() else project / path


def question_key(text):
    # The sampler preserves source formatting instructions. Ignore this known
    # appended section for the exact-question audit, never for model input.
    text = text.split('Output Format Requirements:', 1)[0]
    text = re.sub(r'<image>|<\|image_pad\|>|<\|vision_start\|>|<\|vision_end\|>', '', text)
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def audit_pool(data_root, bench_root, datasets, output, rollout_adapter_root):
    """Verify bytes/schema, and compare exact question + ordered image paths.

    No image opens/stat/hash: copies of an image under different paths and
    paraphrased questions cannot be detected by this bounded audit.
    """
    index = build_index(data_root, output / 'input_index.json', ('reasoning',))
    sampling = read_json(data_root / 'report.json')
    require(sampling['status'] == 'complete' and sampling['selected_queries'] == len(index['entries']),
            'Sampling report/count does not match training pool')
    require(all(sampling['files_sha256'][name] == value for name, value in index['hashes'].items()),
            'Training pool differs from sampling report')
    suite = read_json(bench_root / 'configs/suite.json')
    require(datasets and len(set(datasets)) == len(datasets) and set(datasets) <= set(suite['general']),
            'This experiment evaluates a distinct nonempty subset of the general benchmarks')
    keys, bench_files, bench_ids, image_keys = {}, {}, set(), {}
    for name in datasets:
        path = bench_root / 'data' / (name + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(rows) == suite['counts'][name], 'Benchmark count mismatch: ' + name)
        bench_files[name] = file_hash(path)
        for row in rows:
            require(row['benchmark'] == name and row['sample_id'] not in bench_ids, 'Invalid benchmark ID')
            bench_ids.add(row['sample_id'])
            paths = tuple(str(project_path(media['path'], bench_root).absolute()) for media in row['media'])
            key = (question_key(row['question']), paths)
            keys.setdefault(key, []).append(row['sample_id'])
            if paths:
                image_keys.setdefault(paths, []).append(row['sample_id'])
    sys.path.insert(0, str(rollout_adapter_root))
    from runner.data import normalize
    matches, same_paths = [], []
    with (data_root / 'train.jsonl').open('rb') as stream:
        for i, entry in enumerate(index['entries'], 1):
            stream.seek(entry['offset'])
            row = json.loads(stream.read(entry['length']))
            prepared = normalize(row, entry['provenance'], 'reasoning', {}, defer_media=True)
            initial = []
            for message in row['messages']:
                if message['role'] == 'assistant':
                    break
                if message['role'] == 'user':
                    initial.append(message['content'])
            paths = tuple(prepared['image_paths'])
            key = (question_key('\n'.join(initial)), paths)
            if key in keys:
                matches.append({'sample_id': entry['sample_id'], 'benchmark_ids': keys[key]})
            elif paths and paths in image_keys:
                same_paths.append({'sample_id': entry['sample_id'], 'benchmark_ids': image_keys[paths]})
            if i % 1000 == 0:
                print(f'Reasoning inputs: {i}/{len(index["entries"])}; no media opened', flush=True)
    report = {'queries': len(index['entries']), 'benchmarks': datasets, 'benchmark_questions': len(bench_ids),
              'input_hashes': index['hashes'], 'benchmark_hashes': bench_files,
              'exact_question_and_image_path_matches': matches, 'same_image_paths': same_paths,
              'media_opened': False, 'scope': 'exact normalized question and literal image paths; not a semantic/image-content contamination guarantee'}
    write_json(output / 'data_audit.json', report)
    require(not matches and not same_paths, 'Training/benchmark overlap detected; inspect ' + str(output / 'data_audit.json'))
    return report


def prepare(config_path, project, root):
    config = read_json(config_path)
    require(config.get('domains') == ['reasoning'], 'This launcher requires domains=[reasoning]')
    require(set(config['loss_variants']) == set(LOSSES), 'Specify forward and reverse loss variants')
    require(config['loss_variants']['forward']['loss_mode'] == 'forward_kl_topk'
            and config['loss_variants']['reverse']['loss_mode'] == 'reverse_kl_topk_tail', 'Wrong comparison losses')
    for variant in config['loss_variants'].values():
        distillation_settings({'distillation': variant})
    require(0 < config['max_new_tokens'] < config['max_sequence_tokens'], 'Invalid generation/context budget')
    evaluation = config['evaluation']
    require(config['nodes'] * config['gpus_per_node'] == evaluation['replicas'] * evaluation['tp'],
            'Training and evaluation must use the same GPU count')
    data_root = project_path(config['data_root'], project)
    bench_root = project / 'mopd_bench'
    audit = audit_pool(data_root, bench_root, evaluation['datasets'].split(','), root,
                      project_path(config['rollout_adapter_root'], project))
    contract = {'config': config, 'project': str(project), 'audit': audit}
    contract_path = root / 'pipeline_contract.json'
    if contract_path.exists():
        require(read_json(contract_path) == contract, 'Experiment/data/bench configuration changed; use a new output_root')
    else:
        write_json(contract_path, contract)
    configs = {}
    for loss in LOSSES:
        training = {k: v for k, v in config.items() if k not in ('loss_variants', 'evaluation')}
        training.update(output_root=str(root / loss), data_root=str(data_root), distillation=config['loss_variants'][loss])
        distillation_settings(training)
        path = root / 'configs' / (loss + '.json')
        if path.exists():
            require(read_json(path) == training, 'Generated training config changed: ' + str(path))
        else:
            write_json(path, training)
        configs[loss] = path
    return config, configs, audit
