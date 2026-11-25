#!/usr/bin/env python3
"""Strict, CPU, sharded HF safetensors averaging: (1-alpha)*base + alpha*SFT.

Source checkpoints are read-only. Defaults target the user's two 35B-A3B paths.
Use --check-only before merging. Full checkpoints only; not LoRA or quantized.
If SFT omits or disables the entire mtp.* branch, retain it from base by default.
This fallback is for ordinary decoding with MTP speculative decoding disabled.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

BASE_DEFAULT = '/path/to/student'
SFT_DEFAULT = '/path/to/specialist'
FLOATS = {'F16', 'BF16', 'F32'}
BYTES = {'F16': 2, 'BF16': 2, 'F32': 4, 'I64': 8, 'I32': 4, 'I16': 2,
         'I8': 1, 'U8': 1, 'BOOL': 1}
IGNORE_CONFIG = {'_name_or_path', '_commit_hash', 'transformers_version',
                 'torch_dtype', 'dtype', 'use_cache', 'output_router_logits',
                 'router_aux_loss_coef', 'return_dict', 'output_attentions',
                 'output_hidden_states', 'architectures'}
CONFIG_KEYS = ['model_type', 'text_config', 'vision_config', 'tie_word_embeddings',
               'image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id']
ASSET_NAMES = {'config.json', 'generation_config.json', 'tokenizer.json',
               'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
               'vocab.json', 'merges.txt', 'tokenizer.model', 'chat_template.jinja',
               'chat_template.json', 'preprocessor_config.json', 'processor_config.json',
               'video_preprocessor_config.json', 'LICENSE', 'LICENSE.txt'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in IGNORE_CONFIG}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            name = prefix + '.' + key if prefix else key
            if key not in a or key not in b:
                result.append({'field': name, 'base': a.get(key, '<absent>'), 'sft': b.get(key, '<absent>')})
            else:
                result.extend(differences(a[key], b[key], name))
        return result
    return [] if a == b else [{'field': prefix, 'base': a, 'sft': b}]


def inventory(folder):
    """Read only safetensors headers, validating index, shapes and byte ranges."""
    config = read_json(folder / 'config.json')
    if (folder / 'adapter_config.json').exists():
        raise ValueError(str(folder) + ': adapter checkpoint; export merged full HF weights first')
    if config.get('quantization_config'):
        raise ValueError(str(folder) + ': quantized checkpoint; use original BF16/FP16/FP32 weights')
    index_path = folder / 'model.safetensors.index.json'
    if index_path.exists():
        weight_map = read_json(index_path)['weight_map']
        names = sorted(set(weight_map.values()))
    elif (folder / 'model.safetensors').is_file():
        weight_map, names = None, ['model.safetensors']
    else:
        raise ValueError(str(folder) + ': expected model.safetensors or model.safetensors.index.json')
    tensors, file_info = {}, {}
    for name in names:
        path = (folder / name).resolve()
        if path.parent != folder.resolve():
            raise ValueError('Only checkpoint-root shard files are supported: ' + name)
        stat = path.stat()
        with path.open('rb') as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise ValueError('Truncated safetensors header: ' + str(path))
            length = struct.unpack('<Q', size_bytes)[0]
            if length < 2 or length > 100_000_000 or length + 8 > stat.st_size:
                raise ValueError('Invalid safetensors header size: ' + str(path))
            header_raw = handle.read(length)
            header = json.loads(header_raw)
        file_info[name] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                           'header_sha256': hashlib.sha256(header_raw).hexdigest()}
        for key, spec in header.items():
            if key == '__metadata__':
                continue
            if key in tensors:
                raise ValueError('Duplicate parameter: ' + key)
            dtype, shape = spec['dtype'], spec['shape']
            if dtype not in BYTES:
                raise ValueError('Unsupported/quantized dtype %s at %s' % (dtype, key))
            if any(not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError('Invalid shape: ' + key)
            nbytes = math.prod(shape) * BYTES[dtype]
            start, end = spec['data_offsets']
            if start < 0 or end - start != nbytes or end > stat.st_size - 8 - length:
                raise ValueError('Invalid/truncated tensor byte range: ' + key)
            if weight_map is not None and weight_map.get(key) != name:
                raise ValueError('Index/header disagreement: ' + key)
            tensors[key] = {'file': name, 'shape': shape, 'dtype': dtype, 'nbytes': nbytes}
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError('Index has missing/extra tensor keys')
    if not tensors:
        raise ValueError('Checkpoint contains no tensors: ' + str(folder))
    return config, tensors, file_info


def asset_check(base, sft):
    """Check tokenizer semantics when exported; output always uses base assets."""
    from transformers import AutoTokenizer
    if not (base / 'tokenizer.json').exists():
        raise ValueError('Base tokenizer.json is required')
    a = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    report = {'output_assets_from': str(base), 'base_vocab_entries': len(a.get_vocab()),
              'base_pad_token_id': a.pad_token_id,
              'base_tokenizer_json_sha256': digest(base / 'tokenizer.json')}
    if (sft / 'tokenizer.json').exists():
        if not (sft / 'tokenizer_config.json').exists():
            raise ValueError('SFT has tokenizer.json but lacks tokenizer_config.json; restore its tokenizer assets')
        b = AutoTokenizer.from_pretrained(str(sft), local_files_only=True, trust_remote_code=False)
        if a.get_vocab() != b.get_vocab():
            raise ValueError('Runtime token-ID mappings differ')
        aa, bb = json.loads(a.backend_tokenizer.to_str()), json.loads(b.backend_tokenizer.to_str())
        for key in ('model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens'):
            if aa.get(key) != bb.get(key):
                raise ValueError('Tokenizer behavior differs: ' + key)
        report['sft_tokenizer_status'] = 'runtime mapping and BPE behavior verified'
    else:
        # User identifies this as an SFT descendant of this base. No claim of verification.
        report['sft_tokenizer_status'] = 'absent: inherited base tokenizer under declared same-base SFT assumption'
    report['asset_differences'] = []
    for name in sorted(ASSET_NAMES - {'config.json', 'LICENSE', 'LICENSE.txt'}):
        if (base / name).is_file() and (sft / name).is_file() and digest(base / name) != digest(sft / name):
            report['asset_differences'].append(name)
    return report


def preflight(base, sft, missing_mtp='keep-base', convert_experts=False):
    if missing_mtp not in ('keep-base', 'error'):
        raise ValueError('Unknown missing-MTP policy: ' + str(missing_mtp))
    ac, at, af = inventory(base)
    bc, bt, bf = inventory(sft)
    layout_conversions = []
    if convert_experts:
        from expert_layout import align_experts
        bt, layout_conversions = align_experts(at, bt, bc.get('model_type'))
    base_mtp = {key for key in at if key.startswith('mtp.')}
    source_mtp = {key for key in bt if key.startswith('mtp.')}
    # The exported vision subconfig used the parent type in older Qwen checkpoints.
    # Match the explicit alias in Transformers' Qwen3_5MoeConfig.__post_init__;
    # do not ignore model_type changes elsewhere or any other vision differences.
    vision_alias = (
        ac.get('model_type') == bc.get('model_type') == 'qwen3_5_moe'
        and {ac.get('vision_config', {}).get('model_type'),
             bc.get('vision_config', {}).get('model_type')}
        == {'qwen3_5_moe', 'qwen3_5_moe_vision'}
    )
    # verl disables MTP by setting this count to zero. Preserve the reference
    # branch whether the disabled branch was omitted or exported in full.
    # Partial MTP exports, 1->2 changes, and other model families remain errors.
    base_layers = ac.get('text_config', {}).get('mtp_num_hidden_layers')
    source_layers = bc.get('text_config', {}).get('mtp_num_hidden_layers')
    disabled_mtp = (
        missing_mtp == 'keep-base'
        and ac.get('model_type') == bc.get('model_type')
        and ac.get('model_type') in ('qwen3_5_moe', 'qwen3_5')
        and type(base_layers) is int and base_layers == 1
        and type(source_layers) is int and source_layers == 0
        and bool(base_mtp) and (not source_mtp or source_mtp == base_mtp)
    )
    acmp = {k: normalize(ac[k]) for k in CONFIG_KEYS if k in ac}
    bcmp = {k: normalize(bc[k]) for k in CONFIG_KEYS if k in bc}
    diff, adjustments = [], []
    for item in differences(acmp, bcmp):
        if item['field'] == 'vision_config.model_type' and vision_alias:
            adjustments.append(dict(item, reason='Known Transformers legacy vision model_type alias'))
        elif item['field'] == 'text_config.mtp_num_hidden_layers' and disabled_mtp:
            adjustments.append(dict(item, reason='Source disables MTP; preserve complete base MTP branch'))
        elif (item['field'] == 'text_config.pad_token_id'
              and ac.get('model_type') == bc.get('model_type')
              and ((item['base'] in (None, '<absent>') and type(item['sft']) is int)
                   or (item['sft'] in (None, '<absent>') and type(item['base']) is int))):
            adjustments.append(dict(item, reason='Export omits PAD ID; validated against reference tokenizer below'))
        else:
            diff.append(item)
    if ac.get('architectures') != bc.get('architectures'):
        diff.insert(0, {'field': 'architectures', 'base': ac.get('architectures'), 'sft': bc.get('architectures')})
    if diff:
        raise ValueError('Structural config mismatch:\n' + json.dumps(diff[:20], ensure_ascii=False, indent=2))
    only_base, only_sft = at.keys()-bt.keys(), bt.keys()-at.keys()
    # Only allow the exact known export case: the WHOLE MTP branch is absent
    # from SFT, all base-only tensors are MTP, and all core keys still match.
    copy_base_only = sorted(only_base) if (
        missing_mtp == 'keep-base' and only_base and not only_sft
        and all(key.startswith('mtp.') for key in only_base)
        and not any(key.startswith('mtp.') for key in bt)
        and ac.get('model_type') in ('qwen3_5_moe', 'qwen3_5')
    ) else []
    if (only_base or only_sft) and not copy_base_only:
        raise ValueError('Parameter names differ; no parameters were skipped.\n' + json.dumps({
            'only_base_count': len(only_base), 'only_sft_count': len(only_sft),
            'only_base_examples': sorted(only_base)[:20],
            'only_sft_examples': sorted(only_sft)[:20]}, indent=2))
    preserve_mtp = sorted(base_mtp if disabled_mtp else copy_base_only)
    dtype_changes = []
    for key in at:
        if key in copy_base_only:
            if at[key]['dtype'] not in FLOATS:
                raise ValueError('Expected floating MTP weights: ' + key)
            continue
        if at[key]['shape'] != bt[key]['shape']:
            raise ValueError('Shape mismatch for %s: %s / %s' % (key, at[key]['shape'], bt[key]['shape']))
        ad, bd = at[key]['dtype'], bt[key]['dtype']
        if ad != bd:
            if ad not in FLOATS or bd not in FLOATS:
                raise ValueError('Nonfloating dtype mismatch: ' + key)
            dtype_changes.append(key)
    assets = asset_check(base, sft)
    for item in adjustments:
        if item['field'] == 'text_config.pad_token_id':
            exported_pad = item['base'] if type(item['base']) is int else item['sft']
            if assets['base_pad_token_id'] != exported_pad:
                raise ValueError('Exported pad_token_id differs from reference tokenizer: '
                                 + str(exported_pad) + ' vs ' + str(assets['base_pad_token_id']))
    report = {'base': str(base), 'sft': str(sft), 'tensor_count': len(at),
              'parameters_and_buffer_elements': sum(math.prod(v['shape']) for v in at.values()),
              'output_tensor_bytes': sum(v['nbytes'] for v in at.values()),
              'floating_dtype_difference_count': len(dtype_changes), 'assets': assets,
              'missing_mtp_policy': missing_mtp,
              'config_compatibility_adjustments': adjustments,
              'expert_layout_conversions': layout_conversions,
              'copied_base_only_keys': copy_base_only,
              'copied_base_only_count': len(copy_base_only),
              'preserved_base_mtp_keys': preserve_mtp,
              'preserved_base_mtp_count': len(preserve_mtp),
              'averaged_floating_tensor_count': sum(v['dtype'] in FLOATS for k, v in at.items() if k not in preserve_mtp),
              'mtp_note': ('SFT omits or disables entire MTP branch: copied unchanged from base, not averaged. '
                           'Use ordinary decoding; MTP speculative decoding is not validated for this merged model.')
                          if preserve_mtp else 'No MTP fallback used',
              'base_files': af, 'sft_files': bf,
              'value_checks_pending': 'finite floats and equality of integer/bool buffers are checked during merge'}
    return report, at, bt


def average_tensor(a, b, alpha, chunk_elements):
    import torch
    if a.shape != b.shape:
        raise ValueError('Shape mismatch')
    if not a.is_floating_point() or not b.is_floating_point():
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError('Nonfloating buffers must be identical')
        return a.clone().contiguous()
    output = torch.empty(a.shape, dtype=a.dtype, device='cpu')
    av, bv, ov = a.reshape(-1), b.reshape(-1), output.reshape(-1)
    for start in range(0, av.numel(), chunk_elements):
        aa = av[start:start+chunk_elements].float()
        bb = bv[start:start+chunk_elements].float()
        if not torch.isfinite(aa).all() or not torch.isfinite(bb).all():
            raise ValueError('Source tensor has NaN/Inf')
        # mul (not mul_) is essential: .float() can alias an original FP32 tensor.
        value = aa.mul(1.0-alpha).add_(bb, alpha=alpha).to(a.dtype)
        if not torch.isfinite(value).all():
            raise ValueError('Average produced NaN/Inf or overflow')
        ov[start:start+chunk_elements].copy_(value)
    return output.contiguous()


def copy_tensor_checked(value, chunk_elements):
    """Preserve the base-only MTP tensor exactly; still reject nonfinite values."""
    import torch
    flat = value.reshape(-1)
    for start in range(0, flat.numel(), chunk_elements):
        if not torch.isfinite(flat[start:start+chunk_elements]).all():
            raise ValueError('Base-only MTP tensor has NaN/Inf')
    return value.clone().contiguous()


def shard_plan(tensors, limit):
    shards, current, size = [], [], 0
    for key in sorted(tensors):
        n = tensors[key]['nbytes']
        if current and size+n > limit:
            shards.append(current)
            current, size = [], 0
        current.append(key)
        size += n
    if current:
        shards.append(current)
    return shards


def unchanged(folder, files):
    for name, before in files.items():
        stat = (folder / name).stat()
        if stat.st_size != before['size'] or stat.st_mtime_ns != before['mtime_ns']:
            raise RuntimeError('Source weights changed during merge: ' + str(folder/name))


def merge(base, sft, out, alpha, shard_gb, chunk_elements, threads, missing_mtp='keep-base'):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if out.exists():
        raise ValueError('Output exists; choose a new --out directory')
    if out == base or out == sft or base in out.parents or sft in out.parents:
        raise ValueError('Output must be outside both source checkpoint directories')
    report, at, bt = preflight(base, sft, missing_mtp)
    copy_base_only = set(report['preserved_base_mtp_keys'])
    torch.set_num_threads(threads)
    print(json.dumps({k:report[k] for k in ('tensor_count','output_tensor_bytes','assets')}, ensure_ascii=False, indent=2), flush=True)
    if copy_base_only:
        print('MTP: retaining %d base tensors unchanged; ordinary decoding only. See merge_report.json.' % len(copy_base_only), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    needed = report['output_tensor_bytes'] + 256*1024**2
    if shutil.disk_usage(out.parent).free < needed:
        raise ValueError('Insufficient free space: need at least %.1f GiB' % (needed/1024**3))
    stage = Path(tempfile.mkdtemp(prefix=out.name+'.partial-', dir=str(out.parent)))
    print('Writing temporary checkpoint: ' + str(stage), flush=True)
    shards = shard_plan(at, int(shard_gb*1024**3))
    weight_map = {}
    try:
        with contextlib.ExitStack() as stack:
            aa = {f: stack.enter_context(safe_open(str(base/f), framework='pt', device='cpu')) for f in report['base_files']}
            bb = {f: stack.enter_context(safe_open(str(sft/f), framework='pt', device='cpu')) for f in report['sft_files']}
            for i, keys in enumerate(shards, 1):
                name = 'model-%05d-of-%05d.safetensors' % (i, len(shards))
                values = {}
                for key in keys:
                    try:
                        source_a = aa[at[key]['file']].get_tensor(key)
                        if key in copy_base_only:
                            values[key] = copy_tensor_checked(source_a, chunk_elements)
                        else:
                            values[key] = average_tensor(source_a,
                                bb[bt[key]['file']].get_tensor(key), alpha, chunk_elements)
                        del source_a
                    except Exception as exc:
                        raise ValueError('Failed parameter %s: %s' % (key, exc)) from exc
                    weight_map[key] = name
                save_file(values, str(stage/name), metadata={'format':'pt'})
                del values
                print('Saved shard %d/%d: %s' % (i, len(shards), name), flush=True)
        unchanged(base, report['base_files'])
        unchanged(sft, report['sft_files'])
        for name in sorted(ASSET_NAMES):
            if (base/name).is_file():
                shutil.copy2(base/name, stage/name)
        if (base/'chat_templates').is_dir():
            shutil.copytree(base/'chat_templates', stage/'chat_templates')
        (stage/'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': report['output_tensor_bytes']}, 'weight_map':weight_map}, indent=2)+'\n')
        _, actual, _ = inventory(stage)
        if set(actual) != set(at) or any(actual[k]['shape'] != at[k]['shape'] or actual[k]['dtype'] != at[k]['dtype'] for k in at):
            raise RuntimeError('Output header/index validation failed')
        report.update(status='complete', alpha_sft=alpha, weight_base=1-alpha,
                      formula='Shared floating tensors: (1-alpha_sft)*base + alpha_sft*SFT; FP32 arithmetic, base dtype. MTP tensors listed in preserved_base_mtp_keys are copied unchanged.',
                      output_shards=len(shards), nonfloating_buffers='identical only',
                      copied_asset_hashes={name:digest(stage/name) for name in ASSET_NAMES if (stage/name).is_file()})
        report.pop('value_checks_pending', None)
        report['value_checks'] = 'all source/output floats finite; integer/bool buffers identical'
        (stage/'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if out.exists():
            raise ValueError('Output directory appeared during merge; not replacing it')
        os.rename(str(stage), str(out))
        print('MERGE COMPLETE: ' + str(out), flush=True)
    except Exception:
        print('MERGE FAILED; incomplete files retained at: ' + str(stage), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE_DEFAULT)
    parser.add_argument('--sft', default=SFT_DEFAULT)
    parser.add_argument('--out', default='/path/to/merged-checkpoint')
    parser.add_argument('--alpha', type=float, default=.5, help='SFT coefficient; base coefficient is 1-alpha')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--missing-mtp', choices=['keep-base', 'error'], default='keep-base',
                        help='If SFT omits or disables the entire mtp.* branch: retain base MTP (default), or reject. Other missing keys always fail.')
    parser.add_argument('--shard-gb', type=float, default=2)
    parser.add_argument('--chunk-elements', type=int, default=4_000_000)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error('--alpha must be between 0 and 1')
    if not math.isfinite(args.shard_gb) or args.shard_gb <= 0 or args.chunk_elements < 1 or args.threads < 1:
        parser.error('shard size, chunk elements and threads must be positive')
    base, sft, out = (Path(x).resolve() for x in (args.base, args.sft, args.out))
    print('Base: %s\nSFT: %s\nSFT coefficient: %s' % (base, sft, args.alpha), flush=True)
    if args.check_only:
        report, _, _ = preflight(base, sft, args.missing_mtp)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print('PREFLIGHT PASSED (header/config/tokenizer checks; no weights written)', flush=True)
    else:
        merge(base, sft, out, args.alpha, args.shard_gb, args.chunk_elements, args.threads, args.missing_mtp)


if __name__ == '__main__':
    main()
