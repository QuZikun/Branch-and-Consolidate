#!/usr/bin/env python3
"""Average full HF checkpoints on CPU, with one final cast to the reference dtype."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from expert_layout import load_tensor

from average_weights import (
    ASSET_NAMES, FLOATS, copy_tensor_checked, digest, inventory, preflight,
    shard_plan, unchanged,
)


def coefficients(count, weights=None):
    if count < 2:
        raise ValueError('Provide at least two models')
    weights = [1.0] * count if weights is None else list(weights)
    if len(weights) != count:
        raise ValueError('--weights must contain one value per model')
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Weights must be finite and positive; omit zero-weight models')
    scaled = [w / max(weights) for w in weights]
    total = math.fsum(scaled)
    return [w / total for w in scaled]


def asset_hashes(folder):
    paths = [folder / name for name in sorted(ASSET_NAMES) if (folder / name).is_file()]
    if (folder / 'chat_templates').is_dir():
        paths += sorted(p for p in (folder / 'chat_templates').rglob('*') if p.is_file())
    return {str(p.relative_to(folder)): digest(p) for p in paths}


def preflight_many(models, weights=None, reference_index=0, missing_mtp='keep-base'):
    weights = coefficients(len(models), weights)
    if not 0 <= reference_index < len(models):
        raise ValueError('--reference-index is outside --models')
    if len(set(models)) != len(models):
        raise ValueError('Duplicate model directories; provide each checkpoint once')
    reference = models[reference_index]
    sources, tables, fallback = [], [], set()
    for i, model in enumerate(models):
        print(f'Check model {i+1}/{len(models)}: {model}', flush=True)
        check, _, table = preflight(reference, model, missing_mtp, convert_experts=True)
        conversions = check['expert_layout_conversions']
        if conversions:
            print(f'Expert layout conversion: {sum(c["source_tensor_count"] for c in conversions)} '
                  f'split tensors -> {len(conversions)} packed tensors; no expert weights skipped', flush=True)
        fallback.update(check['preserved_base_mtp_keys'])
        for adjustment in check['config_compatibility_adjustments']:
            print('Compatible export difference: ' + json.dumps(adjustment, ensure_ascii=False), flush=True)
        tables.append(table)
        sources.append({
            'path': str(model), 'weight': weights[i], 'files': check['sft_files'],
            'asset_hashes': asset_hashes(model),
            'tokenizer_status': check['assets']['sft_tokenizer_status'],
            'asset_differences_from_reference': check['assets']['asset_differences'],
            'floating_dtype_difference_count': check['floating_dtype_difference_count'],
            'missing_mtp_keys': check['copied_base_only_keys'],
            'preserved_reference_mtp_keys': check['preserved_base_mtp_keys'],
            'config_compatibility_adjustments': check['config_compatibility_adjustments'],
            'expert_layout_conversions': conversions,
        })
    reference_table = tables[reference_index]
    report = {
        'status': 'preflight_passed', 'sources': sources,
        'reference_index': reference_index, 'assets_and_output_dtype_from': str(reference),
        'tensor_count': len(reference_table),
        'output_tensor_bytes': sum(t['nbytes'] for t in reference_table.values()),
        'averaged_floating_tensor_count': sum(t['dtype'] in FLOATS and k not in fallback
                                             for k, t in reference_table.items()),
        'missing_mtp_policy': missing_mtp,
        'copied_reference_only_keys': sorted({k for s in sources for k in s['missing_mtp_keys']}),
        'preserved_reference_mtp_keys': sorted(fallback),
        'mtp_note': ('At least one source omits or disables the entire MTP branch. Keep the reference '
                     'MTP branch unchanged, even when another source has it. MTP is NOT averaged; '
                     'use ordinary decoding, without MTP speculative decoding.') if fallback
                    else 'All sources have matching keys; no MTP fallback.',
        'formula': 'Shared floating tensors: sum(weight[i] * model[i]); FP32 accumulation, '
                   'one final cast to reference dtype. Integer/bool buffers must be identical.',
        'value_checks_pending': 'Finite floats and identical nonfloating buffers checked during merge',
    }
    return report, tables


def average_tensors(tensors, weights, reference_index, chunk_elements):
    import torch
    reference = tensors[reference_index]
    if len(tensors) != len(weights) or any(t.shape != reference.shape for t in tensors):
        raise ValueError('Tensor count/shape mismatch')
    if not reference.is_floating_point():
        if any(t.dtype != reference.dtype or not torch.equal(t, reference) for t in tensors):
            raise ValueError('Nonfloating buffers must be identical across all models')
        return reference.clone().contiguous()
    if any(not t.is_floating_point() for t in tensors):
        raise ValueError('Floating/nonfloating dtype mismatch')
    output = torch.empty(reference.shape, dtype=reference.dtype, device='cpu')
    flat_sources = [t.reshape(-1) for t in tensors]
    flat_output = output.reshape(-1)
    for start in range(0, reference.numel(), chunk_elements):
        end = min(start + chunk_elements, reference.numel())
        value = torch.zeros(end-start, dtype=torch.float32)
        for tensor, weight in zip(flat_sources, weights):
            part = tensor[start:end].float()
            if not torch.isfinite(part).all():
                raise ValueError('Source tensor has NaN/Inf')
            value.add_(part, alpha=weight)
        cast = value.to(reference.dtype)
        if not torch.isfinite(cast).all():
            raise ValueError('Average produced NaN/Inf or overflow')
        flat_output[start:end].copy_(cast)
    return output.contiguous()


def merge_many(models, out, weights=None, reference_index=0, missing_mtp='keep-base',
               shard_gb=2, chunk_elements=4_000_000, threads=8):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not math.isfinite(shard_gb) or shard_gb <= 0 or chunk_elements < 1 or threads < 1:
        raise ValueError('Shard size, chunk elements and threads must be positive')
    if out.exists() or out.is_symlink():
        raise ValueError('Output exists; choose a new --out directory')
    if any(out == src or src in out.parents or out in src.parents for src in models):
        raise ValueError('Output must be separate from all source directories')
    report, tables = preflight_many(models, weights, reference_index, missing_mtp)
    weights = [s['weight'] for s in report['sources']]
    reference = models[reference_index]
    table = tables[reference_index]
    fallback = set(report['preserved_reference_mtp_keys'])
    torch.set_num_threads(threads)
    print(json.dumps({k: report[k] for k in ('tensor_count', 'output_tensor_bytes',
                      'assets_and_output_dtype_from', 'mtp_note')}, ensure_ascii=False, indent=2), flush=True)
    print('Normalized weights: ' + str(weights), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    needed = report['output_tensor_bytes'] + 256 * 1024**2
    if shutil.disk_usage(out.parent).free < needed:
        raise ValueError(f'Insufficient free space: need at least {needed / 1024**3:.1f} GiB')
    # Keep incomplete files separate; an interrupted merge is never a published model.
    stage = Path(tempfile.mkdtemp(prefix=out.name+'.partial-', dir=str(out.parent)))
    print('Writing temporary checkpoint: ' + str(stage), flush=True)
    shards = shard_plan(table, max(1, int(shard_gb * 1024**3)))
    weight_map = {}
    try:
        with contextlib.ExitStack() as stack:
            handles = [{name: stack.enter_context(safe_open(str(folder/name), framework='pt', device='cpu'))
                        for name in source['files']}
                       for folder, source in zip(models, report['sources'])]
            for number, keys in enumerate(shards, 1):
                filename = f'model-{number:05d}-of-{len(shards):05d}.safetensors'
                values = {}
                for key in keys:
                    try:
                        if key in fallback:
                            tensor = handles[reference_index][table[key]['file']].get_tensor(key)
                            values[key] = copy_tensor_checked(tensor, chunk_elements)
                            del tensor
                        else:
                            tensors = [load_tensor(h, t, key) for h, t in zip(handles, tables)]
                            values[key] = average_tensors(tensors, weights, reference_index, chunk_elements)
                            del tensors
                    except Exception as exc:
                        raise ValueError(f'Failed parameter {key}: {exc}') from exc
                    weight_map[key] = filename
                save_file(values, str(stage/filename), metadata={'format': 'pt'})
                del values
                print(f'Saved shard {number}/{len(shards)}: {filename}', flush=True)
        for folder, source in zip(models, report['sources']):
            unchanged(folder, source['files'])
            if asset_hashes(folder) != source['asset_hashes']:
                raise RuntimeError('Source assets changed during merge: ' + str(folder))
        for name in sorted(ASSET_NAMES):
            if (reference/name).is_file():
                shutil.copy2(reference/name, stage/name)
        if (reference/'chat_templates').is_dir():
            shutil.copytree(reference/'chat_templates', stage/'chat_templates')
        if asset_hashes(stage) != report['sources'][reference_index]['asset_hashes']:
            raise RuntimeError('Copied assets differ from preflight snapshot')
        (stage/'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': report['output_tensor_bytes']}, 'weight_map': weight_map}, indent=2)+'\n')
        _, actual, _ = inventory(stage)
        if set(actual) != set(table) or any(actual[k]['shape'] != table[k]['shape'] or
                                           actual[k]['dtype'] != table[k]['dtype'] for k in table):
            raise RuntimeError('Output header/index validation failed')
        report.update(status='complete', output=str(out), output_shards=len(shards),
                      value_checks='All contributing floats finite; nonfloating buffers identical',
                      copied_asset_hashes=asset_hashes(stage))
        report.pop('value_checks_pending')
        (stage/'merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if out.exists() or out.is_symlink():
            raise ValueError('Output appeared during merge; not replacing it')
        os.rename(stage, out)
        print('MERGE COMPLETE: ' + str(out), flush=True)
    except BaseException:
        print('MERGE INCOMPLETE; partial files retained at: ' + str(stage), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--weights', nargs='+', type=float, help='Positive ratios; default equal weights')
    parser.add_argument('--reference-index', type=int, default=0,
                        help='Zero-based model index supplying assets, output dtype and missing MTP')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--missing-mtp', choices=['keep-base', 'error'], default='keep-base')
    parser.add_argument('--shard-gb', type=float, default=2)
    parser.add_argument('--chunk-elements', type=int, default=4_000_000)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    models = [p.expanduser().resolve() for p in args.models]
    out = args.out.expanduser().resolve()
    try:
        if args.check_only:
            report, _ = preflight_many(models, args.weights, args.reference_index, args.missing_mtp)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            print('PREFLIGHT PASSED (headers/config/tokenizers only; no weights written)', flush=True)
        else:
            merge_many(models, out, args.weights, args.reference_index, args.missing_mtp,
                       args.shard_gb, args.chunk_elements, args.threads)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'ERROR {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
