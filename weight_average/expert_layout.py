"""Read split Qwen3.5 MoE experts in the reference's packed layout."""
import re


def align_experts(reference, source, model_type):
    result = dict(source)
    conversions = []
    pattern = re.compile(r'^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$')
    for key, target in reference.items():
        match = pattern.fullmatch(key)
        if not match:
            continue
        prefix, projection = match.groups()
        split = {k for k in source if k.startswith(prefix + '.') and
                 re.fullmatch(r'\d+\.(gate_proj|up_proj|down_proj)\.weight', k[len(prefix)+1:])}
        if not split:
            continue
        if model_type != 'qwen3_5_moe':
            raise ValueError('Split expert conversion requires qwen3_5_moe')
        if key in source:
            raise ValueError('Mixed packed/split expert export: ' + key)
        shape = target['shape']
        if len(shape) != 3:
            raise ValueError('Expected rank-3 packed experts: ' + key)
        experts, rows, cols = shape
        if projection == 'gate_up_proj' and rows % 2:
            raise ValueError('Odd gate/up dimension: ' + key)
        names = ('gate_proj', 'up_proj') if projection == 'gate_up_proj' else ('down_proj',)
        expected_shape = [rows // len(names), cols]
        pieces = []
        for expert in range(experts):
            for name in names:
                original = f'{prefix}.{expert}.{name}.weight'
                if original not in source:
                    raise ValueError('Missing split expert parameter: ' + original)
                entry = source[original]
                if entry['shape'] != expected_shape:
                    raise ValueError(f'Split expert shape mismatch: {original}: {entry["shape"]} != {expected_shape}')
                pieces.append(dict(entry, key=original, expert=expert, part=names.index(name)))
        dtypes = {p['dtype'] for p in pieces}
        if len(dtypes) != 1 or not dtypes <= {'F16', 'BF16', 'F32'}:
            raise ValueError('Split experts must have one floating dtype: ' + key)
        for piece in pieces:
            result.pop(piece['key'])
        result[key] = {'shape': shape, 'dtype': pieces[0]['dtype'],
                       'nbytes': sum(p['nbytes'] for p in pieces), 'pieces': pieces,
                       'parts_per_expert': len(names)}
        conversions.append({'target': key, 'source_tensor_count': len(pieces),
                            'layout': 'numeric expert order; gate then up; no transpose'})
    # Unknown/extra source keys remain visible to the strict name check.
    return result, conversions


def load_tensor(handles, table, key):
    import torch
    entry = table[key]
    if 'pieces' not in entry:
        return handles[entry['file']].get_tensor(key)
    dtype = {'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32}[entry['dtype']]
    value = torch.empty(entry['shape'], dtype=dtype, device='cpu')
    rows = entry['shape'][1] // entry['parts_per_expert']
    for part in entry['pieces']:
        start = part['part'] * rows
        value[part['expert'], start:start+rows].copy_(handles[part['file']].get_tensor(part['key']))
    return value
