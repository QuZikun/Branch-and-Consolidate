"""Stream verified complete candidate pools; keep only the selected unique rows."""
from collections import Counter
import hashlib
import heapq
from itertools import zip_longest
import json
import shutil
from .sampling import ROUTES


def stratum(meta):
    value = meta.get('sampling_stratum')
    if value not in ROUTES[:-1]:
        raise ValueError('Set sampling_stratum to a specialist route in the dataset metadata')
    return value


def proportional_quotas(total, mix):
    if set(mix) != set(ROUTES[:-1]) or any(v <= 0 for v in mix.values()):
        raise ValueError('Need all three baseline self strata')
    quotas = dict.fromkeys(ROUTES[:-1], 0)
    for _ in range(total):
        s = max(quotas, key=lambda s: total * mix[s] / sum(mix.values()) - quotas[s])
        quotas[s] += 1
    return quotas


def select_full_pool(root, summary, quotas, self_mix, seed):
    self_quotas = proportional_quotas(quotas['self'], self_mix)
    selected, audit, all_base, all_self = {}, {}, set(), set()
    for pool_name in ('base', 'self'):
        limit = self_quotas if pool_name == 'self' else {r: quotas[r] for r in ROUTES[:-1]}
        heaps = {r: [] for r in limit}
        totals, skipped, ids = Counter(), Counter(), set()
        hashes = summary['pools'][pool_name]['files_sha256']
        digests = {n: hashlib.sha256() for n in ('train.jsonl', 'train.meta.jsonl')}
        folder = root / pool_name
        with (folder / 'train.jsonl').open('rb') as data, (folder / 'train.meta.jsonl').open('rb') as metas:
            for line, (raw, sidecar) in enumerate(zip_longest(data, metas), 1):
                if raw is None or sidecar is None:
                    raise ValueError('Candidate JSONL/metadata lengths differ')
                digests['train.jsonl'].update(raw); digests['train.meta.jsonl'].update(sidecar)
                meta = json.loads(sidecar)
                if meta['output_line'] != line or meta['sample_id'] in ids:
                    raise ValueError('Invalid candidate row or duplicate ID')
                ids.add(meta['sample_id'])
                if meta['row_sha256'] not in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.rstrip(b'\n')).hexdigest()):
                    raise ValueError('Candidate row checksum differs')
                evidence = meta['selection_evidence']
                fp = meta['query_fingerprint']
                if pool_name == 'base':
                    if evidence['teacher']['clean_correct'] is not True or evidence['student']['correct'] is not False:
                        raise ValueError('External teacher pool must be teacher-correct/student-incorrect')
                    route = meta['teacher_key']
                    if fp in all_base:
                        skipped['duplicate_base_query'] += 1
                        continue
                    all_base.add(fp)
                else:
                    if meta['teacher_key'] != 'self' or evidence['student']['clean_correct'] is not True:
                        raise ValueError('Self pool must be original-student clean-correct')
                    route = stratum(meta)
                    if fp in all_base or fp in all_self:
                        skipped['self_overlap_or_duplicate'] += 1
                        continue
                    all_self.add(fp)
                    meta = dict(meta, sampling_stratum=route)
                if route not in limit:
                    raise ValueError('Unexpected candidate route: ' + route)
                totals[route] += 1
                rank = int(hashlib.sha256(f'{seed}:{pool_name}:{route}:{meta["sample_id"]}'.encode()).hexdigest(), 16)
                item = (-rank, meta['sample_id'], raw, meta)
                heap = heaps[route]
                if len(heap) < limit[route]:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        for name, digest in digests.items():
            if digest.hexdigest() != hashes[name]:
                raise ValueError('Full candidate pool changed: ' + str(folder / name))
        if hashlib.sha256((folder / 'shared_context.json').read_bytes()).hexdigest() != hashes['shared_context.json']:
            raise ValueError('Agent shared context changed')
        if pool_name == 'self' and hashes['shared_context.json'] != summary['pools']['base']['files_sha256']['shared_context.json']:
            raise ValueError('Base/self Agent tool contexts differ')
        audit[pool_name] = {'available_unique': dict(totals), 'exposure_quotas': limit,
                            'skipped': dict(skipped), 'source_hashes': hashes}
        for route, heap in heaps.items():
            if not heap:
                raise ValueError('No eligible candidates for ' + route)
            rows = [(raw, meta) for _, _, raw, meta in sorted(heap, key=lambda item: (-item[0], item[1]))]
            selected[('self:' if pool_name == 'self' else '') + route] = rows
        print(f'Full {pool_name} pool verified: eligible={dict(totals)}; draw quotas={limit}', flush=True)
    audit.update(self_draw_quotas=self_quotas, baseline_self_mix=dict(self_mix), seed=seed,
                 deduplication='By query fingerprint; base first, then self. First source occurrence retained.')
    return selected, audit


def export_pool(folder, selected, shared):
    folder.mkdir(parents=True)
    counts, entries = Counter(), []
    with (folder / 'train.jsonl').open('wb') as data, (folder / 'train.meta.jsonl').open('wb') as metas:
        for group in (*ROUTES[:-1], *('self:' + r for r in ROUTES[:-1])):
            for raw, original in selected[group]:
                raw = raw if raw.endswith(b'\n') else raw + b'\n'
                meta = dict(original, output_line=len(entries) + 1, row_sha256=hashlib.sha256(raw).hexdigest())
                data.write(raw); metas.write((json.dumps(meta, ensure_ascii=False) + '\n').encode())
                counts[meta['teacher_key']] += 1
                entries.append(meta)
    shutil.copyfile(shared, folder / 'shared_context.json')
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    return {'rows': len(entries), 'by_teacher': dict(counts),
            'files_sha256': {n: digest(folder / n) for n in ('train.jsonl', 'train.meta.jsonl', 'shared_context.json')}}, entries
