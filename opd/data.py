"""Index original JSONL without reading images; deterministic one-epoch order."""
import hashlib
import json
import random
from pathlib import Path
from itertools import zip_longest
from .common import SOURCES, require, file_hash, write_json, read_json


def build_index(data_root, destination, domains=SOURCES, teachers=None):
    root = Path(data_root)
    require(domains and len(set(domains)) == len(domains) and set(domains) <= set(SOURCES), 'Invalid domains')
    entries, seen, counts = [], set(), dict.fromkeys(domains, 0)
    teacher_counts = dict.fromkeys(teachers or (), 0)
    with (root / 'train.jsonl').open('rb') as data, (root / 'train.meta.jsonl').open('rb') as metas:
        offset = 0
        for line, (raw, sidecar) in enumerate(zip_longest(data, metas), 1):
            require(raw is not None and sidecar is not None, 'JSONL and metadata have different lengths')
            meta = json.loads(sidecar)
            sid, mode = meta['sample_id'], meta['mode']
            require(sid not in seen and mode in domains, 'Duplicate ID or invalid domain')
            require(meta['output_line'] == line, 'Metadata line mismatch')
            # Export preserves source bytes; only the final row may have gained a newline.
            require(meta['row_sha256'] in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.removesuffix(b'\n')).hexdigest()), 'Original trajectory changed')
            require(isinstance(json.loads(raw).get('messages'), list), 'Missing messages')
            if teachers:
                require(meta.get('teacher_key') in teacher_counts, 'Missing/unknown teacher route')
                teacher_counts[meta['teacher_key']] += 1
            entries.append(dict(offset=offset, length=len(raw), **meta))
            offset += len(raw)
            seen.add(sid)
            counts[mode] += 1
    require(entries and all(counts.values()), 'Selected query pool has an empty domain')
    if len(domains) == 2 and not teachers:
        require(counts['reasoning'] == counts['agent'], 'Selected query pool must be reasoning : Agent = 1:1')
    if teachers:
        require(all(teacher_counts.values()), 'Every configured teacher needs training queries')
    files = ['train.jsonl', 'train.meta.jsonl']
    if 'agent' in domains:
        files.append('shared_context.json')
    result = {'root': str(root), 'entries': entries, 'counts': counts,
              'hashes': {name: file_hash(root / name) for name in files}}
    if teachers:
        result['teacher_counts'] = teacher_counts
    write_json(destination, result)
    return result


class Pool:
    def __init__(self, path):
        self.index = read_json(path)
        self.entries = self.index['entries']
        self.root = Path(self.index['root'])

    def row(self, index):
        meta = self.entries[int(index)]
        with (self.root / 'train.jsonl').open('rb') as stream:
            stream.seek(meta['offset'])
            raw = stream.read(meta['length'])
        require(meta['row_sha256'] in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.removesuffix(b'\n')).hexdigest()), 'Original row changed during training')
        return json.loads(raw), meta


# Keep GPU-heavy framework imports out of the index-building CLI.
def dataset_class():
    import torch
    from torch.utils.data import Dataset

    class OriginalDataset(Dataset):
        def __init__(self, data_files=None, tokenizer=None, config=None, processor=None, **kwargs):
            self.pool = Pool(config.opd_index)
            self.preserve_order = bool(config.get("opd_preserve_order", False))
            self.sampling_order = None
            if config.get('opd_sampling_order_file'):
                from opd.sampling import read_order
                self.sampling_order = read_order(config.opd_sampling_order_file, self.pool.entries)

        def __len__(self):
            return len(self.pool.entries)

        def __getitem__(self, index):
            entry = self.pool.entries[index]
            row = {'dummy_tensor': torch.tensor([0], dtype=torch.uint8),
                    'query_index': index, 'sample_id': entry['sample_id'], 'data_source': entry['mode'],
                    'agent_name': 'opd_trajectory', 'raw_prompt': [],
                    'reward_model': {'ground_truth': ''}, 'extra_info': {}}
            if 'teacher_key' in entry:
                row['teacher_key'] = entry['teacher_key']
            return row

    return OriginalDataset


def sampler_class():
    from torch.utils.data import Sampler

    class EpochSampler(Sampler):
        def __init__(self, dataset, seed):
            custom = getattr(dataset, 'sampling_order', None)
            self.order = list(custom) if custom is not None else list(range(len(dataset)))
            if custom is None and not getattr(dataset, "preserve_order", False):
                random.Random(seed).shuffle(self.order)

        def __iter__(self):
            return EpochIterator(self.order)

        def __len__(self):
            return len(self.order)

        def state_dict(self):
            return {'order': self.order}

        def load_state_dict(self, state):
            require(state['order'] == self.order, 'Resume query order changed')
    return EpochSampler


class EpochIterator:
    """torchdata checkpoints the active iterator independently from the sampler."""
    def __init__(self, order):
        self.order, self.position = order, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.order):
            raise StopIteration
        value = self.order[self.position]
        self.position += 1
        return value

    def state_dict(self):
        return {'order': self.order, 'position': self.position}

    def load_state_dict(self, state):
        require(state['order'] == self.order, 'Resume iterator order changed')
        require(type(state['position']) is int and 0 <= state['position'] <= len(self.order), 'Invalid iterator cursor')
        self.position = state['position']


def validate_sampler_resume(count, batch_size, seed, preserve_order=False, sampling_order=None,
                            phase_boundary_updates=None, epochs=1):
    """Cheap CPU check against the installed torchdata version, before GPU allocation."""
    from torchdata.stateful_dataloader import StatefulDataLoader
    cls = sampler_class()
    class QueryOrder(list):
        pass
    dataset = QueryOrder(range(count))
    dataset.preserve_order = preserve_order
    dataset.sampling_order = sampling_order
    def loader():
        return StatefulDataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False,
                                  sampler=cls(dataset, seed))
    expected = [batch.tolist() for batch in loader()]
    cuts = {0, min(5, len(expected)), max(0, len(expected) - 1)}
    # At an exhausted epoch checkpoint Trainer._load_checkpoint deliberately
    # constructs a fresh loader. Do not require a version-specific exhausted
    # StatefulDataLoader restoration behavior that the runtime never uses.
    if phase_boundary_updates is not None:
        require(type(phase_boundary_updates) is int and 0 < phase_boundary_updates < len(expected) - 1,
                'Invalid delayed-self resume boundary')
        cuts.update((phase_boundary_updates, phase_boundary_updates + 1))
    for cut in sorted(cuts):
        source = loader()
        iterator = iter(source)
        prefix = [next(iterator).tolist() for _ in range(cut)]
        restored = loader()
        restored.load_state_dict(source.state_dict())
        actual = prefix + [batch.tolist() for batch in restored]
        require(actual == expected, f'Installed torchdata did not restore sampler at batch {cut}')
    exposures = len(sampling_order) if sampling_order is not None else count
    require(sum(map(len, expected)) == exposures, 'Tail exposures lost or duplicated')
    if sampling_order is not None:
        require([i for batch in expected for i in batch] == sampling_order, 'Frozen exposure order changed')
    if epochs > 1:
        repeated = loader()
        actual_epochs = [[batch.tolist() for batch in repeated] for _ in range(epochs)]
        require(actual_epochs == [expected] * epochs,
                'Installed torchdata did not reset the frozen order at an epoch boundary')
    print('CPU dataloader resume/tail check passed against installed torchdata.', flush=True)
