"""Small adapters for one named teacher and a final-only checkpoint policy."""
import math
from pathlib import Path
import re
import shutil

from opd.common import require


def single_teacher_token_weights(mask, coefficient):
    import torch
    require(mask.ndim == 2 and len(mask) > 0 and bool(mask.bool().any(-1).all()),
            'Every real query requires at least one assistant token')
    coefficient = float(coefficient)
    require(math.isfinite(coefficient) and coefficient > 0,
            'Single-teacher coefficient must be finite and positive')
    return torch.full((len(mask),), coefficient / float(mask.bool().sum()),
                      dtype=torch.float32, device=mask.device)


def prune_recovery_checkpoints(root, final_step):
    """Only delete older local recovery points after the final export is valid."""
    root = Path(root)
    require(type(final_step) is int and final_step > 0 and not root.is_symlink(),
            'Invalid final checkpoint root or step')
    final = root / f'global_step_{final_step}'
    hf = final / 'actor/model/huggingface'
    require(final.is_dir() and not final.is_symlink() and hf.is_dir()
            and not hf.is_symlink() and (final / 'data.pt').is_file()
            and (hf / 'config.json').is_file()
            and any(p.is_file() and p.stat().st_size > 0 for p in hf.glob('*.safetensors')),
            'Final checkpoint/export incomplete; recovery points retained')
    obsolete = []
    for path in root.glob('global_step_*'):
        match = re.fullmatch(r'global_step_(\d+)', path.name)
        if match and int(match[1]) < final_step:
            require(path.is_dir() and not path.is_symlink()
                    and path.resolve().parent == root.resolve(),
                    'Unexpected recovery path; refusing checkpoint cleanup')
            obsolete.append(path)
    # Validate all paths before deleting any, including malicious symlink entries.
    for path in obsolete:
        shutil.rmtree(path)
    return final
