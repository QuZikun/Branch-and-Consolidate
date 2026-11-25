"""Explicit, run-scoped execution tuning; the training/data contract stays intact."""
import copy
import os
from pathlib import Path
import time
import uuid

from opd.common import read_json, write_json, require, fingerprint


def configure_runtime(config, settings, profile_path=None):
    from omegaconf import OmegaConf

    run = Path(settings['run_dir']).resolve()
    # Other jobs retain their exact existing config and behavior.
    if not profile_path and not (run / 'performance/latest.json').exists():
        return None
    before = OmegaConf.to_container(config, resolve=True)
    updates = {}
    profile = None
    if profile_path:
        profile = read_json(profile_path)
        required = {'version', 'name', 'run_dir', 'student_concurrency',
                    'dynamic_micro_batches', 'micro_batch_token_budget'}
        require(required <= set(profile) and not set(profile) - required - {'teacher_concurrency'},
                'Unexpected performance profile fields')
        require(profile['version'] == 1 and isinstance(profile['name'], str) and profile['name'],
                'Invalid performance profile version/name')
        require(Path(profile['run_dir']).is_absolute() and Path(profile['run_dir']).resolve() == run,
                'Performance profile belongs to another run; no overrides applied')
        modes = {'mopd3': 'forward_kl_topk', 'mopd6': 'reverse_kl_topk_tail'}
        experiment = Path(settings['output_root']).name
        require(settings['mode'] == 'student_rollout' and experiment in modes,
                'This profile is restricted to MOPD3/MOPD6 student_rollout')
        require(config.distillation.distillation_loss.loss_mode == modes[experiment]
                and set(settings.get('teachers', {})) == {'business', 'general', 'self'},
                'Expected the original MOPD3/MOPD6 loss and three frozen teachers')
        require(config.actor_rollout_ref.rollout.n == 1, 'This profile requires rollout N=1')
        concurrency = profile['student_concurrency']
        dynamic = profile['dynamic_micro_batches']
        budget = profile['micro_batch_token_budget']
        require(type(concurrency) is int and concurrency in (4, 8, 16), 'Student concurrency must be 4, 8 or 16')
        require(type(dynamic) is bool, 'dynamic_micro_batches must be boolean')
        require(type(budget) is int and settings['max_sequence_tokens'] <= budget <= 65536,
                'Micro-batch token budget must cover one full sequence and be at most 65536')
        teacher_concurrency = profile.get('teacher_concurrency')
        if teacher_concurrency is not None:
            require(type(teacher_concurrency) is int and teacher_concurrency in (4, 8, 16),
                    'Teacher concurrency must be 4, 8 or 16')
        if dynamic:
            require(config.actor_rollout_ref.model.use_remove_padding,
                    'Dynamic micro-batches require the native packed-token model path')
            require(config.actor_rollout_ref.actor.megatron.context_parallel_size == 1,
                    'Review token budgets before enabling context parallelism')
        updates = {
            'actor_rollout_ref.rollout.max_num_seqs': concurrency,
            'actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests': concurrency,
            'actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph': False,
            'actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs_decode': concurrency,
            'actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_bs_decode':
                [x for x in (1, 2, 4, 8, 16) if x <= concurrency],
            # kai-verl requires these two flags to agree. Native global token
            # normalization preserves the objective across micro-batch splits.
            'actor_rollout_ref.actor.use_dynamic_bsz': dynamic,
            'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': budget,
            'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': dynamic,
            'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': budget,
        }
        if teacher_concurrency is not None:
            for name in settings['teachers']:
                prefix = 'distillation.teacher_models.' + name + '.inference.'
                updates[prefix + 'max_num_seqs'] = teacher_concurrency
                updates[prefix + 'engine_kwargs.sglang.max_running_requests'] = teacher_concurrency
                updates[prefix + 'engine_kwargs.sglang.cuda_graph_max_bs_decode'] = teacher_concurrency
                updates[prefix + 'engine_kwargs.sglang.cuda_graph_bs_decode'] = [
                    x for x in (1, 2, 4, 8, 16) if x <= teacher_concurrency]
        for key, value in updates.items():
            OmegaConf.update(config, key, value, merge=False, force_add=True)

    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    metrics = {
        'student_concurrency': rollout.engine_kwargs.sglang.max_running_requests,
        'decode_graph_max_batch': rollout.engine_kwargs.sglang.cuda_graph_max_bs_decode,
        'dynamic_micro_batches': int(actor.use_dynamic_bsz),
        'train_micro_batch_token_budget': actor.ppo_max_token_len_per_gpu,
        'logprob_micro_batch_token_budget': rollout.log_prob_max_token_len_per_gpu,
        'rollout_n': rollout.n,
        'global_batch_size': config.data.train_batch_size,
    }
    for name in settings.get('teachers', {}):
        metrics['teacher_concurrency/' + name] = (
            config.distillation.teacher_models[name].inference.engine_kwargs.sglang.max_running_requests)
    # This namespace is execution metadata, not part of config.opd or its frozen
    # contract. Per-step metrics also expose changes when W&B resumes an old run.
    runtime = {'name': profile['name'] if profile else 'baseline', 'profile': profile,
               'metrics': metrics, 'profile_sha256': fingerprint(profile)}
    OmegaConf.update(config, 'opd_performance', runtime, merge=False, force_add=True)
    require(OmegaConf.to_container(config.opd, resolve=True) == before['opd'],
            'Execution tuning must not change the training/data contract')
    changes = []
    for key, value in updates.items():
        old = before
        for part in key.split('.'):
            old = old.get(part) if isinstance(old, dict) else None
        if old != value:
            changes.append({'field': key, 'before': old, 'after': value})
    return {**copy.deepcopy(runtime), 'changes_from_baseline': changes,
            'effective_config_sha256': fingerprint(OmegaConf.to_container(config, resolve=True))}


def record_runtime(report, run_dir, check_only=False):
    if report is None:
        return
    run = Path(run_dir)
    latest = run / 'checkpoints/latest_checkpointed_iteration.txt'
    checkpoint = latest.read_text().strip() if latest.exists() else None
    item = {**report, 'check_only': check_only, 'created_at': time.time(), 'pid': os.getpid(),
            'resume_checkpoint_hint': checkpoint}
    if check_only:
        write_json(run / 'performance/check.json', item)
    else:
        stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        write_json(run / 'performance' / (stamp + '.json'), item)
        write_json(run / 'performance/latest.json', item)
    print('[opd performance] ' + report['name'] + ': ' + str(report['metrics']), flush=True)
    print('[opd performance] existing checkpoint marker: ' + str(checkpoint)
          + '; native resume_mode=auto', flush=True)
