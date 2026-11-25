"""Compose the supplied kai-verl config; all experiment choices live in JSON."""
import math
from pathlib import Path
from opd.common import SOURCES, read_json, require, trajectory_source, active_domains


def teacher_paths(settings):
    """Named fixed teachers; the legacy single-teacher contract stays unchanged."""
    if settings.get('teachers'):
        return {name: spec['path'] for name, spec in settings['teachers'].items()}
    return {'default': settings['teacher']}


def resource_layout(settings):
    """Resolve separate actor/teacher pools from configured nodes, not package names."""
    for key in ('nodes', 'teacher_nodes', 'gpus_per_node', 'train_tp', 'rollout_tp'):
        require(type(settings[key]) is int and settings[key] > 0, key + ' must be a positive integer')
    require(settings['teacher_nodes'] < settings['nodes'], 'Need separate student and teacher resource pools')
    actor = (settings['nodes'] - settings['teacher_nodes']) * settings['gpus_per_node']
    teachers = settings['teacher_nodes'] * settings['gpus_per_node']
    require(actor % settings['train_tp'] == 0, 'Student GPUs must divide training TP')
    require(settings['gpus_per_node'] % settings['rollout_tp'] == 0, 'Rollout TP must fit one node')
    return {'actor_gpus': actor, 'teacher_gpus': teachers, 'attention_dp': actor // settings['train_tp']}


def validate_route_settings(settings):
    """Keep semantic teacher roles beside stable metric keys and explicit weights."""
    names = set(settings.get('teachers', {}))
    coefficients = settings.get('route_loss_coefficients')
    from opd.experiment_controls import AGGREGATIONS, batch_coefficients, negative_feedback_alpha
    require(settings.get('loss_aggregation', 'route') in AGGREGATIONS, 'Unknown configured loss aggregation')
    if settings.get('loss_aggregation') == 'global_token_mean':
        batch_coefficients(settings)
    negative = settings.get('negative_distillation', {})
    if negative.get('enabled'):
        negative_feedback_alpha(negative, normal_progress=0., step=1)
    if coefficients is not None:
        require(settings.get('balanced_routes') is True, 'Explicit route coefficients require balanced_routes')
        require(set(coefficients) == names, 'Route coefficient keys must match the configured teachers')
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value > 0 for value in coefficients.values()),
                'Route coefficients must be finite positive numbers')
    roles = settings.get('teacher_roles')
    if roles is not None:
        require(set(roles) == names and all(isinstance(role, str) and role for role in roles.values()),
                'teacher_roles must describe every configured teacher key')


def validate_checkpoint_steps(settings, total_updates=None):
    steps = settings.get('checkpoint_steps')
    if steps is None:
        return
    require(isinstance(steps, (list, tuple)) and steps
            and all(type(step) is int and step > 0 for step in steps)
            and list(steps) == sorted(set(steps)),
            'checkpoint_steps must be a nonempty strictly increasing list of positive integers')
    require(not settings.get('checkpoint_schedule', {}).get('enabled')
            and not settings.get('final_only_checkpoint'),
            'Explicit checkpoint_steps cannot be combined with another save schedule')
    if total_updates is not None:
        require(steps[-1] == total_updates,
                'checkpoint_steps must end at the final optimizer update')


def distillation_settings(settings):
    """Defaults reproduce the existing forward-KL experiment exactly."""
    defaults = {'loss_mode': 'forward_kl_topk', 'topk': 32,
                'loss_max_clamp': 10.0, 'log_prob_min_clamp': -10.0,
                'chunk_tokens': 1024, 'probability_floor': 1e-8}
    supplied = settings.get('distillation', {})
    require(not set(supplied) - set(defaults), 'Unknown distillation hyperparameters: ' + str(set(supplied) - set(defaults)))
    values = {**defaults, **supplied}
    require(values['loss_mode'] in ('forward_kl_topk', 'reverse_kl_topk_tail'), 'Unsupported OPD loss_mode')
    require(type(values['topk']) is int and values['topk'] > 0, 'Distillation topk must be a positive integer')
    require(type(values['chunk_tokens']) is int and values['chunk_tokens'] > 0, 'KL chunk_tokens must be positive')
    for key in ('loss_max_clamp', 'log_prob_min_clamp'):
        value = values[key]
        require(value is None or isinstance(value, (int, float)) and math.isfinite(value), key + ' must be finite or null')
    require(values['loss_max_clamp'] is None or values['loss_max_clamp'] > 0, 'loss_max_clamp must be positive')
    require(values['log_prob_min_clamp'] is None or values['log_prob_min_clamp'] < 0, 'log_prob_min_clamp must be negative')
    floor = values['probability_floor']
    require(isinstance(floor, (int, float)) and math.isfinite(floor) and 0 < floor < 1 / (values['topk'] + 1),
            'probability_floor must be positive and smaller than 1/(topk+1)')
    if values['loss_mode'] == 'reverse_kl_topk_tail':
        require(values['log_prob_min_clamp'] is None, 'Reverse KL uses probability_floor; set log_prob_min_clamp=null')
    return values


def load_settings(path, project_root, mode, output_root=None):
    settings = read_json(path)
    resource_layout(settings)
    require(type(settings.get('save_epoch_boundaries', True)) is bool,
            'save_epoch_boundaries must be true or false')
    project = Path(project_root).resolve()
    for key in ('repo', 'data_root', 'agent_root', 'step2_root', 'student', 'teacher', 'output_root'):
        p = Path(output_root if key == 'output_root' and output_root else settings[key]).expanduser()
        settings[key] = str(p if p.is_absolute() else project / p)
    if settings.get('teachers'):
        require(len(settings['teachers']) >= 1, 'At least one named teacher is required')
        import re
        for name, spec in settings['teachers'].items():
            require(re.fullmatch('[a-z][a-z0-9_]*', name) and name != 'teacher_model', 'Invalid teacher name')
            p = Path(spec['path']).expanduser()
            spec['path'] = str(p if p.is_absolute() else project / p)
            require(type(spec['replicas']) is int and spec['replicas'] > 0, 'Teacher replicas must be positive integers')
        require(sum(x['replicas'] for x in settings['teachers'].values()) * settings['rollout_tp']
                == settings['teacher_nodes'] * settings['gpus_per_node'], 'Teacher replicas do not fill teacher GPU pool')
        if 'self' in settings['teachers']:
            reference = settings.get('self_reference_model', settings['student'])
            require(settings['teachers']['self']['path'] == reference,
                    'Self teacher must match the explicitly frozen self reference')
    validate_route_settings(settings)
    settings.update(project_root=str(project), mode=mode)
    # Part of contract.json and W&B config: old all-original checkpoints cannot resume.
    settings['trajectory_sources'] = {domain: trajectory_source(mode, domain) for domain in active_domains(settings)}
    settings['run_dir'] = str(Path(settings['output_root']) / mode)
    settings['index'] = str(Path(settings['run_dir']) / 'input_index.json')
    require(type(settings['epochs']) is int and settings['epochs'] > 0,
            'Epochs must be a positive integer')
    if settings.get('online_replay', {}).get('epochs'):
        require(settings['epochs'] == settings['online_replay']['epochs'],
                'Normal external epochs and replay contract differ')
    dynamic_sampling_budget(settings)
    replay_sampling_budget(settings)
    normal_sampling_budget(settings)
    if settings.get('checkpoint_steps') is not None:
        validate_checkpoint_steps(settings)
        normal = normal_sampling_budget(settings)
        if normal is not None:
            total_updates = math.ceil(normal['total_exposures'] / settings['global_batch_size'])
        else:
            with (Path(settings['data_root']) / 'train.jsonl').open('rb') as stream:
                count = sum(bool(line.strip()) for line in stream)
            total_updates = training_budget(count, settings['global_batch_size'], settings['epochs'])['updates']
        validate_checkpoint_steps(settings, total_updates)
    require(0 < settings['teacher_nodes'] < settings['nodes'], 'Need separate student and teacher resource pools')
    require(settings['gpus_per_node'] % settings['rollout_tp'] == 0, 'Rollout TP must fit one node')
    distillation_settings(settings)
    return settings


def training_budget(count, batch_size, epochs):
    """Keep the tail at every epoch boundary; do not concatenate epochs."""
    require(type(count) is int and count > 0, 'Training pool must be nonempty')
    require(type(batch_size) is int and batch_size > 0, 'Batch size must be positive')
    require(type(epochs) is int and epochs > 0, 'Epochs must be positive')
    return {'queries_per_epoch': count, 'epochs': epochs,
            'exposures': count * epochs,
            'updates_per_epoch': math.ceil(count / batch_size),
            'updates': math.ceil(count / batch_size) * epochs,
            'tail_queries': count % batch_size or batch_size}


def normal_sampling_budget(settings):
    """Exact five-pass external and matching self exposure, without feedback."""
    options = settings.get('normal_sampling', {})
    if not options.get('enabled'):
        return None
    require(not settings.get('online_replay', {}).get('enabled')
            and not settings.get('dynamic_sampling')
            and not settings.get('feedback_judge')
            and not settings.get('feedback_reference_index'),
            'Normal sampling must not enable replay, dynamic sampling or a feedback Judge')
    epochs = options.get('epochs')
    require(type(epochs) is int and epochs > 0 and epochs == settings['epochs'],
            'Normal sampling epochs must match the training coverage epochs')
    sizes = options.get('route_sizes')
    require(isinstance(sizes, dict) and sizes and all(type(n) is int and n > 0 for n in sizes.values()),
            'Normal sampling needs positive external route sizes')
    external = sum(sizes.values()) * epochs
    require(options.get('fresh_external_queries') == external,
            'Normal external exposure must cover every selected query in every epoch')
    ratio = options.get('external_self_ratio')
    require(isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
            and math.isfinite(ratio) and ratio > 0,
            'External:self exposure ratio must be positive')
    self_exposures = external / ratio
    require(self_exposures.is_integer(), 'Matching self exposure must be an integer')
    self_exposures = int(self_exposures)
    return {'epochs': epochs, 'external_by_route': {r: n * epochs for r, n in sizes.items()},
            'normal_external_exposures': external, 'self_exposures': self_exposures,
            'total_exposures': external + self_exposures}


def dynamic_sampling_budget(settings):
    """A fixed external budget measured in baseline-equivalent epochs."""
    options = settings.get('dynamic_sampling', {})
    if options.get('mode') != 'd3_composite' or 'epochs' not in options:
        return None
    require(not settings.get('normal_sampling', {}).get('enabled')
            and not settings.get('online_replay', {}).get('enabled'),
            'Dynamic sampling cannot run together with normal sampling or replay')
    epochs, sizes = options['epochs'], options.get('route_sizes')
    require(type(epochs) is int and epochs > 0 and epochs == settings['epochs'],
            'Dynamic budget epochs must match the configured training epochs')
    require(isinstance(sizes, dict) and set(sizes) == {'business_reasoning', 'general', 'business_agent'}
            and all(type(number) is int and number > 0 for number in sizes.values()),
            'Dynamic budget requires the three external pool sizes')
    external = sum(sizes.values()) * epochs
    require(options.get('fresh_external_queries') == external
            and float(options.get('external_self_ratio', 1)) == 1.,
            'Dynamic sampling uses the fixed baseline external budget with matching self exposure')
    require(0 <= float(options.get('ratio_floor', 1 / 6)) < 1 / 3
            <= float(options.get('ratio_cap', .5)) <= 1,
            'Invalid dynamic sampling bounds')
    require(type(options.get('warmup_updates', 17)) is int and options.get('warmup_updates', 17) >= 0
            and all(type(options.get(key, 5)) is int and options.get(key, 5) > 0
                    for key in ('update_every', 'window_updates', 'velocity_windows', 'seed_observations'))
            and math.isfinite(float(options.get('temperature', .5))) and float(options.get('temperature', .5)) > 0
            and math.isfinite(float(options.get('kl_floor', 1e-8))) and float(options.get('kl_floor', 1e-8)) > 0,
            'Invalid dynamic warmup, smoothing, temperature or update interval')
    return {'epochs': epochs, 'external_by_route': {r: size * epochs for r, size in sizes.items()},
            'normal_external_exposures': external, 'self_exposures': external,
            'total_exposures': 2 * external}


def replay_sampling_budget(settings):
    """Native update upper bound; the online controller stops on fresh coverage."""
    options = settings.get('online_replay', {})
    if not options.get('enabled') or 'epochs' not in options:
        return None
    epochs, sizes = options['epochs'], options.get('route_sizes')
    require(type(epochs) is int and epochs > 0 and epochs == settings['epochs']
            and isinstance(sizes, dict) and set(sizes) == {'business_reasoning', 'general', 'business_agent'}
            and all(type(number) is int and number > 0 for number in sizes.values()),
            'Replay budget requires positive epochs and three external pool sizes')
    external = sum(sizes.values()) * epochs
    require(options.get('fresh_external_queries') == external
            and float(options.get('external_self_ratio', 1)) == 1.
            and float(options.get('replay_fraction', .5)) == .5,
            'Replay requires 50% maximum external replay and matching self exposure')
    return {'epochs': epochs, 'external_by_route': {r: size * epochs for r, size in sizes.items()},
            'normal_external_exposures': external, 'maximum_replay_exposures': external,
            'self_exposures': 2 * external, 'total_exposures': 4 * external,
            'coverage_mode': ('fixed_external_budget' if settings.get('token_budget_sampling', {}).get('enabled')
                              else 'complete_external_epochs')}


def compose_config(settings, count):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    s = settings
    loss = distillation_settings(s)
    with initialize_config_dir(config_dir=str(Path(s['repo']) / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=['model_engine=megatron'])
    updates = read_json(Path(__file__).parent / 'configs/native_defaults.json')
    if loss['loss_mode'] == 'reverse_kl_topk_tail':
        updates['distillation.distillation_loss._target_'] = 'opd.losses.make_reverse_distillation_config'
    total, bs = s['max_sequence_tokens'], s['global_batch_size']
    layout = resource_layout(s)
    dp = layout['attention_dp']
    from opd.online_sampling import is_online
    online = is_online(s)
    normal_budget = normal_sampling_budget(s) or dynamic_sampling_budget(s) or replay_sampling_budget(s)
    steps = (math.ceil(normal_budget['total_exposures'] / bs) if normal_budget is not None
             else training_budget(count, bs, s['epochs'])['updates'])
    validate_checkpoint_steps(s, steps)
    tail = bs if online else count % bs or bs
    require(bs % dp == tail % dp == 0, 'Global batch and tail must divide attention-DP; choose another batch/TP')
    require(bs % s['agent_workers'] == tail % s['agent_workers'] == 0,
            'Global batch and tail must be divisible by agent_workers')
    updates.update({
        'data.train_files': str(Path(s['data_root']) / 'train.jsonl'),
        'data.val_files': str(Path(s['data_root']) / 'train.jsonl'),
        'data.opd_preserve_order': bool(s.get('preserve_order', s.get('balanced_routes', False))),
        'data.opd_sampling_order_file': s.get('sampling_order_file'),
        'data.train_batch_size': bs, 'data.gen_batch_size': bs, 'data.val_batch_size': 1,
        'data.custom_cls.path': str(Path(__file__).parent / 'dataset_adapter.py'),
        'data.custom_cls.name': 'OriginalDataset', 'data.opd_index': s['index'],
        # Separate padding widths. The loop enforces prompt + full response <= total.
        'data.max_prompt_length': total, 'data.max_response_length': total,
        'actor_rollout_ref.model.path': s['student'],
        'actor_rollout_ref.model.override_config': s.get('hf_overrides', {}),
        'actor_rollout_ref.actor.ppo_mini_batch_size': bs,
        'actor_rollout_ref.actor.data_loader_seed': s['seed'],
        'actor_rollout_ref.actor.megatron.seed': s['seed'],
        'actor_rollout_ref.rollout.seed': s['seed'],
        'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
        'actor_rollout_ref.actor.optim.lr': s['learning_rate'],
        'actor_rollout_ref.actor.optim.weight_decay': s.get('weight_decay', .01),
        'actor_rollout_ref.actor.optim.clip_grad': s.get('clip_grad', 1.),
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': total,
        'actor_rollout_ref.actor.megatron.tensor_model_parallel_size': s['train_tp'],
        'actor_rollout_ref.actor.megatron.expert_model_parallel_size': s['expert_parallel'],
        'actor_rollout_ref.actor.checkpoint.save_contents': ['model', 'optimizer', 'extra', 'hf_model'],
        'actor_rollout_ref.actor.checkpoint.load_contents': ['model', 'optimizer', 'extra'],
        'actor_rollout_ref.rollout.tensor_model_parallel_size': s['rollout_tp'],
        'actor_rollout_ref.rollout.gpu_memory_utilization': s['rollout_memory_fraction'],
        'actor_rollout_ref.rollout.max_model_len': total + 1,
        'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu': total,
        'actor_rollout_ref.rollout.calculate_log_probs': False,
        # T=1 for the distillation distributions; the loop separately samples with T=.8.
        'actor_rollout_ref.rollout.temperature': 1.,
        'actor_rollout_ref.rollout.max_num_seqs': s['max_running_requests'],
        'actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests': s['max_running_requests'],
        'actor_rollout_ref.rollout.agent.num_workers': s['agent_workers'],
        'actor_rollout_ref.rollout.agent.default_agent_loop': 'opd_trajectory',
        'actor_rollout_ref.rollout.agent.agent_loop_manager_class': 'opd.loops.OPDLoopManager',
        'actor_rollout_ref.rollout.agent.agent_loop_config_path': str(Path(s['run_dir']) / 'agent_loop.yaml'),
        'actor_rollout_ref.rollout.multi_turn.enable': True,
        'distillation.nnodes': s['teacher_nodes'], 'distillation.n_gpus_per_node': s['gpus_per_node'],
        'distillation.distillation_loss.loss_mode': loss['loss_mode'],
        'distillation.distillation_loss.topk': loss['topk'],
        'distillation.distillation_loss.loss_max_clamp': loss['loss_max_clamp'],
        'distillation.distillation_loss.log_prob_min_clamp': loss['log_prob_min_clamp'],
        'distillation.teacher_models.teacher_model.model_path': s['teacher'],
        'distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size': s['rollout_tp'],
        'distillation.teacher_models.teacher_model.inference.gpu_memory_utilization': s['teacher_memory_fraction'],
        'distillation.teacher_models.teacher_model.inference.max_model_len': total + 1,
        # Teacher receives one packed sequence of at most total tokens, not the sum
        # of the student's two independent *padding widths*. Native preparation sums these.
        'distillation.teacher_models.teacher_model.inference.prompt_length': total - 1,
        'distillation.teacher_models.teacher_model.inference.response_length': 1,
        'distillation.teacher_models.teacher_model.inference.max_num_seqs': s['max_running_requests'],
        'distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang': {
            'mm_processor_cpu_only': True, 'chunked_prefill_size': 4096,
            'max_running_requests': s['max_running_requests'], 'disable_prefill_cuda_graph': True},
        'reward.custom_reward_function.path': str(Path(__file__).parent / 'trainer.py'),
        'reward.custom_reward_function.name': 'neutral_task_reward',
        'trainer.nnodes': s['nodes'] - s['teacher_nodes'], 'trainer.n_gpus_per_node': s['gpus_per_node'],
        # Online controller owns all five normal passes in one continuous stream.
        'trainer.total_epochs': 1 if online else s['epochs'],
        'trainer.total_training_steps': steps,
        'trainer.save_freq': 1 if s.get('checkpoint_steps') else s['save_every'], 'trainer.resume_mode': 'auto',
        # Training never invokes the quick evaluator or Judge; stage 4 owns both.
        'trainer.test_freq': -1, 'trainer.val_before_train': False,
        'trainer.default_local_dir': str(Path(s['run_dir']) / 'checkpoints'),
        'trainer.project_name': s['wandb_project'], 'trainer.experiment_name': s.get('wandb_run_name', Path(s['output_root']).name) + '-' + s['mode'],
        'trainer.monitor_trace_dir': str(Path(s['run_dir']) / 'memory_trace'),
        'opd': s,
    })
    for key, value in updates.items():
        OmegaConf.update(config, key, value, merge=True, force_add=True)
    if s.get('teachers'):
        # Keep native teacher_model as the template sentinel: kai-verl removes it
        # while resolving named teachers. Deep copies prevent shared inference state.
        import copy
        template = OmegaConf.to_container(config.distillation.teacher_models.teacher_model, resolve=True)
        for name, spec in s['teachers'].items():
            teacher = copy.deepcopy(template)
            teacher.update(key=name, model_path=spec['path'], num_replicas=spec['replicas'])
            OmegaConf.update(config, 'distillation.teacher_models.' + name, teacher, merge=False, force_add=True)
        OmegaConf.update(config, 'distillation.teacher_key', 'teacher_key')
    return config
