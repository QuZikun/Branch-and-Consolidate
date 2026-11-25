#!/usr/bin/env python3
"""Train exactly one experiment. The sequential launcher runs this twice."""
import argparse
import fcntl
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opd.common import require, read_json, write_json, fingerprint, file_hash, active_domains
from opd.config import load_settings, compose_config, distillation_settings, teacher_paths, training_budget, normal_sampling_budget, dynamic_sampling_budget, replay_sampling_budget
from opd.data import build_index, validate_sampler_resume
from opd.online_sampling import is_online, OnlineScheduler, validate_online_resume


def validate_checkpoint_schedule(config):
    """Reject incompatible save settings on CPU before allocating model workers."""
    options = config.opd.get('checkpoint_schedule', {})
    if not options.get('enabled'):
        return
    from omegaconf import OmegaConf
    from opd.checkpoints import CoverageCheckpointSchedule
    if OmegaConf.is_config(options):
        options = OmegaConf.to_container(options, resolve=True)
    CoverageCheckpointSchedule(options)


def start_feedback_worker(settings):
    """A CPU-only child follows durable jobs; no network probe gates training."""
    if settings.get('normal_sampling', {}).get('enabled'):
        return None
    if not settings.get('online_replay', {}).get('enabled'):
        return None
    run = Path(settings['run_dir'])
    path = run / 'feedback/worker_settings.json'
    selected = dict(settings.get('feedback_judge', {}))
    selected.setdefault('provider', 'internal')
    selected.setdefault('concurrency', 32)
    if selected['provider'] == 'internal':
        selected.setdefault('base_url', os.environ.get('EXP5_JUDGE_BASE', 'http://judge.example.invalid/v1'))
        selected.setdefault('judge_model', os.environ.get('EXP5_JUDGE_MODEL', 'judge-model'))
        selected.setdefault('api_key_env', 'INTERNAL_JUDGE_API_KEY')
    config = dict(settings, feedback_judge=selected)
    try:
        if selected.get('api_key'):
            raise ValueError('Do not serialize a Judge API key; use the shared provider config')
        write_json(path, config)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT), os.environ.get('PYTHONPATH', '')]))
        if selected['provider'] == 'internal':
            env.update(NO_PROXY='*', no_proxy='*')
        log = (run / 'feedback/worker.log').open('a')
        try:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'opd.feedback', 'run',
                                      '--run-dir', str(run), '--settings', str(path)],
                                     cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        print(f"Async feedback Judge started: PID={child.pid}, provider={selected['provider']}, "
              f"concurrency={selected['concurrency']}; training never waits for Judge.", flush=True)
        return child
    except Exception as exc:
        print(f'Async Judge unavailable: {type(exc).__name__}: {exc}; continue fresh training.', flush=True)
        return None


def stop_feedback_worker(child):
    if child is None or child.poll() is not None:
        return
    try:
        child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)
    except Exception as exc:
        print(f'Feedback worker cleanup: {type(exc).__name__}; queued jobs retained.', flush=True)


def print_training_budget(settings, pool_queries, exposures, steps):
    """Describe optimizer exposure rather than the much larger candidate pool."""
    replay = replay_sampling_budget(settings)
    if replay is not None:
        print(f"{settings['mode']}: normal_external_exposures={replay['normal_external_exposures']}, "
              f"normal_external_epochs={replay['epochs']}, "
              f"maximum_replay_exposures={replay['maximum_replay_exposures']}, "
              f"maximum_total_exposures_including_self={replay['total_exposures']}, "
              f"maximum_updates={steps}, GPUs={settings['nodes'] * settings['gpus_per_node']}", flush=True)
        stop_basis = ('the fixed normal external query budget (domain visit counts may differ)'
                      if settings.get('token_budget_sampling', {}).get('enabled')
                      else 'complete three-pass normal external coverage')
        print('Replay shortage uses normal data; self matches all external draws 1:1. '
              'Stop after ' + stop_basis + '; actual update count and replay fraction are recorded.', flush=True)
        return
    normal = normal_sampling_budget(settings) or dynamic_sampling_budget(settings)
    if normal is not None:
        print(f"{settings['mode']}: normal_external_exposures={normal['normal_external_exposures']}, "
              f"self_exposures={normal['self_exposures']}, total_exposures={normal['total_exposures']}, "
              f"normal_external_epochs={normal['epochs']}, BS={settings['global_batch_size']}, "
              f"steps={steps}, GPUs={settings['nodes'] * settings['gpus_per_node']}", flush=True)
        label = ('Reference external budget (dynamic allocation can differ)' if dynamic_sampling_budget(settings)
                 else 'Normal external exposures by route')
        print(label + ': ' + str(normal['external_by_route']), flush=True)
        return
    budget = training_budget(exposures, settings['global_batch_size'], settings['epochs'])
    print(f"{settings['mode']}: pool_queries={pool_queries}, initial_exposures={budget['exposures']}, "
          f"epochs={settings['epochs']}, BS={settings['global_batch_size']}, steps={steps}, "
          f"GPUs={settings['nodes'] * settings['gpus_per_node']}", flush=True)
    if settings.get('online_replay', {}).get('epochs'):
        print(f"Stop after {settings['online_replay']['fresh_external_queries']} normal external "
              f"draws across {settings['epochs']} epochs. Added error replay also receives matching "
              "self draws; record actual exposure, replay fraction and optimizer updates.", flush=True)
    elif settings.get('online_replay', {}).get('enabled'):
        print('Replay steps are an upper bound for progress only. Stop after fresh pools exhaust.', flush=True)


def probe(settings, hashes):
    import importlib
    import socket
    from opd.native import check_cuda_build
    for name in ('torch', 'ray', 'sglang', 'transformers', 'mbridge', 'megatron.core'):
        importlib.import_module(name)
    import torch
    from opd.network import verify_libraries
    from opd.ports import check_store_support
    communication = verify_libraries()
    rendezvous = check_store_support()
    require(torch.cuda.device_count() == settings['gpus_per_node'], 'Wrong visible GPU count on ' + socket.gethostname())
    for path, sha in hashes.items():
        require(file_hash(path) == sha, 'Code/input differs on node: ' + path)
    for path in {settings['student'], *teacher_paths(settings).values()}:
        require((Path(path) / 'config.json').is_file(), 'Missing model path ' + path)
    tools = None
    if 'agent' in active_domains(settings):
        from runner.tools import ToolRunner
        from agent.tools._common import BASE_URL
        from urllib.parse import urlsplit
        tools = ToolRunner(read_json(Path(settings['data_root']) / 'shared_context.json'),
                           {'timeout_seconds': settings['tool_timeout_seconds'], 'max_response_chars': settings['tool_response_chars']})
        backend = urlsplit(BASE_URL)
        with socket.create_connection((backend.hostname, backend.port or 80), timeout=10):
            pass
        import requests
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(os.environ['POI_SEARCH_URL'], json={'query': os.environ['OPD_POI_PROBE_QUERY'], 'topk': 1}, timeout=(5, 60))
            response.raise_for_status()
            require(bool(response.json().get('results')), 'POI query failed on ' + socket.gethostname())
    return {'tool_schemas': len(tools.schemas) if tools else 0, 'hostname': socket.gethostname(), 'cuda': check_cuda_build(),
            'communication': communication, 'rendezvous': rendezvous,
            'versions': {p: importlib.metadata.version(p) for p in ('ray', 'torch', 'transformers', 'sglang', 'mbridge', 'megatron-core')}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('original_trajectory', 'student_rollout'))
    p.add_argument('--config', default=str(PROJECT / 'opd/configs/business.json'))
    p.add_argument('--project-root', default=str(PROJECT))
    p.add_argument('--output-root')
    p.add_argument('--ray-address', default='auto')
    p.add_argument('--check-only', action='store_true', help='Compose/index/dependency checks; do not load GPU models')
    args = p.parse_args()
    s = load_settings(args.config, args.project_root, args.mode, args.output_root)
    run = Path(s['run_dir']); run.mkdir(parents=True, exist_ok=True)
    with (run / '.train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require((Path(s['repo']) / 'verl/trainer/config').is_dir(), 'Configure the compatible internal VERL checkout at: ' + s['repo'])
        index = build_index(s['data_root'], s['index'], active_domains(s), s.get('teachers'))
        sampling_order = None
        if s.get('sampling_order_file'):
            from opd.sampling import read_order
            sampling_order = read_order(s['sampling_order_file'], index['entries'])
            require(read_json(s['sampling_order_file'])['batch_size'] == s['global_batch_size'],
                    'Frozen sampling batch size differs from the optimizer batch size')
        if s.get('feedback_reference_index'):
            s['feedback_reference_index_sha256'] = file_hash(s['feedback_reference_index'])
        contract = {'settings': s, 'inputs': index['hashes']}
        if s.get('sampling_order_file'):
            contract['sampling_order_sha256'] = file_hash(s['sampling_order_file'])
        contract_file = run / 'contract.json'
        if contract_file.exists():
            require(read_json(contract_file) == contract, 'Training/data settings changed; restore config or use a new output root')
        else:
            write_json(contract_file, contract)
        if (run / 'result.json').exists() and read_json(run / 'result.json').get('state') == 'complete':
            print('Already completed: ' + str(run), flush=True)
            return
        sys.path[:0] = [s['repo'], s['agent_root'], s['step2_root']]
        from opd.native import cuda_build_env, logging_env
        os.environ.update(cuda_build_env(os.environ))
        from omegaconf import OmegaConf
        from verl.utils.config import validate_config
        exposures = len(sampling_order) if sampling_order is not None else len(index['entries'])
        config = compose_config(s, exposures)
        online = is_online(s)
        if online:
            config.trainer.total_training_steps = OnlineScheduler(s, index['entries'], sampling_order).maximum_updates
        performance = None
        require(not os.environ.get('OPD_PERFORMANCE_PROFILE'), 'Unset OPD_PERFORMANCE_PROFILE for these frozen ablations')
        if os.environ.get('OPD_PERFORMANCE_PROFILE') or (run / 'performance/latest.json').exists():
            from opd.performance import configure_runtime, record_runtime
            performance = configure_runtime(config, s, os.environ.get('OPD_PERFORMANCE_PROFILE'))
        validate_config(config, use_reference_policy=False, use_critic=False)
        validate_checkpoint_schedule(config)
        if online:
            validate_online_resume(s, index['entries'], sampling_order)
        else:
            validate_sampler_resume(len(index['entries']), s['global_batch_size'], s['seed'],
                                s.get('balanced_routes', False), sampling_order,
                                phase_boundary_updates=18 if s.get('sampling_order_file')
                                and read_json(s['sampling_order_file']).get('schema')
                                == 'mopd_exp2_delayed_self_v1' else None,
                                    epochs=s['epochs'])
        if performance is not None:
            record_runtime(performance, run, check_only=args.check_only)
        write_json(run / 'config.json', OmegaConf.to_container(config, resolve=True))
        (run / 'agent_loop.yaml').write_text('- name: opd_trajectory\n  _target_: opd.loops.OPDTrajectoryLoop\n')
        print_training_budget(s, len(index['entries']), exposures, config.trainer.total_training_steps)
        print('Trajectory sources: ' + ', '.join(f'{domain}={source}' for domain, source in s['trajectory_sources'].items()), flush=True)
        print('Distillation hyperparameters: ' + str(distillation_settings(s)), flush=True)
        if s.get('teachers'):
            print('Teacher routes (fixed weights): ' + str(teacher_paths(s)), flush=True)
            aggregation = s.get('loss_aggregation', 'route')
            print('Loss aggregation: ' + aggregation + '; route coefficients: '
                  + str(s.get('route_loss_coefficients')) + '; global scale: '
                  + str(s.get('global_loss_scale', 4.0 / 3.0)), flush=True)
            if s.get('teacher_roles'):
                print('Teacher roles (stable metric keys): ' + str(s['teacher_roles']), flush=True)
            print('Queries per teacher: ' + str(index['teacher_counts']), flush=True)
        if args.check_only:
            print('Config/index checks passed. No models loaded.')
            return
        require(os.environ.get('WANDB_API_KEY'), 'Set WANDB_API_KEY in the shell before starting')
        if 'agent' in active_domains(s):
            require(os.environ.get('POI_SEARCH_URL'), 'Run scripts/run_opd_pair.py to prepare services, or set POI_SEARCH_URL')
        import ray
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        from opd.network import apply_node_environment, check_collectives, PREFIXES
        env = get_ppo_ray_runtime_env(config)
        worker_env = env.setdefault('env_vars', {})
        require(os.environ.get('OPD_NETWORK_MANIFEST'), 'Use scripts/run_opd_pair.py to capture platform network settings')
        worker_env['OPD_NETWORK_MANIFEST'] = os.environ['OPD_NETWORK_MANIFEST']
        worker_env['OPD_DISTILLATION_LOSS_MODE'] = distillation_settings(s)['loss_mode']
        worker_env['OPD_COMM_TIMEOUT_SECONDS'] = os.environ.get('OPD_COMM_TIMEOUT_SECONDS', '120')
        env['worker_process_setup_hook'] = apply_node_environment
        worker_env['PYTHONPATH'] = os.pathsep.join([str(PROJECT), s['repo'], s['agent_root'], s['step2_root'], os.environ.get('PYTHONPATH', '')])
        worker_env.update(cuda_build_env(os.environ))
        # Internal services bypass proxy settings. W&B credentials/proxies go only to the logging actor.
        worker_env.update({'NO_PROXY': '*', 'no_proxy': '*', 'PYTHONUNBUFFERED': '1'})
        for key, value in os.environ.items():
            if key.startswith(('NVTE_', 'SGL_', 'VIDI_', 'POI_')) or key == 'CUDA_DEVICE_MAX_CONNECTIONS':
                worker_env[key] = value
        # Set LD_PRELOAD before Ray starts Python, not in the post-start hook.
        worker_env.update(read_json(os.environ['OPD_NETWORK_MANIFEST'])['loader'])
        # Network settings are node-specific; the setup hook supplies them.
        for key in list(worker_env):
            if key.startswith(PREFIXES):
                worker_env.pop(key)
        write_json(run / 'progress.json', {'state': 'starting', 'pid': os.getpid(), 'time': time.time()})
        feedback_worker = None
        ray.init(address=args.ray_address, runtime_env=env)
        try:
            nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0)]
            require(len(nodes) == s['nodes'] and all(n['Resources']['GPU'] == s['gpus_per_node'] for n in nodes), 'Ray GPU topology differs from config')
            require(ray.available_resources().get('GPU', 0) >= s['nodes'] * s['gpus_per_node'], 'GPUs are reserved by another Ray job')
            hashes = {str(path): file_hash(path) for path in Path(__file__).parent.glob('*.py')}
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
            pending = [ray.remote(num_cpus=1)(probe).options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'], soft=False)).remote(s, hashes) for n in nodes]
            reports = ray.get(pending, timeout=300)
            require(all(r['versions'] == reports[0]['versions'] for r in reports), 'Node dependency versions differ')
            require(all(r['communication'] == reports[0]['communication'] for r in reports),
                    'Actual loaded communication libraries differ between nodes')
            write_json(run / 'nodes.json', reports)
            check_collectives(nodes, s['gpus_per_node'], s['train_tp'], run / 'nccl_preflight.json')
            from opd.trainer import build_runner
            if not s.get('normal_sampling', {}).get('enabled'):
                feedback_worker = start_feedback_worker(s)
            if s.get('online_replay', {}).get('epochs'):
                require(feedback_worker is not None, 'Could not launch the configured feedback Judge worker')
            runner = ray.remote(num_cpus=1)(build_runner(distillation_settings(s))).remote()
            outcome = ray.get(runner.run.remote(config, logging_env(os.environ)))
            write_json(run / 'result.json', outcome)
            write_json(run / 'progress.json', outcome)
        except BaseException as exc:
            write_json(run / 'progress.json', {'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       'error_type': type(exc).__name__, 'time': time.time(), 'resume': 'Run the same command; latest completed checkpoint is used.'})
            raise
        finally:
            stop_feedback_worker(feedback_worker)
            ray.shutdown()


if __name__ == '__main__':
    main()
