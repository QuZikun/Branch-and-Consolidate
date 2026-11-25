"""Per-node platform communication settings and a bounded pre-model NCCL check."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

PREFIXES = ('NCCL_', 'GLOO_', 'KCCL_', 'TORCH_NCCL_')
LOADER_KEYS = ('LD_PRELOAD', 'LD_LIBRARY_PATH')


def loaded_communication_libraries(with_hash=False):
    """Inspect actual mapped libraries, not just NCCL's compatible version number."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if any(name in path.name.lower() for name in ('nccl', 'kccl')) and '.so' in path.name:
                paths.add(str(path))
    result = {}
    for path in sorted(paths):
        sha = None
        if with_hash:
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1048576), b''):
                    digest.update(block)
            sha = digest.hexdigest()
        result[path] = sha
    return result


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture(directory):
    """Run through the platform MPI wrapper on EVERY node, including the head."""
    name = socket.gethostname()
    env = {k: v for k, v in os.environ.items() if k.startswith(PREFIXES)}
    if not env.get('NCCL_SOCKET_IFNAME'):
        raise RuntimeError('Platform MPI did not provide NCCL_SOCKET_IFNAME on ' + name)
    # HCA names are local to each machine. Never broadcast the head's HCA list.
    save(Path(directory) / (name + '.json'), {
        'hostname': name, 'environment': env,
        'loader': {k: os.environ.get(k, '') for k in LOADER_KEYS},
        'preloaded_libraries': loaded_communication_libraries(with_hash=True)})
    print('Captured platform communication environment: ' + name, flush=True)


def apply_node_environment():
    """Ray setup hook: executes before importing/constructing training workers."""
    path = os.environ.get('DIST_RUNTIME_CONFIG')
    if not path:
        raise RuntimeError('Set DIST_RUNTIME_CONFIG to the distributed runtime configuration file')
    manifest = json.loads(Path(path).read_text())
    name = socket.gethostname()
    if name not in manifest['nodes']:
        raise RuntimeError('No platform communication environment for ' + name)
    missing = set(manifest['preloaded_libraries']) - set(loaded_communication_libraries())
    if missing:
        raise RuntimeError('Platform communication library was not preloaded when this Ray worker '
                           'started: ' + str(sorted(missing)))
    for key in list(os.environ):
        if key.startswith(PREFIXES):
            os.environ.pop(key)
    os.environ.update(manifest['nodes'][name])
    os.environ.update(manifest['overrides'])
    # Keep this hook stdlib-only: Ray invokes it before task/actor GPU isolation.
    # Importing torch/verl here can cache the parent's full CUDA device list.
    # Reverse loss registration happens in the Hydra config factory instead.


def verify_libraries():
    """Once per node, also verify the bytes of the preloaded platform library."""
    manifest = json.loads(Path(os.environ['DIST_RUNTIME_CONFIG']).read_text())
    actual = loaded_communication_libraries(with_hash=True)
    for path, sha in manifest['preloaded_libraries'].items():
        if actual.get(path) != sha:
            raise RuntimeError('Communication library differs from platform capture: ' + path)
    return {'loader': {k: os.environ.get(k, '') for k in LOADER_KEYS},
            'loaded_libraries': actual}


def probe_device_index(assigned, visible):
    """Map Ray's physical ID into this process's CUDA-visible ordinal."""
    assigned = [str(value) for value in assigned]
    if len(assigned) != 1:
        raise RuntimeError('Expected one Ray GPU assignment, got ' + str(assigned))
    gpu = assigned[0]
    if visible is not None:
        devices = [value.strip() for value in visible.split(',') if value.strip()]
        if len(devices) != len(set(devices)) or gpu not in devices:
            raise RuntimeError(f'Ray GPU {gpu} conflicts with CUDA_VISIBLE_DEVICES={visible!r}')
        return devices.index(gpu)
    if gpu.isdecimal():
        return int(gpu)
    raise RuntimeError('A UUID Ray GPU assignment requires CUDA_VISIBLE_DEVICES: ' + gpu)


def validate_probe_devices(devices, world):
    if len(devices) != world or sorted(row['rank'] for row in devices) != list(range(world)):
        raise RuntimeError('Incomplete communication-probe GPU inventory')
    seen = {}
    for row in devices:
        uuid = row['gpu_uuid']
        if not uuid:
            raise RuntimeError('Missing physical GPU UUID on rank ' + str(row['rank']))
        if uuid in seen:
            raise RuntimeError(f"Duplicate physical GPU before NCCL: ranks {seen[uuid]} and {row['rank']}, UUID={uuid}")
        seen[uuid] = row['rank']


class CollectiveProbe:
    def device(self, rank):
        # Called after Ray assigns the actor's resources, before importing CUDA.
        apply_node_environment()
        import ray
        assigned = ray.get_runtime_context().get_accelerator_ids().get('GPU', [])
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        index = probe_device_index(assigned, visible)
        import torch
        count = torch.cuda.device_count()
        if visible is not None and count != len(visible.split(',')):
            raise RuntimeError(f'CUDA saw {count} GPUs but CUDA_VISIBLE_DEVICES={visible!r}; '
                               'CUDA may have been initialized before Ray GPU assignment')
        if not 0 <= index < count:
            raise RuntimeError(f'Ray GPU assignment maps to cuda:{index}, but CUDA sees {count} GPUs')
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, 'uuid', None)
        if not uuid:
            raise RuntimeError('PyTorch did not provide a physical GPU UUID for the communication probe')
        self.assignment = dict(rank=rank, hostname=socket.gethostname(),
                               node_id=ray.get_runtime_context().get_node_id(),
                               ray_gpu_ids=[str(value) for value in assigned],
                               cuda_visible_devices=visible, cuda_device=index, visible_gpu_count=count,
                               gpu_uuid=str(uuid), gpu_name=properties.name)
        return self.assignment

    def endpoint(self):
        import ray
        with socket.socket() as sock:
            sock.bind(('', 0))
            port = sock.getsockname()[1]
        return f'tcp://{ray.util.get_node_ip_address()}:{port}'

    def run(self, rank, world, init_method, tp, seconds):
        # Bound even communicator initialization/driver calls stuck inside native code.
        watchdog = threading.Timer(seconds, lambda: os._exit(86))
        watchdog.daemon = True
        watchdog.start()
        dist = None
        try:
            apply_node_environment()
            import torch
            import torch.distributed as dist
            torch.set_num_threads(1)
            device = torch.device('cuda', self.assignment['cuda_device'])
            torch.cuda.set_device(device)
            dist.init_process_group('nccl', init_method=init_method, rank=rank,
                                    world_size=world, timeout=timedelta(seconds=seconds),
                                    device_id=device)
            value = torch.tensor([rank + 1.0], device=device)
            dist.all_reduce(value)
            if value.item() != world * (world + 1) / 2:
                raise RuntimeError('Global all_reduce returned an incorrect sum')
            # Exercise strided TP groups across the whole pool. This is a small
            # communication check, not an exact replay of Megatron's group topology.
            own_group = None
            for offset in range(tp):
                ranks = list(range(offset, world, tp))
                group = dist.new_group(ranks, timeout=timedelta(seconds=seconds))
                if rank in ranks:
                    own_group = group
            for _ in range(3):
                value.fill_(1)
                dist.all_reduce(value, group=own_group)
                if value.item() != world // tp:
                    raise RuntimeError('Data-parallel all_reduce returned an incorrect sum')
            torch.cuda.synchronize()
            result = dict(self.assignment, status='PASS',
                          libraries=list(loaded_communication_libraries()),
                          network={k: v for k, v in os.environ.items() if k.startswith(PREFIXES)})
            return result
        finally:
            try:
                if dist is not None and dist.is_initialized():
                    dist.destroy_process_group()
            finally:
                watchdog.cancel()


def check_collectives(nodes, gpus, tp, output):
    """Reserve tiny test actors on the idle pool, release them before model loading."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    seconds = int(os.environ.get('OPD_COMM_TIMEOUT_SECONDS', '120'))
    if not 30 <= seconds <= 600:
        raise ValueError('OPD_COMM_TIMEOUT_SECONDS must be 30..600')
    world = len(nodes) * gpus
    if world % tp:
        raise ValueError('Communication-check ranks must divide TP')
    actors, devices = [], []
    save(output, {'state': 'checking', 'ranks': world, 'time': time.time()})
    print(f'[opd] NCCL preflight: {world} ranks, global + DP all_reduce; no models; timeout={seconds}s', flush=True)
    try:
        cls = ray.remote(num_cpus=1, num_gpus=1, max_restarts=0)(CollectiveProbe)
        for node in sorted(nodes, key=lambda n: n['NodeManagerAddress']):
            for _ in range(gpus):
                actors.append(cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node['NodeID'], soft=False)).remote())
        devices = ray.get([actor.device.remote(rank) for rank, actor in enumerate(actors)], timeout=seconds)
        save(output, {'state': 'checking', 'ranks': world, 'devices': devices, 'time': time.time()})
        validate_probe_devices(devices, world)
        print(f'[opd] GPU assignment PASS: {world} distinct physical GPU UUIDs', flush=True)
        endpoint = ray.get(actors[0].endpoint.remote(), timeout=seconds)
        pending = [actor.run.remote(rank, world, endpoint, tp, seconds) for rank, actor in enumerate(actors)]
        reports = ray.get(pending, timeout=seconds + 30)
        save(output, {'state': 'passed', 'ranks': world, 'devices': devices, 'nodes': reports, 'time': time.time()})
        print(f'[opd] NCCL preflight PASS: {world} ranks; loading training models is now allowed.', flush=True)
    except Exception as exc:
        save(output, {'state': 'failed', 'ranks': world, 'devices': devices,
                      'error': str(exc)[-8000:], 'time': time.time()})
        raise RuntimeError('NCCL preflight failed BEFORE model loading; see ' + str(output)) from exc
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                print('NCCL probe cleanup: ' + str(exc)[-500:], flush=True)
    deadline = time.monotonic() + 45
    while ray.available_resources().get('GPU', 0) < world:
        if time.monotonic() >= deadline:
            raise RuntimeError('NCCL probe actors did not release GPUs; do not load models yet')
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    capture(parser.parse_args().capture)
