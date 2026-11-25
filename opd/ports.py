"""Keep each single-node SGLang replica's rendezvous port bound for its lifetime.

The server actor owns a CPU TCPStore. All TP ranks (including rank 0) connect
as clients, using PyTorch's existing agent-store mode. No socket hand-off or
fixed port list is involved. Importing this module does not import Torch/Ray.
"""
from contextlib import contextmanager
from datetime import timedelta
import os
import socket


def start_store():
    import torch.distributed as dist
    # port=0 lets the kernel select AND bind in one operation. Keep the store
    # object alive; closing a temporary socket would reintroduce the race.
    for attempt in range(5):
        try:
            return dist.TCPStore('127.0.0.1', 0, world_size=None, is_master=True,
                                 timeout=timedelta(seconds=60), wait_for_workers=False)
        except RuntimeError as exc:
            if 'EADDRINUSE' not in str(exc) or attempt == 4:
                raise


@contextmanager
def agent_store_environment():
    # Scope these to the SGLang server's spawn call. Training workers and the
    # driver must continue to use their own normal rendezvous configuration.
    updates = {'TORCHELASTIC_USE_AGENT_STORE': 'True',
               'SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE': ''}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def check_store_support():
    """Exercise rank-0/client rendezvous before GPU/model startup on each node."""
    import torch.distributed as dist
    store = start_store()
    try:
        store.set('opd_port_probe', 'ready')
        with agent_store_environment():
            for rank in (0, 1):
                client, actual_rank, world = next(dist.rendezvous(
                    f'tcp://127.0.0.1:{store.port}', rank=rank, world_size=2,
                    timeout=timedelta(seconds=10)))
                if (actual_rank, world) != (rank, 2) or client.get('opd_port_probe') != b'ready':
                    raise RuntimeError('PyTorch external TCPStore rendezvous check failed')
                del client
        return {'status': 'PASS', 'method': 'actor_owned_tcpstore', 'port': store.port}
    finally:
        del store


def managed_replica_class():
    """Use verl's backend loader (including its optional vLLM compatibility)."""
    import ray
    from verl.workers.rollout.replica import _load_sglang
    native_replica = _load_sglang()
    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    class ManagedServer(SGLangHttpServer):
        async def launch_server(self, master_address=None, master_port=None):
            if self.nnodes != 1:
                raise ValueError('OPD managed ports require each SGLang replica to fit one node')
            if getattr(self, '_opd_store', None) is not None:
                raise RuntimeError('SGLang replica is already starting or running')
            engine = dict(self.config.engine_kwargs or {})
            options = dict(engine.get('sglang') or {})
            if options.get('enable_dp_attention') or self.config.data_parallel_size != 1:
                raise ValueError('OPD managed ports support TP replicas with inference DP=1')
            if options.get('nccl_port') is not None or options.get('dist_init_addr') is not None:
                raise ValueError('Remove manual nccl_port/dist_init_addr; OPD allocates each replica its own store')
            self._opd_store = start_store()
            port = self._opd_store.port
            options.update(nccl_port=port, dist_init_addr=f'127.0.0.1:{port}')
            engine['sglang'] = options
            # Base launch pops options; never mutate a config shared by replicas.
            original_engine = self.config.engine_kwargs
            self.config.engine_kwargs = engine
            print(f'[opd] SGLang rendezvous bound: host={socket.gethostname()} '
                  f'pid={os.getpid()} replica={self.replica_rank} port={port}; TP ranks use clients', flush=True)
            try:
                with agent_store_environment():
                    await super().launch_server(master_address, master_port)
            except BaseException:
                self._opd_store = None
                raise
            finally:
                self.config.engine_kwargs = original_engine
            # Retain the store until Ray destroys this actor, including sleep /
            # wake-up and weight updates. The OS closes the listener on exit.
            print(f'[opd] SGLang rendezvous ready: replica={self.replica_rank} port={port}', flush=True)

    class ManagedReplica(native_replica):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server_class = ray.remote(ManagedServer)

    return ManagedReplica


def install_managed_ports():
    # Called only inside the training TaskRunner, after Ray resource assignment.
    # Both the teacher manager and student LLMServerManager use this registry.
    from verl.workers.rollout.replica import RolloutReplicaRegistry
    RolloutReplicaRegistry.register('sglang', managed_replica_class)
    print('[opd] SGLang ports: actor-owned TCPStore, automatic allocation, no released-port hand-off', flush=True)


if __name__ == '__main__':
    import json
    print(json.dumps(check_store_support(), indent=2))
