"""Small shared I/O helpers; no model or cluster imports."""
import hashlib
import json
import os
from pathlib import Path
import uuid

SOURCES = ('reasoning', 'agent')
EXPERIMENTS = ('student_rollout', 'original_trajectory')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trajectory_source(experiment, domain):
    """Reasoning is always on-policy; the experiment only selects Agent history."""
    require(experiment in EXPERIMENTS, 'Unknown OPD experiment: ' + str(experiment))
    require(domain in SOURCES, 'Unknown OPD domain: ' + str(domain))
    return 'student_rollout' if domain == 'reasoning' else experiment


def active_domains(settings):
    domains = tuple(settings.get('domains', SOURCES))
    require(domains and len(set(domains)) == len(domains) and set(domains) <= set(SOURCES),
            'domains must be a distinct nonempty subset of reasoning, agent')
    return domains


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
    return sha.hexdigest()
