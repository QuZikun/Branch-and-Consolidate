#!/usr/bin/env python3
"""Durable asynchronous answer-only Judge feedback; never waits inside training.

Task-specific correctness rules are supplied through feedback_judge.task_rubrics.
No GPU/framework dependency is imported.
"""
import json
import re

def question_text(messages):
    result, image_index = [], 0
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            parts = []
            for block in content:
                if block.get('type') == 'text':
                    parts.append(block['text'])
                elif block.get('type') in ('image', 'image_url'):
                    image_index += 1
                    parts.append(f'[图片{image_index}：本次不提供图像，仅依据参考答案评判]')
                else:
                    raise ValueError('Unsupported question content block')
            content = '\n'.join(parts)
        if not isinstance(content, str):
            raise ValueError('Question content must be text or content blocks')
        result.append({'role_in_original_question': message['role'], 'text': content})
    return result


def base_payload(sample, overrides):
    reference = sample.get('gt')
    if reference is None or (isinstance(reference, str) and not reference.strip()):
        raise ValueError('missing_reference')
    mode, task = sample['mode'], sample['task']
    if mode not in ('reasoning', 'agent'):
        raise ValueError('unknown_mode: ' + mode)
    rubric = overrides.get(task)
    if not isinstance(rubric, str) or not rubric.strip():
        raise ValueError('Supply a dataset-owned correctness rubric for task: ' + task)
    return {'mode': mode, 'task': task, 'question': question_text(sample['messages']),
            'reference_answer': reference, 'rubric': rubric,
            'evaluation_scope': 'reference_based_final_answer_only'}


def parse_verdict(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    value = json.loads(text)
    if (not isinstance(value, dict) or 'correct' not in value
            or (value['correct'] is not None and type(value['correct']) is not bool)
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()):
        raise ValueError('Judge must return boolean/null correct and nonempty reason')
    return {'correct': value['correct'], 'reason': value['reason'].strip()}


import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import math
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit

DEFAULT_JUDGE = {'base_url': 'http://judge.example.invalid/v1', 'judge_model': 'judge-model',
                 'concurrency': 32, 'temperature': 0.0, 'max_tokens': 1024, 'enable_thinking': False,
                 'timeout_seconds': 120, 'retries': 0, 'retry_backoff_seconds': 5,
                 'max_request_chars': 0, 'max_response_bytes': 2 * 1024 * 1024,
                 'api_key_env': 'Judge_API_KEY', 'use_proxy': False, 'task_rubrics': {},
                 'system_prompt': None,
                 'provider': 'internal', 'strict_response_model': True}
_REFERENCES = {}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temporary.open('w') as stream:
            stream.write(dumps(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def judge_config(settings):
    """Resolve an explicit internal Judge, preserving legacy provider compatibility.

    Credentials remain in process memory. Worker settings contain only the
    provider configuration path and public choices, never a copied API key.
    """
    config = dict(DEFAULT_JUDGE)
    supplied = dict(settings.get('feedback_judge', settings))
    config.update({k: supplied[k] for k in DEFAULT_JUDGE if k in supplied})
    if config['provider'] not in ('internal', 'aliyun'):
        raise ValueError('Unsupported feedback Judge provider')
    if supplied.get('api_key'):
        raise ValueError('Feedback credentials must use the provider file or API key environment')
    if config['provider'] == 'aliyun':
        bench_root = supplied.get('bench_root') or settings.get('feedback_bench_root') or settings.get('bench_root')
        if not bench_root:
            bench_root = str(Path(settings.get('project_root', Path(__file__).resolve().parents[1])) / 'mopd_bench')
        provider_file = Path(bench_root) / 'configs/judge.json'
        saved = read_json(provider_file)
        provider = saved.get('providers', {}).get('aliyun')
        if not isinstance(provider, dict):
            raise ValueError('Benchmark config has no Aliyun Judge provider')
        for field in ('base_url', 'judge_model', 'api_key_env', 'timeout_seconds'):
            if field in provider:
                config[field] = provider[field]
        key = provider.get('api_key') or os.getenv(config['api_key_env'])
        if not key or key == 'EMPTY':
            raise ValueError('Aliyun Judge API key missing from benchmark config/environment')
        config['api_key'] = key
        config['provider_config'] = str(provider_file)
        # Public DashScope follows the launcher proxy; a response may name a
        # resolved model revision, as the existing benchmark client permits.
        config['use_proxy'] = config['base_url'].lower().startswith('https://')
        config['strict_response_model'] = False
    else:
        config['use_proxy'] = False
    config['retries'] = 0  # Retry scheduling is persisted by the coordinator.
    if type(config['concurrency']) is not int or not 1 <= config['concurrency'] <= 32:
        raise ValueError('Feedback Judge concurrency must be in [1,32]')
    endpoint = urlsplit(config['base_url']) if isinstance(config['base_url'], str) else None
    if (endpoint is None or endpoint.scheme not in ('http', 'https') or not endpoint.netloc
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError('Feedback Judge requires an HTTP endpoint without embedded credentials')
    if (type(config['timeout_seconds']) not in (int, float) or not math.isfinite(config['timeout_seconds'])
            or config['timeout_seconds'] <= 0 or not isinstance(config['judge_model'], str)
            or not config['judge_model'].strip()):
        raise ValueError('Feedback Judge requires a model and positive timeout')
    if not isinstance(config['system_prompt'], str) or not config['system_prompt'].strip():
        raise ValueError('Supply the dataset-owned feedback_judge.system_prompt')
    if not isinstance(config['api_key_env'], str) or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', config['api_key_env']):
        raise ValueError('Invalid feedback Judge API key environment variable')
    return config


def reference_for(entry, settings):
    if entry.get('feedback_reference'):
        return dict(entry['feedback_reference'])
    path = Path(settings['feedback_reference_index'])
    st = path.stat(); key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _REFERENCES:
        index = read_json(path)
        if index.get('schema') != 'mopd_exp5_feedback_references_v1':
            raise ValueError('Unexpected feedback reference index')
        if settings.get('feedback_reference_index_sha256') and file_hash(path) != settings['feedback_reference_index_sha256']:
            raise ValueError('Frozen feedback reference index changed')
        _REFERENCES.clear(); _REFERENCES[key] = index['references']
    result = dict(_REFERENCES[key][entry['sample_id']])
    if result.get('query_fingerprint') != entry['query_fingerprint']:
        raise ValueError('Effective query/reference fingerprint differs')
    return result


def enqueue_job(run, step, entry, trajectory_path, report, settings):
    """Publish one optimizer-committed trajectory without HTTP or Judge waits.

    Returns a deterministic step/slot job ID, or None on any local failure.
    Re-generating a step changes its version; old in-flight results remain audit
    records but cannot become replay feedback for the replacement trajectory.
    """
    try:
        if type(step) is not int or step <= 0 or report.get('step') != step:
            raise ValueError('Feedback requires a committed positive optimizer step')
        if entry['sample_id'] != report['effective_id']:
            raise ValueError('Feedback entry must describe the effective replacement query')
        trajectory_path = Path(trajectory_path)
        raw = trajectory_path.read_bytes(); trajectory = json.loads(raw)
        stored = trajectory.get('report', {})
        if any(stored.get(k) != report.get(k) for k in ('step', 'slot_id', 'effective_id', 'domain')):
            raise ValueError('Feedback report/trajectory identity differs')
        reference = reference_for(entry, settings)
        root = Path(run) / 'feedback'
        job_id = digest([step, report['slot_id']])
        content = {'schema': 'mopd_exp5_feedback_job_v1', 'job_id': job_id, 'step': step,
                   'query_fingerprint': entry['query_fingerprint'], 'slot_id': report['slot_id'],
                   'effective_id': report['effective_id'], 'teacher_key': entry.get('teacher_key'),
                   'mode': report['domain'], 'task': report.get('task'), 'reference': reference,
                   'trajectory_path': str(trajectory_path), 'trajectory_sha256': hashlib.sha256(raw).hexdigest(),
                   'trajectory': trajectory, 'report': dict(report)}
        version = digest(content)
        payload_path = root / 'jobs' / (job_id + '.' + version + '.json')
        if not payload_path.exists():
            atomic(payload_path, dict(content, job_sha256=version, published_at=time.time()))
        header = {k: content[k] for k in ('job_id', 'step', 'query_fingerprint', 'slot_id', 'effective_id', 'teacher_key')}
        header.update(job_sha256=version, payload_path=str(payload_path))
        target = root / 'outbox' / (job_id + '.json')
        if not target.exists() or read_json(target) != header:
            atomic(target, header)
        return job_id
    except Exception as exc:
        # Feedback is ancillary: a missing reference, disk or Judge problem must
        # never turn an otherwise successful optimizer step into a training error.
        print('Feedback enqueue skipped: ' + type(exc).__name__ + ': ' + str(exc)[:250], file=sys.stderr, flush=True)
        return None


def result_path(root, header):
    return root / 'results' / (header['job_id'] + '.' + header['job_sha256'] + '.json')


def state_path(root, header):
    return root / 'state' / (header['job_id'] + '.' + header['job_sha256'] + '.json')


def successful_feedback(run, committed_step):
    """Latest valid boolean judgement per teacher/query, from committed steps.

    A newer success removes an older wrong flag when its verdict is correct.
    Late older answers never replace a newer successful generation. Unknown,
    input errors and HTTP errors are absent from this view, not interpreted as
    wrong. Filesystem errors return the remaining valid records, never raise.
    """
    latest = {}
    try:
        root = Path(run) / 'feedback'
        for path in (root / 'outbox').glob('*.json'):
            try:
                header = read_json(path)
                if type(header.get('step')) is not int or header['step'] > committed_step:
                    continue
                target = result_path(root, header)
                if not target.exists():
                    continue
                result = read_json(target)
                if (result.get('status') != 'judged' or type(result.get('correct')) is not bool
                        or any(result.get(k) != header.get(k) for k in
                               ('job_id', 'job_sha256', 'step', 'query_fingerprint', 'effective_id', 'teacher_key'))):
                    continue
                key = (result.get('teacher_key') or '', result['query_fingerprint'])
                # Completion wall time is deliberately not the ordering key.
                rank = (result['step'], result['job_id'])
                previous = latest.get(key)
                if previous is None or rank > (previous['step'], previous['job_id']):
                    latest[key] = result
            except (OSError, ValueError, KeyError, TypeError):
                continue
    except Exception:
        pass
    return [latest[key] for key in sorted(latest)]


def load_job(header):
    job = read_json(header['payload_path'])
    version = job.pop('job_sha256'); job.pop('published_at', None)
    if version != header['job_sha256'] or digest(job) != version:
        raise ValueError('Feedback job snapshot checksum differs')
    return job


def final_answer(job):
    detail = job['trajectory'].get('detail', {})
    if job['report'].get('trajectory_source') != 'student_rollout':
        raise ValueError('Feedback only evaluates current student on-policy trajectories')
    turns = detail.get('turns')
    if not isinstance(turns, list) or not turns or turns[-1].get('calls'):
        raise ValueError('Current rollout has no completed final assistant turn')
    answer = turns[-1].get('answer')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Current rollout has no final answer')
    return answer


def evaluate_job(config, header, stopped):
    try:
        job = load_job(header)
        reference = job['reference']; path = Path(reference['path'])
        if file_hash(path) != reference['sha256']:
            raise ValueError('Frozen feedback reference changed')
        sample = read_json(path)
        if sample.get('mode') != job['mode'] or sample.get('task') != job['task']:
            raise ValueError('Feedback reference task/domain differs from effective query')
        fingerprint = hashlib.sha256(json.dumps({k: sample.get(k) for k in ('messages', 'image_paths', 'mode')},
                                                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if fingerprint != job['query_fingerprint']:
            raise ValueError('Feedback reference question differs from effective query')
        answer = final_answer(job)
        payload = base_payload(sample, config['task_rubrics'])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'status': 'input_error', 'correct': None, 'reason': str(exc), 'attempts': []}
    result = judge(config, payload, answer, stopped)
    return dict(result, candidate_answer=answer)


def publish(root, header, result, config, now=None):
    now = time.time() if now is None else now
    state_file = state_path(root, header)
    old = read_json(state_file) if state_file.exists() else {}
    attempt = int(old.get('attempt_count', 0)) + 1
    result = dict(result, **{k: header[k] for k in ('job_id', 'job_sha256', 'step', 'query_fingerprint',
                                                 'slot_id', 'effective_id', 'teacher_key')},
                  attempt_count=attempt, finished_at=now,
                  judge_model=config['judge_model'], judge_base=config['base_url'])
    credential = config.get('api_key') or os.getenv(config['api_key_env'])
    if credential and credential != 'EMPTY':
        result = json.loads(dumps(result).replace(credential, '[REDACTED]'))
    # Errors cannot persist a false verdict even if a malformed client returned one.
    if result.get('status') == 'judged' and type(result.get('correct')) is not bool:
        result.update(status='judge_error', reason='Judge success did not contain a boolean verdict')
    if result.get('status') != 'judged' or type(result.get('correct')) is not bool:
        result['correct'] = None
    atomic(root / 'attempts' / (header['job_id'] + '.' + header['job_sha256'] + '.' + str(attempt) + '.json'), result)
    terminal = result['status'] in ('judged', 'input_error', 'unjudgeable')
    state = dict(attempt_count=attempt, terminal=terminal, last_status=result['status'],
                 next_attempt_at=None if terminal else now + min(5 * 2 ** min(attempt - 1, 6), 300),
                 updated_at=now, job_sha256=header['job_sha256'])
    # Only boolean successes become replay feedback. Everything else is retained
    # separately as an audit record or persisted retry state.
    if result['status'] == 'judged' and type(result['correct']) is bool:
        atomic(result_path(root, header), result)
    atomic(state_file, state)
    return result


def status(run):
    root = Path(run) / 'feedback'; counts = Counter()
    for path in (root / 'outbox').glob('*.json'):
        try:
            header = read_json(path)
            state_file = state_path(root, header)
            state = read_json(state_file) if state_file.exists() else {}
            counts[state.get('last_status', 'pending')] += 1
        except (OSError, ValueError, KeyError, TypeError):
            counts['invalid_local_record'] += 1
    success = successful_feedback(run, 2 ** 63 - 1)
    return {'updated_at': time.time(), 'jobs': sum(counts.values()), 'statuses': dict(counts),
            'latest_successful_queries': len(success), 'latest_correct': sum(r['correct'] for r in success),
            'latest_incorrect': sum(not r['correct'] for r in success),
            'note': 'Errors, unknown and unavailable references do not mean wrong. Training does not wait for this worker.'}


def run_worker(run, settings, once=False, poll_seconds=2):
    root = Path(run) / 'feedback'; root.mkdir(parents=True, exist_ok=True)
    config = judge_config(settings); stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        executor = ThreadPoolExecutor(max_workers=config['concurrency'])
        futures, active, attempted, last_print = {}, set(), set(), 0
        try:
            while not stopped.is_set():
                scanned_all = True
                for path in sorted((root / 'outbox').glob('*.json')):
                    if stopped.is_set() or len(futures) >= config['concurrency']:
                        scanned_all = False
                        break
                    try:
                        header = read_json(path); key = header['job_id'], header['job_sha256']
                        if key in active or (once and key in attempted) or result_path(root, header).exists():
                            continue
                        state_file = state_path(root, header)
                        state = read_json(state_file) if state_file.exists() else {}
                        if state.get('terminal') or state.get('next_attempt_at', 0) > time.time():
                            continue
                        futures[executor.submit(evaluate_job, config, header, stopped)] = header
                        active.add(key)
                        attempted.add(key)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        print('Feedback local job skipped: ' + str(exc)[:200], flush=True)
                done, _ = wait(futures, timeout=.2, return_when=FIRST_COMPLETED) if futures else ([], [])
                for future in done:
                    header = futures.pop(future); active.discard((header['job_id'], header['job_sha256']))
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {'status': 'judge_error', 'correct': None, 'reason': type(exc).__name__ + ': ' + str(exc), 'attempts': []}
                    try:
                        publish(root, header, result, config)
                    except Exception as exc:
                        print('Feedback result retained as pending: ' + type(exc).__name__ + ': ' + str(exc)[:200], flush=True)
                if time.monotonic() - last_print >= 15:
                    progress = dict(status(run), state='following_training', in_flight=len(futures), concurrency=config['concurrency'])
                    atomic(root / 'progress.json', progress)
                    print('Online Judge: ' + dumps(progress), flush=True); last_print = time.monotonic()
                if once and not futures and scanned_all:
                    break
                if not futures:
                    stopped.wait(poll_seconds)
        finally:
            stopped.set()
            # Publish only already-finished requests; never drain pending Judge
            # work at the cost of delaying the training/evaluation sequence.
            for future, header in list(futures.items()):
                if future.done() and not future.cancelled():
                    try:
                        publish(root, header, future.result(), config)
                    except Exception:
                        pass
            executor.shutdown(wait=False, cancel_futures=True)
            atomic(root / 'progress.json', dict(status(run), state='stopped', pending_preserved=True))


def request_body(config, payload, answer):
    data = dict(payload, candidate_answer=answer)
    body = {'model': config['judge_model'], 'temperature': config['temperature'],
            'max_tokens': config['max_tokens'],
            'chat_template_kwargs': {'enable_thinking': config['enable_thinking']},
            'messages': [{'role': 'system', 'content': config['system_prompt']},
                         {'role': 'user', 'content': dumps(data)}]}
    if config.get('provider') == 'aliyun':
        # Match the previous benchmark DashScope OpenAI-compatible client.
        body.pop('chat_template_kwargs')
        body['enable_thinking'] = False
        body['response_format'] = {'type': 'json_object'}
    return body


def judge(config, payload, answer, stopped):
    body = request_body(config, payload, answer)
    encoded = dumps(body).encode()
    request_hash = digest(body)
    attempts = []
    key = config.get('api_key') or os.getenv(config['api_key_env'], 'EMPTY')
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if not config['use_proxy'] else urllib.request.build_opener()
    url = config['base_url'].rstrip('/') + '/chat/completions'
    if config['max_request_chars'] and len(encoded.decode()) > config['max_request_chars']:
        return {'status': 'input_error', 'correct': None, 'reason': 'max_request_chars_exceeded',
                'attempts': [], 'request_sha256': request_hash}
    for attempt_no in range(config['retries'] + 1):
        if stopped.is_set():
            return {'status': 'pending', 'correct': None, 'reason': 'stopped_before_request',
                    'attempts': attempts, 'request_sha256': request_hash}
        started = time.monotonic()
        evidence = {'attempt': attempt_no + 1, 'started_at': time.time()}
        retryable = True
        try:
            request = urllib.request.Request(url, data=encoded, headers=headers, method='POST')
            with opener.open(request, timeout=config['timeout_seconds']) as response:
                evidence['http_status'] = response.status
                raw = response.read(config['max_response_bytes'] + 1)
            if len(raw) > config['max_response_bytes']:
                raise ValueError('Judge HTTP response exceeds max_response_bytes')
            evidence['raw_response'] = raw.decode(errors='replace')
            data = json.loads(raw)
            evidence['response'] = data
            evidence['usage'] = data.get('usage') or {}
            if config.get('strict_response_model', True) and data.get('model') != config['judge_model']:
                raise ValueError('Judge response model does not match the configured model')
            choices = data.get('choices')
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError('Expected exactly one judge choice')
            choice = choices[0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Judge did not finish normally: ' + str(choice.get('finish_reason')))
            content = choice.get('message', {}).get('content')
            if not isinstance(content, str):
                raise ValueError('Missing judge text content')
            verdict = parse_verdict(content)
            evidence.update(seconds=round(time.monotonic() - started, 6), status='ok')
            if key and key != 'EMPTY':
                evidence = json.loads(dumps(evidence).replace(key, '[REDACTED]'))
            attempts.append(evidence)
            return {**verdict, 'status': 'unjudgeable' if verdict['correct'] is None else 'judged',
                    'attempts': attempts, 'request_sha256': request_hash}
        except urllib.error.HTTPError as exc:
            evidence.update(http_status=exc.code, error_kind='http_error',
                            error=exc.read(2048).decode(errors='replace'))
            exc.close()
            retryable = exc.code in (408, 429) or 500 <= exc.code < 600
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            evidence.update(error_kind='connection_error', error=str(exc))
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            evidence.update(error_kind='response_parse_error', error=str(exc))
        evidence.update(status='error', seconds=round(time.monotonic() - started, 6))
        # Avoid storing credentials even if an upstream gateway echoes a header in its error.
        if key and key != 'EMPTY':
            evidence = json.loads(dumps(evidence).replace(key, '[REDACTED]'))
        attempts.append(evidence)
        if not retryable or attempt_no == config['retries']:
            break
        if stopped.wait(min(config['retry_backoff_seconds'] * (2 ** attempt_no), 30)):
            break
    return {'status': 'judge_error', 'correct': None,
            'reason': attempts[-1].get('error', 'Judge request failed'),
            'attempts': attempts, 'request_sha256': request_hash}


def probe_judge(settings):
    """One real CPU-only chat request before allocating training GPUs.

    Return public connection choices only. The API key and complete HTTP
    response never enter the launcher's metadata or output.
    """
    config = judge_config(settings)
    config['max_tokens'] = 128
    sample = {'mode': 'reasoning', 'task': 'arithmetic', 'gt': '2',
              'messages': [{'role': 'user', 'content': '1+1=?'}]}
    result = judge(config, base_payload(sample, {}), '<answer>2</answer>', threading.Event())
    if result.get('status') != 'judged' or result.get('correct') is not True:
        raise ValueError('Feedback Judge chat/JSON probe failed: ' + str(result.get('status'))
                         + ': ' + str(result.get('reason', 'unexpected verdict'))[:300])
    return {'provider': config['provider'], 'model': config['judge_model'],
            'base_url': config['base_url'], 'status': 'judged', 'correct': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'status'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--once', action='store_true', help='Judge the currently pending due jobs once, then exit')
    args = parser.parse_args()
    if args.command == 'status':
        print(dumps(status(args.run_dir)))
    else:
        settings = read_json(args.settings) if args.settings else {}
        run_worker(args.run_dir, settings, args.once)


if __name__ == '__main__':
    main()
