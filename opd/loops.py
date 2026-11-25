"""On-policy reasoning in both experiments; Agent uses original or on-policy trajectories."""
import asyncio
import copy
import json
import random
import time
from pathlib import Path
from uuid import uuid4
from functools import lru_cache
from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template
from opd.common import SOURCES, fingerprint, read_json, write_json, require, trajectory_source, active_domains
from opd.data import Pool
from runner.data import normalize
from runner.images import prepare_images_async, apply_image_selection, IMAGE_BUDGET_DEFAULTS
from runner.parsing import parse_calls
from runner.tokens import split_reasoning


@lru_cache(maxsize=4)
def cached_pool(path):
    return Pool(path)


class BadQuery(ValueError):
    """A bounded query failure; eligible for replacement, not a process failure."""


class OPDTrajectoryLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = dict(self.config.opd)
        require(dict(self.settings['trajectory_sources']) == {
            domain: trajectory_source(self.settings['mode'], domain) for domain in active_domains(self.settings)
        }, 'Trajectory policy mismatch: reasoning must always use student rollout')
        self.pool = cached_pool(self.settings['index'])
        self.shared, self.tools = {}, None
        if 'agent' in active_domains(self.settings):
            from runner.tools import ToolRunner
            self.shared = read_json(self.pool.root / 'shared_context.json')
            self.tools = ToolRunner(self.shared, {
                'selection': 'all', 'timeout_seconds': self.settings['tool_timeout_seconds'],
                'max_response_chars': self.settings['tool_response_chars'],
                'concurrency_per_replica': self.settings['max_calls_per_turn']})

    def _cap_text_prompt_length(self, ids):
        # Never silently left-truncate vision tokens or a stored conversation.
        if len(ids) > self.settings['max_sequence_tokens']:
            raise BadQuery('context_overflow')
        return ids

    def render(self, messages, schemas, images, generation=False):
        text = apply_chat_template(self.processor or self.tokenizer, messages,
                                   tools=schemas or None, tokenize=False,
                                   add_generation_prompt=generation,
                                   **self.apply_chat_template_kwargs)
        if self.processor is None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            inputs = build_multimodal_processor_inputs(self.processor, text=[text], images=images or None,
                        mm_processor_kwargs=dict(self.mm_processor_kwargs or {}))
            ids = normalize_token_ids(inputs['input_ids'])
        return ids, text

    async def prepare(self, index):
        row, meta = self.pool.row(index)
        try:
            sample = normalize(row, meta['provenance'], meta['mode'], self.shared, defer_media=True)
            prepared = await prepare_images_async(sample['image_paths'], sample['media_stats'],
                        IMAGE_BUDGET_DEFAULTS, 'first_n_long_side',
                        max_images=self.settings['max_images'], max_side=self.settings['max_image_side'])
            try:
                if prepared.preprocessing.get('skipped_images'):
                    raise BadQuery('unreadable_selected_image')
                apply_image_selection(sample['messages'], prepared.preprocessing)
                images = []
                for path in prepared.paths:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB').copy())
                info = copy.deepcopy(prepared.preprocessing)
            finally:
                prepared.close()
            return row, meta, sample, images, info
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BadQuery('input_error: ' + str(exc)) from exc

    async def original(self, row, sample, images, schemas):
        require(trajectory_source(self.settings['mode'], sample['mode']) == 'original_trajectory',
                'Stored trajectories are allowed only for Agent in original_trajectory mode')
        start = next((i for i, m in enumerate(row['messages']) if m['role'] == 'assistant'), None)
        if start is None or row['messages'][-1]['role'] != 'assistant':
            raise BadQuery('original_trajectory_has_no_final_assistant')
        messages = copy.deepcopy(sample['messages']) + copy.deepcopy(row['messages'][start:])
        # Teacher and student receive exactly this stored conversation. No tool execution.
        ids, _ = await asyncio.to_thread(self.render, messages, schemas, images, False)
        if len(ids) > self.settings['max_sequence_tokens']:
            raise BadQuery('context_overflow')
        mask, spans = [0] * len(ids), []
        first = len(sample['messages'])
        for i in range(first, len(messages)):
            if messages[i]['role'] != 'assistant':
                continue
            before, _ = await asyncio.to_thread(self.render, messages[:i], schemas, images, False)
            end, _ = await asyncio.to_thread(self.render, messages[:i + 1], schemas, images, False)
            header, _ = await asyncio.to_thread(self.render, messages[:i], schemas, images, True)
            # Prefix checking makes template reasoning removal or altered history visible.
            if ids[:len(end)] != end or end[:len(before)] != before:
                raise BadQuery('non_append_only_chat_template: stored history changed while rendering')
            a = len(before)
            while a < min(len(header), len(end)) and header[a] == end[a]:
                a += 1
            b = len(end)
            # Inter-turn separator is template context, not an assistant target.
            sep = self.turn_separator
            if sep and end[-len(sep):] == sep:
                b -= len(sep)
            if b <= a:
                raise BadQuery('empty_stored_assistant')
            mask[a:b] = [1] * (b - a)
            spans.append([a, b])
        split = spans[0][0]
        if not split or not any(mask):
            raise BadQuery('empty_original_loss_mask')
        return ids[:split], ids[split:], mask[split:], len(spans), {
            'tool_calls': sum(len(m.get('tool_calls') or []) for m in messages),
            'tool_seconds': 0., 'generation_seconds': 0., 'messages': messages,
            'assistant_spans': spans}

    async def online(self, sample, images, schemas, seed):
        prompt, prompt_text = await asyncio.to_thread(self.render, sample['messages'], schemas, images, True)
        full, mask, calls_total, tool_time, gen_time = list(prompt), [], 0, 0., 0.
        turns = []
        stop_ids = set(self.tokenizer.encode('<|im_end|>', add_special_tokens=False))
        if isinstance(self.tokenizer.eos_token_id, int):
            stop_ids.add(self.tokenizer.eos_token_id)
        max_turns = self.settings['max_turns'] if sample['mode'] == 'agent' else 1
        for turn in range(max_turns):
            room = self.settings['max_sequence_tokens'] - len(full)
            generation_limit = self.settings.get('max_new_tokens_by_domain', {}).get(sample['mode'], self.settings['max_new_tokens'])
            limit = min(room, generation_limit)
            if limit <= 0:
                raise BadQuery('context_overflow')
            params = dict(temperature=self.settings['temperature'], top_p=self.settings['top_p'],
                          top_k=self.settings['top_k'], max_new_tokens=limit,
                          sampling_seed=(seed + turn) % (2**31), no_stop_trim=True,
                          skip_special_tokens=False, stop_token_ids=sorted(stop_ids))
            begin = time.monotonic()
            self.phase = 'generation'
            # Do not turn engine/transport failures into bad data. They must surface for resume.
            output = await self.server_manager.generate(request_id=uuid4().hex,
                        prompt_ids=list(full), sampling_params=params, image_data=images or None,
                        mm_processor_kwargs=dict(self.mm_processor_kwargs or {}))
            self.phase = 'parse'
            gen_time += time.monotonic() - begin
            generated = list(output.token_ids)
            if not generated:
                raise BadQuery('empty_generation')
            if output.stop_reason in ('length', 'max_tokens') or generated[-1] not in stop_ids:
                raise BadQuery('truncated')
            if len(generated) > limit:
                raise RuntimeError('Rollout engine exceeded max_new_tokens')
            split = split_reasoning(self.tokenizer, generated, prompt_text, stop_ids)
            if split['parse_problems']:
                raise BadQuery('parse_error: ' + ','.join(split['parse_problems']))
            try:
                visible, calls = parse_calls(split['answer'], schemas)
            except (ValueError, TypeError) as exc:
                raise BadQuery('parse_error: ' + str(exc)) from exc
            full.extend(generated)
            mask.extend([1] * len(generated))
            trace = {'generated_ids': generated, 'reasoning': split['cot'], 'answer': visible, 'calls': calls}
            turns.append(trace)
            if not calls:
                if not visible.strip():
                    raise BadQuery('empty_final_answer')
                return prompt, full[len(prompt):], mask, turn + 1, {
                    'tool_calls': calls_total, 'tool_seconds': tool_time,
                    'generation_seconds': gen_time, 'turns': turns}
            if sample['mode'] != 'agent' or len(calls) > self.settings['max_calls_per_turn']:
                raise BadQuery('invalid_tool_calls')
            if turn + 1 == max_turns:
                raise BadQuery('max_turns')
            begin = time.monotonic()
            self.phase = 'tools'
            observations = await asyncio.gather(*(self.tools.execute(call) for call in calls))
            tool_time += time.monotonic() - begin
            calls_total += len(calls)
            trace['observations'] = observations
            if any(o['error'] for o in observations):
                bad = next(o for o in observations if o['error'])
                raise BadQuery('tool_error: ' + bad['name'] + ': ' + bad.get('error_kind', 'unknown'))
            messages = [{'role': 'tool', 'content': o['text']} for o in observations]
            # Same incremental tool-turn protocol as native verl ToolAgentLoop. The generated
            # assistant token IDs remain unchanged; only tool/context tokens are appended.
            append = self.turn_separator + await self.apply_chat_template(messages, remove_system_prompt=True)
            full.extend(append)
            mask.extend([0] * len(append))
            if len(full) >= self.settings['max_sequence_tokens']:
                raise BadQuery('context_overflow')
            prompt_text = self.tokenizer.decode(full, skip_special_tokens=False)
        raise BadQuery('max_turns')

    async def run(self, sampling_params, **kwargs):
        slot = int(kwargs['query_index'])
        step = int(kwargs.get('opd_step', 0))
        original = self.pool.entries[slot]
        seed_key = original['sample_id'] if self.settings.get('balanced_routes') else slot
        seed = int(fingerprint([self.settings['seed'], seed_key, step])[:8], 16)
        replacement_pool = [i for i, item in enumerate(self.pool.entries)
                            if item.get('teacher_key') == original.get('teacher_key')
                            and item['mode'] == original['mode']] if self.settings.get('balanced_routes') else None
        rng = random.Random(seed)
        rejected, index = [], slot
        started = time.monotonic()
        for attempt in range(self.settings['max_replacements'] + 1):
            images = []
            self.phase = 'prepare'
            try:
                async with asyncio.timeout(self.settings['query_timeout_seconds']):
                    row, meta, sample, images, media = await self.prepare(index)
                    schemas = self.tools.schemas if meta['mode'] == 'agent' else []
                    # Re-evaluate after each replacement, which may change the domain.
                    source = self.settings['trajectory_sources'][meta['mode']]
                    if source == 'original_trajectory':
                        result = await self.original(row, sample, images, schemas)
                    else:
                        result = await self.online(sample, images, schemas, seed + attempt)
                prompt, response, mask, nturns, detail = result
                if len(prompt) + len(response) > self.settings['max_sequence_tokens'] or not any(mask):
                    raise BadQuery('context_overflow_or_empty_loss_mask')
                report = {'slot_id': self.pool.entries[slot]['sample_id'], 'effective_id': meta['sample_id'],
                          'domain': meta['mode'], 'trajectory_source': source,
                          'task': sample['task'], 'step': step, 'replacements': rejected,
                          'prompt_tokens': len(prompt), 'response_tokens': len(response),
                          'assistant_tokens': sum(mask), 'context_tokens_in_response': len(mask) - sum(mask),
                          'num_turns': nturns, 'image_count': len(images),
                          'input_image_count': len(sample['image_paths']), 'tool_calls': detail['tool_calls'],
                          'tool_seconds': detail['tool_seconds'], 'generation_seconds': detail['generation_seconds'],
                          'wall_seconds': time.monotonic() - started}
                if self.settings.get('teachers'):
                    require(meta.get('teacher_key') in self.settings['teachers'], 'Effective query has no teacher route')
                    report.update(teacher_key=meta['teacher_key'], data_group=meta['data_group'],
                                  original_sample_id=meta['original_sample_id'])
                trajectory_path = Path(self.settings['run_dir']) / 'trajectories' / f'step_{step}' / (str(slot) + '.' + uuid4().hex[:8] + '.json')
                report.update(slot_index=slot, effective_index=index,
                              query_fingerprint=meta.get('query_fingerprint'),
                              slot_query_fingerprint=original.get('query_fingerprint'),
                              sampling_source=kwargs.get('sampling_source', 'frozen'),
                              sampling_plan_step=int(kwargs.get('sampling_plan_step', step)),
                              trajectory_path=str(trajectory_path))
                write_json(trajectory_path,
                           {'report': report, 'prompt_ids': prompt, 'response_ids': response,
                            'response_mask': mask, 'media': media, 'detail': detail})
                return AgentLoopOutput(prompt_ids=prompt, response_ids=response, response_mask=mask,
                        multi_modal_data={'images': images} if images else {}, reward_score=0., num_turns=nturns,
                        mm_processor_kwargs=dict(self.mm_processor_kwargs or {}),
                        metrics=AgentLoopMetrics(generate_sequences=detail['generation_seconds'],
                                                 tool_calls=detail['tool_seconds'], num_preempted=0),
                        extra_fields={'opd_report': report})
            except (BadQuery, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) and self.phase == 'generation':
                    raise RuntimeError('Student engine timed out; stopping this job to drain outstanding requests before resume') from exc
                reason = 'query_timeout' if isinstance(exc, TimeoutError) else str(exc)
                rejected.append({'sample_id': self.pool.entries[index]['sample_id'],
                                 'domain': self.pool.entries[index]['mode'], 'reason': reason})
                for image in images:
                    image.close()
                index = rng.choice(replacement_pool) if replacement_pool is not None else rng.randrange(len(self.pool.entries))
        write_json(Path(self.settings['run_dir']) / 'failures' / (uuid4().hex + '.json'),
                   {'slot': slot, 'step': step, 'rejected': rejected})
        raise RuntimeError(f'Cannot fill query slot after {len(rejected)} attempts; see failures/. Resume checkpoint after fixing data/tools.')


from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
import ray


class OPDLoopWorker(AgentLoopWorker):
    async def _compute_teacher_logprobs(self, output, prompt_ids, response_ids, validate, sample_kwargs=None):
        if self.config.opd.get('teachers') and not validate:
            # Replacement can change the domain AND teacher. Native routing uses
            # the original dataloader slot; override it with the effective query.
            report = output.extra_fields['opd_report']
            route = report['teacher_key']
            require(route in self.config.opd.teachers, 'Unknown effective teacher route')
            sample_kwargs = dict(sample_kwargs or {}, teacher_key=route)
        try:
            async with asyncio.timeout(self.config.opd.query_timeout_seconds):
                return await super()._compute_teacher_logprobs(output, prompt_ids, response_ids, validate, sample_kwargs)
        except TimeoutError as exc:
            raise RuntimeError('Teacher engine timed out; resume this job from its checkpoint') from exc


class OPDLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(OPDLoopWorker)
        super().__init__(*args, **kwargs)


async def preflight_trajectories(config, model):
    """One usable query per domain before GPU loading; no model generation or benchmark."""
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    loop = OPDTrajectoryLoop(trainer_config=DictConfigWrap(config), server_manager=None,
             tokenizer=model.tokenizer, processor=model.processor, dataset_cls=None,
             data_config=DictConfigWrap(config.data))
    report = {}
    for domain in active_domains(loop.settings):
        errors = []
        candidates = [i for i, e in enumerate(loop.pool.entries) if e['mode'] == domain][:8]
        for index in candidates:
            images = []
            try:
                async with asyncio.timeout(120):
                    row, meta, sample, images, _ = await loop.prepare(index)
                    schemas = loop.tools.schemas if domain == 'agent' else []
                    source = loop.settings['trajectory_sources'][domain]
                    if source == 'original_trajectory':
                        prompt, response, mask, turns, _ = await loop.original(row, sample, images, schemas)
                        value = {'prompt_tokens': len(prompt), 'response_tokens': len(response),
                                 'assistant_tokens': sum(mask), 'turns': turns}
                    else:
                        prompt, _ = await asyncio.to_thread(loop.render, sample['messages'], schemas, images, True)
                        if len(prompt) >= config.opd.max_sequence_tokens:
                            raise BadQuery('context_overflow')
                        value = {'prompt_tokens': len(prompt)}
                report[domain] = dict(value, trajectory_source=source, sample_id=meta['sample_id'], skipped=errors)
                break
            except (BadQuery, TimeoutError) as exc:
                errors.append({'index': index, 'error': str(exc)})
            finally:
                for image in images:
                    image.close()
        else:
            write_json(Path(config.opd.run_dir) / 'input_preflight.json', dict(report, failed_domain=domain, errors=errors))
            raise RuntimeError(f'No usable {domain} input among the first {len(candidates)}; inspect input_preflight.json before loading GPUs')
    write_json(Path(config.opd.run_dir) / 'input_preflight.json', report)
    print('Actual tokenizer/template input checks passed for ' + ', '.join(report) + '; no rollout or benchmark performed.', flush=True)
