# Branch and Consolidate — Initial VERL Implementation

This release contains the initial implementation used with our own datasets, with three specialist routes (business reasoning, general reasoning, and business Agent tasks) and a frozen self teacher.

**We release the initial three-specialist implementation. Experiments with six or more specialists can be implemented by extending the routes, query pools, teacher weights, sampling allocation, and resource configuration in this code.**

The industrial training framework and its internal VERL extensions cannot be open-sourced. This repository provides the application and algorithm code that integrates with that framework. Native distributed training requires the compatible internal runtime, Agent tools, and data adapters. CPU tests exercise the loss and sampler independently.

## Data confidentiality

Our industrial datasets are company-confidential and cannot be released. This code package excludes the original training and evaluation questions, reference answers, labels, images and videos, model-generated trajectories, and dataset-owned task mappings and scoring rules. The data-related Python modules expose file-reading and sampling interfaces. Test fixtures are generated synthetically in the test code. Historical training hyperparameters and the generic input schema are documented for implementation purposes.

## Method and source

The code comes from the archived `exp3-1` implementation. The historical configuration uses the equal-weight average of the original student and three specialists as the initialization. A frozen copy of that initialization acts as the self teacher. Student-generated completions receive reverse-KL supervision on the teacher's Top32 tokens and one residual probability bucket.

| Component | Source |
|---|---|
| Checkpoint averaging | `weight_average/average_many.py` |
| Candidate selection and data indexing | `opd/candidates.py`, `opd/data.py` |
| Sampling, normal coverage, and error replay | `opd/online_sampling.py`, `opd/sampling.py` |
| Per-query, per-route loss reduction | `opd/balanced.py` |
| Full-vocabulary student normalization and Top32-plus-tail reverse KL | `opd/losses.py` |
| Student rollout and teacher scoring | `opd/loops.py` |
| Asynchronous correctness feedback | `opd/feedback.py` |
| VERL actor and trainer integration | `opd/native.py`, `opd/trainer.py`, `opd/train.py` |
| Historical training configuration with deployment placeholders | `configs/three_specialists.template.json` |

The historical preset uses batch size 128, learning rate `1e-6`, and three complete normal-data passes over 1,180 selected prompts per specialist. Self queries match the total specialist-query exposure 1:1. After 17 warmup updates, replay can occupy up to half the specialist slots, with fresh queries filling unused slots. Replay generates new student answers and extends training until normal coverage is complete.

The loss averages generated assistant tokens within a query and queries within a route. The historical four route coefficients are each `1/3`, giving a total coefficient of `4/3`. These values preserve the original implementation's objective scale.

`SOURCE_MANIFEST.json` records released file hashes and publication edits. The loss, sampler, and native adapter retain their original file contents. Publication edits externalize dataset-owned prompts, scoring rules, source strata, and service probe inputs, and replace private deployment defaults. The unused task-specific evaluation report renderer is excluded. The package includes the archived application modules. The six-capability paper results require the corresponding experiment configuration and runtime.

## Configure the data and teachers

Copy `configs/three_specialists.template.json` and replace its `/path/to/...` values with your own paths. The model map has four keys.

- `business_reasoning`, `general`, and `business_agent` point to the three specialists.
- `self` points to the same initial merged checkpoint as `student`. It remains frozen during training.

The data root contains aligned `train.jsonl` and `train.meta.jsonl`. Each input row has `messages`. Metadata includes `sample_id`, `mode`, `teacher_key`, `output_line`, `row_sha256`, and the original query identity and self-stratum fields used by the sampler. Modes are `reasoning` and `agent`; the three teacher routes remain distinct. Agent inputs also require `shared_context.json` and the corresponding external tool environment.

The original query-selection code consumes previously judged candidate pools. Supply the student/specialist judgments and answer references using your dataset's evaluation rules. Set `feedback_judge.base_url`, `judge_model`, and `api_key_env` for your Judge. Supply your private Judge system message as `feedback_judge.system_prompt` and the correctness rubric for each task through `feedback_judge.task_rubrics`. Self-pool metadata provides `sampling_stratum` explicitly as a specialist route, replacing the private source-group mapping. These values belong to the dataset owner's external configuration.

## Use the original entry points

Checkpoint averaging runs on CPU with PyTorch and safetensors. Supply the student first, followed by the three specialists.

```bash
python3 weight_average/average_many.py \
  --models "$STUDENT" "$BUSINESS_REASONING" "$GENERAL_REASONING" "$BUSINESS_AGENT" \
  --reference-index 0 --out "$MEAN4"
```

For native training, use the matching original Python environment and complete the template's paths. The training environment must already provide Ray resources, `OPD_NETWORK_MANIFEST`, Agent services including `POI_SEARCH_URL`, and logging and Judge credentials. Supply a dataset-owner-approved service probe query via `OPD_POI_PROBE_QUERY`. The `repo`, `agent_root`, and `step2_root` settings locate the external framework and adapters. Then validate the configuration through the existing entry point.

```bash
python3 -u -m opd.train --mode student_rollout \
  --config /absolute/path/to/your-config.json --project-root "$PWD" --check-only
```

With those dependencies and services prepared, start training through the same module.

```bash
python3 -u -m opd.train --mode student_rollout \
  --config /absolute/path/to/your-config.json --project-root "$PWD" \
  --ray-address auto
```

CPU regression tests require PyTorch. They cover the original reverse-KL values and gradients, query weighting, sampler coverage, replay, and restore behavior. Distributed-runtime boundaries in these CPU tests use test doubles.

Release verification passed 23 CPU tests. One native packing test was skipped because the internal framework checkout was unavailable. No new multi-GPU training run was performed for this release.

```bash
python3 -m unittest discover -s tests -v
```

## Extend to more specialists

Update the route definitions in `opd/online_sampling.py`, `opd/sampling.py`, `opd/balanced.py`, and `opd/experiment_controls.py`, and the corresponding validations in `opd/config.py`. Extend the teacher map, pool sizes, self strata, loss coefficients, and per-batch allocation consistently. Adapt the multimodal and Agent adapters and GPU allocation to the additional tasks. The historical source contains assumptions about three specialist pools, four routes, and attention data-parallel divisibility that need updating together.
