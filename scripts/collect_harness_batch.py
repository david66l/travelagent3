"""Cloud-only resumable migration collection with one shared, batched student GPU."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from agentic.batched_policy import BatchedCheckpointEngine
from agentic.local_policy import LocalCheckpointAgentPolicy
from agentic.trajectory import redact_pii
from evaluation.migration_fixture import MigrationResearchExecutor, VERSION
from evaluation.curriculum_fixture import CurriculumResearchExecutor, VERSION as CURRICULUM_VERSION, quality_audit
from evaluation.scale_curriculum import VERSION as SCALE_VERSION
from evaluation.named_curriculum import VERSION as NAMED_VERSION
from evaluation.named_dining_dev import VERSION as NAMED_DEV_VERSION
from evaluation.expanded_dining_dev import VERSION as EXPANDED_DEV_VERSION
from evaluation.dining_recovery_curriculum import VERSION as DINING_RECOVERY_VERSION
from evaluation.multiday_training_curriculum import VERSION as MULTIDAY_VERSION
from evaluation.posttrain_scale_curriculum import TRAIN_VERSION as FINAL_TRAIN_VERSION, TEST_VERSION as FINAL_TEST_VERSION
from scripts import evaluate_harness_base as base


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def source_hashes():
    paths = [* (base.ROOT / "backend/src/agentic").glob("*.py"),
             * (base.ROOT / "backend/src/evaluation").glob("*.py"),
             * (base.ROOT / "backend/src/tools").glob("*.py"),
             * (base.ROOT / "backend/src/vrp_solver_service").glob("*.py"),
             * (base.ROOT / "backend/src/planner").rglob("*.py"),
             *[base.ROOT / ('backend/src/core/'+n) for n in ('api_budget.py','glm_tool_client.py','inference_metrics.py') if (base.ROOT / ('backend/src/core/'+n)).exists()],
             Path(__file__), base.ROOT / "scripts/evaluate_harness_base.py"]
    return {str(p.relative_to(base.ROOT)):base.sha(p) for p in paths}


def initialize_jobs(output, cases, config, max_attempts):
    manifest = output / "batch-manifest.json"
    jobs_path = output / "jobs.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Cannot resume changed inputs, weights, source or batch configuration")
    else:
        atomic_json(manifest, config)
    if jobs_path.exists():
        jobs = json.loads(jobs_path.read_text())
        if set(jobs) != {c["id"] for c in cases}:
            raise ValueError("Job index differs from input cases")
    else:
        jobs = {c["id"]:{"status":"pending", "attempts":0, "history":[]} for c in cases}
    for job in jobs.values():
        if job["status"] == "running":
            job["status"] = "retry_pending" if job["attempts"] < max_attempts else "infrastructure_failed"
            job["history"].append({"event":"interrupted_attempt_preserved"})
    atomic_json(jobs_path, jobs)
    return jobs


def pending_cases(cases, jobs):
    pending = [c for c in cases if jobs[c['id']]['status'] in {'pending', 'retry_pending'}]
    if not pending and any(j['status'] != 'complete' for j in jobs.values()):
        raise RuntimeError('No retry budget remains for unfinished jobs; inspect retained attempts')
    return pending


def completed_summaries(output, jobs):
    summaries = []
    for case_id, job in jobs.items():
        if job['status'] != 'complete':
            continue
        target = (output / job['evidence']).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError('Evidence path escapes this run directory')
        expected = job.get('evidence_sha256', {})
        required = {'episode.json', 'summary.json', 'model-calls.jsonl', 'tool-calls.json'}
        if set(expected) != required or any(not (target/name).is_file() or base.sha(target/name) != expected[name] for name in required):
            raise ValueError('Completed evidence missing or changed; refusing to silently skip it')
        summary = json.loads((target/'summary.json').read_text())
        if summary['case'] != case_id:
            raise ValueError('Completed evidence belongs to a different case')
        summaries.append(summary)
    return summaries


def validate_case_partition(cases, split):
    if not cases:
        raise ValueError('Empty evaluation or collection set')
    revision=cases[0]['dataset_version']
    if split not in {'train','validation','test'} or (split=='validation') != (revision in {NAMED_DEV_VERSION,EXPANDED_DEV_VERSION}) or (split=='test') != (revision==FINAL_TEST_VERSION):
        raise ValueError('Each partition requires its explicit dataset version')
    if revision not in {VERSION, CURRICULUM_VERSION, SCALE_VERSION,NAMED_VERSION,NAMED_DEV_VERSION,EXPANDED_DEV_VERSION,DINING_RECOVERY_VERSION,MULTIDAY_VERSION,FINAL_TRAIN_VERSION,FINAL_TEST_VERSION} or not all(c['split']==split and c['dataset_version']==revision for c in cases):
        raise ValueError('Only homogeneous preflighted fixtures of the declared split are supported')
    return revision


def validate_locked_test_release(config, authorization):
    if authorization is None:
        raise ValueError('Locked test requires the frozen model release manifest')
    release=json.loads(authorization.read_text())
    if release.get('schema_version')!='locked-test-release.v1' or release.get('cases_sha256')!=config['cases_sha256']:
        raise ValueError('Locked test cases or release contract changed')
    if release.get('source_hashes')!=config['source_hashes']:
        raise ValueError('Locked test runtime changed after model freeze')
    if config['weight_hashes'] not in [v['weight_hashes'] for v in release.get('models',[])]:
        raise ValueError('Checkpoint was not frozen before locked test opening')
    return base.sha(authorization)


async def main(args):
    import fcntl  # This launcher intentionally runs on the Linux cloud host.

    if not 1 <= args.batch_size <= args.concurrency <= 16:
        raise ValueError("Require 1 <= batch-size <= concurrency <= 16")
    if not 0 <= args.retries <= 2:
        raise ValueError("Infrastructure retries must be bounded to 0..2")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".collection.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases = base.load_cases(args.cases_file)
    revision = cases[0]['dataset_version']
    split=getattr(args,'split','train')
    revision=validate_case_partition(cases,split)
    if revision == VERSION and len({c['source_group'] for c in cases}) != len(cases):
        raise ValueError("This pilot queue requires source-group-deduplicated cases")
    if len({c['request'] for c in cases}) != len(cases):
        raise ValueError("Duplicate requests are not independent collection jobs")
    if revision in {CURRICULUM_VERSION,SCALE_VERSION,NAMED_VERSION,NAMED_DEV_VERSION,EXPANDED_DEV_VERSION,DINING_RECOVERY_VERSION,MULTIDAY_VERSION,FINAL_TRAIN_VERSION,FINAL_TEST_VERSION}:
        preflight_path = args.cases_file.parent/'preflight-summary.json'
        preflight = json.loads(preflight_path.read_text())
        if preflight['cases_sha256'] != base.sha(args.cases_file) or not preflight['all_passed']:
            raise ValueError('Curriculum environment preflight is missing, stale or failed')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise ValueError("A cloud-local checkpoint is required")
    adapter_file = checkpoint / "adapter_config.json"
    model_base = Path(json.loads(adapter_file.read_text())["base_model_name_or_path"]) if adapter_file.exists() else checkpoint
    if not model_base.is_dir():
        raise ValueError("A cloud-local base checkpoint is required")
    weight_files = list(model_base.glob('*.safetensors')) + list(checkpoint.glob('*.safetensors'))
    weight_files += [p for folder in {model_base,checkpoint} for name in ['config.json','adapter_config.json','tokenizer.json','tokenizer_config.json'] if (p:=folder/name).exists()]
    config = {"schema_version":"harness-batch-collection.v1", "cases_sha256":base.sha(args.cases_file),
              "checkpoint":str(checkpoint), "weight_hashes":{str(p):base.sha(p) for p in weight_files},
              "fixture_revision":revision, "harness_revision":base.HARNESS_REVISION,
              "source_hashes":source_hashes(), "batch_size":args.batch_size, "concurrency":args.concurrency,
              "wait_ms":args.wait_ms, "max_padded_tokens":args.max_padded_tokens,
              "max_new_tokens":384, "greedy":True, "seed":42, "infrastructure_retries":args.retries,
              "teacher_api_enabled":False, "training_enabled":False,"split":split}
    config['inference_engine']=args.engine
    if split=='test':
        config['locked_test_release_sha256']=validate_locked_test_release(config,getattr(args,'test_release',None))
    jobs = initialize_jobs(out, cases, config, args.retries+1)
    atomic_json(out/'cases.json', cases)
    completed = completed_summaries(out, jobs)
    pending = pending_cases(cases, jobs)
    if not pending:
        atomic_json(out/'summary.json', completed)
        print('RESUME_NO_PENDING_JOBS: no model loaded, no inference repeated', flush=True)
        return
    start_load = time.perf_counter()
    if args.engine=='vllm':
        from agentic.vllm_batch_policy import VLLMCheckpointEngine
        engine=VLLMCheckpointEngine(str(checkpoint),concurrency=args.concurrency)
        model=None
    else:
        model = LocalCheckpointAgentPolicy(str(checkpoint), max_new_tokens=384, seed=42,
                                          do_sample=False, load_in_4bit=False, structured_decoding='native')
        model._torch.cuda.reset_peak_memory_stats()
        engine = BatchedCheckpointEngine(model, batch_size=args.batch_size, wait_ms=args.wait_ms,
                                        max_padded_tokens=args.max_padded_tokens)
    load_seconds = time.perf_counter()-start_load
    base.settings.agentic_guard_mode = 'enforce'
    semaphore = asyncio.Semaphore(args.concurrency)
    collection_started = time.perf_counter()

    async def run_case(case):
        async with semaphore:
            job = jobs[case['id']]
            while job['attempts'] < args.retries+1:
                job['attempts'] += 1
                target = out/case['id']/f"attempt-{job['attempts']:02d}"
                job.update(status='running', evidence=str(target.relative_to(out)))
                atomic_json(out/'jobs.json',jobs)
                target.mkdir(parents=True, exist_ok=False)
                backend = CurriculumResearchExecutor(case) if revision in {CURRICULUM_VERSION,SCALE_VERSION,NAMED_VERSION,NAMED_DEV_VERSION,EXPANDED_DEV_VERSION,DINING_RECOVERY_VERSION,MULTIDAY_VERSION,FINAL_TRAIN_VERSION,FINAL_TEST_VERSION} else MigrationResearchExecutor(case)
                session = engine.session()
                policy = base.AuditedPolicy(session, target/'model-calls.jsonl')

                class Recorder(base.EpisodeRecorder):
                    def record_step(self, **kwargs):
                        super().record_step(**kwargs)
                        step = self.episode.steps[-1]
                        snapshot={
                            'schema_version':'migration-recovery-state.v1','case_id':case['id'],
                            'split':case['split'],'source_group':case['source_group'],'step_index':step.step_index,
                            'state':redact_pii(kwargs['state_after'].model_dump(mode='json')),
                            'state_after_hash':step.state_after_hash,
                            'provider_state':{'counts':dict(backend.counts),'seen_fault_indices':sorted(backend._seen_faults),
                                              'tool_call_count':len(backend.calls),
                                              'fault_events':getattr(backend,'fault_events',[]),
                                              'provider_attempts':getattr(backend,'provider_attempts',{})},'provider_revision':revision,
                            'error_code':step.verification.get('error_code')}
                        if revision in {NAMED_VERSION,NAMED_DEV_VERSION,EXPANDED_DEV_VERSION,DINING_RECOVERY_VERSION,MULTIDAY_VERSION,FINAL_TRAIN_VERSION,FINAL_TEST_VERSION}:
                            from evaluation.curriculum_recovery import provider_snapshot
                            snapshot['provider_state']=provider_snapshot(backend)
                        base.dump(target/f'state-after-{step.step_index:03d}.json',snapshot)
                try:
                    started = time.perf_counter()
                    with base.frozen_reference_time(base.MOMENT):
                        state = base.initial(case)
                        recorder = Recorder(state, environment_version=base.ARCHITECTURE_VERSION,
                            validator_version=base.VALIDATOR_VERSION,policy_name='Qwen3-student-batched',policy_version='v1')
                        await base.BoundedAgentLoop().run(state, policy=base.SelfRepairingAgentPolicy(policy),
                                                         executor=base.TravelActionExecutor(backend), recorder=recorder)
                    episode = recorder.episode
                    base.dump(target/'episode.json',episode.model_dump(mode='json'))
                    base.dump(target/'tool-calls.json',backend.calls)
                    replay = base.EpisodeReplayVerifier().verify(episode)
                    if replay:
                        raise RuntimeError('Trajectory integrity check failed: '+str(replay))
                    infrastructure_errors = [c['error'] for c in policy.calls if c.get('error') and c['error']['type'] not in {'PolicyOutputError'}]
                    if infrastructure_errors:
                        raise RuntimeError('Generation infrastructure failure: '+str(infrastructure_errors))
                    summary = base.judge(case,episode,policy.calls)
                    summary.update(wall_seconds=time.perf_counter()-started,
                        reward=base.HierarchicalRewardEngine().score(episode).model_dump(mode='json'),
                        replay_errors=replay,training_ready=False)
                    if revision in {CURRICULUM_VERSION,SCALE_VERSION,NAMED_VERSION,NAMED_DEV_VERSION,EXPANDED_DEV_VERSION,DINING_RECOVERY_VERSION,MULTIDAY_VERSION,FINAL_TRAIN_VERSION,FINAL_TEST_VERSION}:
                        summary.update(quality_audit(case,summary,backend))
                    base.dump(target/'summary.json',summary)
                    # Model task failures are completed measurements, never retried
                    # automatically to select a lucky successful trajectory.
                    job.update(status='complete',criterion_passed=summary['criterion_passed'],
                        evidence_sha256={name:base.sha(target/name) for name in ('episode.json','summary.json','model-calls.jsonl','tool-calls.json')})
                    job['history'].append({'attempt':job['attempts'],'event':'complete','criterion_passed':summary['criterion_passed']})
                    atomic_json(out/'jobs.json',jobs)
                    print(json.dumps({'event':'case_done','case':case['id'],'passed':summary['criterion_passed'],'attempt':job['attempts']},ensure_ascii=False),flush=True)
                    return
                except Exception as exc:
                    base.dump(target/'infrastructure-error.json',{'type':type(exc).__name__,'message':str(exc)})
                    job['history'].append({'attempt':job['attempts'],'event':'infrastructure_failure','type':type(exc).__name__})
                    job['status'] = 'retry_pending' if job['attempts'] < args.retries+1 else 'infrastructure_failed'
                    atomic_json(out/'jobs.json',jobs)
                    if job['status']=='retry_pending':
                        await asyncio.sleep(min(2**job['attempts'],8))
            print(json.dumps({'event':'case_infrastructure_failed','case':case['id']}),flush=True)
    try:
        await asyncio.gather(*(run_case(c) for c in pending))
    finally:
        await engine.aclose()
        elapsed=time.perf_counter()-collection_started
        atomic_json(out/'batch-metrics.json',{'model_load_seconds':load_seconds,'collection_wall_seconds':elapsed,
            'batch_events':engine.batches,'peak_gpu_memory_bytes':(model or engine)._torch.cuda.max_memory_allocated(),
            'event_unit':'completed_request' if args.engine=='vllm' else 'gpu_generate_batch',
            'inference_engine':args.engine,
            'cases_this_invocation':len(pending),'concurrency':args.concurrency,'batch_size':args.batch_size})
        if model is not None:model.close()
    summaries=completed_summaries(out, jobs)
    atomic_json(out/'summary.json',summaries)
    print(json.dumps({'event':'batch_complete','completed':len(summaries),'collection_wall_seconds':elapsed}),flush=True)
    if any(j['status']!='complete' for j in jobs.values()):
        raise RuntimeError('Some jobs remain unfinished; retained job history is authoritative')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases-file',type=Path,required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--wait-ms',type=int,default=15)
    parser.add_argument('--max-padded-tokens',type=int,default=24000)
    parser.add_argument('--retries',type=int,default=1)
    parser.add_argument('--engine',choices=['transformers','vllm'],default='transformers')
    parser.add_argument('--split',choices=['train','validation','test'],default='train')
    parser.add_argument('--test-release',type=Path)
    asyncio.run(main(parser.parse_args()))

