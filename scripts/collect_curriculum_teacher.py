"""Budgeted teacher demonstrations on representative rejected train tasks.

Starts from the task's initial state, not a student-state continuation. Never
exports teacher reasoning as supervision. All batches share one CNY ledger.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

from scripts import evaluate_harness_base as base
from scripts.evaluate_harness_teacher import AuditableNativePolicy
from scripts.collect_harness_batch import atomic_json, source_hashes
from core.api_budget import SharedAPIBudget,BudgetedGLMToolClient
from evaluation.curriculum_fixture import CurriculumResearchExecutor,quality_audit,VERSION
from agentic.trajectory import redact_pii


async def main(args):
    import fcntl
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    lock=(out/'.teacher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=os.environ.get('ZAI_API_KEY')
    if not key:raise ValueError('Existing teacher credential is required')
    cases=base.load_cases(args.cases_file)
    if len(cases)>12 or not all(c['split']=='train' and c['dataset_version']==VERSION for c in cases):
        raise ValueError('Teacher pilot must contain at most 12 train representatives')
    if len({c['source_group'] for c in cases})!=len(cases):raise ValueError('One teacher task per source group')
    config=json.loads(args.budget_config.read_text())
    budget=SharedAPIBudget(args.budget_ledger,config)
    atomic_json(out/'cases.json',cases)
    hashes=source_hashes()
    for name in ['scripts/collect_curriculum_teacher.py','backend/src/core/api_budget.py','backend/src/core/glm_tool_client.py']:
        hashes[name]=base.sha(base.ROOT/name)
    atomic_json(out/'manifest.json',dict(schema_version='curriculum-teacher-pilot.v1',
        model=config['model'],cases_sha256=base.sha(args.cases_file),source_hashes=hashes,
        budget_ledger=str(args.budget_ledger),budget_config=config,
        concurrency=4,max_output_tokens=8192,http_retries=0,episode_timeout_seconds=600,
        collection_mode='initial-state-demonstration-not-student-continuation',teacher_thinking=True,
        private_reasoning_is_training_target=False))
    base.settings.agentic_guard_mode='enforce'
    semaphore=asyncio.Semaphore(4);stop=asyncio.Event();summaries=[]
    async def run(case):
        async with semaphore:
            if stop.is_set():return
            target=out/case['id'];target.mkdir()
            client=BudgetedGLMToolClient(api_key=key,budget=budget,budget_task=case['id'],timeout=180)
            native=AuditableNativePolicy(client,model=config['model'],temperature=1,max_tokens=8192)
            policy=base.AuditedPolicy(native,target/'model-calls.jsonl')
            backend=CurriculumResearchExecutor(case)
            class Recorder(base.EpisodeRecorder):
                def record_step(self, **kwargs):
                    super().record_step(**kwargs);step=self.episode.steps[-1]
                    base.dump(target/f'state-after-{step.step_index:03d}.json',dict(
                        case_id=case['id'],source_group=case['source_group'],state_after_hash=step.state_after_hash,
                        state=redact_pii(kwargs['state_after'].model_dump(mode='json')),
                        provider_state=dict(counts=dict(backend.counts),provider_attempts=backend.provider_attempts,
                            fault_events=backend.fault_events,tool_call_count=len(backend.calls))))
            started=time.perf_counter()
            try:
                with base.frozen_reference_time(base.MOMENT):
                    state=base.initial(case);state.budget=state.budget.model_copy(update={'timeout_ms':600000})
                    recorder=Recorder(state,environment_version=base.ARCHITECTURE_VERSION,
                        validator_version=base.VALIDATOR_VERSION,policy_name=config['model'],policy_version='budgeted-curriculum.v1')
                    await base.BoundedAgentLoop().run(state,policy=base.SelfRepairingAgentPolicy(policy),
                        executor=base.TravelActionExecutor(backend),recorder=recorder)
                episode=recorder.episode
                base.dump(target/'episode.json',episode.model_dump(mode='json'));base.dump(target/'tool-calls.json',backend.calls)
                errors=base.EpisodeReplayVerifier().verify(episode)
                row=base.judge(case,episode,policy.calls)
                row['parse_or_schema_errors']=sum((r.get('error') or {}).get('type')=='PolicyOutputError' for r in policy.calls)
                row.update(reward=base.HierarchicalRewardEngine().score(episode).model_dump(mode='json'),
                    replay_errors=errors,wall_seconds=time.perf_counter()-started,
                    provider_failures=sum(bool((r.get('generation') or {}).get('provider_error')) for r in policy.calls))
                row.update(quality_audit(case,row,backend))
                infrastructure=[r['error'] for r in policy.calls if r.get('error') and r['error']['type']!='PolicyOutputError']
                row['infrastructure_errors']=infrastructure
                if infrastructure or errors:
                    row['quality_candidate']=False;row['rejection_reasons'].append('INFRASTRUCTURE_OR_REPLAY_ERROR');stop.set()
                base.dump(target/'summary.json',row);summaries.append(row)
                atomic_json(out/'summary.json',summaries)
                print(json.dumps({'event':'teacher_case_done','case':case['id'],'passed':row['criterion_passed'],
                    'candidate':row['quality_candidate'],'calls':row['model_calls'],'reasons':row['rejection_reasons']},ensure_ascii=False),flush=True)
            finally:await client.aclose()
    await asyncio.gather(*(run(c) for c in cases))
    atomic_json(out/'budget-snapshot.json',budget.snapshot())
    print('TEACHER_COMPLETE',len(summaries),len(cases),flush=True)
    if stop.is_set() or len(summaries)!=len(cases):raise RuntimeError('Teacher collection stopped; evidence retained, no automatic rerun')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['output','cases-file','budget-config','budget-ledger']:p.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
