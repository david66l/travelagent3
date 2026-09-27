"""Budgeted full demonstrations and expert continuations on frozen student states."""
import argparse,asyncio,json,os,time
from copy import deepcopy
from pathlib import Path
from scripts import evaluate_harness_base as base
from scripts.collect_harness_batch import atomic_json,source_hashes,completed_summaries
from scripts.evaluate_harness_teacher import AuditableNativePolicy
from evaluation.named_curriculum import VERSION
from evaluation.curriculum_fixture import CurriculumResearchExecutor,quality_audit
from evaluation.curriculum_recovery import provider_snapshot,restore
from agentic.trajectory import AgentEpisode,_canonical_hash,redact_pii
from core.api_budget import SharedAPIBudget,BudgetedGLMToolClient


def select_branches(student):
    cases=base.load_cases(student/'cases.json');jobs=json.loads((student/'jobs.json').read_text())
    if len(cases)>20 or any(c['split']!='train' or c['dataset_version']!=VERSION for c in cases):
        raise ValueError('Only the 20-task named-dining train pilot is allowed')
    if len(completed_summaries(student,jobs))!=len(cases):raise ValueError('Student collection is incomplete')
    branches={};excluded=[]
    for case in cases:
        folder=student/jobs[case['id']]['evidence'];ep=AgentEpisode(**json.loads((folder/'episode.json').read_text()))
        assert not base.EpisodeReplayVerifier().verify(ep)
        options=[]
        for step in ep.steps:
            path=folder/f'state-after-{step.step_index:03d}.json';snap=json.loads(path.read_text())
            state=base.AgentLedgerState(**snap['state'])
            if state.task_graph.tasks[0].status!='ready' or state.budget.remaining_episode_steps<2:continue
            error=step.verification.get('error_code')
            priority=0 if error=='REPEATED_NO_PROGRESS_ACTION' else 1 if error and error!='RESEARCH_EVIDENCE_INSUFFICIENT' else 2 if step.action.action=='get_poi_detail' else 3
            options.append((priority,step.step_index,path,snap,error))
        if not options:
            excluded.append({'case':case['id'],'reason':'no_real_nonterminal_student_checkpoint'});continue
        priority,index,path,snap,error=min(options,key=lambda v:v[:2])
        records=json.loads((folder/'tool-calls.json').read_text());restore(case,snap,records)
        branches[case['id']]={'snapshot':snap,'snapshot_path':str(path),'snapshot_sha256':base.sha(path),
            'student_evidence':str(folder),'source_episode_sha256':base.sha(folder/'episode.json'),
            'step_index':index,'source_error':error,'kind':'observed_error_continuation' if error else 'student_prefix_expert_takeover'}
    return cases,branches,excluded


async def main(args):
    import fcntl
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    key=os.environ.get('ZAI_API_KEY');assert key,'Teacher credential missing'
    cases,branches,excluded=select_branches(args.student_run)
    hashes=source_hashes();student_manifest=json.loads((args.student_run/'batch-manifest.json').read_text())
    assert hashes==student_manifest['source_hashes'],'Environment changed since student collection'
    config=json.loads(args.budget_config.read_text());budget=SharedAPIBudget(args.budget_ledger,config)
    manifest={'version':'paired-expert-collection.v1','student_manifest_sha256':base.sha(args.student_run/'batch-manifest.json'),
        'source_hashes':hashes,'teacher_script_sha256':base.sha(Path(__file__)),
        'budget_ledger':str(args.budget_ledger),'budget_config':config,'concurrency':4,'max_tokens':8192,
        'teacher_model':config['model'],'student_prefix_labels':0,'private_reasoning_labels':0}
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==manifest
    else:
        atomic_json(out/'manifest.json',manifest);atomic_json(out/'budget-before.json',budget.snapshot())
        atomic_json(out/'cases.json',cases);atomic_json(out/'branches.json',branches);atomic_json(out/'excluded-branches.json',excluded)
    tasks=[(c,mode) for c in cases for mode in ['demo','correction'] if mode=='demo' or c['id'] in branches]
    jobs=json.loads((out/'jobs.json').read_text()) if (out/'jobs.json').exists() else {f'{c["id"]}/{mode}':{'status':'pending'} for c,mode in tasks}
    for name,job in jobs.items():
        if job['status']=='running':raise RuntimeError(f'Interrupted teacher attempt retained: {name}; no automatic paid rerun')
        if job['status']=='complete':
            for file,digest in job['evidence_sha256'].items():assert base.sha(out/name/file)==digest
    atomic_json(out/'jobs.json',jobs)
    if all(j['status']=='complete' for j in jobs.values()):print('TEACHER_RESUME_NO_PENDING_JOBS',flush=True);return
    base.settings.agentic_guard_mode='enforce';semaphore=asyncio.Semaphore(4);stop=asyncio.Event()
    async def run(case,mode):
        name=f'{case["id"]}/{mode}';job=jobs[name]
        async with semaphore:
            if job['status']=='complete' or stop.is_set():return
            target=out/name;target.mkdir(parents=True,exist_ok=False)
            job['status']='running';atomic_json(out/'jobs.json',jobs)
            client=BudgetedGLMToolClient(api_key=key,budget=budget,budget_task=f'oec20/{name}',timeout=180)
            policy=base.AuditedPolicy(AuditableNativePolicy(client,model=config['model'],temperature=1,max_tokens=8192),target/'model-calls.jsonl')
            if mode=='correction':
                branch=branches[case['id']];source=Path(branch['student_evidence'])
                assert base.sha(source/'episode.json')==branch['source_episode_sha256']
                assert base.sha(Path(branch['snapshot_path']))==branch['snapshot_sha256']
                state,backend=restore(case,branch['snapshot'],json.loads((source/'tool-calls.json').read_text()))
                provenance={**{k:v for k,v in branch.items() if k!='snapshot'},'source_state_hash':branch['snapshot']['state_after_hash']}
            else:
                with base.frozen_reference_time(base.MOMENT):state=base.initial(case)
                backend=CurriculumResearchExecutor(case);provenance={'kind':'initial_state_demonstration'}
            original=state.model_dump(mode='json');prefix=len(backend.calls)
            provenance.update(mode=mode,case_id=case['id'],source_group=case['source_group'],lineage_group=case['lineage_group'],
                prefix_tool_calls=prefix,provider_initial_state=provider_snapshot(backend),original_state_hash=_canonical_hash(original),
                changed_initial_fields=['budget.timeout_ms'],student_prefix_targets=0)
            state.budget=state.budget.model_copy(update={'timeout_ms':state.budget.used_latency_ms+600000})
            atomic_json(target/'provenance.json',provenance)
            class Recorder(base.EpisodeRecorder):
                def record_step(self,**kwargs):
                    super().record_step(**kwargs);step=self.episode.steps[-1]
                    atomic_json(target/f'state-after-{step.step_index:03d}.json',dict(case_id=case['id'],split='train',source_group=case['source_group'],
                        state=redact_pii(kwargs['state_after'].model_dump(mode='json')),state_after_hash=step.state_after_hash,
                        provider_state=provider_snapshot(backend),error_code=step.verification.get('error_code')))
            started=time.perf_counter()
            try:
                with base.frozen_reference_time(base.MOMENT):
                    recorder=Recorder(state,environment_version=base.ARCHITECTURE_VERSION,validator_version=base.VALIDATOR_VERSION,
                        policy_name=config['model'],policy_version='paired-expert.v1')
                    await base.BoundedAgentLoop().run(state,policy=base.SelfRepairingAgentPolicy(policy),executor=base.TravelActionExecutor(backend),recorder=recorder)
                ep=recorder.episode;errors=base.EpisodeReplayVerifier().verify(ep);assert not errors,errors
                row=base.judge(case,ep,policy.calls)
                row.update(mode=mode,reward=base.HierarchicalRewardEngine().score(ep).model_dump(mode='json'),replay_errors=errors,
                    provider_failures=sum(bool((c.get('generation') or {}).get('provider_error')) for c in policy.calls),
                    parse_or_schema_errors=sum((c.get('error') or {}).get('type')=='PolicyOutputError' for c in policy.calls),
                    wall_seconds=time.perf_counter()-started,teacher_tool_calls=len(backend.calls)-prefix)
                row.update(quality_audit(case,row,backend))
                infrastructure=[c['error'] for c in policy.calls if c.get('error') and c['error']['type']!='PolicyOutputError']
                if infrastructure or row['provider_failures']:
                    row['quality_candidate']=False;row['rejection_reasons'].append('TEACHER_PROVIDER_ERROR');stop.set()
                atomic_json(target/'episode.json',ep.model_dump(mode='json'));atomic_json(target/'tool-calls.json',backend.calls);atomic_json(target/'summary.json',row)
                job.update(status='complete',evidence_sha256={f:base.sha(target/f) for f in ['episode.json','tool-calls.json','model-calls.jsonl','summary.json','provenance.json']})
                atomic_json(out/'jobs.json',jobs)
                print(json.dumps({'event':'paired_teacher_done','case':case['id'],'mode':mode,'passed':row['criterion_passed'],'candidate':row['quality_candidate'],'calls':row['model_calls'],'reasons':row['rejection_reasons']},ensure_ascii=False),flush=True)
            finally:await client.aclose()
    await asyncio.gather(*(run(c,m) for c,m in tasks))
    atomic_json(out/'budget-after.json',budget.snapshot())
    print('PAIRED_TEACHER_COMPLETE',sum(j['status']=='complete' for j in jobs.values()),len(jobs),flush=True)
    if stop.is_set() or any(j['status']!='complete' for j in jobs.values()):raise RuntimeError('Provider issue: paid queue stopped with evidence retained')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['student-run','output','budget-config','budget-ledger']:p.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
