import asyncio
import json
from types import SimpleNamespace
import pytest
from agentic.batched_policy import BatchedCheckpointEngine, decode_action
from agentic.policy import PolicyOutputError
from scripts.collect_harness_batch import initialize_jobs, atomic_json, pending_cases, completed_summaries


@pytest.mark.asyncio
async def test_dynamic_batches_keep_session_audits_separate():
    engine=BatchedCheckpointEngine(SimpleNamespace(do_sample=False,structured_decoding_mode='native'),batch_size=4,wait_ms=10)
    calls=[]
    def generate(requests):
        calls.append(len(requests))
        return [(r.messages[0],{'request':r.messages[0]},None) for r in requests]
    engine._generate=generate
    sessions=[engine.session() for _ in range(4)]
    results=await asyncio.gather(*(s.propose_from_history([i],tools=[],allowed_actions=['search_pois']) for i,s in enumerate(sessions)))
    await engine.aclose()
    assert calls==[4] and results==list(range(4))
    assert [s.last_generation_audit for s in sessions]==[{'request':i} for i in range(4)]


@pytest.mark.asyncio
async def test_worker_failure_finishes_all_waiters_and_can_continue():
    engine=BatchedCheckpointEngine(SimpleNamespace(do_sample=False,structured_decoding_mode='native'),batch_size=2)
    def fail(requests):raise RuntimeError('simulated generation failure')
    engine._generate=fail
    results=await asyncio.gather(*(engine.submit([i],[],[],{}) for i in range(2)),return_exceptions=True)
    assert all(isinstance(r,RuntimeError) for r in results)
    engine._generate=lambda rs:[('ok',{},None) for r in rs]
    assert (await engine.submit([],[],[],{}))[0]=='ok'
    await engine.aclose()


def test_batch_parser_enforces_same_action_and_argument_authority():
    audit={}
    action=decode_action('<tool_call>{"name":"search_pois","arguments":{"keywords":["公园"]}}</tool_call>',audit,['search_pois'],{},'fixture',123,20,5)
    assert action.arguments=={'keywords':['公园']}
    with pytest.raises(PolicyOutputError):
        decode_action('{"name":"finish","arguments":{}}',{},['search_pois'],{},'fixture',10,3,1)
    with pytest.raises(PolicyOutputError):
        decode_action('{"name":"search_pois","arguments":{"keywords":"公园"}}',{},['search_pois'],{},'fixture',10,3,1)


def test_resume_preserves_completed_measurements_and_bounds_interrupted_attempts(tmp_path):
    cases=[{'id':'a'},{'id':'b'}];config={'immutable':'v1'}
    jobs=initialize_jobs(tmp_path,cases,config,2)
    jobs['a'].update(status='complete',attempts=1,criterion_passed=False)
    jobs['b'].update(status='running',attempts=1,evidence='b/attempt-01')
    atomic_json(tmp_path/'jobs.json',jobs)
    resumed=initialize_jobs(tmp_path,cases,config,2)
    assert resumed['a']['status']=='complete' and not resumed['a']['criterion_passed']
    assert resumed['b']['status']=='retry_pending' and resumed['b']['attempts']==1
    resumed['b'].update(status='running',attempts=2)
    atomic_json(tmp_path/'jobs.json',resumed)
    assert initialize_jobs(tmp_path,cases,config,2)['b']['status']=='infrastructure_failed'
    with pytest.raises(ValueError,match='Cannot resume changed'):
        initialize_jobs(tmp_path,cases,{'immutable':'v2'},2)


def test_resume_does_not_report_exhausted_queue_as_success():
    with pytest.raises(RuntimeError, match='No retry budget'):
        pending_cases([{'id':'a'}], {'a':{'status':'infrastructure_failed'}})
    assert pending_cases([{'id':'a'}], {'a':{'status':'complete'}}) == []


def test_completed_evidence_is_checked_before_resume_skip(tmp_path):
    import hashlib
    target=tmp_path/'a'/'attempt-01'
    target.mkdir(parents=True)
    files={'summary.json':'{"case":"a","criterion_passed":true}',
           'episode.json':'{}','model-calls.jsonl':'{}\n','tool-calls.json':'[]'}
    for name,text in files.items():(target/name).write_text(text)
    hashes={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in files}
    jobs={'a':{'status':'complete','evidence':'a/attempt-01','evidence_sha256':hashes}}
    assert completed_summaries(tmp_path,jobs)==[{'case':'a','criterion_passed':True}]
    (target/'model-calls.jsonl').write_text('corrupted')
    with pytest.raises(ValueError,match='missing or changed'):
        completed_summaries(tmp_path,jobs)
