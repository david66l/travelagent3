import asyncio,json
import pytest
from agentic.loop import ActionOutcome,BoundedAgentLoop,PolicyAction,gather_or_cancel
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from agentic.trajectory import EpisodeRecorder,EpisodeReplayVerifier,episode_content_hash
from agentic.episode_accounting import verify_episode_accounting
from agentic.observations import ObservationEnvelope

def ledger(**budget):
    state=AgentLedgerState(**initialize_agent_ledger({'user_input':'Plan a trip',
        'slots':{'destination':'Hangzhou','travel_days':1}},mode='agent')['agent_ledger'])
    state.budget=state.budget.model_copy(update=budget);return state

def recorder(state):return EpisodeRecorder(state,environment_version='test-accounting',validator_version='test',policy_name='scripted-test',policy_version='1')

class Policy:
    async def propose(self,context):return PolicyAction(action='search_pois',arguments={'keywords':['alice@example.test']},token_usage=3)

class Executor:
    def __init__(self,calls=1):self.calls=0;self.charge=calls
    async def execute(self,*,task,action,ledger):
        self.calls+=1
        return ActionOutcome(tool_calls_used=self.charge,observations=[ObservationEnvelope.failure(tool='search_pois',code='TEST_EMPTY',message='none',retryable=False,tool_call_id=action.action_id)],status='failed',error_code='TEST_EMPTY')

def event(rec):
    entries=[v.payload for v in rec.episode.events if v.event_type=='agent_uncommitted_batch']
    assert len(entries)==1
    assert not EpisodeReplayVerifier().verify(rec.episode)
    assert not verify_episode_accounting(rec.episode)
    return entries[0]

@pytest.mark.asyncio
async def test_admission_failure_preserves_paid_proposal_without_invented_execution():
    state=ledger(max_tool_calls=0);rec=recorder(state);executor=Executor()
    result=await BoundedAgentLoop().run(state,policy=Policy(),executor=executor,recorder=rec)
    e=event(rec)
    assert result.status=='failed' and result.termination_reason=='budget_exhausted_fallback'
    assert not rec.episode.steps and executor.calls==0
    assert e['budget_after']['used_episode_steps']==1 and e['budget_after']['used_tokens']==3
    assert not e['execution_attempts'][0]['executor_started']
    assert e['execution_attempts'][0]['phase']=='resource_admission'
    assert 'alice@example.test' not in json.dumps(e) and e['training_labels_authorized'] is False

@pytest.mark.asyncio
async def test_token_limit_keeps_returned_proposal_and_denied_charge_separate():
    state=ledger(max_tokens=0);rec=recorder(state);executor=Executor()
    await BoundedAgentLoop().run(state,policy=Policy(),executor=executor,recorder=rec)
    e=event(rec)
    assert e['phase']=='proposal_budget' and e['requested_charge']=={'episode_steps':1,'tokens':3}
    assert e['budget_after']['used_episode_steps']==0 and e['policy_attempts'][0]['status']=='returned'
    assert executor.calls==0

@pytest.mark.asyncio
async def test_post_execution_overflow_retains_observation_and_actual_reported_tool_usage():
    state=ledger(max_tool_calls=1);rec=recorder(state);executor=Executor(calls=2)
    await BoundedAgentLoop().run(state,policy=Policy(),executor=executor,recorder=rec)
    e=event(rec)
    assert e['phase']=='post_execution_budget' and not rec.episode.steps
    assert e['execution_attempts'][0]['outcome']['tool_calls_used']==2
    assert len(e['execution_attempts'][0]['outcome']['observations'])==1
    assert e['budget_after']['used_tool_calls']==1 and e['requested_charge']['tool_calls']==1

@pytest.mark.asyncio
async def test_executor_timeout_is_drained_before_final_state_is_hashed():
    state=ledger(timeout_ms=20);rec=recorder(state);cancelled=[]
    class Slow:
        async def execute(self,**kwargs):
            try:await asyncio.sleep(10)
            finally:cancelled.append(True)
    result=await BoundedAgentLoop().run(state,policy=Policy(),executor=Slow(),recorder=rec)
    e=event(rec)
    assert cancelled and result.termination_reason=='agent_deadline_exceeded'
    assert e['execution_attempts'][0]['status']=='cancelled' and e['execution_attempts'][0]['executor_started']
    assert 'outcome' not in e['execution_attempts'][0]

@pytest.mark.asyncio
async def test_failed_parallel_operation_cancels_and_awaits_sibling_cleanup():
    started=asyncio.Event();cleaned=[]
    async def slow():
        started.set()
        try:await asyncio.sleep(10)
        finally:cleaned.append(True)
    async def failed():
        await started.wait();raise ValueError('failed')
    with pytest.raises(ValueError):await gather_or_cancel(failed(),slow())
    assert cleaned

@pytest.mark.asyncio
async def test_external_cancellation_preserves_evidence_and_propagates_to_caller():
    state=ledger();rec=recorder(state);started=asyncio.Event();cleaned=[]
    class Slow:
        async def execute(self,**kwargs):
            started.set()
            try:await asyncio.sleep(10)
            finally:cleaned.append(True)
    task=asyncio.create_task(BoundedAgentLoop().run(state,policy=Policy(),executor=Slow(),recorder=rec))
    await started.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert cleaned and rec.episode.termination_reason=='agent_cancelled'
    assert event(rec)['execution_attempts'][0]['status']=='cancelled'

@pytest.mark.asyncio
async def test_terminal_budget_witness_must_match_final_state():
    state=ledger(max_tool_calls=0);rec=recorder(state)
    await BoundedAgentLoop().run(state,policy=Policy(),executor=Executor(),recorder=rec)
    e=event(rec);e['budget_after']['used_tokens']=0
    rec.episode.content_hash=episode_content_hash(rec.episode)
    assert 'UNCOMMITTED_FINAL_BUDGET_MISMATCH' in EpisodeReplayVerifier().verify(rec.episode)

@pytest.mark.asyncio
async def test_terminal_diagnostic_events_never_authorize_sft_targets():
    from agentic.sft_dataset import SFTDatasetBuilder,EpisodeCandidate
    state=ledger(max_tool_calls=0);rec=recorder(state)
    await BoundedAgentLoop().run(state,policy=Policy(),executor=Executor(),recorder=rec)
    candidate=EpisodeCandidate(scenario_id='terminal',source='teacher',template_family='test',city='Hangzhou',episode=rec.episode)
    assert 'L1_UNCOMMITTED_TERMINAL_BATCH' in SFTDatasetBuilder()._review(candidate)

@pytest.mark.asyncio
async def test_completed_step_stays_normal_and_deleted_terminal_evidence_is_detected():
    state=ledger(max_tool_calls=1);rec=recorder(state)
    result=await BoundedAgentLoop().run(state,policy=Policy(),executor=Executor(),recorder=rec,max_batches=1)
    assert result.status=='running' and len(rec.episode.steps)==1
    assert not [e for e in rec.episode.events if e.event_type=='agent_uncommitted_batch']
    assert not EpisodeReplayVerifier().verify(rec.episode)
    failed=ledger(max_tool_calls=0);fr=recorder(failed)
    await BoundedAgentLoop().run(failed,policy=Policy(),executor=Executor(),recorder=fr)
    fr.episode.events=[e for e in fr.episode.events if e.event_type!='agent_uncommitted_batch']
    fr.episode.content_hash=episode_content_hash(fr.episode)
    assert 'EPISODE_STEP_ACCOUNTING_MISMATCH' in EpisodeReplayVerifier().verify(fr.episode)
