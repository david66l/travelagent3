from copy import deepcopy
from uuid import uuid4

import pytest

from agentic.harness import preflight_action, invalidate_derived_evidence
from agentic.loop import BoundedAgentLoop, PolicyAction
from agentic.action_executor import TravelActionExecutor
from agentic.trajectory import EpisodeRecorder, EpisodeReplayVerifier
from agentic.reward import HierarchicalRewardEngine
from agentic.state import ArtifactRecord
from agentic.clock import frozen_reference_time
from evaluation.validator import ItineraryValidator
from scripts.evaluate_harness_base import build_cases, initial, FrozenResearchExecutor, MOMENT


class ScriptedAudit:
    """Infrastructure test policy, never measured teacher/student results."""
    def __init__(self, *, solver=False, valid_reference=True, direct=False):
        self.actions=iter(['search_pois','get_poi_detail'] + (['get_route_matrix','solve_itinerary'] if solver else []) + ([] if direct else ['validate_itinerary']) + ['abort'])
        self.valid_reference=valid_reference
        self.contexts=[]
    async def propose(self,context):
        self.contexts.append(context)
        name=next(self.actions)
        arguments={}
        if name=='abort':
            arguments={'reason':'保持您的要求，本次无法完成该行程。',
                       'evidence_ids':context.capability['evidence_ids'] if self.valid_reference else ['made-up']}
        return PolicyAction(action=name,arguments=arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', [c for c in build_cases() if c['family']=='constraint'], ids=lambda c:c['id'])
@pytest.mark.parametrize('solver',[False,True])
async def test_real_infeasible_tasks_support_grounded_stop_with_or_without_solving(case,solver):
    with frozen_reference_time(MOMENT):
        state=initial(case); backend=FrozenResearchExecutor(case)
        policy=ScriptedAudit(solver=solver)
        recorder=EpisodeRecorder(state,environment_version='test',validator_version='test',policy_name='scripted-test',policy_version='test')
        await BoundedAgentLoop().run(state,policy=policy,executor=TravelActionExecutor(backend),recorder=recorder)
        assert not EpisodeReplayVerifier().verify(recorder.episode)
        score=HierarchicalRewardEngine().score(recorder.episode)
        assert score.gate_status=='passed'
        assert recorder.episode.steps[-1].action.action=='abort'
        assert bool(state.goal.capability.evidence_ids)
        # Evidence permits a stop; the model can still choose other actions.
        assert {'search_pois','solve_itinerary','ask_user','abort'} <= set(policy.contexts[-1].allowed_actions)
        assert state.budget.used_solver_calls == int(solver)


def certified_state():
    case=next(c for c in build_cases() if c['id']=='constraint-03')
    state=initial(case)
    report=ItineraryValidator().validate([],{'must_visit':['p0'],'total_budget':300},case['pois'])
    artifact=ArtifactRecord(artifact_id=str(uuid4()),artifact_type='validation_report',
        payload=report.model_dump(mode='json'),goal_version=1,plan_version=1)
    state.artifacts[artifact.artifact_id]=artifact
    BoundedAgentLoop._refresh_post_validation_capability(state,[artifact])
    return state


def test_stale_forged_or_missing_refs_cannot_authorize_stop():
    state=certified_state()
    valid=state.goal.capability.evidence_ids
    assert preflight_action(state,PolicyAction(action='abort',arguments={'reason':'不能满足当前约束','evidence_ids':valid})) is None
    for ids in [None,['made-up'],valid+['made-up']]:
        error=preflight_action(state,PolicyAction(action='abort',arguments={'reason':'不能满足当前约束','evidence_ids':ids}))
        assert error.error_code=='TERMINAL_EVIDENCE_REFERENCE_INVALID'
    invalidate_derived_evidence(state,'search_pois')
    assert state.goal.capability.status=='undetermined'
    assert state.goal.capability.evidence_ids==[]
    assert preflight_action(state,PolicyAction(action='abort',arguments={'reason':'停止','evidence_ids':valid})).error_code=='TERMINAL_EVIDENCE_REFERENCE_INVALID'


def test_retrieval_omission_is_undetermined_not_infeasible_or_forced_clarification():
    state=certified_state()
    payload=ItineraryValidator().validate([],{'must_visit':['missing']},[]).model_dump(mode='json')
    artifact=ArtifactRecord(artifact_id=str(uuid4()),artifact_type='validation_report',payload=payload,goal_version=1,plan_version=1)
    BoundedAgentLoop._refresh_post_validation_capability(state,[artifact])
    assert state.goal.capability.status=='undetermined'
    assert not state.goal.capability.evidence_ids


def test_only_user_supplied_relaxation_options_are_exposed():
    state=certified_state()
    state.goal.hard_constraints['constraint_flexibility']={
        'relaxable_constraints':['total_budget'],'relaxation_options':{'total_budget':['提高总预算至2500元']}}
    artifact=next(iter(state.artifacts.values()))
    BoundedAgentLoop._refresh_post_validation_capability(state,[artifact])
    assert state.goal.capability.actionable_alternatives is True
    assert state.goal.capability.alternatives==['提高总预算至2500元']


@pytest.mark.asyncio
async def test_closed_fixture_search_and_details_do_not_contradict():
    case=next(c for c in build_cases() if c['id']=='constraint-01')
    backend=FrozenResearchExecutor(case)
    data=await backend.research_handler('search_current_info')({'query':case['pois'][0]['name']+' 开放时间'})
    assert '闭馆' in data.data['results'][0]['snippet']
    assert '08:00' not in data.data['results'][0]['snippet']


def test_tampered_capability_ref_does_not_override_actual_report():
    state=certified_state()
    state.goal.capability.evidence_ids=['fabricated-capability-id']
    result=preflight_action(state,PolicyAction(action='abort',arguments={'reason':'停止','evidence_ids':['fabricated-capability-id']}))
    assert result.error_code=='TERMINAL_EVIDENCE_REFERENCE_INVALID'


@pytest.mark.asyncio
@pytest.mark.parametrize('case',[c for c in build_cases() if c['family']=='constraint'],ids=lambda c:c['id'])
async def test_model_can_stop_immediately_after_verified_detail_evidence(case):
    with frozen_reference_time(MOMENT):
        state=initial(case); policy=ScriptedAudit(direct=True)
        recorder=EpisodeRecorder(state,environment_version='test',validator_version='test',policy_name='scripted',policy_version='test')
        await BoundedAgentLoop().run(state,policy=policy,executor=TravelActionExecutor(FrozenResearchExecutor(case)),recorder=recorder)
        assert [s.action.action for s in recorder.episode.steps]==['search_pois','get_poi_detail','abort']
        assert HierarchicalRewardEngine().score(recorder.episode).gate_status=='passed'


def test_normal_state_schema_remains_compatible_and_refs_appear_only_with_proof():
    from agentic.policy_actions import policy_action_schemas_for_state
    for capability,expected in [({},False),({'evidence_ids':['proof']},True)]:
        properties=policy_action_schemas_for_state(['abort'],capability=capability)[0]['function']['parameters']['properties']
        assert ('evidence_ids' in properties)==expected


@pytest.mark.asyncio
async def test_rejected_abort_returns_to_model_for_reference_repair():
    class RepairReference(ScriptedAudit):
        def __init__(self):
            super().__init__(direct=True)
            self.actions=iter(['search_pois','get_poi_detail','abort','abort'])
            self.abort_count=0
        async def propose(self,context):
            action=await super().propose(context)
            if action.action=='abort':
                self.abort_count+=1
                if self.abort_count==1:action.arguments['evidence_ids']=['forged']
            return action
    with frozen_reference_time(MOMENT):
        case=next(c for c in build_cases() if c['id']=='constraint-03')
        state=initial(case);policy=RepairReference()
        recorder=EpisodeRecorder(state,environment_version='test',validator_version='test',policy_name='scripted-test',policy_version='test')
        await BoundedAgentLoop().run(state,policy=policy,executor=TravelActionExecutor(FrozenResearchExecutor(case)),recorder=recorder)
        assert policy.abort_count==2
        assert recorder.episode.steps[-2].verification['error_code']=='TERMINAL_EVIDENCE_REFERENCE_INVALID'
        assert 'TERMINAL_EVIDENCE_REFERENCE_INVALID' in str(policy.contexts[-1].failure_summary)
        assert recorder.episode.steps[-1].action.arguments['evidence_ids']==state.goal.capability.evidence_ids
        assert HierarchicalRewardEngine().score(recorder.episode).gate_status=='passed'
        assert not EpisodeReplayVerifier().verify(recorder.episode)


@pytest.mark.asyncio
async def test_replaced_detail_with_affordable_cost_clears_old_proof():
    from agentic.harness import refresh_constraint_evidence
    with frozen_reference_time(MOMENT):
        case=next(c for c in build_cases() if c['id']=='constraint-03')
        state=initial(case); backend=FrozenResearchExecutor(case)
        executor=TravelActionExecutor(backend)
        for name in ['search_pois','get_poi_detail','get_poi_detail']:
            outcome=await executor.execute(task=state.task_graph.tasks[0],action=PolicyAction(action=name),ledger=state)
            invalidate_derived_evidence(state,name,artifacts=outcome.artifacts,facts=outcome.facts)
            for artifact in outcome.artifacts:state.artifacts[artifact.artifact_id]=artifact
            for fact in outcome.facts:state.facts[fact.fact_id]=fact
            refresh_constraint_evidence(state)
            if name=='get_poi_detail' and backend.case['pois'][0]['ticket_price']==2000:
                old_ids=state.goal.capability.evidence_ids
                assert old_ids
                backend.case['pois'][0]['ticket_price']=20
        assert state.goal.capability.status=='undetermined'
        assert not any(a.artifact_type=='constraint_evidence' for a in state.artifacts.values())
        assert preflight_action(state,PolicyAction(action='abort',arguments={'reason':'停止','evidence_ids':old_ids})).error_code=='TERMINAL_EVIDENCE_REFERENCE_INVALID'
