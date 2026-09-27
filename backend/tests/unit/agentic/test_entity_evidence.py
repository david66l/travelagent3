from copy import deepcopy
from datetime import timedelta
import pytest
from agentic.clock import frozen_reference_time
from agentic.current_evidence import fresh_searches,opening_target_covered
from agentic.react import ResearchSufficiencyVerifier
from agentic.reward import _arguments_grounded
from agentic.action_executor import TravelActionExecutor
from agentic.loop import PolicyAction
from agentic.harness import invalidate_derived_evidence,refresh_constraint_evidence
from scripts.evaluate_harness_base import FrozenResearchExecutor,initial,MOMENT
from scripts.build_harness_training_pilot import build


async def act(state,executor,name,**args):
    outcome=await executor.execute(task=state.task_graph.tasks[0],action=PolicyAction(action=name,arguments=args),ledger=state)
    invalidate_derived_evidence(state,name,artifacts=outcome.artifacts,facts=outcome.facts)
    for a in outcome.artifacts:state.artifacts[a.artifact_id]=a
    for f in outcome.facts:state.facts[f.fact_id]=f
    refresh_constraint_evidence(state)
    return outcome


@pytest.mark.asyncio
async def test_fresh_second_entity_cannot_cover_stale_first_entity():
    with frozen_reference_time(MOMENT):
        case=build()[20];state=initial(case);executor=TravelActionExecutor(FrozenResearchExecutor(case))
        for name in ['search_pois','get_poi_detail','get_route_matrix']:await act(state,executor,name)
        for query in case['slots']['current_info_queries']:
            await act(state,executor,'search_current_info',query=query,info_type='opening_hours')
        report=ResearchSufficiencyVerifier().evaluate(state)
        assert not report.sufficient
        assert any(case['pois'][0]['name'] in m for m in report.missing)
        assert not any(case['pois'][1]['name'] in m for m in report.missing)
        # A new query supplies fresh evidence; old stale history does not veto it.
        await act(state,executor,'search_current_info',query=case['slots']['current_info_queries'][0]+' 官方公告',info_type='opening_hours')
        await act(state,executor,'get_route_matrix')
        assert ResearchSufficiencyVerifier().evaluate(state).sufficient


@pytest.mark.asyncio
async def test_stale_closure_never_mutates_solver_facts():
    with frozen_reference_time(MOMENT):
        case=build()[0];state=initial(case);executor=TravelActionExecutor(FrozenResearchExecutor(case))
        for name in ['search_pois','get_poi_detail']:await act(state,executor,name)
        result=await act(state,executor,'search_current_info',query=case['slots']['current_info_queries'][0],info_type='closure')
        artifact=result.artifacts[0]
        artifact.payload['queried_at']=(MOMENT-timedelta(days=3)).isoformat()
        artifact.payload['results'][0]['snippet']=case['pois'][0]['name']+' '+case['slots']['start_date']+' 临时闭馆'
        assert not fresh_searches(state)
        facts=TravelActionExecutor._planning_candidate_items_with_evidence(state)
        assert case['slots']['start_date'] not in facts[0].get('closed_dates',[])


def test_generated_queries_are_not_asserted_facts_but_dates_remain_grounded():
    context={'hard_constraints':{'start_date':'2026-10-21'}}
    assert _arguments_grounded({'keywords':['古建园林']},context,action='search_pois')
    assert _arguments_grounded({'query':'场馆 最新开放公告','date':'2026-10-21'},context,action='search_current_info')
    assert not _arguments_grounded({'query':'场馆','date':'2028-01-01'},context,action='search_current_info')
    assert not _arguments_grounded({'constraints':{}},context,action='search_pois')
