from copy import deepcopy
import pytest
from evaluation.research_fixture import catalog,search_pois,detail,current_info
from scripts.evaluate_harness_base import build_cases,MOMENT,FrozenResearchExecutor,initial
from scripts.build_harness_training_pilot import build
from agentic.loop import BoundedAgentLoop,PolicyAction
from agentic.action_executor import TravelActionExecutor
from agentic.trajectory import EpisodeRecorder,EpisodeReplayVerifier
from agentic.clock import frozen_reference_time


def case():return build_cases()[0]


def test_dining_search_detail_and_web_agree():
    c=case();restaurants=search_pois(c,{'city':'上海','keywords':['餐厅']})
    assert len(restaurants)==2
    assert all(r['category']=='restaurant' for r in restaurants)
    r=restaurants[0]; assert detail(c,r['name'])==r
    web=current_info(c,{'query':r['name'],'info_type':'restaurant','date':'2026-09-18'},MOMENT)
    assert len(web['results'])==1
    assert str(r['average_cost']) in web['results'][0]['snippet']
    assert r['address'] in web['results'][0]['snippet']
    assert web['results'][0]['url']==r['source_url']


def test_wrong_city_unknown_entity_wrong_date_remain_empty():
    c=case()
    assert not search_pois(c,{'city':'不存在城','keywords':['餐厅']})
    assert not search_pois(c,{'keywords':['不存在的独角兽店']})
    assert not search_pois(c,{'keywords':['上海不存在的独角兽餐厅']})
    assert all('素食' in r['tags'] for r in search_pois(c,{'keywords':['素食餐厅']}))
    for args in [{'query':'上海不存在的独角兽餐厅 营业时间','info_type':'restaurant'},
                 {'query':c['pois'][0]['name']+' 2026-09-19','date':'2026-09-18'},
                 {'query':'无关网页'}]:
        assert current_info(c,args,MOMENT)['results']==[]
    with pytest.raises(ValueError):detail(c,'不存在的店')


def test_dates_staleness_and_closure_do_not_depend_on_call_counter():
    c=build()[12]; name=c['pois'][0]['name'];d=c['slots']['start_date']
    args={'query':name+' '+d,'date':d}
    assert '闭馆' in current_info(c,args,MOMENT)['results'][0]['snippet']
    changed=deepcopy(c);changed['pois'][0]['closed_dates']=[]
    assert '开放时间' in current_info(changed,args,MOMENT)['results'][0]['snippet']
    assert current_info(c,args,MOMENT,stale=True)['queried_at']!=current_info(c,args,MOMENT)['queried_at']


def test_pilot_groups_do_not_reuse_diagnostic_entities_or_requests():
    pilot=build();old=build_cases()
    assert len(pilot)==24 and len({c['id'] for c in pilot})==24
    assert not {p['name'] for c in pilot for p in c['pois']} & {p['name'] for c in old for p in c['pois']}
    assert {c['split'] for c in pilot}=={'train'}
    assert all(len(c['slots']['must_visit'])==2 and c['restaurants'] for c in pilot)


@pytest.mark.asyncio
async def test_restaurant_research_preserves_attractions_and_real_solver_meal_slots():
    class AuditPolicy:
        def __init__(self):self.actions=iter([
            PolicyAction(action='search_pois'),
            PolicyAction(action='search_pois',arguments={'keywords':['餐厅']}),
            PolicyAction(action='search_current_info',arguments={'query':'上海 餐厅 推荐','info_type':'restaurant'}),
            *[PolicyAction(action=n) for n in ['get_poi_detail','get_route_matrix','solve_itinerary','validate_itinerary','finish']]])
        async def propose(self,context):return next(self.actions)
    with frozen_reference_time(MOMENT):
        state=initial(case());backend=FrozenResearchExecutor(case())
        recorder=EpisodeRecorder(state,environment_version='test',validator_version='test',policy_name='scripted-infrastructure',policy_version='test')
        await BoundedAgentLoop().run(state,policy=AuditPolicy(),executor=TravelActionExecutor(backend),recorder=recorder)
        assert not EpisodeReplayVerifier().verify(recorder.episode)
        assert recorder.episode.steps[-1].action.action=='finish'
        assert recorder.episode.termination_reason=='awaiting_user'
        candidates=[a for a in state.artifacts.values() if a.artifact_type=='poi_candidate_set'][-1]
        assert {'restaurant','attraction'}=={p['category'] for p in candidates.payload['pois']}
        solver=[s for s in recorder.episode.steps if s.action.action=='solve_itinerary'][0]
        assert all(p['category']!='restaurant' for p in solver.action.executed_arguments['pois'])
        # A successful itinerary only certifies meal slots; no invented venue assignment.
        report=[a for a in state.artifacts.values() if a.artifact_type=='validation_report'][-1]
        assert report.payload['hard_pass'] is True
