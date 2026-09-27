from copy import deepcopy

import pytest

from evaluation.feasibility import assess_required_facts
from evaluation.validator import ItineraryValidator


def facts():
    return [{"id":"museum", "name":"城市博物馆", "ticket_price":2000,
             "closed_dates":["2026-09-18"], "closed_weekdays":[], "date_opening_hours":{}}]


def constraints(**changes):
    return {"must_visit":["museum"],"total_budget":300,"travel_days":1,
            "trip_start_date":"2026-09-18", **changes}


def test_proves_only_required_cost_and_entire_travel_window():
    report=assess_required_facts(constraints(),facts())
    assert report['status']=='infeasible'
    assert {w['code'] for w in report['witnesses']} == {
        'REQUIRED_COST_EXCEEDS_BUDGET','REQUIRED_POI_CLOSED_ALL_DATES'}
    assert report==assess_required_facts(constraints(),facts())
    longer=assess_required_facts(constraints(travel_days=2,total_budget=2500),facts())
    assert longer['status']=='undetermined'


@pytest.mark.parametrize('config',[
    constraints(must_visit=[]), constraints(must_visit=['not-found']),
    constraints(trip_start_date=None,total_budget=0),
    constraints(trip_start_date='bad-date',total_budget=2000),
    constraints(travel_days=0,total_budget=2000),
])
def test_unknown_optional_or_just_affordable_facts_do_not_prove_infeasibility(config):
    assert assess_required_facts(config,facts())['status']=='undetermined'


def test_aliases_do_not_double_count_and_unknown_entity_is_not_invented():
    f=facts(); f[0]['ticket_price']=200
    result=assess_required_facts(constraints(must_visit=['museum','城市博物馆','missing'],trip_start_date=None),f+deepcopy(f))
    assert result['status']=='undetermined'
    assert result['unresolved_required_entities']==['missing']


def test_date_override_or_conflicting_records_cannot_certify_closure():
    f=facts(); f[0]['date_opening_hours']={'2026-09-18':['09:00','17:00']}
    assert assess_required_facts(constraints(total_budget=5000),f)['status']=='undetermined'
    conflicting=facts(); conflicting[0]['ticket_price']=20
    assert assess_required_facts(constraints(),facts()+conflicting)['status']=='undetermined'


def test_budget_lower_bound_can_use_known_subset_but_not_bad_values():
    assert assess_required_facts(constraints(must_visit=['museum','unknown']),facts())['status']=='infeasible'
    for value in [float('nan'),float('inf'),-10,None,True]:
        f=facts(); f[0]['ticket_price']=value
        assert assess_required_facts(constraints(trip_start_date=None),f)['status']=='undetermined'


def test_missing_required_poi_in_one_draft_does_not_prove_unsolvable_task():
    f=facts(); f[0]['ticket_price']=20; f[0]['closed_dates']=[]
    report=ItineraryValidator().validate([],constraints(),f)
    assert not report.hard_pass
    assert report.evaluation_scope=='evidence_only'
    assert report.feasibility['status']=='undetermined'


def test_proof_id_changes_when_cost_or_dates_change():
    old=assess_required_facts(constraints(),facts())
    new=assess_required_facts(constraints(total_budget=400),facts())
    assert old['witnesses'][-1]['evidence_id']!=new['witnesses'][-1]['evidence_id']


def test_weekly_closure_assumptions_are_not_dated_proof():
    f=facts(); f[0].update(ticket_price=20,closed_dates=[],closed_weekdays=[4])
    assert assess_required_facts(constraints(),f)['status']=='undetermined'
