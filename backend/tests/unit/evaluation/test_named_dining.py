from copy import deepcopy
import pytest
from vrp_solver_service.models import POIInput,ConstraintsInput,SolverRequest
from vrp_solver_service.solver import TravelVRPSolver
from planner.preprocessing.transport_selector import TransportSelector
from evaluation.validator import ItineraryValidator


def solve_named(*,days=1,closed=False,budget=500):
    pois=[POIInput(id='p',name='河岸公园',lat=31.23,lng=121.47,ticket_price=10,duration_minutes=60,close_time='21:00'),
          POIInput(id='r',name='河岸家常菜馆',category='restaurant',lat=31.24,lng=121.48,average_cost=35.25,
                   open_time='10:00',close_time='22:00',closed_dates=['2026-10-02'] if closed else [])]
    c=ConstraintsInput(travel_days=days,trip_start_date='2026-10-02',include_restaurant=True,meals_per_day=2,
        require_named_restaurants=True,must_visit=['p'],total_budget=budget,food_day=999)
    hotel=POIInput(id='__hotel',name='Hotel')
    dist,costs=TransportSelector().build_matrices([hotel,*pois],c)
    result=TravelVRPSolver().solve(SolverRequest(pois=deepcopy(pois),constraints=c,dist_matrix=dist,tc_matrix=costs,strategy='greedy'))
    config={**c.model_dump(),'route_poi_ids':['__hotel','p','r'],'route_time_matrix':dist,'route_cost_matrix':costs}
    facts=[p.model_dump() for p in pois]
    return result,config,facts


@pytest.mark.parametrize('days',[1,2])
def test_named_restaurant_routed_with_exact_cost_and_reusable_across_days(days):
    result,config,facts=solve_named(days=days)
    assert result.status in {'optimal','feasible'}
    itinerary=[d.model_dump() for d in result.days]
    report=ItineraryValidator().validate(itinerary,config,facts)
    assert report.hard_pass,report.model_dump()
    for d in itinerary:
        meals=[a for a in d['activities'] if a['category']=='restaurant']
        assert len(meals)==2 and all(a['poi_id']=='r' and a['meal_cost']==35.25 for a in meals)
        assert d['total_cost']<200  # no duplicate 999 food_day charge
    assert any(a['transit_from_prev']['duration_min']>0 for d in itinerary for a in d['activities'])


@pytest.mark.parametrize('kwargs',[{'closed':True},{'budget':50}])
def test_named_meals_cannot_fallback_to_generic_or_ignore_budget(kwargs):
    result,_,_=solve_named(**kwargs)
    assert result.status=='infeasible' and not result.days


def test_validator_rejects_forged_missing_or_underpriced_meals_and_transit():
    result,config,facts=solve_named();original=[d.model_dump() for d in result.days]
    for kind in ['missing','cost','identity','transit']:
        days=deepcopy(original);meal=next(a for a in days[0]['activities'] if a['category']=='restaurant')
        if kind=='missing':days[0]['activities'].remove(meal)
        if kind=='cost':meal['meal_cost']=0
        if kind=='identity':meal['poi_id']='__meal_d1_fake'
        if kind=='transit':meal['transit_from_prev']['duration_min']=999
        assert not ItineraryValidator().validate(days,config,facts).hard_pass
