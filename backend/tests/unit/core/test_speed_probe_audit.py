from types import SimpleNamespace
from scripts.audit_teacher_speed_probe import accounting_errors,raw_policy_arguments
import pytest

def test_counter_gap_is_an_explicit_rejection_even_when_terminal_fallback_exists():
    ep=SimpleNamespace(steps=[1,2],initial_state={'budget':{'used_episode_steps':0}},
        final_state={'budget':{'used_episode_steps':3,'used_tool_calls':1,'used_solver_calls':0}})
    records=[{'call':{'function':{'name':'search_pois'}}}]
    assert accounting_errors(ep,records)==['EPISODE_STEP_COUNTER_MISMATCH']

def test_all_actual_provider_calls_are_accounted_including_solver():
    ep=SimpleNamespace(steps=[1],initial_state={'budget':{'used_episode_steps':4}},
        final_state={'budget':{'used_episode_steps':5,'used_tool_calls':1,'used_solver_calls':1}})
    records=[{'call':{'function':{'name':'solve_itinerary'}}}]
    assert accounting_errors(ep,records)==[]
    ep.final_state['budget']['used_tool_calls']=0;ep.final_state['budget']['used_solver_calls']=0
    assert accounting_errors(ep,records)==['TOOL_CALL_COUNTER_MISMATCH','SOLVER_CALL_COUNTER_MISMATCH']

def test_normalized_default_is_not_misrepresented_as_model_emitted_argument():
    call={'action':{'action':'solve_itinerary','arguments':{'strategy':'cpsat'},'model_arguments':{}},
        'generation':{'response':{'choices':[{'message':{'tool_calls':[{'function':{'name':'solve_itinerary','arguments':'{}'}}]}}]}}}
    assert raw_policy_arguments(call)=={}
    call['action']['model_arguments']={'strategy':'cpsat'}
    with pytest.raises(AssertionError):raw_policy_arguments(call)
