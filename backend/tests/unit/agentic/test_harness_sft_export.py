from types import SimpleNamespace
from scripts.export_harness_sft import recorded_preflight_rejection
from agentic.loop import PolicyAction

def test_only_recorded_unexecuted_research_preflight_can_lack_provider_observation():
    step=SimpleNamespace(action=PolicyAction(action='solve_itinerary'),observations=[],verification={'task_status':'ready','error_code':'RESEARCH_EVIDENCE_INSUFFICIENT'})
    assert recorded_preflight_rejection(step)
    step.action.executed_arguments={'strategy':'auto'}
    assert not recorded_preflight_rejection(step)
    step.action.executed_arguments=None
    step.verification['error_code']='TOOL_TIMEOUT'
    assert not recorded_preflight_rejection(step)
    step.verification={'task_status':'succeeded','error_code':None}
    assert not recorded_preflight_rejection(step)
