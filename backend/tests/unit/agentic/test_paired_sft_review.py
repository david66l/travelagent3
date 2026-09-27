from types import SimpleNamespace as NS

from agentic.loop import PolicyAction, PolicyContext
from agentic.sft_dataset import SFTDatasetBuilder
from scripts.audit_paired_teacher import paired_structural_errors, executed_decision_signature


def step(index=0, *, executed=None, error=None, observations=True):
    context=PolicyContext(trajectory_id='t',goal_version=1,plan_version=1,
        original_request='规划一天旅行',current_subtask={'task_id':'research'},
        hard_constraints={},soft_preferences={},relevant_fact_refs=[],relevant_artifact_refs=[],
        relevant_facts=[],relevant_artifacts=[],failure_summary=[],remaining_tasks=1,remaining_steps=8,
        allowed_actions=['solve_itinerary'])
    return NS(step_index=index,context=context,
        action=PolicyAction(action='solve_itinerary',arguments={},executed_arguments=executed),
        verification={'task_status':'ready','error_code':error},
        observations=[NS(ok=True)] if observations else [])


def candidate(steps):
    return NS(scenario_id='case',source='teacher',episode=NS(steps=steps,trajectory_id='t',content_hash='abc',
        environment_version='env',policy_name='teacher',policy_version='v1'))


def test_unexecuted_preflight_is_not_a_provider_retry_or_training_target():
    c=candidate([step(0,error='RESEARCH_EVIDENCE_INSUFFICIENT',observations=False),
        step(1,error='RESEARCH_EVIDENCE_INSUFFICIENT',observations=False),step(2,executed={'pois':['a']})])
    errors=paired_structural_errors(c,['L2_TOOL_OBSERVATION_MISSING','L2_EXCESSIVE_IDENTICAL_RETRIES','L1_PII_DETECTED'])
    assert errors==['L1_PII_DETECTED']
    examples=SFTDatasetBuilder._examples(c,'train','validated_plan',signature_fn=executed_decision_signature)
    assert [e.step_index for e in examples]==[2]


def test_missing_observation_after_actual_execution_remains_rejected():
    c=candidate([step(executed={},error='RESEARCH_EVIDENCE_INSUFFICIENT',observations=False)])
    assert 'L2_TOOL_OBSERVATION_MISSING' in paired_structural_errors(c,['L2_TOOL_OBSERVATION_MISSING'])


def test_unrecorded_missing_observation_remains_rejected():
    c=candidate([step(error='TOOL_TIMEOUT',observations=False)])
    assert 'L2_TOOL_OBSERVATION_MISSING' in paired_structural_errors(c,['L2_TOOL_OBSERVATION_MISSING'])


def test_repeated_actual_identical_requests_still_rejected_and_deduplicated():
    c=candidate([step(i,executed={'pois':['a']}) for i in range(3)])
    assert 'L2_EXCESSIVE_IDENTICAL_RETRIES' in paired_structural_errors(c,['L2_EXCESSIVE_IDENTICAL_RETRIES'])
    assert len(SFTDatasetBuilder._examples(c,'train','validated_plan',signature_fn=executed_decision_signature))==1


def test_distinct_hydrated_inputs_preserve_verified_decisions_without_changing_legacy_default():
    c=candidate([step(i,executed={'pois':[str(i)]}) for i in range(3)])
    assert paired_structural_errors(c,['L2_EXCESSIVE_IDENTICAL_RETRIES'])==[]
    assert len(SFTDatasetBuilder._examples(c,'train','validated_plan',signature_fn=executed_decision_signature))==3
    assert len(SFTDatasetBuilder._examples(c,'train','validated_plan'))==1
