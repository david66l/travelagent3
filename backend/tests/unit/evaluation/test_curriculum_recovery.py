from copy import deepcopy
import json
import pytest
from evaluation.named_curriculum import build_cases
from evaluation.curriculum_fixture import CurriculumResearchExecutor
from evaluation.curriculum_recovery import provider_snapshot,restore
from agentic.trajectory import _canonical_hash
from scripts import evaluate_harness_base as base


@pytest.mark.asyncio
async def test_restore_preserves_timeout_consumption_and_rejects_changed_environment():
    case=next(c for c in build_cases() if c['scenario']=='search-timeout')
    backend=CurriculumResearchExecutor(case)
    call={'id':'call-1','type':'function','function':{'name':'search_pois','arguments':json.dumps({'city':case['slots']['destination'],'keywords':['历史','餐厅']})}}
    first=await backend.execute([call]);assert not first[0]['observation']['ok']
    state=base.initial(case);state.budget=state.budget.model_copy(update={'used_tool_calls':1})
    snapshot={'case_id':case['id'],'source_group':case['source_group'],'state':state.model_dump(mode='json'),'provider_state':provider_snapshot(backend)}
    snapshot['state_after_hash']=_canonical_hash(snapshot['state'])
    recovered,provider=restore(case,snapshot,backend.calls)
    call['id']='call-2'
    second=await provider.execute([call]);assert second[0]['observation']['ok']
    assert len(provider.fault_events)==1 and provider.provider_attempts['search_pois']==2
    assert recovered.budget.used_tool_calls==1
    changed=deepcopy(case);changed['restaurants'][0]['average_cost']+=1
    with pytest.raises(ValueError,match='environment'):restore(changed,snapshot,backend.calls)
    changed=deepcopy(snapshot);changed['state']['budget']['used_tool_calls']=0
    with pytest.raises(ValueError,match='hash'):restore(case,changed,backend.calls)


def test_named_requests_are_train_only_and_have_explicit_parent_lineage():
    cases=build_cases()
    assert len(cases)==20 and len({c['request'] for c in cases})==20
    assert all(c['split']=='train' and c['slots']['require_named_restaurants'] for c in cases)
    assert all(c['provenance']['environment_changed'] and not c['provenance']['old_rollout_reused'] for c in cases)
