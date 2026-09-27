import asyncio
from copy import deepcopy
import pytest
from evaluation.curriculum_fixture import build_cases, CurriculumResearchExecutor, quality_audit
from evaluation import research_fixture


def test_preference_is_not_copied_into_provider_tags():
    original=build_cases()[0]; changed=deepcopy(original)
    changed['slots']['interests']=['不存在的兴趣']
    assert research_fixture.catalog(original)==research_fixture.catalog(changed)
    assert research_fixture.search_pois(changed,{'keywords':['不存在的兴趣']})==[]
    assert len({tuple(p['tags']) for p in original['pois']})==6


def test_grouped_train_variants_and_real_missing_fields():
    cases=build_cases()
    assert len(cases)==len({c['request'] for c in cases})==100
    assert len({c['source_group'] for c in cases})==20
    assert all(c['split']=='train' and not c['training_targets_present'] for c in cases)
    for case in cases:
        if case['missing_field']:
            assert case['missing_field'] not in case['slots']


def test_timeout_occurs_once_regardless_of_search_wording():
    case=next(c for c in build_cases() if c['scenario']=='search-timeout')
    backend=CurriculumResearchExecutor(case)
    async def run():
        with pytest.raises(TimeoutError): await backend._handlers['search_pois']({'keywords':['历史']})
        result=await backend._handlers['search_pois']({'keywords':['历史博物馆']})
        assert result.data and len(backend.fault_events)==1
        assert backend.provider_attempts['search_pois']==2
    asyncio.run(run())


def test_unexercised_recovery_cannot_become_quality_candidate():
    case=next(c for c in build_cases() if c['family']=='recovery')
    row=dict(criterion_passed=True,reward={'gate_status':'passed'},validation={'soft_scores':{'preference_match':1}},actions=['finish'])
    audit=quality_audit(case,row,CurriculumResearchExecutor(case))
    assert not audit['quality_candidate']
    assert 'CONFIGURED_FAULT_NOT_EXERCISED' in audit['rejection_reasons']


def test_provider_identifiers_do_not_disclose_hidden_scenario():
    import json
    cases=build_cases()
    private_tokens={c['scenario'] for c in cases}|{'grounded_stop','missing_field','provider_faults'}
    for case in cases:
        public=json.dumps(research_fixture.catalog(case),ensure_ascii=False)
        assert case['id'] not in public
        assert not any(token in public for token in private_tokens)
