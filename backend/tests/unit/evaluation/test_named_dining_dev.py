from copy import deepcopy
import json
import pytest
from evaluation.named_dining_dev import build_cases,VERSION
from evaluation.named_curriculum import build_cases as train_cases
from scripts.collect_harness_batch import validate_case_partition


def test_development_sources_and_provider_entities_are_separate_from_training():
    cases=build_cases();train=train_cases()
    assert len(cases)==12 and len({c['source_group'] for c in cases})==6
    assert not ({c['lineage_group'] for c in cases}&{c['lineage_group'] for c in train})
    assert not ({c['request'] for c in cases}&{c['request'] for c in train})
    assert all(not c['provenance']['training_authorized'] and c['split']=='validation' for c in cases)
    visible=json.dumps([c['pois']+c['restaurants'] for c in cases],ensure_ascii=False)
    assert 'dining-dev' not in visible and 'validation' not in visible and 'grounded_stop' not in visible
    assert all(r['close_time']=='14:00' for c in cases for r in c['restaurants'][:1])


def test_declared_partition_is_enforced_and_test_access_is_unavailable():
    cases=build_cases();assert validate_case_partition(cases,'validation')==VERSION
    for split in ['train','test']:
        with pytest.raises(ValueError):validate_case_partition(cases,split)
    changed=deepcopy(cases);changed[0]['split']='train'
    with pytest.raises(ValueError):validate_case_partition(changed,'validation')
    with pytest.raises(ValueError):validate_case_partition(train_cases(),'validation')
