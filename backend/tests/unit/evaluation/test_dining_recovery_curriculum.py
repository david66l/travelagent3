import json
from collections import Counter
from evaluation.dining_recovery_curriculum import build_cases,VERSION
from evaluation.named_dining_dev import build_cases as dev_cases
from evaluation.named_curriculum import build_cases as old_cases
from scripts.collect_harness_batch import validate_case_partition
import pytest

def test_new_catalogs_have_declared_correlation_and_no_dev_source_entities():
    cases=build_cases();other=dev_cases()+old_cases()
    assert len(cases)==96 and len({c['source_group'] for c in cases})==8
    assert Counter(c['family'] for c in cases)=={'normal':64,'recovery':16,'constraint':16}
    assert Counter(c['expected'] for c in cases)=={'plan':80,'grounded_stop':16}
    for field in ['request','lineage_group']:
        assert not ({c[field] for c in cases}&{c[field] for c in other})
    for field in ['id','name','source_url']:
        assert not ({v[field] for c in cases for v in c['pois']+c['restaurants']}&{v[field] for c in other for v in c['pois']+c['restaurants']})
    public=json.dumps([c['pois']+c['restaurants'] for c in cases])
    assert 'dining96' not in public and 'timeout' not in public and 'composition' not in public
    assert validate_case_partition(cases,'train')==VERSION
    with pytest.raises(ValueError):validate_case_partition(cases,'validation')
