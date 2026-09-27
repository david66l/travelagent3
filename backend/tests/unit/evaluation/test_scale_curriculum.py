from collections import Counter
import json
from evaluation.scale_curriculum import build_cases
from evaluation.research_fixture import catalog


def test_scale_unique_requests_lineage_and_opaque_provider_records():
    cases=build_cases()
    assert len(cases)==len({c['id'] for c in cases})==len({c['request'] for c in cases})==1000
    assert len({c['source_group'] for c in cases})==40
    assert len({c['lineage_group'] for c in cases})==20
    assert Counter(c['family'] for c in cases)==dict(normal=400,recovery=300,constraint=200,clarification=100)
    for case in cases:
        public=json.dumps(catalog(case),ensure_ascii=False)
        assert 'scale1k-' not in public and 'b01-' not in public
        assert not any(s in public for s in ['wrong-entity','stale-hours','hard-budget','required-closed'])
        assert not case['training_targets_present'] and case['split']=='train'


def test_pair_cost_conflict_is_fact_grounded_not_a_fake_solver_answer():
    cases=[c for c in build_cases() if c['scenario']=='hard-budget' and c['provenance']['composition_constraints_added']]
    assert len(cases)==25
    for case in cases:
        selected=[p for p in case['pois'] if p['name'] in case['slots']['must_visit']]
        assert len(selected)==2
        assert max(p['ticket_price'] for p in selected)<case['slots']['budget_range']
        assert sum(p['ticket_price'] for p in selected)>case['slots']['budget_range']
        assert all(p['name'] in case['request'] for p in selected)
