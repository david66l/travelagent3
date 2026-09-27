"""A bounded paired-collection pilot rebuilt from the frozen scale corpus."""
from copy import deepcopy
from evaluation.scale_curriculum import build_cases as scale_cases
from evaluation.migration_fixture import fingerprint

VERSION='travel-named-dining20.v1'


def build_cases():
    source=scale_cases()
    # Eighteen non-clarification parent groups plus two compound constraints.
    selected=[c for c in source if c['variant']==0 and c['composition']%2==0 and c['expected']!='clarify']
    selected += [c for c in source if c['variant']==1 and c['composition']%2==1 and c['scenario'] in {'history','hard-budget'}]
    assert len(selected)==20
    result=[]
    for index,original in enumerate(selected):
        case=deepcopy(original)
        case.update(id=f'named20-{index:02d}',dataset_version=VERSION)
        case['slots']['require_named_restaurants']=True
        case['request']+='午晚餐都要安排具名餐厅，并核实营业时段、人均餐费和往返通勤。'
        case['provenance'].update(parent_instance_id=original['id'],parent_case_sha256=original['source_case_sha256'],
            environment_changed=True,old_rollout_reused=False,named_dining_required=True)
        case.pop('source_case_sha256');case['source_case_sha256']=fingerprint(case)
        result.append(case)
    return result
