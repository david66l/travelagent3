"""Restore a frozen provider and student ledger without replaying tool calls."""
from copy import deepcopy
from collections import Counter
from agentic.state import AgentLedgerState
from agentic.trajectory import _canonical_hash
from evaluation.curriculum_fixture import CurriculumResearchExecutor
from evaluation.migration_fixture import fingerprint


def provider_snapshot(backend):
    return dict(schema_version='curriculum-provider-state.v1',counts=dict(backend.counts),
        seen_fault_indices=sorted(backend._seen_faults),provider_attempts=deepcopy(backend.provider_attempts),
        fault_events=deepcopy(backend.fault_events),first_info_query=backend.first_info_query,
        tool_call_count=len(backend.calls),case_sha256=fingerprint(backend.case))


def restore(case,snapshot,records):
    if case['split']!='train' or snapshot['case_id']!=case['id'] or snapshot['source_group']!=case['source_group']:
        raise ValueError('Checkpoint task/source/split mismatch')
    state=AgentLedgerState(**deepcopy(snapshot['state']))
    if _canonical_hash(snapshot['state'])!=snapshot['state_after_hash']:
        raise ValueError('Checkpoint state hash changed')
    provider=snapshot['provider_state']
    if provider.get('schema_version')!='curriculum-provider-state.v1' or provider['case_sha256']!=fingerprint(case):
        raise ValueError('Provider environment differs from student collection')
    count=provider['tool_call_count']
    if count!=state.budget.used_tool_calls or not 0<=count<=len(records):raise ValueError('Tool counter mismatch')
    backend=CurriculumResearchExecutor(case)
    backend.counts=Counter(provider['counts']);backend._seen_faults=set(provider['seen_fault_indices'])
    backend.provider_attempts=deepcopy(provider['provider_attempts']);backend.fault_events=deepcopy(provider['fault_events'])
    backend.first_info_query=provider['first_info_query'];backend.calls=deepcopy(records[:count])
    if sum(r['call']['function']['name']=='solve_itinerary' for r in backend.calls)!=state.budget.used_solver_calls:
        raise ValueError('Solver counter mismatch')
    assert provider_snapshot(backend)==provider
    return state,backend
