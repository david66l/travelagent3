"""Audit paired dining96 teacher evidence; accepted labels remain an isolated candidate release."""
import argparse,json,hashlib
from pathlib import Path
from copy import deepcopy
from collections import Counter
from types import SimpleNamespace
from agentic.trajectory import AgentEpisode,EpisodeReplayVerifier,_canonical_hash
from agentic.episode_accounting import verify_episode_accounting
from agentic.policy_prompts import policy_prompt_payload,AGENT_TOOL_POLICY_SYSTEM_PROMPT
from core.glm_tool_client import SINGLE_CALL_REMINDER
from agentic.loop import NO_TOOL_ACTIONS
from scripts.export_harness_sft import recorded_preflight_rejection
from agentic.sft_dataset import SFTDatasetBuilder,EpisodeCandidate
from evaluation.curriculum_fixture import quality_audit
from evaluation.curriculum_recovery import restore
from evaluation.research_fixture import catalog
from evaluation.validator import ItineraryValidator
from scripts.evaluate_harness_base import judge,HierarchicalRewardEngine,sha,dump,AgentLedgerState,TravelActionExecutor


def accounting_errors(ep, records):
    errors=verify_episode_accounting(ep)
    if len(records)!=ep.final_state['budget']['used_tool_calls']:errors.append('TOOL_CALL_COUNTER_MISMATCH')
    if sum(r['call']['function']['name']=='solve_itinerary' for r in records)!=ep.final_state['budget']['used_solver_calls']:
        errors.append('SOLVER_CALL_COUNTER_MISMATCH')
    return errors


def raw_policy_arguments(call):
    action=call['action']
    arguments=action.get('model_arguments')
    if arguments is None:arguments=action['arguments']
    native=call['generation']['response']['choices'][0]['message']['tool_calls']
    assert len(native)==1 and native[0]['function']['name']==action['action']
    emitted=native[0]['function']['arguments']
    if isinstance(emitted,str):emitted=json.loads(emitted)
    assert emitted==arguments,'Recorded model arguments differ from actual provider response'
    return arguments


def executed_decision_signature(step):
    """Same empty policy arguments can hydrate to different source entities."""
    return _canonical_hash({'goal_version':step.context.goal_version,'plan_version':step.context.plan_version,
        'action':step.action.action,'arguments':step.action.model_arguments if step.action.model_arguments is not None else step.action.arguments,
        'executed_arguments':step.action.executed_arguments})


def paired_structural_errors(candidate, errors, max_exact_retries=1):
    steps=[s for s in candidate.episode.steps if s.action.decision_source!='controller' and s.action.action not in NO_TOOL_ACTIONS]
    missing=[s for s in steps if not s.observations]
    if missing and all(recorded_preflight_rejection(s) for s in missing):
        errors=[e for e in errors if e!='L2_TOOL_OBSERVATION_MISSING']
    # Preflight refusals never invoked a provider. Count actual attempts, and
    # keep requests with different hydrated source inputs distinct.
    attempts=Counter(executed_decision_signature(s) for s in steps if not recorded_preflight_rejection(s))
    if not any(n>max_exact_retries+1 for n in attempts.values()):
        errors=[e for e in errors if e!='L2_EXCESSIVE_IDENTICAL_RETRIES']
    return errors


class PairedBuilder(SFTDatasetBuilder):
    def __init__(self,case_index,audits):super().__init__();self.case_index=case_index;self.audits=audits
    def _split_group(self,candidate):return self.case_index[candidate.scenario_id]['lineage_group']
    def _split_for_group(self,group):return 'train'
    def _review(self,candidate):
        errors=paired_structural_errors(candidate,super()._review(candidate),self.max_exact_retries)
        if not self.audits[candidate.scenario_id]['quality_candidate']:errors.append('PAIRED_INDEPENDENT_QUALITY_REJECTED')
        return errors
    @staticmethod
    def _examples(candidate,split,quality_label):
        return [ex for ex in SFTDatasetBuilder._examples(candidate,split,quality_label,signature_fn=executed_decision_signature)
                if candidate.episode.steps[ex.step_index].action.repair_attempts==0]


def main(args):
    run=args.teacher_run;out=args.output;out.mkdir(exist_ok=False)
    cases=json.loads((run/'cases.json').read_text());by_case={c['id']:c for c in cases}
    jobs=json.loads((run/'jobs.json').read_text());assert all(j['status']=='complete' for j in jobs.values())
    branches=json.loads((run/'branches.json').read_text())
    ledger=json.loads(args.budget_ledger.read_text());before=json.loads((run/'budget-before.json').read_text())
    audits={};case_index={};candidates=[];model_records={};budget_ids=[];steps=0;lineage_checks=0;restaurant_routes=0
    for name,job in jobs.items():
        case_id,mode,effort=name.split('/');case=by_case[case_id];case_index[name]=case;target=run/name
        for file,digest in job['evidence_sha256'].items():assert sha(target/file)==digest
        ep=AgentEpisode(**json.loads((target/'episode.json').read_text()));assert not EpisodeReplayVerifier().verify(ep)
        stored=json.loads((target/'summary.json').read_text());provenance=json.loads((target/'provenance.json').read_text())
        records=json.loads((target/'tool-calls.json').read_text())
        counter_errors=accounting_errors(ep,records)
        if mode=='correction':
            branch=branches[case_id];snapshot=json.loads(Path(branch['snapshot_path']).read_text());source=Path(branch['student_evidence'])
            assert sha(source/'episode.json')==branch['source_episode_sha256'] and sha(Path(branch['snapshot_path']))==branch['snapshot_sha256']
            original_records=json.loads((source/'tool-calls.json').read_text());state,provider=restore(case,snapshot,original_records)
            actual=deepcopy(ep.initial_state);actual['budget']['timeout_ms']=state.budget.timeout_ms
            assert _canonical_hash(actual)==snapshot['state_after_hash']==provenance['source_state_hash']
            prefix=provenance['prefix_tool_calls'];assert records[:prefix]==original_records[:prefix]
            assert provenance['provider_initial_state']==snapshot['provider_state'];lineage_checks+=1
        calls=[json.loads(l) for l in (target/'model-calls.jsonl').read_text().splitlines()]
        byid={c['action']['action_id']:c for c in calls if c.get('action')};model_records[name]=byid
        row=judge(case,ep,calls);assert row['criterion_passed']==stored['criterion_passed']
        row.update(reward=HierarchicalRewardEngine().score(ep).model_dump(mode='json'),replay_errors=[],
            parse_or_schema_errors=sum((c.get('error') or {}).get('type')=='PolicyOutputError' for c in calls))
        row.update(quality_audit(case,row,SimpleNamespace(fault_events=stored['fault_events'])))
        row['accounting_errors']=counter_errors
        row['rejection_reasons'].extend(counter_errors)
        for call in calls:
            generation=call.get('generation') or {};request=generation.get('request') or {}
            if request:
                assert request['reasoning_effort']==effort
                assert len(request['messages'])==3
                assert request['messages'][0]=={'role':'system','content':AGENT_TOOL_POLICY_SYSTEM_PROMPT}
                assert request['messages'][1]['role']=='user'
                assert request['messages'][2]=={'role':'user','content':SINGLE_CALL_REMINDER}
                assert hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==generation['request_sha256']
                assert json.loads(request['messages'][1]['content'])==call['state']
                visible=json.dumps(request,ensure_ascii=False)
                assert not any(t in visible for t in ['dining96-','dining-recovery-composition:','grounded_stop','training_authorized','named20-','scale1k-','scale-composition:','stale-hours','wrong-entity','hard-budget','budget-tradeoff','required-closed','optional-closed'])
            request_id=generation.get('budget_request_id')
            if request_id is None:
                assert not request and not generation.get('response')
                assert generation.get('queue_stopped_before_reservation') or generation.get('budget_denied_before_request')
            else:
                assert request_id in ledger['requests']
                assert ledger['requests'][request_id]['task']==f'dining96pilot8/{name}';budget_ids.append(request_id)
        if any(c.get('error') and c['error']['type']!='PolicyOutputError' for c in calls):row['rejection_reasons'].append('TEACHER_PROVIDER_ERROR')
        source_reports=[];previous_provider=provenance['provider_initial_state'];previous_budget=ep.initial_state['budget']
        for step in ep.steps:
            snapshot=json.loads((target/f'state-after-{step.step_index:03d}.json').read_text())
            assert _canonical_hash(snapshot['state'])==snapshot['state_after_hash']==step.state_after_hash
            if recorded_preflight_rejection(step):
                assert snapshot['provider_state']==previous_provider
                assert all(snapshot['state']['budget'][k]==previous_budget[k] for k in ['used_tool_calls','used_solver_calls'])
            previous_provider=snapshot['provider_state'];previous_budget=snapshot['state']['budget']
            if step.action.decision_source=='controller':
                steps+=1
                continue
            raw=byid[step.action.action_id]
            assert raw_policy_arguments(raw)==(step.action.model_arguments if step.action.model_arguments is not None else step.action.arguments)
            expected=policy_prompt_payload(step.context)
            if raw['state']!=expected:
                assert {k for k in set(raw['state'])|set(expected) if raw['state'].get(k)!=expected.get(k)}=={'policy_feedback'} and step.action.repair_attempts>0
            if step.action.action=='solve_itinerary':
                args_used=step.action.executed_arguments or {}
                for observation in step.observations:
                    if observation.ok and isinstance(observation.data,dict) and observation.data.get('days'):
                        source_reports.append(ItineraryValidator().validate(observation.data['days'],args_used['constraints'],catalog(case)).model_dump(mode='json'))
            steps+=1
        if case['expected']=='plan':
            if not source_reports:
                state=AgentLedgerState(**ep.final_state)
                solver=TravelActionExecutor._latest_artifact(state,'solver_result')
                if solver and solver.payload.get('days'):
                    source_reports.append(ItineraryValidator().validate(solver.payload['days'],TravelActionExecutor._trusted_constraints(state),catalog(case)).model_dump(mode='json'))
            if not source_reports or not source_reports[-1]['hard_pass']:row['rejection_reasons'].append('ORIGINAL_CATALOG_CONSTRAINT_CHECK_FAILED')
            else:restaurant_routes+=1
        row.update(quality_candidate=not row['rejection_reasons'],mode=mode,reasoning_effort=effort,source_catalog_validation=source_reports,episode_sha256=sha(target/'episode.json'))
        audits[name]=row
        candidates.append(EpisodeCandidate(scenario_id=name,source='teacher',template_family=case['lineage_group'],city=case['slots']['destination'],episode=ep))
    assert len(set(budget_ids))==len(budget_ids)
    added=set(ledger['requests'])-set(before['requests']);assert added==set(budget_ids)
    assert not any(r['status']=='reserved' for r in ledger['requests'].values())
    builder=PairedBuilder(case_index,audits);dataset=builder.build(candidates)
    lookup={c.scenario_id:c for c in candidates}
    for ex in dataset.examples:
        step=lookup[ex.scenario_id].episode.steps[ex.step_index];raw=model_records[ex.scenario_id][step.action.action_id]
        assert json.loads(ex.messages[1].content)==raw['state']
        assert ex.messages[-1].tool_calls[0].function.arguments==raw_policy_arguments(raw)
        request=raw['generation']['request']
        assert ex.messages[0].content==request['messages'][0]['content'] and ex.tools==request['tools']
        assert ex.split=='train' and len(ex.messages)==3 and ex.messages[-1].content is None
    from agentic.sft_dataset import _step_verified_success
    duplicate_steps=0
    for candidate in candidates:
        signatures=[executed_decision_signature(s) for s in candidate.episode.steps
            if s.action.decision_source!='controller' and s.action.action not in NO_TOOL_ACTIONS and _step_verified_success(s)]
        duplicate_steps+=len(signatures)-len(set(signatures))
    dataset.manifest.excluded_duplicate_policy_steps=duplicate_steps
    assert all(not recorded_preflight_rejection(lookup[ex.scenario_id].episode.steps[ex.step_index]) for ex in dataset.examples)
    builder.export(dataset,out/'sft')
    # Keep the actual exported schema, with explicit task/source provenance beside it.
    modes={}
    for mode in ['demo/max','correction/max']:
        group=[r for k,r in audits.items() if k.endswith('/'+mode)]
        examples=[ex for ex in dataset.examples if ex.scenario_id.endswith('/'+mode)]
        modes[mode]={'trajectories':len(group),'task_passed':sum(r['criterion_passed'] for r in group),'quality_candidates':sum(r['quality_candidate'] for r in group),
            'exported_actions':len(examples),'accepted_episodes':sum(r.accepted for r in dataset.reviews if r.scenario_id.endswith('/'+mode))}
        with (out/'sft'/f"{mode.replace('/','-')}-train.jsonl").open('w') as f:
            for ex in examples:f.write(ex.model_dump_json()+'\n')
    cost=sum(r['charged_micros'] for r in ledger['requests'].values())/1e6
    summary={'schema_version':'dining96-paired-teacher-audit.v1','training_release_authorized':False,'modes':modes,'states_verified':steps,'student_checkpoint_lineages_verified':lineage_checks,
        'structural_review_contract':'recorded-unexecuted-preflight-excluded; executed-input-aware-retry-and-dedup.v1',
        'auditor_sha256':sha(Path(__file__)),'source_teacher_manifest_sha256':sha(run/'manifest.json'),
        'named_restaurant_routes_verified':restaurant_routes,'new_api_calls':len(added),'new_conservative_cost_cny':sum(ledger['requests'][k]['charged_micros'] for k in added)/1e6,
        'program_conservative_cost_cny':cost,'remaining_cny':100-cost,'new_unknown_usage_calls':sum(ledger['requests'][k]['status']!='settled' for k in added),
        'exported_actions':len(dataset.examples),'student_prefix_targets':0,'private_reasoning_targets':0,'new_training_runs':0,
        'teacher_only_protocol_reminder':'Exact fixed one-native-call reminder audited; omitted from student prompt; no task facts or next-action hints',
        'rejection_reasons':dict(Counter(x for r in audits.values() for x in r['rejection_reasons'])),
        'label_mask_audit_pending':True,'limits':['synthetic fixtures; not live facts','eight train compositions selected by scripted checkpoint feasibility before teacher results; not independent validation','intent slots predeclared; no multi-turn user response evaluation']}
    dump(out/'summary.json',summary);dump(out/'case-audits.json',audits);dump(out/'provenance.json',{k:{'source_group':v['source_group'],'lineage_group':v['lineage_group']} for k,v in case_index.items()})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['teacher-run','output','budget-ledger']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
