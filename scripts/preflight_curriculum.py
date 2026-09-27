"""Cloud-only environment checks. Scripted probes are excluded from training."""
import argparse
import asyncio
from collections import Counter
from pathlib import Path

from scripts import evaluate_harness_base as base
from evaluation.curriculum_fixture import build_cases, CurriculumResearchExecutor, quality_audit


class InfrastructureProbe:
    def __init__(self, case):
        self.case = case
        self.actions = []
        def add(name, args=None, retry=False):
            self.actions.append(base.PolicyAction(action=name, arguments=args or {}))
            if retry: self.actions.append(base.PolicyAction(action=name, arguments=args or {}))
        scenario = case['scenario']
        if case['expected'] == 'clarify':
            question = '请问目的地是哪里？' if case['missing_field'] == 'destination' else '请问要玩几天？'
            add('ask_user',{'question':question})
        else:
            add('search_pois',{'keywords':case['slots']['interests']+['餐厅']},scenario=='search-timeout')
            add('get_poi_detail',retry=scenario=='detail-timeout')
            if scenario == 'weather-timeout': add('get_weather',retry=True)
            if scenario in {'stale-hours','wrong-entity'}:
                query = case['slots']['current_info_queries'][0]
                add('search_current_info',{'query':query,'info_type':'opening_hours'})
                add('search_current_info',{'query':query+' 当日入馆时段','info_type':'opening_hours'})
            add('get_route_matrix',retry=scenario=='matrix-timeout')
            for action in ['solve_itinerary','validate_itinerary','finish']: add(action)
        self.cursor = 0

    async def propose(self, context):
        if context.capability.get('status') == 'infeasible':
            reason = '；'.join(context.capability.get('evidence') or [])
            action = 'propose_tradeoff' if context.capability.get('actionable_alternatives') else 'abort'
            return base.PolicyAction(action=action,arguments={'reason':reason,'evidence_ids':context.capability.get('evidence_ids',[])})
        if self.cursor >= len(self.actions):
            raise RuntimeError('Infrastructure probe exhausted; fixture needs inspection')
        action = self.actions[self.cursor]; self.cursor += 1
        return action


async def main(output, size=100):
    output.mkdir(parents=True,exist_ok=False)
    if size==20:
        from evaluation.named_curriculum import build_cases as named_cases
        cases=named_cases()
    elif size==1000:
        from evaluation.scale_curriculum import build_cases as scale_cases
        cases=scale_cases()
    else:cases=build_cases()
    base.dump(output/'cases.json',cases)
    assert len(cases)==size and len({c['request'] for c in cases})==size
    if size!=20:assert Counter(c['family'] for c in cases)==dict(normal=size*4//10,recovery=size*3//10,constraint=size//10*2,clarification=size//10)
    rows=[]; base.settings.agentic_guard_mode='enforce'
    for case in cases:
        target=output/case['id'];target.mkdir()
        backend=CurriculumResearchExecutor(case)
        try:
            with base.frozen_reference_time(base.MOMENT):
                state=base.initial(case)
                recorder=base.EpisodeRecorder(state,environment_version=base.ARCHITECTURE_VERSION,
                    validator_version=base.VALIDATOR_VERSION,policy_name='environment-probe-not-training',policy_version='v1')
                await base.BoundedAgentLoop().run(state,policy=InfrastructureProbe(case),
                    executor=base.TravelActionExecutor(backend),recorder=recorder)
            episode=recorder.episode
            row=base.judge(case,episode,[])
            row.update(reward=base.HierarchicalRewardEngine().score(episode).model_dump(mode='json'),
                       replay_errors=base.EpisodeReplayVerifier().verify(episode))
            row.update(quality_audit(case,row,backend))
            base.dump(target/'episode.json',episode.model_dump(mode='json'))
            base.dump(target/'tool-calls.json',backend.calls)
            row['preflight_passed']=row['quality_candidate']
        except Exception as exc:
            row=dict(case=case['id'],preflight_passed=False,error=type(exc).__name__,message=str(exc))
            import traceback
            (target/'error.txt').write_text(traceback.format_exc())
        base.dump(target/'summary.json',row);rows.append(row)
        print(case['id'],row['preflight_passed'],row.get('rejection_reasons',row.get('message')),flush=True)
    summary=dict(cases_sha256=base.sha(output/'cases.json'),all_passed=all(r['preflight_passed'] for r in rows),
        instances=len(cases),source_groups=len({c['source_group'] for c in cases}),scripted_probes_are_training_data=False,rows=rows)
    base.dump(output/'preflight-summary.json',summary)
    print('PREFLIGHT_RESULT',sum(r['preflight_passed'] for r in rows),len(rows),flush=True)
    return 0 if summary['all_passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,choices=[20,100,1000],default=100)
    args=parser.parse_args()
    raise SystemExit(asyncio.run(main(args.output,args.size)))
