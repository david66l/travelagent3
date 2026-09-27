"""Scripted infrastructure reachability only; NEVER teacher/student scores."""
import argparse,asyncio,json
from pathlib import Path
from scripts.evaluate_harness_base import (load_cases,initial,FrozenResearchExecutor,MOMENT,
    frozen_reference_time,TravelActionExecutor,BoundedAgentLoop,PolicyAction,
    EpisodeRecorder,EpisodeReplayVerifier,judge,dump)


class InfrastructurePolicy:
    def __init__(self,case):
        self.case=case; actions=[]
        if case['expected']=='clarify':
            question={'destination':'您想去哪个城市？','travel_days':'您计划旅行几天？','origin':'您从哪个城市出发？'}[case['missing_field']]
            self.actions=iter([PolicyAction(action='ask_user',arguments={'question':question})]);return
        for query in case['slots'].get('current_info_queries',[]):
            actions.append(PolicyAction(action='search_current_info',arguments={'query':query,'info_type':'opening_hours'}))
        if case['family']=='evidence':
            actions.append(PolicyAction(action='search_current_info',arguments={
                'query':' '.join(case['slots']['must_visit'])+' 最新官方开放公告','info_type':'opening_hours'}))
        if 'weather' in case['slots'].get('information_needs',[]):actions.append(PolicyAction(action='get_weather'))
        actions += [PolicyAction(action='search_pois'),PolicyAction(action='get_poi_detail')]
        if case['family']=='recovery':actions.append(PolicyAction(action='get_poi_detail'))
        if case['expected']=='plan':
            actions.append(PolicyAction(action='get_route_matrix'))
            if case['family']=='recovery' and case['variant']==2:actions.append(PolicyAction(action='get_route_matrix'))
            actions += [PolicyAction(action=n) for n in ['solve_itinerary','validate_itinerary','finish']]
        else:actions.append(PolicyAction(action='abort'))
        self.actions=iter(actions)
    async def propose(self,context):
        action=next(self.actions)
        if action.action=='abort':action.arguments={'reason':'当前事实证明无法满足全部要求。','evidence_ids':context.capability.get('evidence_ids',[])}
        return action


async def main(args):
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    for case in load_cases(args.cases_file):
        with frozen_reference_time(MOMENT):
            state=initial(case);backend=FrozenResearchExecutor(case)
            recorder=EpisodeRecorder(state,environment_version='infrastructure-audit',validator_version='travel-validator.v2',policy_name='scripted-infrastructure-only',policy_version='v1')
            await BoundedAgentLoop().run(state,policy=InfrastructurePolicy(case),executor=TravelActionExecutor(backend),recorder=recorder)
            errors=EpisodeReplayVerifier().verify(recorder.episode);assert not errors,errors
            row=judge(case,recorder.episode,[]);rows.append(row)
            dump(args.output/(case['id']+'.json'),recorder.episode.model_dump(mode='json'))
            print(case['id'],row['criterion_passed'],row['termination_reason'],row['failures'],flush=True)
    dump(args.output/'summary.json',rows)
    assert all(r['criterion_passed'] for r in rows),'Infrastructure reachability failed; inspect retained traces'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
