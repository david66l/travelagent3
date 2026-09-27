"""Export explicitly split, rejudged teacher episodes through the existing builder."""
from pathlib import Path
import argparse,json,hashlib
from collections import Counter
from agentic.sft_dataset import SFTDatasetBuilder,EpisodeCandidate
from agentic.trajectory import AgentEpisode,EpisodeReplayVerifier
from agentic.reward import HierarchicalRewardEngine
from agentic.loop import NO_TOOL_ACTIONS
from scripts.evaluate_harness_base import judge,JUDGING_VERSION
from evaluation.validator import ItineraryValidator
from vrp_solver_service.models import POIInput,ConstraintsInput
from planner.preprocessing.play_time_manager import PlayTimeManager

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def recorded_preflight_rejection(step):
    return (step.action.action in {'solve_itinerary','validate_itinerary'}
            and step.action.executed_arguments is None
            and not step.observations
            and step.verification.get('task_status')=='ready'
            and step.verification.get('error_code')=='RESEARCH_EVIDENCE_INSUFFICIENT')

def audit_episode(root,case):
    raw=json.loads((root/case['id']/'episode.json').read_text(encoding='utf-8'))
    assert not EpisodeReplayVerifier().verify(raw),case['id']
    ep=AgentEpisode(**raw)
    calls=[json.loads(line) for line in (root/case['id']/'model-calls.jsonl').read_text(encoding='utf-8').splitlines()]
    row=judge(case,ep,calls)
    row['reward']=HierarchicalRewardEngine().score(ep).model_dump(mode='json')
    row['provider_failures']=sum(bool(c.get('generation',{}).get('provider_error')) for c in calls)
    row['independent_plans']=[]
    for step in raw['steps']:
        if step['action']['action']!='solve_itinerary':continue
        args=step['action'].get('executed_arguments') or {}
        for obs in step['observations']:
            if obs['ok'] and isinstance(obs.get('data'),dict) and obs['data'].get('days'):
                facts=[p.model_dump(mode='json') for p in PlayTimeManager().adjust([POIInput(**p) for p in args['pois']],ConstraintsInput(**args['constraints']))]
                report=ItineraryValidator().validate(obs['data']['days'],constraints=args['constraints'],facts=facts)
                row['independent_plans'].append(report.hard_pass)
    if row['submitted_valid_plan']:assert any(row['independent_plans']),case['id']
    records=json.loads((root/case['id']/'tool-calls.json').read_text(encoding='utf-8'))
    budget=raw['final_state']['budget']
    assert len(records)==budget['used_tool_calls']
    assert sum(r['call']['function']['name']=='solve_itinerary' for r in records)==budget['used_solver_calls']
    row['budget_verified']=True
    row['episode_sha256']=sha(root/case['id']/'episode.json')
    return ep,row

class ExplicitSplitBuilder(SFTDatasetBuilder):
    def __init__(self,case_index,audits):
        super().__init__(max_steps=24,max_exact_retries=1)
        self.cases=case_index;self.audits=audits
        self.groups={c['source_group']:c['split'] for c in case_index.values()}
        assert len({(c['source_group'],c['split']) for c in case_index.values()})==len(self.groups)
    def _split_group(self,candidate):return self.cases[candidate.scenario_id]['source_group']
    def _split_for_group(self,group):return self.groups[group]
    def _review(self,candidate):
        errors=super()._review(candidate);audit=self.audits[candidate.scenario_id]
        # A recorded preflight refusal never called a provider, so it should
        # have no provider observation. Preserve it in the source trajectory,
        # exclude it from SFT targets, and retain later verified recovery steps.
        missing=[s for s in candidate.episode.steps if s.action.action not in NO_TOOL_ACTIONS and not s.observations]
        if missing and all(recorded_preflight_rejection(s) for s in missing):
            errors=[e for e in errors if e!='L2_TOOL_OBSERVATION_MISSING']
        if not audit['criterion_passed']:errors.append('L3_INDEPENDENT_TASK_CRITERION_FAILED')
        if audit['reward']['gate_status']!='passed':errors.append('L3_TERMINAL_REWARD_GATE_FAILED')
        if audit['provider_failures']:errors.append('L1_PROVIDER_FAILURE')
        return errors

def main(args):
    candidates=[];audits={};cases={};sources={}
    for split,root in [('train',args.train_run),('validation',args.dev_run)]:
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['harness_revision']=='entity-evidence-v6' and manifest['judging_version']==JUDGING_VERSION
        sources[split]={'path':str(root),'manifest_sha256':sha(root/'manifest.json'),'cases_sha256':sha(root/'cases.json')}
        for case in json.loads((root/'cases.json').read_text(encoding='utf-8')):
            assert case['split']==split
            if not (root/case['id']/'summary.json').exists():raise RuntimeError('Incomplete teacher collection: '+case['id'])
            ep,audit=audit_episode(root,case);audits[case['id']]=audit;cases[case['id']]=case
            candidates.append(EpisodeCandidate(scenario_id=case['id'],source='teacher',template_family=case['source_group'],city=case['slots'].get('destination','unspecified'),episode=ep))
    builder=ExplicitSplitBuilder(cases,audits);result=builder.build(candidates)
    targets={(e.trajectory_id,e.step_index) for e in result.examples}
    for candidate in candidates:
        for step in candidate.episode.steps:
            if recorded_preflight_rejection(step):assert (candidate.episode.trajectory_id,step.step_index) not in targets
    assert not args.output.exists(), 'Use an immutable new export directory'
    builder.export(result,args.output)
    counts=Counter(e.messages[-1].tool_calls[0].function.name for e in result.examples if e.split=='train')
    audit={'judging_version':JUDGING_VERSION,'sources':sources,'case_reviews':audits,'train_actions':dict(counts),'test_examples':0,'test_rollouts_used':False,'target_format':'one model-owned native action from observed state; no private teacher reasoning','source_group_split':'explicit composition groups; not random row/city splitting','limits':['synthetic pilot only','no named restaurant routing','shared skills across distinct compositions']}
    (args.output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result.manifest.model_dump_json(indent=2));print(json.dumps(dict(counts)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train-run',type=Path,required=True);p.add_argument('--dev-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
