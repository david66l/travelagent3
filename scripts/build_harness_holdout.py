"""Freeze pilot dev/test task compositions before SFT; never use test for targets."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
from scripts.build_harness_training_pilot import build

ROOT=Path(__file__).resolve().parents[1]
VERSION='harness-holdout16.v2'

def build_holdout():
    rows=[]
    for split,cities in [('validation',[('杭州',30.27,120.15),('成都',30.67,104.06)]),
                         ('test',[('武汉',30.59,114.30),('青岛',36.07,120.38)])]:
        for group in range(4):
            for variant,(city,lat,lng) in enumerate(cities):
                case=deepcopy(build()[0]);ident=f'{split}-{group+1:02d}-{variant+1:02d}'
                day=f'2026-11-{5+group*4+variant:02d}' if split=='validation' else f'2026-12-{6+group*4+variant:02d}'
                names=[f'{city}澄明{group+1}号科技体验中心',f'{city}竹影{group+1}号植物园',f'{city}星河{group+1}号艺术空间',f'{city}临风{group+1}号广场']
                for i,p in enumerate(case['pois']):
                    p.update(id=f'{ident}-p{i}',name=names[i],lat=lat+.002*i,lng=lng+.002*i,source_url=f'https://holdout.example.invalid/{ident}/p{i}',tags=['科技','植物','美术','散步'][i:i+1])
                case.update(id=ident,split=split,dataset_version=VERSION,restaurants=[],family='holdout',variant=0)
                hard=case['slots'];hard.update(destination=city,start_date=day,end_date=day,must_visit=names[:2],interests=['科技','植物'])
                hard.pop('current_info_queries',None);hard.pop('information_needs',None)
                request=f'{day}在{city}玩一天，总预算1000元，必须去{names[0]}和{names[1]}。'
                if split=='validation':
                    scene=['two_hours_plus_weather','optional_closed_plus_detail_fault','three_ticket_joint_floor','missing_destination_with_interests'][group]
                    if group==0:
                        hard.update(information_needs=['opening_hours','weather'],current_info_queries=[f'{n} {day} 开放时间' for n in names[:2]])
                        request+='请分别核实两个地方的当天开放时间和天气。'
                    elif group==1:
                        case.update(family='recovery',variant=1);case['pois'][2]['closed_dates']=[day]
                        request+=f'也对{names[2]}感兴趣，如果闭馆可以省略。'
                    elif group==2:
                        hard.update(must_visit=names[:3],budget_range=250)
                        for p in case['pois'][:3]:p['ticket_price']=90
                        case['expected']='grounded_stop'
                        request=f'{day}在{city}玩一天，预算严格不超过250元，必须买票进入'+ '、'.join(names[:3])+'，三项都不能取消，预算和日期不能调整。'
                    else:
                        hard.pop('destination');hard.pop('must_visit');hard['interests']=['科技','植物']
                        case.update(expected='clarify',missing_slots=['destination'],missing_field='destination')
                        request=f'{day}想玩一天，预算1000元，喜欢科技和植物，还没确定去哪个城市。请帮我安排。'
                else:
                    scene=['closure_and_ticket_conflict','optional_closed_plus_matrix_fault','missing_duration_with_named_venues','three_hours_stale_plus_weather'][group]
                    if group==0:
                        case['pois'][0]['closed_dates']=[day]
                        for p in case['pois'][:2]:p['ticket_price']=600
                        case['expected']='grounded_stop';request+='必须买票进入场馆，不能取消任何一个，日期和预算不能改变。'
                    elif group==1:
                        case.update(family='recovery',variant=2);case['pois'][2]['closed_dates']=[day]
                        request+=f'如果{names[2]}当天开放，也可以顺路去。'
                    elif group==2:
                        hard.pop('travel_days');hard.pop('end_date')
                        case.update(expected='clarify',missing_slots=['travel_days'],missing_field='travel_days')
                        request=f'{day}开始在{city}旅行，预算1000元，想去{names[0]}和{names[1]}，待几天还没决定。'
                    else:
                        case['family']='evidence';hard.update(must_visit=names[:3],information_needs=['opening_hours','weather'],current_info_queries=[f'{n} {day} 开放时间' for n in names[:3]])
                        request=f'{day}在{city}玩一天，预算1000元，必须去'+ '、'.join(names[:3])+'。请核实三个地方的当天开放时间和天气，需要能确认有效的资料。'
                case.update(scene_template=scene,source_group=f'{split}-composition-{group+1}',request=request+' 请预留午晚餐时段，本次不要求指定餐厅。')
                rows.append(case)
    return rows

if __name__=='__main__':
    folder=ROOT/'ML/agentic/data/harness_holdout16_v2';folder.mkdir(parents=True,exist_ok=True)
    rows=build_holdout();train=build();manifest={'version':VERSION,'frozen_before_sft':True,'test_targets_forbidden':True,'files':{},'counts':{},'scope':'synthetic pilot; composition-separated, shared skills; no statistical independence claim'}
    for split in ['validation','test']:
        selected=[c for c in rows if c['split']==split]
        content=json.dumps(selected,ensure_ascii=False,indent=2).encode('utf-8');path=folder/(split+'-cases.json')
        if path.exists() and path.read_bytes()!=content:raise RuntimeError('Immutable dataset; create new version')
        path.write_bytes(content);manifest['files'][path.name]=hashlib.sha256(content).hexdigest();manifest['counts'][split]=len(selected)
    groups=[{c['source_group'] for c in rows if c['split']==s} for s in ['validation','test']]+[{c['source_group'] for c in train}]
    entities=[{p['name'] for c in rows if c['split']==s for p in c['pois']} for s in ['validation','test']]+[{p['name'] for c in train for p in c['pois']}]
    assert all(not a&b for pools in [groups,entities] for i,a in enumerate(pools) for b in pools[i+1:])
    manifest.update(source_group_overlap=False,entity_overlap=False)
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest))
