"""Build a small synthetic training-only corpus; never relabel it as a test set.

Different scene compositions, cities and entities from diagnostic24. Shared
decision skills are intentional. No teacher actions or answers enter policy state.
"""
from pathlib import Path
from datetime import date,timedelta
import json,hashlib
from evaluation.research_fixture import catalog

VERSION="harness-training-pilot24.v1"
ROOT=Path(__file__).resolve().parents[1]
SCENES=[
    ("two_venues_hours","两个场馆都要去，先核实它们当天的开放时间。"),
    ("optional_closed","除必去项目外，我也对{optional}感兴趣；若它当天关闭，可以不安排。"),
    ("combined_ticket_floor","预算是不可提高的硬上限，两项必去活动都不能取消。"),
    ("two_day_closure","这两天内必须进入{required}，不接受外观打卡；日期和必去项都不能改。"),
    ("detail_timeout_and_weather","请核对天气；不要把一次服务超时当成景点不存在。"),
    ("stale_hours_and_optional_closed","要核实两项必去活动的开放时间；过期页面不能作为依据，其他候选可选。"),
]
CITIES=[("苏州",31.30,120.58),("南京",32.06,118.79),("长沙",28.22,112.94),("西安",34.27,108.95)]


def build():
    cases=[]
    for index,(scene,wording) in enumerate(SCENES):
        for variant,(city,lat,lng) in enumerate(CITIES):
            ident=f"pilot-{index+1:02d}-{variant+1:02d}"
            start=date(2026,10,6)+timedelta(days=3*index+variant)
            days=2 if index==3 else 1
            names=[f"{city}青禾{index+1}号工艺展馆",f"{city}望江{index+1}号古建园",f"{city}拾光{index+1}号摄影馆",f"{city}映水{index+1}号步道"]
            pois=[dict(id=f"{ident}-p{i}",name=n,category="attraction",lat=lat+.002*i,lng=lng+.002*i,
                ticket_price=40,open_time="09:00",close_time="20:00",duration_minutes=75,
                tags=["工艺","古建","摄影","散步"][i:i+1],score=.95-.03*i,
                source_url=f"https://pilot.example.invalid/{ident}/p{i}",fixture_record=True) for i,n in enumerate(names)]
            budget=220 if index==2 else 1000
            slots=dict(destination=city,travel_days=days,start_date=start.isoformat(),end_date=(start+timedelta(days=days-1)).isoformat(),budget_range=budget,must_visit=names[:2],interests=["工艺","古建"])
            expected="plan";family="training"; fault_variant=variant
            if index in {0,5}:
                slots.update(information_needs=["opening_hours"],current_info_queries=[f"{n} {start.isoformat()} 开放时间" for n in names[:2]])
            if index in {1,5}:pois[2]['closed_dates']=[start.isoformat()]
            if index==2:
                pois[0]['ticket_price']=130;pois[1]['ticket_price']=140;expected="grounded_stop"
            if index==3:
                pois[0]['closed_dates']=[(start+timedelta(days=d)).isoformat() for d in range(days)];expected="grounded_stop"
            if index==4:
                family="recovery";fault_variant=1;slots['information_needs']=['weather']
            if index==5:family="evidence"
            request=f"{start.isoformat()}开始在{city}游玩{days}天，总预算{budget}元，必须参观{names[0]}和{names[1]}。"+wording.format(optional=names[2],required=names[0])+"请安排可行行程，并预留午晚餐时段；本次不要求指定餐厅或预订。"
            case=dict(id=ident,family=family,variant=fault_variant,request=request,slots=slots,missing_slots=[],missing_field=None,
                pois=pois,expected=expected,reference_time="2026-09-05T00:00:00+00:00",
                dataset_version=VERSION,split="train",scene_template=scene,source_group=f"pilot-composition-{index+1}",
                synthetic=True,training_targets_present=False)
            case['restaurants']=[p for p in catalog(case) if p['category']=='restaurant']
            for i,r in enumerate(case['restaurants']):
                r['id']=f"{ident}-r{i}";r['name']=f"{city}青禾{index+1}号"+('素食餐厅' if i else '家常菜馆')
                r['source_url']=f"https://pilot.example.invalid/{ident}/r{i}"
            cases.append(case)
    return cases


if __name__=='__main__':
    from scripts.evaluate_harness_base import build_cases
    cases=build(); old=build_cases()
    names=lambda rows:{p['name'] for c in rows for p in c['pois']}
    assert not names(cases)&names(old)
    target=ROOT/'ML/agentic/data/harness_training_pilot24_v1'
    target.mkdir(parents=True,exist_ok=True)
    content=json.dumps(cases,ensure_ascii=False,indent=2).encode('utf-8')
    path=target/'cases.json'
    if path.exists() and path.read_bytes()!=content:raise RuntimeError('Use a new dataset version')
    path.write_bytes(content)
    audit=dict(version=VERSION,split='train',count=len(cases),sha256=hashlib.sha256(content).hexdigest(),
        source_groups=sorted({c['source_group'] for c in cases}),scene_templates=sorted({c['scene_template'] for c in cases}),
        diagnostic_entity_overlap=sorted(names(cases)&names(old)),
        diagnostic_request_overlap=sorted({c['request'] for c in cases}&{c['request'] for c in old}),
        limits=['synthetic pilot, not independent test','shared decision skills; scene compositions differ, no statistical independence claim',
                'no final test cases constructed or used','not SFT-ready until actual execution and example review',
                'meal time slots only; named restaurant routing not evaluated'])
    (target/'manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))
