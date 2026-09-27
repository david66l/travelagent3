"""1,000 new synthetic task instances with explicit template lineage.

Forty compositions reuse twenty parent templates. City/date/wording changes do
not create independent structural groups. All records and coordinates are
simulated; this corpus is for action-policy research, not live travel facts.
"""
from copy import deepcopy
from datetime import date,timedelta
import random
from evaluation.curriculum_fixture import build_cases as pilot_cases
from evaluation.migration_fixture import fingerprint

VERSION='travel-curriculum1000.v1'
CITIES=[('上海',31.23,121.47),('杭州',30.25,120.15),('成都',30.66,104.06),('北京',39.90,116.40),
 ('苏州',31.30,120.58),('南京',32.06,118.79),('武汉',30.59,114.30),('长沙',28.23,112.94),
 ('西安',34.34,108.94),('重庆',29.56,106.55),('广州',23.13,113.26),('深圳',22.54,114.06),
 ('厦门',24.48,118.09),('福州',26.07,119.30),('青岛',36.07,120.38),('济南',36.65,117.12),
 ('天津',39.08,117.20),('大连',38.91,121.61),('沈阳',41.80,123.43),('长春',43.82,125.32),
 ('昆明',25.04,102.72),('贵阳',26.65,106.63),('南宁',22.82,108.37),('南昌',28.68,115.86),('合肥',31.82,117.23)]


def build_cases():
    templates=[c for c in pilot_cases() if c['variant']==0]
    cases=[]
    for composition, (template, compound) in enumerate((t,c) for t in templates for c in [False,True]):
        for variant,(city,lat,lng) in enumerate(CITIES):
            case=deepcopy(template);old_city=CITIES[0][0];scenario=case['scenario']
            case_id=f'scale1k-{composition:02d}-{variant:02d}'
            public=fingerprint({'instance':case_id})[:16]
            rng=random.Random(int(public,16))
            start=date(2026,10,2)+timedelta(days=composition*3+variant*2)
            if start.weekday()==0:start+=timedelta(days=1)
            day=start.isoformat();old_day=case['slots']['start_date']
            old_budget=case['slots']['budget_range']
            budget=old_budget if scenario in {'hard-budget','budget-tradeoff'} else rng.choice([900,1100,1300,1500])
            request=case['request'].replace(old_city,city).replace(old_day,day).replace(f'总预算{old_budget}元',f'总预算{budget}元').replace(f'预算{old_budget}元',f'预算{budget}元')
            slots=case['slots']
            for key in ['destination']:
                if key in slots:slots[key]=city
            slots['start_date']=day
            if 'end_date' in slots:slots['end_date']=day
            slots['budget_range']=budget
            for key in ['must_visit','must_not_visit','current_info_queries']:
                if key in slots:slots[key]=[s.replace(old_city,city).replace(old_day,day) for s in slots[key]]
            for index,poi in enumerate(case['pois']+case['restaurants']):
                poi['id']=f'entity-{public}-{index}'
                poi['name']=poi['name'].replace(old_city,city)
                poi['source_url']=f'https://curriculum.example.invalid/{public}/record-{index}'
                poi['lat']=round(lat+rng.uniform(-.009,.009),6)
                poi['lng']=round(lng+rng.uniform(-.009,.009),6)
                if poi.get('address'):poi['address']=poi['address'].replace(old_city,city)
                poi['closed_dates']=[day for _ in poi.get('closed_dates',[])]
                if poi.get('date_opening_hours'):poi['date_opening_hours']={day:['08:00','21:00']}
                if poi.get('category')!='restaurant' and poi['ticket_price']<800:
                    poi['ticket_price']=max(0,poi['ticket_price']+rng.choice([-5,0,5]))
                if poi.get('category')=='restaurant':poi['average_cost']+=rng.choice([-5,0,5])
            if compound:
                required=slots.get('must_visit',[])
                # These additions change constraints or provider facts, not just wording.
                if scenario in {'history','mixed-culture','detail-timeout','weather-timeout','required-closed','budget-tradeoff'}:
                    extra=case['pois'][4 if scenario=='history' else 2]
                    if extra['name'] not in required:
                        slots['must_visit']=[*required,extra['name']]
                        request+=f"另外必须参观{extra['name']}，不可自行删掉。"
                elif scenario=='hard-budget':
                    first=next(p for p in case['pois'] if p['name'] in required)
                    first['ticket_price']=180;extra=case['pois'][0];extra['ticket_price']=180
                    slots['must_visit']=[*required,extra['name']]
                    request+=f"还必须参观{extra['name']}；总预算300元不变。"
                elif scenario in {'art','matrix-timeout','avoid-venue','wrong-entity'}:
                    forbidden=next(p for p in reversed(case['pois']) if p['name'] not in required)
                    slots['must_not_visit']=list(dict.fromkeys([*slots.get('must_not_visit',[]),forbidden['name']]))
                    request+=f"不要安排{forbidden['name']}。"
                elif scenario in {'nature','search-timeout','optional-closed'}:
                    optional=[p for p in case['pois'] if p['name'] not in required][:2]
                    for p in optional:p['closed_dates']=[day];p['date_opening_hours']={}
                    request+='可选场馆如果当天闭馆，请跳过并保持所有必去项。'
                elif scenario in {'architecture','science','stale-hours'}:
                    first=next(p for p in case['pois'] if p['name'] in required)
                    first['close_time']='15:30';first['date_opening_hours']={day:['08:00','15:30']}
                    request+='请按场馆当天可入馆时段安排。'
                elif scenario=='botany':
                    slots['budget_range']=250;request=request.replace(f'总预算{budget}元','总预算250元')
                    request+='预算250元是硬上限，优先低票价场馆。'
                elif scenario=='missing-destination':
                    slots['interests']=['自然'];request+='偏好自然景观，目的地仍需先确定。'
                elif scenario=='missing-days':
                    slots['must_not_visit']=[case['pois'][1]['name']]
                    request+=f"不要去{case['pois'][1]['name']}，游玩天数仍未确定。"
            case.update(id=case_id,request=request,variant=variant,dataset_version=VERSION,
                source_group=f'scale-composition:{scenario}:{"compound" if compound else "single"}',
                lineage_group=template['source_group'],composition=composition,
                provenance={**case['provenance'],'parent_template':template['source_group'],
                    'composition_constraints_added':compound,'shared_catalog_archetypes':True,
                    'independent_structural_group_claim':False,'generation_seed':public})
            case.pop('source_case_sha256',None)
            case['source_case_sha256']=fingerprint(case)
            cases.append(case)
    return cases
