"""Ninety-six catalog-first train instances, eight related composition groups.

New source facts; shared tool skills with prior training and development.
Instance count is not an independent structure count or scale acceptance.
"""
from datetime import date,timedelta
from evaluation.migration_fixture import fingerprint

VERSION='travel-dining-recovery96.v1'
VENUES=[('古籍展览馆',['文化','历史'],38,65),('版画艺术中心',['文化','艺术'],48,70),
        ('水生植物园',['生态','自然'],12,60),('机械探索馆',['科学','工业'],58,80),
        ('光学实验馆',['科学','技术'],42,65)]
GROUPS=[('dining_price_choice',16),('early_lunch_research',16),('dining_transit_tradeoff',16),
        ('two_day_science_research',16),('detail_timeout_dining',8),('matrix_timeout_dining',8),
        ('locked_ticket_lower_bound',8),('authorized_budget_proposal',8)]
CITIES=[('苏州',31.30,120.58),('长沙',28.23,112.94),('福州',26.08,119.30),('合肥',31.82,117.23)]

def build_cases():
    cases=[]
    for gi,(group,count) in enumerate(GROUPS):
        for variant in range(count):
            city,lat,lng=CITIES[variant%len(CITIES)]
            private_id=f'dining96-{gi:02d}-{variant:02d}';key=fingerprint({'catalog':private_id})[:16]
            first=date(2027,5,4)+timedelta(days=gi*17+variant*2);days=2 if gi in {3,4,7} else 1
            dates=[(first+timedelta(days=d)).isoformat() for d in range(days)]
            pois=[dict(id=f'entity-{key}-{i}',name=city+'合成'+name,category='attraction',tags=tags,
                ticket_price=price+variant%3,duration_minutes=duration,open_time='08:00',close_time='19:00',
                lat=lat+i*.001,lng=lng+i*.0012,score=.88-i*.012,date_opening_hours={d:['08:00','19:00'] for d in dates},
                fixture_record=True,source_url=f'https://frozen.example.invalid/{key}/v{i}',description='合成研究场馆：'+name)
                for i,(name,tags,price,duration) in enumerate(VENUES)]
            restaurants=[dict(id=f'entity-{key}-r{i}',name=city+'合成'+name,category='restaurant',tags=['餐厅','美食','家常菜'],
                ticket_price=0,average_cost=price+variant%4,duration_minutes=45,open_time=start,close_time=end,
                lat=lat+.002+i*.001,lng=lng+.002+i*.001,address=city+'虚构研究街'+str(i+1)+'号',
                fixture_record=True,source_url=f'https://frozen.example.invalid/{key}/r{i}')
                for i,(name,price,start,end) in enumerate([('稻香午餐店',29,'10:30','14:30'),
                    ('灯下家常馆',43,'17:00','21:00'),('锦味食坊',190,'09:00','21:00')])]
            required=[3,4] if gi==3 else [0,1] if gi in {5,6} else [2] if gi==2 else [3] if gi==7 else [0]
            interests=['科学'] if gi in {3,7} else ['自然'] if gi==2 else ['文化']
            budget=1000 if days==2 else 430
            scenario='train-dining-composition';expected='plan';family='normal';extra=''
            if gi==0:restaurants[2]['average_cost']=500;extra='餐厅价位不同，请在总预算内选择。'
            if gi==1:restaurants[0]['close_time']='13:15';extra='请核实午餐店的收市时间。'
            if gi==2:
                restaurants[2].update(average_cost=18,lat=lat+.09,lng=lng+.09)
                extra='较远餐厅的餐费和通勤应一起计入预算。'
            if gi in {4,5}:family='recovery';scenario='detail-timeout' if gi==4 else 'matrix-timeout'
            if gi==6:
                family='constraint';budget=200;expected='grounded_stop';pois[0]['ticket_price']=125;pois[1]['ticket_price']=135
                extra='总预算和两项入馆要求均固定，不接受删除必去项或只在外面打卡。'
            if gi==7:
                family='constraint';scenario='budget-tradeoff';expected='grounded_stop';budget=350;pois[3]['ticket_price']=950
                extra='若无法满足预算，可提出将预算提高至1600元，等我确认后再改；不得删除必去项。'
            slots=dict(destination=city,travel_days=days,start_date=dates[0],end_date=dates[-1],budget_range=budget,
                interests=interests,must_visit=[pois[i]['name'] for i in required],require_named_restaurants=True)
            if gi==7:slots['constraint_flexibility']=dict(schema_version='constraint-flexibility.v1',locked_constraints=['activity_set'],
                solver_adjustable_constraints=['activity_schedule'],relaxable_constraints=['total_budget'],relaxation_options={'total_budget':['将总预算提高至1600元']})
            request=f"从{dates[0]}起去{city}游玩{days}天，总预算{budget}元，偏好{'、'.join(interests)}，必须入内参观{'、'.join(slots['must_visit'])}。每天午晚餐安排具名餐厅，核实餐费、营业时间和相邻交通。"+extra
            case=dict(id=private_id,dataset_version=VERSION,split='train',family=family,scenario=scenario,variant=variant,
                request=request,slots=slots,missing_slots=[],missing_field=None,expected=expected,pois=pois,restaurants=restaurants,
                source_group='dining-recovery-composition:'+group,lineage_group='dining-recovery-composition:'+group,
                synthetic=True,reference_time='2026-09-05T00:00:00+00:00',provider_faults=[],provider_query_facts=[],training_targets_present=False,
                provenance=dict(kind='newly_authored_training_catalog',parent_instance_id=None,training_authorized=True,
                    catalog_blueprint_sha256=fingerprint(VENUES),variants_share_composition=True,live_facts=False,
                    intent_slots='predeclared_from_request',limits='eight related groups; shared tool skills with dev; not independent structural generalization'))
            case['source_case_sha256']=fingerprint(case);cases.append(case)
    return cases
