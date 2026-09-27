"""Source-disjoint synthetic development compositions; never training targets.

Catalogs are newly authored here, not copied from the training curriculum.
The twelve instances share six composition families and the same tool/solver
skills as training. This is a small development set, not an independent test
of live travel facts or twelve independent families.
"""
from datetime import date,timedelta
from evaluation.migration_fixture import fingerprint

VERSION='travel-named-dining-dev.v1'
VENUES=[('城南文史馆',['历史','文化'],45,75),('手作工艺坊',['艺术','文化'],55,75),
        ('湿地观察园',['自然','生态'],15,60),('天文体验馆',['科学','天文'],65,75),
        ('地质科普园',['科学','自然'],20,60)]
GROUPS=['split_dining_windows','two_day_cultural_visits','early_nature_visit',
        'optional_cost_and_closure','joint_must_visit_budget','authorized_two_day_budget']


def build_cases():
    cases=[]
    for gi,group in enumerate(GROUPS):
        for variant,(city,lat,lng) in enumerate([('南京',32.06,118.79),('武汉',30.58,114.30)]):
            private_id=f'dining-dev-{gi:02d}-{variant}'
            key=fingerprint({'new_catalog_instance':private_id})[:16]
            start=date(2027,3,9)+timedelta(days=gi*3+variant*20)
            days=2 if gi in {1,4,5} else 1
            dates=[(start+timedelta(days=d)).isoformat() for d in range(days)]
            pois=[dict(id=f'place-{key}-{i}',name=city+'合成'+name,category='attraction',tags=list(tags),
                ticket_price=price,duration_minutes=duration,open_time='08:00',close_time='19:00',
                lat=lat+i*.0015,lng=lng+i*.001,score=.85-i*.015,
                date_opening_hours={d:['08:00','19:00'] for d in dates},fixture_record=True,
                source_url=f'https://frozen.example.invalid/{key}/p{i}',description='合成场馆：'+name)
                for i,(name,tags,price,duration) in enumerate(VENUES)]
            restaurants=[dict(id=f'place-{key}-r{i}',name=city+'合成'+name,category='restaurant',
                tags=['餐厅','美食','家常菜'],ticket_price=0,average_cost=price,duration_minutes=45,
                open_time=opening,close_time=closing,lat=lat+.002+i*.001,lng=lng+.002+i*.001,
                address=city+'虚构街'+str(i+1)+'号',fixture_record=True,
                source_url=f'https://frozen.example.invalid/{key}/r{i}')
                for i,(name,price,opening,closing) in enumerate([
                    ('午间小馆',32+variant,'10:30','14:00'),('晚间食堂',46+variant,'17:00','21:00')])]
            required=[0,1] if gi in {1,4} else [2] if gi==2 else [3] if gi in {3,5} else [0]
            interests=['自然'] if gi==2 else ['科学'] if gi in {3,5} else ['文化']
            budget=900 if days==2 else 550
            expected='plan';extra='';scenario='dev-composition'
            if gi==2:
                pois[2].update(close_time='13:00',date_opening_hours={d:['08:00','13:00'] for d in dates})
                extra='必去场馆午后不再接待，请以工具查到的开放时间安排。'
            if gi==3:
                pois[0].update(ticket_price=700,closed_dates=dates,date_opening_hours={})
                extra='非必去场馆可以不安排；请避免超预算或安排停业场馆。'
            if gi==4:
                pois[0]['ticket_price']=110;pois[1]['ticket_price']=140
                budget=200;expected='grounded_stop';extra='预算、日期及两项入馆要求都不能改变，不接受只在门口打卡。'
            if gi==5:
                pois[3]['ticket_price']=900;budget=300;expected='grounded_stop';scenario='budget-tradeoff'
                extra='若无法满足预算，可以提出提高到1500元，由我确认后再改；必去项不能删。'
            slots=dict(destination=city,travel_days=days,start_date=dates[0],end_date=dates[-1],
                budget_range=budget,interests=interests,must_visit=[pois[i]['name'] for i in required],require_named_restaurants=True)
            if gi==5:
                slots['constraint_flexibility']=dict(schema_version='constraint-flexibility.v1',locked_constraints=['activity_set'],
                    solver_adjustable_constraints=['activity_schedule'],relaxable_constraints=['total_budget'],
                    relaxation_options={'total_budget':['将总预算提高至1500元']})
            request=f"从{dates[0]}起在{city}游玩{days}天，预算{budget}元，偏好{'、'.join(interests)}，必须入内参观{'、'.join(slots['must_visit'])}。每天午晚餐都安排具名餐厅，核实营业时段、餐费和通勤。"+extra
            c=dict(id=private_id,dataset_version=VERSION,split='validation',family='constraint' if gi>=4 else 'normal',
                scenario=scenario,variant=variant,request=request,slots=slots,missing_slots=[],missing_field=None,expected=expected,
                pois=pois,restaurants=restaurants,source_group='dining-dev-composition:'+group,lineage_group='dining-dev-composition:'+group,
                synthetic=True,reference_time='2026-09-05T00:00:00+00:00',provider_faults=[],provider_query_facts=[],training_targets_present=False,
                provenance={'kind':'newly_authored_catalog_and_development_composition','parent_instance_id':None,
                    'catalog_blueprint_sha256':fingerprint(VENUES),'variants_share_composition':True,
                    'live_facts':False,'intent_slots':'predeclared_from_request','training_authorized':False,
                    'limits':'six development compositions share tool skills with training; not locked test'})
            c['source_case_sha256']=fingerprint(c);cases.append(c)
    return cases
