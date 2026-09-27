"""Factor-held-out synthetic sources for the final post-training experiment.

Fifty planning compositions per split share individual factors and tools.
Ten city/date variants are related instances, not independent task structures.
No model output, validation cases, or old catalogs are used to construct facts.
"""
from datetime import date, timedelta
from evaluation.migration_fixture import fingerprint

TRAIN_VERSION = 'travel-posttrain550.v1'
TEST_VERSION = 'travel-locked240.v1'
SHAPES = [
    ('culture_pair',2,[0,1],['文化']), ('science_pair',2,[3,4],['科学']),
    ('nature_pair',2,[6,7],['自然']), ('culture_triple',3,[0,1,2],['文化']),
    ('science_triple',3,[3,4,5],['科学']), ('nature_triple',3,[6,7,8],['自然']),
    ('culture_nature_pair',2,[1,6],['文化','自然']),
    ('science_nature_pair',2,[4,8],['科学','自然']),
    ('culture_science_triple',3,[0,3,5],['文化','科学']),
    ('three_theme_triple',3,[1,4,7],['文化','科学','自然']),
]
CONDITIONS = ['plain','morning','late','optional_closed_expensive','excluded_optional',
              'search-timeout','detail-timeout','matrix-timeout','weather-timeout','stale-hours']
NAMES = {
    'train':['陶印文化讲堂','织纹传习所','篆刻史料堂','光谱研学馆','电流体验馆','风能实验室','苔藓生态园','湿地观察圃','竹林研习院'],
    'test':['剪纸纹样展厅','漆艺文献所','石雕工艺堂','声波探索馆','齿轮实验所','潮汐工程馆','藤本生态苑','水杉观察园','蕈类研习馆'],
}
CITIES = [('常州',31.81,119.97),('绍兴',30.00,120.58),('九江',29.71,115.99),
          ('柳州',24.32,109.42),('宜昌',30.69,111.29),('漳州',24.51,117.65),
          ('芜湖',31.35,118.43),('潍坊',36.71,119.16),('保定',38.87,115.48),('遵义',27.70,106.93)]


def _case(split, si, ci, variant, *, constraint=False):
    shape,days,required,interests=SHAPES[si]
    condition=('hard-budget' if ci==0 else 'required-closed') if constraint else CONDITIONS[ci]
    cid=f"pt{'tr' if split=='train' else 'te'}-{'c' if constraint else 'p'}{si:02d}{ci:02d}-{variant:02d}"
    key=fingerprint({'catalog':cid})[:16]
    city,lat,lng=CITIES[variant]
    start=date(2029 if split=='train' else 2031,1,11)+timedelta(days=si*7+ci*35+variant*400)
    dates=[(start+timedelta(days=i)).isoformat() for i in range(days)]
    prices=[31,43,39,52,46,35,18,26,15]
    pois=[]
    for i,name in enumerate(NAMES[split]):
        theme=['文化','科学','自然'][i//3]
        pois.append(dict(id=f'venue-{key}-{i}',name=city+'合成'+name,category='attraction',tags=[theme],
            ticket_price=prices[i]+variant*2,duration_minutes=60+(i%3)*10,
            open_time='08:00',close_time='19:30',lat=lat+i*.0012,lng=lng+i*.0011,score=.92-i*.009,
            fixture_record=True,date_opening_hours={d:['08:00','19:30'] for d in dates},
            source_url=f'https://catalog.example.invalid/{key}/p{i}',description='离线合成场馆：'+name))
    meal_names=['禾露午膳坊','夕照晚食屋','云樽宴食馆'] if split=='train' else ['麦穗午食馆','暮桥晚膳坊','雅庭宴食轩']
    restaurants=[dict(id=f'venue-{key}-r{i}',name=city+'合成'+name,category='restaurant',tags=['餐厅','美食','家常菜'],
        ticket_price=0,average_cost=price+variant,duration_minutes=45,open_time=opening,close_time=closing,
        lat=lat+.002+i*.001,lng=lng+.003+i*.001,address=city+'虚构巷'+str(i+1)+'号',fixture_record=True,
        source_url=f'https://catalog.example.invalid/{key}/r{i}')
        for i,(name,price,opening,closing) in enumerate(zip(meal_names,[25,38,190],['10:30','17:00','10:00'],['14:30','21:30','21:30']))]
    budget=1400 if days==2 else 2100
    slots=dict(destination=city,travel_days=days,start_date=dates[0],end_date=dates[-1],budget_range=budget,
        interests=interests.copy(),must_visit=[pois[i]['name'] for i in required],require_named_restaurants=True)
    optional=next(i for i in range(9) if i not in required)
    extra='';scenario=condition if condition in CONDITIONS[5:] else 'normal'
    if condition=='morning':
        pois[required[0]].update(close_time='12:30',date_opening_hours={d:['08:00','12:30'] for d in dates})
        extra='第一处必去场馆只在上午接待，请核实营业时段。'
    elif condition=='late':
        pois[required[0]].update(open_time='11:00',date_opening_hours={d:['11:00','19:30'] for d in dates})
        extra='第一处必去场馆较晚开放，请核实后安排。'
    elif condition=='optional_closed_expensive':
        pois[optional].update(closed_dates=dates,date_opening_hours={})
        expensive=next(i for i in range(9) if i not in required and i!=optional);pois[expensive]['ticket_price']=budget+500
        extra='可选场馆允许舍弃，不要安排停业或超预算项目。'
    elif condition=='excluded_optional':
        slots['must_not_visit']=[pois[optional]['name']];extra=f"不要安排{pois[optional]['name']}。"
    elif condition=='weather-timeout':
        slots['information_needs']=['weather'];extra='请先核实出行当天的天气。'
    elif condition=='stale-hours':
        slots['information_needs']=['opening_hours'];slots['current_info_queries']=[f"{pois[required[0]]['name']} {dates[0]} 开放时间"]
        extra='请用有效且属于目标场馆的资料核实开放时间。'
    expected='plan'
    if constraint:
        expected='grounded_stop';scenario=condition
        if condition=='hard-budget':
            pois[required[0]]['ticket_price']=900;slots['budget_range']=100
            if variant%2:
                slots['constraint_flexibility']=dict(schema_version='constraint-flexibility.v1',locked_constraints=['activity_set'],
                    solver_adjustable_constraints=['activity_schedule'],relaxable_constraints=['total_budget'],
                    relaxation_options={'total_budget':['将总预算提高至2400元']})
                extra='预算不足时可提出提高至2400元的方案，必须由我确认，不能直接放宽。'
            else:extra='总预算为硬上限，日期、天数和必去项均不可变。'
        else:
            pois[required[0]].update(closed_dates=dates,date_opening_hours={})
            extra='日期、天数、必去项均不可变，不接受外观打卡。'
    request=f"从{dates[0]}开始在{city}游玩{days}天，总预算{slots['budget_range']}元，偏好{'、'.join(interests)}，必须入内参观{'、'.join(slots['must_visit'])}。每天安排具名午餐和晚餐，核实营业时间、餐费与通勤。"+extra
    group=f"posttrain-{'constraint' if constraint else 'composition'}:{shape}:{condition}"
    c=dict(id=cid,dataset_version=TRAIN_VERSION if split=='train' else TEST_VERSION,split=split,
        family='constraint' if constraint else ('recovery' if scenario in CONDITIONS[5:] else 'normal'),scenario=scenario,
        variant=variant,request=request,slots=slots,missing_slots=[],missing_field=None,expected=expected,
        pois=pois,restaurants=restaurants,source_group=group,lineage_group=group,synthetic=True,
        reference_time='2026-09-05T00:00:00+00:00',provider_faults=[],provider_query_facts=[],training_targets_present=False,
        provenance=dict(kind='factor_heldout_catalog_first',shape=shape,condition=condition,
            split_rule='planning: parity(shape_index+condition_index); constraints: parity(shape_index+constraint_index)',
            related_variants=True,shared_factors_and_tools=True,intent_slots='predeclared_from_request',live_facts=False,training_authorized=split=='train'))
    c['source_case_sha256']=fingerprint(c);return c


def build_cases(split='train'):
    if split not in {'train','test'}:raise ValueError('Only train or locked test')
    parity=0 if split=='train' else 1
    cases=[_case(split,si,ci,v) for si in range(10) for ci in range(10)
           if (si+ci)%2==parity for v in range(10 if split=='train' else 4)]
    cases += [_case(split,si,ci,v,constraint=True) for si in range(10) for ci in range(2)
              if (si+ci)%2==parity for v in range(5 if split=='train' else 4)]
    return cases
