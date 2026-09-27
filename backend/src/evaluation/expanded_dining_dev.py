"""48 synthetic validation instances in 16 related composition groups.

New catalog and requests, frozen before model evaluation. Shared tool skills
mean this is expanded development evidence, not an independent final test.
Construction and all evaluation run only on the existing cloud host.
"""
from datetime import date, timedelta
from evaluation.migration_fixture import fingerprint

VERSION = 'travel-expanded-dining-dev48.v1'
VENUES = [
    ('织造陈列馆', ['文化', '历史'], 36, 70),
    ('木刻收藏馆', ['文化', '艺术'], 52, 65),
    ('古乐体验馆', ['文化', '音乐'], 44, 75),
    ('声学互动馆', ['科学', '技术'], 62, 80),
    ('能源实验馆', ['科学', '工业'], 48, 70),
    ('材料发现馆', ['科学', '技术'], 39, 65),
    ('苔藓生态园', ['自然', '生态'], 18, 65),
    ('候鸟观察园', ['自然', '生态'], 24, 70),
    ('岩石景观园', ['自然', '地质'], 16, 60),
]
# name, days, interests, required catalog indices, provider scenario
GROUPS = [
    ('two_day_culture', 2, ['文化'], [0, 1], 'dev-composition'),
    ('two_day_science', 2, ['科学'], [3, 4], 'dev-composition'),
    ('two_day_nature', 2, ['自然'], [6, 7], 'dev-composition'),
    ('two_day_mixed', 2, ['文化', '科学'], [0, 3], 'dev-composition'),
    ('three_day_mixed', 3, ['文化', '科学', '自然'], [0, 3, 6], 'dev-composition'),
    ('early_visit_split_meals', 2, ['自然'], [6], 'dev-composition'),
    ('optional_closed_and_expensive', 2, ['科学'], [3], 'dev-composition'),
    ('excluded_culture_venue', 2, ['文化'], [1], 'dev-composition'),
    ('search_retry_multiday', 2, ['自然'], [6, 7], 'search-timeout'),
    ('detail_retry_multiday', 2, ['文化'], [0, 1], 'detail-timeout'),
    ('matrix_retry_multiday', 2, ['科学'], [3, 4], 'matrix-timeout'),
    ('weather_retry_named_meals', 1, ['自然'], [6], 'weather-timeout'),
    ('required_closed_all_dates', 2, ['文化'], [0], 'required-closed'),
    ('joint_ticket_locked_budget', 2, ['科学'], [3, 4], 'dev-composition'),
    ('authorized_budget_multiday', 2, ['自然'], [6], 'budget-tradeoff'),
    ('three_day_locked_ticket_budget', 3, ['文化'], [0, 1, 2], 'hard-budget'),
]


def build_cases():
    cases = []
    for gi, (group, days, interests, required, scenario) in enumerate(GROUPS):
        for variant, (city, lat, lng) in enumerate([
            ('宁波', 29.87, 121.55), ('南昌', 28.68, 115.86), ('贵阳', 26.65, 106.63)
        ]):
            cid = f'expanded-dev-{gi:02d}-{variant}'
            key = fingerprint({'expanded_catalog_instance': cid})[:16]
            start = date(2027, 8, 3) + timedelta(days=gi * 4 + variant * 70)
            dates = [(start + timedelta(days=d)).isoformat() for d in range(days)]
            pois = [dict(
                id=f'place-{key}-{i}', name=city+'合成'+name, category='attraction', tags=list(tags),
                ticket_price=price+variant*2, duration_minutes=duration, open_time='08:00', close_time='19:00',
                lat=lat+i*.001, lng=lng+i*.0012, score=.88-i*.008,
                date_opening_hours={d: ['08:00', '19:00'] for d in dates}, fixture_record=True,
                source_url=f'https://frozen.example.invalid/{key}/p{i}', description='合成场馆：'+name,
            ) for i, (name, tags, price, duration) in enumerate(VENUES)]
            restaurants = [dict(
                id=f'place-{key}-r{i}', name=city+'合成'+name, category='restaurant',
                tags=['餐厅', '美食', '家常菜'], ticket_price=0, average_cost=price+variant,
                duration_minutes=45, open_time=opening, close_time=closing,
                lat=lat+.003+i*.001, lng=lng+.004+i*.001, address=city+'虚构巷'+str(i+1)+'号',
                fixture_record=True, source_url=f'https://frozen.example.invalid/{key}/r{i}',
            ) for i, (name, price, opening, closing) in enumerate([
                ('青禾午食铺', 31, '10:30', '14:00'), ('灯暖晚食铺', 41, '17:00', '21:00'),
                ('云席食府', 170, '10:00', '21:00'),
            ])]
            budget = 1400 if days == 3 else 900 if days == 2 else 500
            expected = 'plan'; extra = ''
            if gi == 5:
                pois[6].update(close_time='12:30', date_opening_hours={d: ['08:00', '12:30'] for d in dates})
                extra = '必去园区只在上午接待，请核实开放时段。'
            if gi == 6:
                pois[0].update(closed_dates=dates, date_opening_hours={})
                pois[1]['ticket_price'] = 1200
                extra = '非必去项目可不选，请避开闭馆或超预算的场馆。'
            if gi == 12:
                pois[0].update(closed_dates=dates, date_opening_hours={})
                expected = 'grounded_stop'; extra = '日期、天数、必去项均不可改，不接受外观打卡。'
            if gi in {13, 15}:
                for i in required: pois[i]['ticket_price'] = 160
                budget = 250; expected = 'grounded_stop'
                extra = '预算、日期及所有入馆要求都是硬约束，不接受删掉必去项或只看外观。'
            if gi == 14:
                pois[6]['ticket_price'] = 850; budget = 300; expected = 'grounded_stop'
                extra = '若预算不足，可提出提高到1600元，由我确认后再改；必去项不可删除。'
            slots = dict(destination=city, travel_days=days, start_date=dates[0], end_date=dates[-1],
                budget_range=budget, interests=list(interests), must_visit=[pois[i]['name'] for i in required],
                require_named_restaurants=True)
            if gi == 7:
                slots['must_not_visit'] = [pois[0]['name']]
                extra = f"不要安排{pois[0]['name']}。"
            if scenario == 'weather-timeout':
                slots['information_needs'] = ['weather']; extra = '请核实当天的天气。'
            if gi == 14:
                slots['constraint_flexibility'] = dict(schema_version='constraint-flexibility.v1',
                    locked_constraints=['activity_set'], solver_adjustable_constraints=['activity_schedule'],
                    relaxable_constraints=['total_budget'], relaxation_options={'total_budget': ['将总预算提高至1600元']})
            request = (f"从{dates[0]}起在{city}玩{days}天，总预算{budget}元，喜欢{'、'.join(interests)}，"
                f"必须入内参观{'、'.join(slots['must_visit'])}。每天午晚餐都安排具名餐厅，核实营业时间、餐费和通勤。"+extra)
            case = dict(id=cid, dataset_version=VERSION, split='validation',
                family='constraint' if gi >= 12 else 'recovery' if gi >= 8 else 'normal',
                scenario=scenario, variant=variant, request=request, slots=slots, missing_slots=[], missing_field=None,
                expected=expected, pois=pois, restaurants=restaurants, source_group='expanded-dev-composition:'+group,
                lineage_group='expanded-dev-composition:'+group, synthetic=True, reference_time='2026-09-05T00:00:00+00:00',
                provider_faults=[], provider_query_facts=[], training_targets_present=False,
                provenance=dict(kind='newly_authored_expanded_development_catalog', parent_instance_id=None,
                    catalog_blueprint_sha256=fingerprint(VENUES), variants_share_composition=True, live_facts=False,
                    intent_slots='predeclared_from_request', training_authorized=False,
                    limits='48 instances in 16 related groups; shared tool skills; not locked final test'))
            case['source_case_sha256'] = fingerprint(case); cases.append(case)
    return cases
