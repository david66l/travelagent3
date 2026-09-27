"""New synthetic planning-only train sources; cloud construction and auditing.

80 instances share 20 composition groups. They are not 80 independent task
structures. Catalog facts and checkpoints from validation are never imported.
"""
from datetime import date, timedelta
from evaluation.migration_fixture import fingerprint

VERSION = 'travel-multiday-train80.v1'
VENUES = [
    ('纸艺传习馆', ['文化', '艺术'], 34, 65), ('民俗器物馆', ['文化', '历史'], 46, 75),
    ('戏曲资料馆', ['文化', '音乐'], 42, 70), ('磁力探索馆', ['科学', '技术'], 56, 80),
    ('水动力展馆', ['科学', '工业'], 44, 70), ('晶体实验馆', ['科学', '地质'], 38, 65),
    ('蕨类观赏园', ['自然', '生态'], 22, 65), ('溪谷研习园', ['自然', '生态'], 28, 75),
    ('芳草标本园', ['自然', '植物'], 14, 60),
]
# composition, days, interests, must-visit indices, observable provider scenario
GROUPS = [
    ('culture_pair_two_days', 2, ['文化'], [0, 1], 'normal'),
    ('science_pair_two_days', 2, ['科学'], [3, 4], 'normal'),
    ('nature_pair_two_days', 2, ['自然'], [6, 7], 'normal'),
    ('culture_triple_three_days', 3, ['文化'], [0, 1, 2], 'normal'),
    ('science_triple_three_days', 3, ['科学'], [3, 4, 5], 'normal'),
    ('nature_triple_three_days', 3, ['自然'], [6, 7, 8], 'normal'),
    ('culture_nature_two_days', 2, ['文化', '自然'], [1, 7], 'normal'),
    ('three_themes_three_days', 3, ['文化', '科学', '自然'], [2, 4, 8], 'normal'),
    ('morning_culture_two_days', 2, ['文化'], [0, 2], 'normal'),
    ('late_science_two_days', 2, ['科学'], [3, 5], 'normal'),
    ('closed_optional_nature', 2, ['自然'], [6, 8], 'normal'),
    ('excluded_culture_two_days', 2, ['文化'], [1, 2], 'normal'),
    ('search_retry_culture_pair', 2, ['文化'], [0, 2], 'search-timeout'),
    ('search_retry_nature_triple', 3, ['自然'], [6, 7, 8], 'search-timeout'),
    ('detail_retry_science_pair', 2, ['科学'], [3, 5], 'detail-timeout'),
    ('detail_retry_culture_triple', 3, ['文化'], [0, 1, 2], 'detail-timeout'),
    ('matrix_retry_nature_pair', 2, ['自然'], [7, 8], 'matrix-timeout'),
    ('matrix_retry_three_themes', 3, ['文化', '科学', '自然'], [1, 3, 6], 'matrix-timeout'),
    ('weather_retry_science_day', 1, ['科学'], [4], 'weather-timeout'),
    ('stale_hours_science_pair', 2, ['科学'], [3, 4], 'stale-hours'),
]


def build_cases():
    cases = []
    for gi, (group, days, interests, required, scenario) in enumerate(GROUPS):
        for variant, (city, lat, lng) in enumerate([
            ('泉州', 24.87, 118.67), ('珠海', 22.27, 113.58),
            ('绵阳', 31.47, 104.74), ('洛阳', 34.62, 112.45),
        ]):
            cid = f'multiday80-{gi:02d}-{variant}'
            key = fingerprint({'new_multiday_training_catalog': cid})[:16]
            start = date(2028, 2, 8) + timedelta(days=gi*5+variant*110)
            dates = [(start+timedelta(days=d)).isoformat() for d in range(days)]
            pois = [dict(id=f'place-{key}-{i}', name=city+'合成'+name, category='attraction', tags=list(tags),
                ticket_price=price+variant*3, duration_minutes=duration, open_time='08:00', close_time='19:30',
                lat=lat+i*.0013, lng=lng+i*.0009, score=.89-i*.009, fixture_record=True,
                date_opening_hours={d: ['08:00', '19:30'] for d in dates},
                source_url=f'https://frozen.example.invalid/{key}/p{i}', description='合成场馆：'+name)
                for i, (name, tags, price, duration) in enumerate(VENUES)]
            restaurants = [dict(id=f'place-{key}-r{i}', name=city+'合成'+name, category='restaurant',
                tags=['餐厅', '美食', '家常菜'], ticket_price=0, average_cost=price+variant*2, duration_minutes=45,
                open_time=opening, close_time=closing, lat=lat+.002+i*.0012, lng=lng+.003+i*.001,
                address=city+'虚构里'+str(i+1)+'号', fixture_record=True,
                source_url=f'https://frozen.example.invalid/{key}/r{i}')
                for i, (name, price, opening, closing) in enumerate([
                    ('谷香午餐屋', 27, '10:30', '14:30'), ('星灯晚膳屋', 39, '17:00', '21:30'),
                    ('锦庭餐轩', 185, '10:00', '21:30')])]
            extra = ''
            if gi == 8:
                pois[0].update(close_time='12:30', date_opening_hours={d: ['08:00', '12:30'] for d in dates})
                extra = '纸艺场馆只在上午接待，请核实后安排。'
            if gi == 9:
                pois[3].update(open_time='11:00', date_opening_hours={d: ['11:00', '19:30'] for d in dates})
                extra = '磁力场馆上午较晚开放，请以工具事实为准。'
            if gi == 10:
                pois[0].update(closed_dates=dates, date_opening_hours={})
                pois[1]['ticket_price'] = 1800
                extra = '非必去场馆可舍弃，避开停业或超预算的项目。'
            budget = 1800 if days == 3 else 1000 if days == 2 else 550
            slots = dict(destination=city, travel_days=days, start_date=dates[0], end_date=dates[-1],
                budget_range=budget, interests=list(interests), must_visit=[pois[i]['name'] for i in required],
                require_named_restaurants=True)
            if gi == 11:
                slots['must_not_visit'] = [pois[0]['name']]; extra = f"不要安排{pois[0]['name']}。"
            if scenario == 'weather-timeout':
                slots['information_needs'] = ['weather']; extra = '请核实当天的天气。'
            if scenario == 'stale-hours':
                slots['information_needs'] = ['opening_hours']
                slots['current_info_queries'] = [f"{pois[3]['name']} {dates[0]} 开放时间"]
                extra = f"请用有效且属于该场馆的资料核实{pois[3]['name']}当天开放时间。"
            request = (f"从{dates[0]}开始在{city}游玩{days}天，总预算{budget}元，偏好{'、'.join(interests)}，"
                f"必须入内参观{'、'.join(slots['must_visit'])}。每天安排具名午餐和晚餐，核实餐厅营业时段、餐费及通勤。"+extra)
            c = dict(id=cid, dataset_version=VERSION, split='train', family='recovery' if gi >= 12 else 'normal',
                scenario=scenario, variant=variant, request=request, slots=slots, missing_slots=[], missing_field=None,
                expected='plan', pois=pois, restaurants=restaurants, source_group='multiday-training-composition:'+group,
                lineage_group='multiday-training-composition:'+group, synthetic=True, reference_time='2026-09-05T00:00:00+00:00',
                provider_faults=[], provider_query_facts=[], training_targets_present=False,
                provenance=dict(kind='newly_authored_planning_only_training_catalog', parent_instance_id=None,
                    catalog_blueprint_sha256=fingerprint(VENUES), variants_share_composition=True, live_facts=False,
                    intent_slots='predeclared_from_request', training_authorized=True,
                    limits='80 instances in 20 related compositions; shared tool skills; synthetic facts; no dev checkpoints'))
            c['source_case_sha256'] = fingerprint(c); cases.append(c)
    return cases
