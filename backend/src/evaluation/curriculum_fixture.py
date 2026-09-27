"""Catalog-first synthetic train curriculum; never a source of policy targets.

Run construction and validation on the cloud. Entity facts are declared before
requests; tags describe the catalog, not the requested preference. Geographic
coordinates and all venue records are simulated, not live travel information.
"""
from copy import deepcopy
from datetime import datetime, timezone

from evaluation import research_fixture
from evaluation.migration_fixture import MigrationResearchExecutor, fingerprint
from schemas import ToolResult

VERSION = "travel-curriculum100.v2"
MOMENT = datetime(2026, 9, 5, tzinfo=timezone.utc)
# Fixed venue types, independent of the user request.
CATALOG = [
    ("历史博物馆", ["历史", "博物馆", "文化"], 40, 90),
    ("现代艺术馆", ["艺术", "美术馆", "文化"], 60, 90),
    ("滨河公园", ["自然", "公园", "散步"], 0, 60),
    ("植物园", ["自然", "植物", "公园"], 20, 90),
    ("建筑展览馆", ["建筑", "历史", "文化"], 35, 60),
    ("科学探索馆", ["科学", "科技", "亲子"], 50, 90),
]


def build_cases():
    cities = [("上海",31.23,121.47),("杭州",30.25,120.15),
              ("成都",30.66,104.06),("北京",39.90,116.40),("苏州",31.30,120.58)]
    scenarios = [
        ("normal", "history", ["历史"], 0),
        ("normal", "art", ["艺术"], 1),
        ("normal", "nature", ["自然"], 2),
        ("normal", "botany", ["植物"], 3),
        ("normal", "architecture", ["建筑"], 4),
        ("normal", "science", ["科学"], 5),
        ("normal", "mixed-culture", ["历史", "艺术"], 0),
        ("normal", "avoid-venue", ["自然"], 2),
        ("recovery", "search-timeout", ["历史"], 0),
        ("recovery", "detail-timeout", ["艺术"], 1),
        ("recovery", "matrix-timeout", ["自然"], 2),
        ("recovery", "weather-timeout", ["植物"], 3),
        ("recovery", "stale-hours", ["建筑"], 4),
        ("recovery", "wrong-entity", ["科学"], 5),
        ("constraint", "required-closed", ["历史"], 0),
        ("constraint", "hard-budget", ["艺术"], 1),
        ("constraint", "optional-closed", ["自然"], 2),
        ("constraint", "budget-tradeoff", ["科学"], 5),
        ("clarification", "missing-destination", [], 0),
        ("clarification", "missing-days", ["历史"], 0),
    ]
    cases = []
    for family, scenario, interests, required_index in scenarios:
        for variant, (city, lat, lng) in enumerate(cities):
            case_id = f"b01-{scenario}-{variant+1:02d}"
            # Scenario names are private audit metadata. Provider-visible IDs
            # and URLs must not reveal the hidden fault or expected terminal.
            public_key = fingerprint({'catalog_instance':case_id})[:16]
            # Recovery variants use a separate travel window; identical requests
            # must not masquerade as distinct task instances.
            day = f"2026-09-{[18,19,20,22,23][variant]+(7 if family == 'recovery' else 0):02d}"
            pois = []
            for i, (name, tags, price, duration) in enumerate(CATALOG):
                pois.append(dict(id=f"poi-{public_key}-{i}", name=f"{city}示例{name}",
                    category="attraction", tags=list(tags), ticket_price=price,
                    duration_minutes=duration, open_time="08:00", close_time="21:00",
                    lat=lat+i*.002, lng=lng+i*.002, score=.9-i*.01,
                    source_url=f"https://curriculum.example.invalid/{public_key}/p{i}",
                    fixture_record=True, closed_weekdays=[0] if i in {0,4} else [],
                    date_opening_hours={day:['08:00','21:00']},
                    description=f"合成场馆：{name}，提供{'、'.join(tags)}主题体验。"))
            restaurants = [dict(id=f"restaurant-{public_key}-{i}", name=f"{city}示例{name}",
                category="restaurant", tags=["餐厅", "美食", cuisine], ticket_price=0,
                average_cost=price, duration_minutes=60, open_time="10:00", close_time="22:00",
                lat=lat+.003+i*.001, lng=lng+.003+i*.001,
                address=f"{city}虚构街{i+1}号", fixture_record=True,
                source_url=f"https://curriculum.example.invalid/{public_key}/r{i}")
                for i,(name,cuisine,price) in enumerate([("家常菜馆","家常菜",45),("素食餐厅","素食",35)])]
            required = pois[required_index]["name"]
            budget = 1000 + variant*100
            slots = dict(destination=city, travel_days=1, start_date=day, end_date=day,
                         budget_range=budget, interests=list(interests), must_visit=[required])
            request = f"{day}在{city}玩一天，总预算{budget}元，喜欢{'、'.join(interests)}，必须进入{required}参观。请安排含午晚餐的完整行程。"
            expected, missing = "plan", []
            if scenario == "avoid-venue":
                slots['must_not_visit'] = [pois[1]['name']]
                request += f"不要安排{pois[1]['name']}。"
            if scenario == "weather-timeout":
                slots['information_needs'] = ['weather']
                request += "请核实当天的天气。"
            if scenario in {'stale-hours','wrong-entity'}:
                slots['information_needs'] = ['opening_hours']
                slots['current_info_queries'] = [f"{required} {day} 开放时间"]
                request += f"请核实{required}当天的开放时间，用有效且属于该场馆的资料。"
            if scenario == 'required-closed':
                pois[required_index]['closed_dates'] = [day]
                pois[required_index]['date_opening_hours'] = {}
                expected = 'grounded_stop'
                request += "日期、天数和必去项均不可更改，也不接受外观打卡。"
            if scenario in {'hard-budget','budget-tradeoff'}:
                pois[required_index]['ticket_price'] = 800
                slots['budget_range'] = 300
                request = request.replace(f'总预算{budget}元','总预算300元')
                expected = 'grounded_stop'
                if scenario == 'hard-budget':
                    request += "预算是硬上限，日期和必去项也不能更改。"
                else:
                    slots['constraint_flexibility'] = dict(schema_version='constraint-flexibility.v1',
                        locked_constraints=['activity_set'], solver_adjustable_constraints=['activity_schedule'],
                        relaxable_constraints=['total_budget'],
                        relaxation_options={'total_budget':['将总预算提高至1200元']})
                    request += "若预算不足，可以向我提出将总预算提高至1200元的方案，由我确认；不可擅自更改。"
            if scenario == 'optional-closed':
                pois[0]['closed_dates'] = [day]
                pois[0]['date_opening_hours'] = {}
                request += "其他场馆只是可选项，闭馆的请跳过。"
            if scenario == 'missing-destination':
                slots.pop('destination'); slots.pop('must_visit'); slots.pop('interests')
                missing = ['destination']; expected = 'clarify'
                request = f"{day}想出去玩一天，预算{budget}元，目的地还没确定，帮我规划。"
            if scenario == 'missing-days':
                slots.pop('travel_days'); slots.pop('end_date')
                missing = ['travel_days']; expected = 'clarify'
                request = f"{day}开始去{city}玩，预算{budget}元，喜欢历史，必须去{required}，还没决定待几天。"
            case = dict(id=case_id, family=family, scenario=scenario, variant=variant,
                request=request, slots=slots, missing_slots=missing, missing_field=missing[0] if missing else None,
                expected=expected, pois=pois, restaurants=restaurants, split='train',
                source_group=f'curriculum-template:{scenario}', synthetic=True,
                dataset_version=VERSION, reference_time=MOMENT.isoformat(),
                provider_faults=[], provider_query_facts=[], training_targets_present=False,
                provenance={'kind':'synthetic_catalog_first','catalog_sha256':fingerprint(CATALOG),
                    'group_unit':'scenario_template','variants_share_template':True,
                    'intent_slots':'predeclared_from_request','live_facts':False})
            case['source_case_sha256'] = fingerprint(case)
            cases.append(case)
    return cases


class CurriculumResearchExecutor(MigrationResearchExecutor):
    """Closed provider registry with one observable transient fault per recovery case."""
    def __init__(self, case):
        super().__init__(case)
        self.fault_events = []
        self.provider_attempts = {}
        self._handlers['get_weather'] = self._curriculum_weather
        self._handlers['search_current_info'] = self._info
        self._handlers['retrieve_city_knowledge'] = self._curriculum_knowledge
        fault_tool = {'search-timeout':'search_pois','detail-timeout':'get_poi_detail',
                      'matrix-timeout':'get_route_matrix','weather-timeout':'get_weather'}.get(case['scenario'])
        if fault_tool:
            original = self._handlers[fault_tool]
            async def flaky(args):
                self.provider_attempts[fault_tool] = self.provider_attempts.get(fault_tool, 0)+1
                if self.provider_attempts[fault_tool] == 1:
                    self.fault_events.append({'tool':fault_tool,'kind':'timeout','attempt':1})
                    raise TimeoutError('Synthetic provider timed out; a subsequent request may succeed')
                return await original(args)
            self._handlers[fault_tool] = flaky

    async def _curriculum_knowledge(self, args):
        self.counts['retrieve_city_knowledge'] += 1
        return ToolResult(data={'city':self.case['slots'].get('destination'),
            'summary':'离线合成旅行环境，下列资料仅在本场景内有效。',
            'pois':research_fixture.catalog(self.case)},data_source='built_in',confidence=1.0)

    async def _curriculum_weather(self, args):
        self.counts['get_weather'] += 1
        return ToolResult(data=[{'date':args.get('date') or self.case['slots']['start_date'],
            'condition':'晴','temperature':'20-27℃'}],data_source='built_in',confidence=1.0)

    async def _info(self, args):
        self.counts['search_current_info'] += 1
        scenario = self.case['scenario']
        first = self.counts['search_current_info'] == 1
        stale = first and scenario == 'stale-hours'
        data = research_fixture.current_info(self.case,args,MOMENT,stale=stale)
        if first and scenario == 'wrong-entity':
            other = next(p for p in self.case['pois'] if p['name'] not in str(args.get('query','')))
            alternate = dict(args,query=f"{other['name']} {self.case['slots']['start_date']} 开放时间")
            data = research_fixture.current_info(self.case,alternate,MOMENT)
        if first and scenario in {'stale-hours','wrong-entity'}:
            self.fault_events.append({'tool':'search_current_info','kind':scenario,'attempt':1})
        return ToolResult(data=data,data_source='built_in',confidence=1.0)


def quality_audit(case, summary, backend):
    preference = ((summary.get('validation') or {}).get('soft_scores') or {}).get('preference_match')
    reasons = []
    if not summary['criterion_passed']: reasons.append('TASK_CRITERION_FAILED')
    if summary['reward']['gate_status'] != 'passed': reasons.append('REWARD_GATE_FAILED')
    if summary.get('replay_errors'): reasons.append('REPLAY_FAILED')
    if summary.get('parse_or_schema_errors'): reasons.append('RAW_POLICY_ERROR')
    if case['expected'] == 'plan' and (preference is None or preference < .5):
        reasons.append('PREFERENCE_MATCH_BELOW_0_5')
    if case['family'] == 'recovery' and not backend.fault_events:
        reasons.append('CONFIGURED_FAULT_NOT_EXERCISED')
    if case['scenario'] == 'budget-tradeoff' and (not summary['actions'] or summary['actions'][-1] != 'propose_tradeoff'):
        reasons.append('AUTHORIZED_CHOICE_NOT_PROPOSED')
    return dict(source_group=case['source_group'], scenario=case['scenario'],
        preference_match=preference, fault_events=deepcopy(backend.fault_events),
        rejection_reasons=reasons, quality_candidate=not reasons,
        training_ready=False, review_status='candidate_only_not_exported')
