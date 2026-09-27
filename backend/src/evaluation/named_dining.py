"""Independent source-based validation for named dining and routed transitions."""
from math import isclose


def validate_named_dining(activities, facts, config, minutes):
    problems=[]
    def add(code,message,activity=None):
        problems.append(dict(code=code,message=message,poi_id=(activity or {}).get('poi_id')))
    meals=[a for a in activities if a.get('category') in {'restaurant','meal'}]
    count=int(config.get('meals_per_day') or 2)
    if len(meals)!=count:add('NAMED_MEAL_COUNT_MISMATCH',f'需要{count}次具名餐厅用餐，实际{len(meals)}次')
    windows=[config.get('lunch_window',(690,810)),config.get('dinner_window',(1050,1200))][:count]
    for window in windows:
        matches=[a for a in meals if minutes(a.get('start_time')) is not None and minutes(a.get('end_time')) is not None
                 and window[0]<=minutes(a['start_time'])<minutes(a['end_time'])<=window[1]]
        if len(matches)!=1:add('NAMED_MEAL_WINDOW_MISMATCH','每个用餐时段必须恰好安排一餐')
    for a in meals:
        fact=facts.get(str(a.get('poi_id') or '')) or {}
        if fact.get('category')!='restaurant' or fact.get('name')!=a.get('poi_name') or not fact.get('lat') or not fact.get('lng'):
            add('UNGROUNDED_RESTAURANT','餐厅必须有同实体名称和坐标证据',a)
        if fact.get('average_cost') is None or not isclose(float(a.get('meal_cost',0))+float(a.get('ticket_price',0)),float(fact.get('average_cost') or 0),abs_tol=.011):
            add('RESTAURANT_COST_MISMATCH','餐费必须与源目录一致',a)
        start,end=minutes(a.get('start_time')),minutes(a.get('end_time'))
        if start is not None and end is not None and end-start<int(fact.get('duration_minutes') or 30):
            add('RESTAURANT_DURATION_TOO_SHORT','用餐时长不足',a)
    ids=config.get('route_poi_ids') or [];times=config.get('route_time_matrix') or [];costs=config.get('route_cost_matrix') or []
    positions={v:i for i,v in enumerate(ids)}
    if not ids or any(len(m)!=len(ids) or any(len(row)!=len(ids) for row in m) for m in (times,costs)):
        add('MISSING_DINING_ROUTE_EVIDENCE','需要完整的餐厅路线矩阵');return problems
    previous='__hotel';end=None
    for a in sorted(activities,key=lambda x:minutes(x.get('start_time')) or 0):
        current=a.get('poi_id');start=minutes(a.get('start_time'))
        if current not in positions or previous not in positions:
            add('DINING_ROUTE_ENTITY_MISSING','路线矩阵缺少活动实体',a)
        else:
            i,j=positions[previous],positions[current]
            transit=a.get('transit_from_prev') or {}
            if transit.get('poi_id')!=previous or transit.get('duration_min')!=times[i][j] or not isclose(float(a.get('transport_cost',0)),costs[i][j],abs_tol=.011):
                add('DINING_ROUTE_COST_TIME_MISMATCH','通勤时间或费用与源矩阵不一致',a)
            if end is not None and start is not None and start-end<times[i][j]:
                add('INSUFFICIENT_TRANSIT_GAP','活动之间没有足够通勤时间',a)
        previous=current;end=minutes(a.get('end_time'))
    return problems
