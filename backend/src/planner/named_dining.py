"""Source-preserving dining alternatives: real routed nodes per day/meal slot."""
from vrp_solver_service.models import POIInput


def expand_dining_nodes(pois, constraints):
    restaurants=[p for p in pois if p.category=='restaurant']
    result=[p for p in pois if p.category!='restaurant']
    for d in range(constraints.travel_days):
        for slot in range(min(constraints.meals_per_day,2)):
            for index,poi in enumerate(restaurants):
                if poi.average_cost is None or not poi.lat or not poi.lng:
                    continue
                result.append(poi.model_copy(deep=True,update={
                    'id':f'__dining_d{d+1}_m{slot+1}_r{index}',
                    'source_poi_id':poi.id,'dining_day':d,'dining_slot':slot,
                    'ticket_price':poi.average_cost,'must_visit':False}))
    return result


def expand_source_matrices(source_pois, expanded_pois, times, costs):
    """Reuse every measured source edge when restaurant visits are duplicated."""
    ids=['__hotel',*[p.id for p in source_pois]]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate source POI IDs')
    if any(len(m)!=len(ids) or any(len(r)!=len(ids) for r in m) for m in (times,costs)):
        raise ValueError('Source route matrices do not match supplied POI order')
    positions={key:i for i,key in enumerate(ids)}
    index=[positions[p.source_poi_id or p.id] for p in expanded_pois]
    return ([[times[i][j] for j in index] for i in index],[[costs[i][j] for j in index] for i in index])
