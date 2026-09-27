"""Current external evidence selection shared by verification and solver hydration."""
from datetime import UTC,datetime,timedelta
import re

from agentic.clock import reference_now


def fresh_searches(ledger):
    now=reference_now();result=[]
    for artifact in ledger.artifacts.values():
        if (artifact.artifact_type!='current_info_search'
                or artifact.goal_version!=ledger.goal.goal_version
                or artifact.plan_version!=ledger.task_graph.plan_version
                or (artifact.expires_at is not None and artifact.expires_at<=now)):
            continue
        p=artifact.payload
        if p.get('_evidence_source') in {'fallback','unavailable'} or p.get('_is_fallback') is True:continue
        try:
            stamp=datetime.fromisoformat(str(p.get('queried_at') or p.get('retrieved_at')))
            if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=UTC)
        except (ValueError,TypeError):continue
        if now-timedelta(hours=6)<stamp<=now+timedelta(minutes=5):result.append(artifact)
    return result


def source_results(artifact):
    return [r for r in artifact.payload.get('results') or [] if isinstance(r,dict) and r.get('url')]


def required_opening_targets(ledger,candidates):
    """Entity/date requirements come from user constraints, not model search order."""
    from agentic.react import _entity_key
    hard=ledger.goal.hard_constraints
    if not set(hard.get('information_needs') or []) & {'opening_hours','closure'}:return []
    names=list(dict.fromkeys([str(n) for n in hard.get('must_visit') or []]+
        [p['name'] for p in candidates if isinstance(p,dict) and p.get('name')]))
    queries=hard.get('current_info_queries') or []
    targets=[]
    for q in queries:
        dates=re.findall(r'20\d\d-\d\d-\d\d',str(q)) or [str(hard.get('start_date') or '')]
        for n in names:
            if _entity_key(n) and _entity_key(n) in _entity_key(q):targets.extend((n,d) for d in dates)
    if not targets:
        targets=[(str(n),str(hard.get('start_date') or '')) for n in hard.get('must_visit') or []]
    return list(dict.fromkeys(targets))


def opening_target_covered(artifact,name,day):
    from agentic.react import _entity_key
    if artifact.payload.get('info_type') not in {'opening_hours','closure','general'}:return False
    for result in source_results(artifact):
        text=f"{result.get('title') or ''} {result.get('snippet') or ''}"
        if _entity_key(name) not in _entity_key(text):continue
        dates=re.findall(r'20\d\d-\d\d-\d\d',text)
        if day and ((dates and day not in dates) or (not dates and str(artifact.payload.get('date') or '')!=day)):continue
        hours=re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d',text)
        if len(hours)>=2 or any(w in text for w in ('临时闭馆','暂停开放','闭馆','停业','关闭')):return True
    return False
