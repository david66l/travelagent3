"""Conservative infeasibility witnesses over the supplied constraint facts.

These are sufficient conditions, not a solver and not an action policy. Absence
of a witness means undetermined. A rejected candidate plan is never a proof.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import math
from typing import Any

FEASIBILITY_SCHEMA = "required-facts-feasibility.v1"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def _witness(code, dimension, message, basis):
    payload = {"code": code, "dimension": dimension, "message": message, "basis": basis}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return {"evidence_id": "feasibility:" + digest, **payload}


def assess_required_facts(constraints: dict, facts: list[dict]) -> dict[str, Any]:
    # Duplicate aliases for one record must not double-count mandatory cost.
    records = {}
    for fact in facts:
        if isinstance(fact, dict):
            identity = str(fact.get("id") or fact.get("poi_id") or fact.get("name") or "")
            if identity:
                records.setdefault(identity, []).append(fact)
    required = {}
    unresolved = []
    for target in dict.fromkeys(str(x) for x in constraints.get("must_visit") or []):
        matched = [key for key, group in records.items() if any(target in {
            str(f.get("id") or ""), str(f.get("poi_id") or ""),
            str(f.get("name") or ""), str(f.get("poi_name") or "")
        } for f in group)]
        if len(matched) != 1:
            unresolved.append(target)
            continue
        key = matched[0]
        group = records[key]
        # Conflicting structured records cannot establish a hard stop.
        signatures = {json.dumps({k:f.get(k) for k in (
            "ticket_price", "closed_dates", "closed_weekdays", "date_opening_hours"
        )}, sort_keys=True, default=str) for f in group}
        if len(signatures) != 1:
            unresolved.append(target)
            continue
        required[key] = group[0]
    witnesses = []
    try:
        start = date.fromisoformat(str(constraints.get("trip_start_date")))
        days = int(constraints.get("travel_days") or 0)
        # Unknown or unreasonable horizons are not certificates.
        dates = [start + timedelta(days=i) for i in range(days)] if 0 < days <= 366 else []
    except (TypeError, ValueError, OverflowError):
        dates = []
    for identity, fact in required.items():
        closed = set(fact.get("closed_dates") or [])
        overrides = fact.get("date_opening_hours") or {}
        # Use explicit dated closure facts only. Planner preprocessing may
        # infer weekly museum closures; assumptions cannot certify a stop.
        if dates and all(
            d.isoformat() in closed
            and d.isoformat() not in overrides for d in dates
        ):
            names = str(fact.get("name") or identity)
            date_text = dates[0].isoformat() if len(dates) == 1 else f"{dates[0].isoformat()}至{dates[-1].isoformat()}"
            witnesses.append(_witness("REQUIRED_POI_CLOSED_ALL_DATES", "activity_set",
                f"必去场馆{names}在{date_text}的全部可用旅行日期均闭馆。",
                {"poi_id":identity,"poi_name":names,"dates":[d.isoformat() for d in dates],
                 "closed_dates":[d.isoformat() for d in dates]}))
    budget = _number(constraints.get("total_budget"))
    costs = [(key, fact, _number(fact.get("ticket_price"))) for key,fact in required.items()]
    # A known subset alone can exceed the budget; unknown costs add no lower bound.
    payable = [(key,f,c) for key,f,c in costs if c is not None and c > 0]
    lower_bound = sum(c for _,_,c in payable)
    if budget and lower_bound > budget + 1e-6:
        names = "、".join(str(f.get("name") or key) for key,f,_ in payable)
        witnesses.append(_witness("REQUIRED_COST_EXCEEDS_BUDGET", "total_budget",
            f"必去项{names}的门票合计至少{lower_bound:g}元，已经超过总预算{budget:g}元。",
            {"required_cost_lower_bound":lower_bound,"total_budget":budget,
             "items":[{"poi_id":key,"poi_name":str(f.get("name") or key),"ticket_price":c} for key,f,c in payable]}))
    return {"schema_version":FEASIBILITY_SCHEMA,
            "status":"infeasible" if witnesses else "undetermined",
            "scope":"current_required_facts", "witnesses":witnesses,
            "unresolved_required_entities":unresolved}
