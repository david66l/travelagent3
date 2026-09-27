"""Argument-sensitive synthetic database for development and data collection.

No live facts: each record belongs to a versioned fixture. Retrieval never
selects the agent's next action. Empty matches remain empty.
"""
from copy import deepcopy
from datetime import timedelta
import re

FIXTURE_REVISION = "research-database-v4"
DINING = ("餐厅", "餐馆", "餐饮", "午餐", "晚餐", "吃饭", "本帮菜", "美食", "面馆", "素食")


def catalog(case):
    if "restaurants" in case:
        return deepcopy(case["pois"] + case["restaurants"])
    city = case["slots"].get("destination") or "示例城"
    anchor = case["pois"][0]
    restaurants = [dict(id=f"dining-{i}", name=f"{city}{name}", category="restaurant",
        lat=anchor["lat"]+.001*i, lng=anchor["lng"]+.001*i, ticket_price=0,
        average_cost=price, open_time=opening, close_time=closing,
        address=f"{city}示例街{20+i}号", tags=["餐厅", "美食", cuisine],
        source_url=f"https://frozen.example.invalid/{city}/dining-{i}",
        fixture_record=True) for i,(name,price,opening,closing,cuisine) in enumerate([
            ("巷口家常菜馆",55,"10:30","21:00","家常菜"),
            ("河畔素食餐厅",45,"11:00","20:30","素食")],1)]
    return deepcopy(case["pois"]) + restaurants


def _matches(record, words):
    corpus = " ".join([record["name"], *record.get("tags", [])])
    return any(word and (word in corpus or record["name"] in word) for word in words)


def search_pois(case, args):
    if args.get("city") and args["city"] != case["slots"].get("destination"):
        return []
    records=catalog(case); words=[str(w).strip() for w in args.get("keywords") or []]
    required=set(args.get("required_pois") or [])
    dining=any(word in DINING or any(t in word for t in ("餐厅推荐","餐厅 推荐","附近餐厅","附近 餐厅")) for word in words)
    vegetarian=any("素食" in word for word in words)
    return [r for r in records if r["name"] in required or r["id"] in required
        or ((not vegetarian or r.get("category")!="restaurant" or "素食" in r.get("tags",[]))
            and (not words or _matches(r,words)
                 or (dining and r.get("category")=="restaurant")))]


def detail(case, name):
    found=[r for r in catalog(case) if r["name"]==name]
    if len(found)!=1:
        raise ValueError("Requested POI is not uniquely present in this frozen database")
    return found[0]


def current_info(case,args,moment,*,stale=False):
    query=str(args.get("query") or "")
    info_type=args.get("info_type","general")
    day=str(args.get("date") or case["slots"].get("start_date") or "")
    query_dates=re.findall(r"20\d\d-\d\d-\d\d",query)
    if query_dates and day not in query_dates:
        return {"query":query,"info_type":info_type,"date":day,"queried_at":moment.isoformat(),"results":[]}
    stamp=moment-timedelta(days=3) if stale else moment
    dining=info_type=="restaurant" or any(w in query for w in DINING)
    records=catalog(case)
    if args.get("city") and args["city"]!=case["slots"].get("destination"):
        records=[]
    if dining:
        # A city-only restaurant request can browse that city's dining catalog.
        city=case["slots"].get("destination") or ""
        browse=bool(city and city in query and (
            any(w in query for w in ("推荐", "附近", "午餐", "晚餐", "美食", "素食"))
            or re.fullmatch(re.escape(city)+r"\s*餐厅",query)))
        records=[r for r in records if r.get("category")=="restaurant"
            and (r["name"] in query or browse)]
        if "素食" in query:records=[r for r in records if "素食" in r.get("tags",[])]
    else:
        records=[r for r in records if r.get("category")!="restaurant" and r["name"] in query]
    results=[]
    for r in records:
        closed=day in r.get("closed_dates",[])
        availability="当日临时闭馆，无法入馆参观" if closed else f"开放时间{r['open_time']}-{r['close_time']}"
        if r.get("category")=="restaurant":
            availability="当日停业" if closed else f"营业时间{r['open_time']}-{r['close_time']}"
            snippet=f"{r['name']} {day} {availability}；人均{r['average_cost']}元；地址{r['address']}；位置{r['lat']},{r['lng']}。"
        else:snippet=f"{r['name']} {day} {availability}；门票{r.get('ticket_price',0)}元。"
        results.append({"title":r["name"],"snippet":snippet,
            "url":r.get("source_url") or f"https://frozen.example.invalid/{r['id']}/hours",
            "published_at":stamp.isoformat(),"score":.99})
    return {"query":query,"info_type":info_type,"date":day,"queried_at":stamp.isoformat(),"results":results}
