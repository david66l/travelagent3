from scripts.benchmark_pure_agent_vs_verified import (
    build_catalog,
    build_report,
    parse_final_itinerary,
    search_catalog,
    stratified_sample,
    validate_itinerary,
)


def test_stratified_sample_covers_cities_before_repeating():
    cases = [
        {"case_id": f"{city}-{index}", "destination": city, "family": "solvable_plan"}
        for city in ("A", "B", "C")
        for index in range(3)
    ]
    selected = stratified_sample(cases, 3, 7)
    assert {row["destination"] for row in selected} == {"A", "B", "C"}


def test_catalog_search_and_grounded_validation_contract():
    catalog = build_catalog("北京")
    selected = search_catalog(catalog, ["历史"], 4)
    assert len(selected) == 4
    poi = selected[0]
    itinerary = [
        {
            "day_number": 1,
            "date": "2026-09-23",
            "activities": [
                {
                    "poi_id": poi["id"],
                    "poi_name": poi["name"],
                    "category": poi["category"],
                    "start_time": "10:00",
                    "end_time": "11:00",
                    "duration_min": 60,
                    "ticket_price": poi["ticket_price"],
                }
            ],
            "total_cost": poi["ticket_price"],
        }
    ]
    case = {"travel_days": 1, "budget": 3000, "interests": ["历史"]}
    validator_pass, grounding_pass, route_pass, codes = validate_itinerary(case, itinerary, catalog)
    assert validator_pass
    assert grounding_pass
    assert route_pass
    assert codes == []


def test_report_uses_provider_total_tokens_and_both_percent_directions():
    base = {
        "schema_version": "pure-agent-vs-verified.v1",
        "destination": "北京",
        "input_hash": "same",
        "model": "test",
        "status": "completed",
        "hard_pass": True,
        "validator_hard_pass": True,
        "grounding_pass": True,
        "route_pass": True,
        "cached_prompt_tokens": 0,
        "model_calls": 1,
        "tool_calls": 1,
        "latency_ms": 100,
        "itinerary_days": 1,
        "activity_count": 1,
        "solver_calls": 0,
        "solver_status": None,
        "violation_codes": [],
        "error": None,
        "trace": [],
        "itinerary": [],
    }
    rows = [
        {
            **base,
            "case_id": "a",
            "mode": "pure_agent",
            "prompt_tokens": 800,
            "completion_tokens": 200,
            "total_tokens": 1000,
        },
        {
            **base,
            "case_id": "a",
            "mode": "verified_planner",
            "prompt_tokens": 400,
            "completion_tokens": 100,
            "total_tokens": 500,
        },
    ]
    report = build_report(rows, model="test", seed=1, requested_size=1)
    assert report["paired_delta"]["pure_token_excess_vs_verified_percent"] == 100
    assert report["paired_delta"]["verified_token_reduction_vs_pure_percent"] == 50


def test_final_itinerary_parser_skips_prose_and_trailing_json():
    content = '计划如下：\n{"itinerary":[{"day_number":1,"activities":[]}]}\n{"note":"done"}'
    assert parse_final_itinerary(content) == [{"day_number": 1, "activities": []}]
