import json
import pytest
from core.api_budget import SharedAPIBudget,BudgetExceeded


def config():
    return dict(currency='CNY',limit_micros=100_000_000,input_cny_per_million='2.8',
        output_cny_per_million='2.8',max_input_tokens=1_048_576,max_output_tokens=8192,
        max_requests=3000,model='glm-5.3-flash',rate_evidence='user supplied 2026-09-06; maximum published tier')


def test_unknown_and_crashed_requests_stay_charged_across_restart(tmp_path):
    path=tmp_path/'budget.json';budget=SharedAPIBudget(path,config())
    first=budget.reserve(8192,'batch1');budget.settle(first,None,'provider_error')
    second=budget.reserve(8192,'batch1')
    restored=SharedAPIBudget(path,config());state=restored.snapshot()
    assert state['requests'][second]['status']=='reserved'
    assert sum(r['charged_micros'] for r in state['requests'].values())==2*budget.cost(1_048_576,8192)


def test_actual_usage_releases_excess_and_all_batches_share_limit(tmp_path):
    budget=SharedAPIBudget(tmp_path/'budget.json',config())
    first=budget.reserve(8192,'batch1');budget.settle(first,{'prompt_tokens':1000,'completion_tokens':500},'success')
    assert budget.snapshot()['requests'][first]['charged_micros']==4200
    accepted=0
    while True:
        try:budget.reserve(8192,'batch2');accepted+=1
        except BudgetExceeded:break
    assert accepted>0
    state=budget.snapshot()
    assert sum(r['charged_micros'] for r in state['requests'].values())<=100_000_000
    with pytest.raises(BudgetExceeded):SharedAPIBudget(budget.path,config()).reserve(8192,'batch3')


def test_changed_price_and_duplicate_settlement_are_rejected(tmp_path):
    budget=SharedAPIBudget(tmp_path/'budget.json',config())
    with pytest.raises(ValueError):SharedAPIBudget(budget.path,{**config(),'output_cny_per_million':'3'})
    request=budget.reserve(8192,'case');budget.settle(request,{},'unknown')
    with pytest.raises(ValueError):budget.settle(request,{},'retry')


def test_provider_bound_violation_freezes_program(tmp_path):
    budget=SharedAPIBudget(tmp_path/'budget.json',config())
    request=budget.reserve(8192,'case')
    with pytest.raises(BudgetExceeded):budget.settle(request,{'prompt_tokens':2_000_000,'completion_tokens':1},'success')
    assert budget.snapshot()['frozen']
    with pytest.raises(BudgetExceeded):budget.reserve(8192,'next')


def test_mock_api_failure_cannot_refund_previous_request_usage(tmp_path):
    import asyncio,httpx
    from core.api_budget import BudgetedGLMToolClient
    budget=SharedAPIBudget(tmp_path/'budget.json',config())
    attempts=0
    def respond(request):
        nonlocal attempts
        attempts+=1
        if attempts==2:return httpx.Response(500,json={'error':{'code':'test'}})
        return httpx.Response(200,json={'usage':{'prompt_tokens':100,'completion_tokens':20},
            'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'function':{'name':'finish','arguments':'{}'}}]}}]})
    async def run():
        client=BudgetedGLMToolClient(api_key='mock-key-not-real',budget=budget,budget_task='mock',transport=httpx.MockTransport(respond))
        try:
            await client.tool_call([],[],model_override='glm-5.3-flash')
            with pytest.raises(RuntimeError):await client.tool_call(messages=[],tools=[],model_override='glm-5.3-flash')
        finally:await client.aclose()
    asyncio.run(run())
    rows=list(budget.snapshot()['requests'].values())
    assert rows[0]['charged_micros']==336
    assert rows[1]['charged_micros']==rows[1]['reserved_micros']


def test_parallel_reservations_cannot_overbook_shared_limit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    budget=SharedAPIBudget(tmp_path/'budget.json',config())
    def reserve(index):
        try:return budget.reserve(8192,str(index))
        except BudgetExceeded:return None
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(reserve,range(80)))
    rows=budget.snapshot()['requests']
    assert 0<len(rows)<80
    assert len(rows)==sum(r is not None for r in results)
    assert sum(r['charged_micros'] for r in rows.values())<=100_000_000
