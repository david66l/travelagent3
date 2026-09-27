import asyncio,json
import httpx,pytest
from core.glm_tool_client import GLMToolClient,GLMProviderError
from core.api_budget import SharedAPIBudget,BudgetExceeded
from core.teacher_queue import TeacherQueueCircuit,QueueBudgetedGLMToolClient

def config():
    return dict(currency='CNY',limit_micros=100_000_000,input_cny_per_million='0.8',output_cny_per_million='2.8',
        max_input_tokens=1_048_576,max_output_tokens=8192,max_requests=3000,model='glm-5.3-flash',rate_evidence='test')

def reply():return dict(usage=dict(prompt_tokens=100,completion_tokens=20),choices=[dict(finish_reason='tool_calls',
    message=dict(tool_calls=[dict(function=dict(name='finish',arguments='{}'))]))])

@pytest.mark.asyncio
async def test_effort_is_explicit_in_request_and_rejects_invalid_before_network():
    seen=[]
    for effort in ['max','high','low']:
        client=GLMToolClient('mock',reasoning_effort=effort,transport=httpx.MockTransport(lambda r:(seen.append(json.loads(r.content)) or httpx.Response(200,json=reply()))))
        try:await client.tool_call([],[])
        finally:await client.aclose()
    assert [r['reasoning_effort'] for r in seen]==['max','high','low']
    assert all({k:v for k,v in r.items() if k!='reasoning_effort'}=={k:v for k,v in seen[0].items() if k!='reasoning_effort'} for r in seen)
    with pytest.raises(ValueError):GLMToolClient('mock',reasoning_effort='disabled')

@pytest.mark.asyncio
async def test_isolated_failure_does_not_stop_next_task_or_retry_failed_request(tmp_path):
    budget=SharedAPIBudget(tmp_path/'ledger.json',config());circuit=TeacherQueueCircuit();attempts=[]
    def response(r):
        attempts.append(1)
        return httpx.Response(500,json={}) if len(attempts)==1 else httpx.Response(200,json=reply())
    client=QueueBudgetedGLMToolClient(api_key='mock',budget=budget,budget_task='probe/a',circuit=circuit,transport=httpx.MockTransport(response))
    try:
        with pytest.raises(GLMProviderError):await client.tool_call([],[],model_override='glm-5.3-flash')
        assert len(attempts)==1 and circuit.stopped_reason is None
        await client.tool_call([],[],model_override='glm-5.3-flash')
    finally:await client.aclose()
    rows=list(budget.snapshot()['requests'].values())
    assert len(rows)==2 and rows[0]['charged_micros']==rows[0]['reserved_micros'] and rows[1]['status']=='settled'

@pytest.mark.asyncio
async def test_rate_limit_cooldown_is_honored_without_sleeping_test():
    now=[0.0];waits=[]
    async def sleep(s):waits.append(s);now[0]+=s
    circuit=TeacherQueueCircuit(clock=lambda:now[0],sleeper=sleep)
    client=GLMToolClient('mock',transport=httpx.MockTransport(lambda r:httpx.Response(429,headers={'Retry-After':'2.5'},json={})))
    try:
        with pytest.raises(GLMProviderError):await client.tool_call([],[])
        circuit.record(client.last_generation_audit)
    finally:await client.aclose()
    await circuit.before_request()
    assert sum(waits)==2.5 and max(waits)<=1 and circuit.stopped_reason is None

@pytest.mark.asyncio
@pytest.mark.parametrize('status',[401,403])
async def test_authentication_blocks_future_requests_before_reservation(tmp_path,status):
    budget=SharedAPIBudget(tmp_path/'ledger.json',config());circuit=TeacherQueueCircuit()
    client=QueueBudgetedGLMToolClient(api_key='mock',budget=budget,budget_task='probe/a',circuit=circuit,
        transport=httpx.MockTransport(lambda r:httpx.Response(status,json={})))
    try:
        for _ in range(2):
            with pytest.raises(GLMProviderError):await client.tool_call([],[],model_override='glm-5.3-flash')
        assert client.last_generation_audit=={'queue_stopped_before_reservation':True}
    finally:await client.aclose()
    assert len(budget.snapshot()['requests'])==1

@pytest.mark.asyncio
async def test_rolling_failures_trip_circuit_but_old_failures_age_out():
    circuit=TeacherQueueCircuit()
    circuit.record({'provider_error':'ReadTimeout'})
    for _ in range(10):circuit.record({})
    assert not sum(circuit.window)
    for _ in range(3):circuit.record({'provider_error':'ReadTimeout'})
    with pytest.raises(GLMProviderError):await circuit.before_request()

def test_scoped_budget_remains_atomic_and_resume_preserves_unknown_reservations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'ledger.json';kw=dict(scope_prefix='probe/',scope_limit_micros=2_000_000,scope_max_requests=10)
    budget=SharedAPIBudget(path,config(),**kw)
    def reserve(i):
        try:return budget.reserve(8192,'probe/'+str(i))
        except BudgetExceeded:return None
    with ThreadPoolExecutor(max_workers=6) as pool:ids=list(pool.map(reserve,range(10)))
    accepted=[v for v in ids if v];assert len(accepted)==2
    for rid in accepted:budget.settle(rid,None,'unknown')
    restored=SharedAPIBudget(path,config(),**kw)
    with pytest.raises(BudgetExceeded):restored.reserve(8192,'probe/resume')
    with pytest.raises(ValueError):restored.reserve(8192,'outside')
    assert len(restored.snapshot()['requests'])==2
    # Another authorized batch uses the SAME ledger/global cap, not a reset.
    SharedAPIBudget(path,config()).reserve(8192,'other/a')
    assert len(restored.snapshot()['requests'])==3

def test_scope_request_limit_counts_settled_requests_across_restart(tmp_path):
    kw=dict(scope_prefix='probe/',scope_limit_micros=10_000_000,scope_max_requests=1)
    b=SharedAPIBudget(tmp_path/'ledger.json',config(),**kw);rid=b.reserve(8192,'probe/first');b.settle(rid,{'prompt_tokens':1,'completion_tokens':1},'ok')
    with pytest.raises(BudgetExceeded):SharedAPIBudget(b.path,config(),**kw).reserve(8192,'probe/second')
