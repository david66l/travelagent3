"""Bounded teacher dispatch; failures remain observations, never paid rerolls."""
import asyncio
from collections import deque
import time

from core.api_budget import BudgetedGLMToolClient, BudgetExceeded
from core.glm_tool_client import GLMProviderError


class TeacherQueueCircuit:
    def __init__(self, *, clock=time.monotonic, sleeper=asyncio.sleep):
        self.clock=clock;self.sleeper=sleeper
        self.window=deque(maxlen=10);self.stopped_reason=None;self.cooldown_until=0.0
        self.events=[]

    async def before_request(self):
        while True:
            if self.stopped_reason:raise GLMProviderError('Teacher queue stopped: '+self.stopped_reason)
            remaining=self.cooldown_until-self.clock()
            if remaining<=0:return
            await self.sleeper(min(remaining,1.0))

    def record(self, audit, *, budget_error=False):
        status=audit.get('http_status')
        failure=bool(audit.get('provider_error'))
        self.window.append(failure)
        if budget_error:self.stopped_reason='budget_limit'
        elif status in {401,403}:self.stopped_reason='authentication_or_permission'
        elif status==429:
            delay=audit.get('retry_after_seconds',30.0)
            self.cooldown_until=max(self.cooldown_until,self.clock()+delay)
        if failure and sum(self.window)>=3 and self.stopped_reason is None:
            self.stopped_reason='three_provider_failures_in_last_ten_requests'
        if failure or budget_error:
            self.events.append(dict(http_status=status,provider_error=audit.get('provider_error'),
                stopped_reason=self.stopped_reason,cooldown_remaining=max(0,self.cooldown_until-self.clock())))


class QueueBudgetedGLMToolClient(BudgetedGLMToolClient):
    def __init__(self, *, circuit, **kwargs):
        super().__init__(**kwargs);self.circuit=circuit

    async def tool_call(self, messages, tools, **kwargs):
        self.last_generation_audit={}
        try:
            await self.circuit.before_request()
        except GLMProviderError:
            self.last_generation_audit={'queue_stopped_before_reservation':True}
            raise
        budget_error=False
        try:
            return await super().tool_call(messages,tools,**kwargs)
        except BudgetExceeded:
            budget_error=True
            if not self.last_generation_audit:self.last_generation_audit={'budget_denied_before_request':True}
            raise
        finally:
            self.circuit.record(self.last_generation_audit or {},budget_error=budget_error)
