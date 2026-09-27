"""Crash-safe CNY reservations shared by all batches of one cloud program.

Conservative rates can exceed the provider price. Unknown usage retains the
entire reservation; restart never refunds an outstanding request implicitly.
"""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from core.glm_tool_client import GLMToolClient

try:  # POSIX: advisory whole-file locks via flock.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows: msvcrt.locking provides the equivalent exclusive lock.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

if fcntl is None and msvcrt is None:  # pragma: no cover - exotic platform
    raise RuntimeError("SharedAPIBudget requires fcntl (POSIX) or msvcrt (Windows) locking")


@contextmanager
def _exclusive_file_lock(handle) -> Any:
    """Hold an exclusive advisory lock on an open file across platforms.

    The reservation ledger is shared by concurrent batches, so the lock is the
    only thing preventing two writers from overspending the same budget. On
    Windows ``msvcrt.locking`` locks a byte range rather than the whole file, so
    lock one byte at the start and seek back before yielding.
    """
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
        return
    # Windows: LK_LOCK retries for ~10s then raises, which is enough for the
    # short critical sections used here.
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    try:
        handle.seek(0)
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def _fsync_directory(path: Path) -> Any:
    """Best-effort directory fsync.

    Windows cannot open a directory with ``os.open`` and does not need this to
    make the rename durable, so the call is skipped there instead of raising.
    """
    if os.name == "nt":  # pragma: no cover - Windows has no directory fsync
        yield
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        yield
        os.fsync(fd)
    finally:
        os.close(fd)


class BudgetExceeded(RuntimeError):
    pass


class SharedAPIBudget:
    def __init__(self, path, config, *, scope_prefix=None, scope_limit_micros=None, scope_max_requests=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.config=config
        self.scope_prefix=scope_prefix
        self.scope_limit_micros=scope_limit_micros
        self.scope_max_requests=scope_max_requests
        if scope_prefix is not None and (not scope_prefix or not scope_prefix.endswith('/') or
            type(scope_limit_micros) is not int or scope_limit_micros<=0 or
            type(scope_max_requests) is not int or scope_max_requests<=0):
            raise ValueError('A scoped pilot requires a task prefix and positive cost/request limits')
        if scope_prefix is None and (scope_limit_micros is not None or scope_max_requests is not None):
            raise ValueError('Scoped limits require a task prefix')
        if config['currency']!='CNY' or config['limit_micros']!=100_000_000:
            raise ValueError('This program has one fixed 100 CNY total limit')
        if not config.get('rate_evidence') or config['max_input_tokens']<=0:
            raise ValueError('Rate evidence and billable input bound are required')
        for key in ['input_cny_per_million','output_cny_per_million']:
            if Decimal(str(config[key]))<0:raise ValueError('Invalid token price')
        with self._locked() as ledger:
            if ledger and ledger['config']!=config:raise ValueError('Budget configuration changed')
            if not ledger:ledger.update(config=config,requests={},frozen=False)

    @contextmanager
    def _locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            with _exclusive_file_lock(lock):
                ledger=json.loads(self.path.read_text()) if self.path.exists() else {}
                yield ledger
                temporary=self.path.with_suffix('.tmp')
                with temporary.open('w') as f:
                    json.dump(ledger,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
                temporary.replace(self.path)
                with _fsync_directory(self.path.parent):
                    pass

    def cost(self, prompt_tokens, completion_tokens):
        # 1 yuan / million tokens equals 1 micro-yuan per token.
        return int((Decimal(str(self.config['input_cny_per_million']))*prompt_tokens+
                    Decimal(str(self.config['output_cny_per_million']))*completion_tokens).to_integral_value(rounding=ROUND_CEILING))

    def reserve(self, max_output_tokens, task):
        if not 1<=max_output_tokens<=self.config['max_output_tokens']:raise ValueError('Unbounded output request')
        reserved=self.cost(self.config['max_input_tokens'],max_output_tokens)
        with self._locked() as ledger:
            if self.scope_prefix is not None:
                if not task.startswith(self.scope_prefix):raise ValueError('Task outside pilot budget scope')
                scoped=[v for v in ledger['requests'].values() if v['task'].startswith(self.scope_prefix)]
                if (sum(v['charged_micros'] for v in scoped)+reserved>self.scope_limit_micros or
                    len(scoped)>=self.scope_max_requests):raise BudgetExceeded('Pilot scope reservation limit reached')
            committed=sum(r['charged_micros'] for r in ledger['requests'].values())
            if ledger['frozen'] or committed+reserved>self.config['limit_micros'] or len(ledger['requests'])>=self.config['max_requests']:
                raise BudgetExceeded('Shared 300-task API reservation limit reached')
            request_id=str(uuid4())
            ledger['requests'][request_id]=dict(status='reserved',task=task,created_at=time.time(),
                reserved_micros=reserved,charged_micros=reserved,max_output_tokens=max_output_tokens)
        return request_id

    def settle(self, request_id, usage, outcome):
        violation=False
        with self._locked() as ledger:
            row=ledger['requests'][request_id]
            if row['status']!='reserved':raise ValueError('Request already settled')
            valid=isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ['prompt_tokens','completion_tokens'])
            row.update(status='settled' if valid else 'unknown_usage_charged_reservation',outcome=outcome)
            if valid:
                charge=self.cost(usage['prompt_tokens'],usage['completion_tokens'])
                violation=(usage['prompt_tokens']>self.config['max_input_tokens'] or
                           usage['completion_tokens']>row['max_output_tokens'] or charge>row['reserved_micros'])
                row.update(usage=usage,charged_micros=charge)
                if violation:ledger['frozen']=True
        if violation:raise BudgetExceeded('Provider exceeded declared billing bound; program frozen')

    def snapshot(self):
        with self._locked() as ledger:
            return json.loads(json.dumps(ledger))


class BudgetedGLMToolClient(GLMToolClient):
    def __init__(self, *, budget, budget_task, **kwargs):
        super().__init__(**kwargs)
        if self.base_url!='https://open.bigmodel.cn/api/paas/v4':
            raise ValueError('CNY rate contract only applies to the domestic endpoint')
        self.budget=budget;self.budget_task=budget_task

    async def tool_call(self, messages, tools, **kwargs):
        if kwargs.get('model_override')!=self.budget.config['model']:
            raise ValueError('Model differs from the budget rate contract')
        self.last_generation_audit=None
        request_id=self.budget.reserve(kwargs.get('max_tokens',8192),self.budget_task)
        outcome='exception';usage=None
        # Avoid settling from the previous successful call if this one fails early.
        self.last_generation_audit=None
        try:
            result=await super().tool_call(messages,tools,**kwargs)
            outcome='success'
            return result
        finally:
            audit=self.last_generation_audit or {}
            usage=(audit.get('response') or {}).get('usage')
            self.budget.settle(request_id,usage,outcome)
            if isinstance(self.last_generation_audit,dict):
                self.last_generation_audit['budget_request_id']=request_id
