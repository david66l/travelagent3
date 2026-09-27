"""Cloud async vLLM inference with the exact native prompt/parser contract."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from agentic.batched_policy import decode_action
from agentic.local_policy import LocalCheckpointAgentPolicy, _tokenizer_compatibility_kwargs
from agentic.chat_template_contract import install_agent_chat_template, render_agent_tool_prompt


class VLLMCheckpointEngine:
    def __init__(self, checkpoint, *, concurrency=16, max_new_tokens=384, prefix_cache=True):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        from vllm.lora.request import LoRARequest
        import torch
        self.checkpoint=str(checkpoint);self.max_new_tokens=max_new_tokens;self._torch=torch
        config=json.loads((Path(checkpoint)/'adapter_config.json').read_text())
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=False,
            **_tokenizer_compatibility_kwargs(checkpoint))
        install_agent_chat_template(self.tokenizer)
        self.engine=AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=config['base_model_name_or_path'],tokenizer=str(checkpoint),
            skip_tokenizer_init=True,
            dtype='bfloat16',max_model_len=8192,max_num_seqs=concurrency,
            max_num_batched_tokens=8192,gpu_memory_utilization=.65,
            enable_prefix_caching=prefix_cache,enable_chunked_prefill=True,
            enable_lora=True,max_loras=1,max_lora_rank=config['r'],
            seed=42,trust_remote_code=False,disable_log_requests=True,disable_log_stats=True,
            generation_config='vllm'))
        self.lora=LoRARequest('student-sft',1,str(checkpoint))
        self.params=SamplingParams(temperature=0,max_tokens=max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],skip_special_tokens=False,detokenize=False)
        self.batches=[];self.active=set()

    def session(self):return VLLMSessionPolicy(self)

    async def submit(self,messages,tools,allowed,capability):
        ids=render_agent_tool_prompt(self.tokenizer,messages,tools,
            add_generation_prompt=True,tokenize=True)
        if len(ids)+self.max_new_tokens>8192:raise ValueError('Prompt exceeds vLLM context bound; no truncation')
        request_id=str(uuid4());started=time.perf_counter();self.active.add(request_id)
        try:
            final=None
            async for output in self.engine.generate({'prompt_token_ids':ids},self.params,request_id,lora_request=self.lora):
                final=output
            if final is None or len(final.outputs)!=1:raise RuntimeError('Incomplete vLLM request')
            generated=final.outputs[0]
            output_ids=list(generated.token_ids)
            raw=self.tokenizer.decode(output_ids,skip_special_tokens=False)
            elapsed=(time.perf_counter()-started)*1000
            audit=dict(schema_version='local-policy-generation-audit.vllm-v1',request_id=request_id,
                prompt_input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                finish_reason=generated.finish_reason,backend='vllm-continuous',generation_ms=elapsed)
            # Entries are completed requests, NOT physical GPU generation batches.
            self.batches.append(dict(event='vllm_request_completed',batch_size=1,prompt_tokens=[len(ids)],
                completion_tokens=len(output_ids),request_wall_ms=elapsed))
            try:
                action=decode_action(raw,audit,allowed,capability,self.checkpoint,len(ids),len(output_ids),elapsed)
                action.inference_metrics.backend='vllm'
                audit['inference_metrics']=action.inference_metrics.model_dump(mode='json')
                return action,audit,None
            except Exception as exc:return None,audit,exc
        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise
        finally:self.active.discard(request_id)

    async def aclose(self):
        for request_id in list(self.active):await self.engine.abort(request_id)
        self.engine.shutdown_background_loop()


class VLLMSessionPolicy(LocalCheckpointAgentPolicy):
    def __init__(self,engine):self.engine=engine;self._last_generation_audit=None
    async def propose_from_history(self,messages,*,tools,allowed_actions,capability=None):
        self._last_generation_audit=None
        action,audit,error=await self.engine.submit(messages,tools,allowed_actions,capability or {})
        self._last_generation_audit=audit
        if error:raise error
        return action
