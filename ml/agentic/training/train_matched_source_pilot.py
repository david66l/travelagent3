"""Cloud pilot: complete action targets, exactly matched tokens per source update.

This is a small, source-matched experiment, not the formal training entrypoint.
Every microbatch contributes its summed completion loss divided by the shared
source-update token count. No microbatch-mean averaging or partial-label masks.
"""
from pathlib import Path
import argparse,json,hashlib,sys,time,random,os
from collections import defaultdict


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    from peft import PeftModel
    sys.path[:0]=[str(args.runtime/'backend/src'),str(args.runtime),'/root/autodl-tmp/agent-harness-refactor-20260905/test-deps']
    from agentic.training import to_rendered_prompt_completion
    from agentic.chat_template_contract import install_agent_chat_template,AGENT_CHAT_TEMPLATE_KWARGS
    from agentic.local_policy import parse_local_tool_call
    read=lambda p:json.loads(p.read_text())
    data=args.inputs;manifest=read(data/'manifest.json');freeze=read(data/'FROZEN.json')
    for name,value in freeze['files'].items():assert digest(data/name)==value
    assert freeze['training_script_sha256']==digest(__file__)
    out=args.output
    if out.exists():
        if not (out/'training-report.json').exists():raise RuntimeError('Incomplete run retained; no automatic restart')
        report=read(out/'training-report.json')
        assert report['inputs_frozen_sha256']==digest(data/'FROZEN.json')
        for name,value in report['checkpoint_files'].items():assert digest(out/'checkpoint'/name)==value
        print('TRAINING_RESUME_NO_PENDING_WORK',flush=True);return
    out.mkdir(parents=True)
    seed=manifest['training_seed'];set_seed(seed);os.environ['TOKENIZERS_PARALLELISM']='false'
    tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen3-4B',local_files_only=True);install_agent_chat_template(tok)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    pool=read(data/'unique-examples.json');meta=read(data/'example-provenance.json');schedule=read(data/(args.arm+'-schedule.json'))
    rendered=to_rendered_prompt_completion(list(pool.values()),tok,chat_template_kwargs=AGENT_CHAT_TEMPLATE_KWARGS);encoded={}
    for (key,row),item in zip(pool.items(),rendered,strict=True):
        completion=item['completion'].rstrip();prefix=tok.encode(item['prompt'],add_special_tokens=False);full=tok.encode(item['prompt']+completion,add_special_tokens=False)
        n=len(full)-len(prefix);assert full[:len(prefix)]==prefix and len(full)<=manifest['max_length'] and n==meta[key]['supervised_tokens']
        assert full[-n:].count(tok.eos_token_id)==1 and full[-1]==tok.eos_token_id
        parse_local_tool_call(tok.decode(full[-n:],skip_special_tokens=False));encoded[key]=(full,n)
    groups=defaultdict(list)
    for entry in schedule:groups[entry['lineage_group']].append(entry['example_key'])
    budgets={v['lineage_group']:v['supervised_tokens_per_arm'] for v in manifest['groups']}
    assert all(sum(encoded[k][1] for k in keys)==budgets[group] for group,keys in groups.items())
    base='/root/autodl-tmp/models/Qwen3-4B';checkpoint=Path(manifest['initial_checkpoint'])
    original_files={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file() and p.name in ['adapter_model.safetensors','adapter_config.json']}
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model=PeftModel.from_pretrained(model,str(checkpoint),is_trainable=True)
    model.config.use_cache=False
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in n for n,p in parameters)
    def trainable_hash():
        h=hashlib.sha256()
        for name,p in parameters:h.update(name.encode());h.update(p.detach().cpu().float().numpy().tobytes())
        return h.hexdigest()
    initial_hash=trainable_hash()
    # Actual 4B proof that restricted output projection preserves full masked CE.
    key=schedule[0]['example_key'];ids,n=encoded[key];x=torch.tensor([ids],device='cuda');labels=x.clone();labels[:,:-n]=-100
    model.eval()
    with torch.no_grad():
        full=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).logits
        reference=F.cross_entropy(full[:,:-1].float().reshape(-1,full.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')
        reference_value=float(reference);del full,reference
        tail=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=n+1).logits
        restricted=F.cross_entropy(tail[:,:-1].float().reshape(-1,tail.shape[-1]),x[:,-n:].reshape(-1),reduction='sum')
        restricted_value=float(restricted);assert abs(reference_value-restricted_value)<=max(1e-4,abs(reference_value)*1e-5)
        del tail,restricted
    del x,labels;torch.cuda.empty_cache()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
    optimizer=torch.optim.AdamW([p for _,p in parameters],lr=manifest['proposed_learning_rate'],weight_decay=0.0)
    steps=[];total_tokens=0;total_rows=0;load_seconds=time.perf_counter()-started;train_started=time.perf_counter()
    for epoch in range(manifest['epochs']):
        order=sorted(groups);random.Random(seed+epoch).shuffle(order)
        for group in order:
            optimizer.zero_grad(set_to_none=True);keys=list(groups[group]);random.Random(str(seed)+str(epoch)+group).shuffle(keys)
            denominator=budgets[group];count=0;loss_sum=0.0
            for key in keys:
                ids,n=encoded[key];x=torch.tensor([ids],device='cuda');output=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,logits_to_keep=n+1)
                logits=output.logits[:,:-1].float();assert logits.shape[1]==n
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,-n:].reshape(-1),reduction='sum')
                assert torch.isfinite(loss);loss_sum+=float(loss.detach());(loss/denominator).backward();count+=n;total_rows+=1
                del output,logits,loss,x
            assert count==denominator
            grad_norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.0,error_if_nonfinite=True))
            assert not any(p.grad is not None for name,p in model.named_parameters() if not p.requires_grad)
            optimizer.step();total_tokens+=count
            entry=dict(step=len(steps)+1,epoch=epoch,lineage_group=group,rows=len(keys),supervised_tokens=count,mean_token_loss=loss_sum/count,grad_norm=grad_norm)
            steps.append(entry)
            with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            print(json.dumps(dict(event='matched_sft_update',arm=args.arm,**entry)),flush=True)
    expected=manifest['arms'][args.arm]['supervised_tokens']*manifest['epochs'];assert total_tokens==expected and len(steps)==len(groups)*manifest['epochs']
    final_hash=trainable_hash();assert final_hash!=initial_hash
    saved=out/'checkpoint';model.save_pretrained(saved,safe_serialization=True);tok.save_pretrained(saved)
    assert all(digest(checkpoint/name)==value for name,value in original_files.items())
    report=dict(schema_version='matched-source-sft-pilot.v1',arm=args.arm,seed=seed,epochs=manifest['epochs'],optimizer_updates=len(steps),
        processed_rows=total_rows,supervised_tokens=total_tokens,initial_trainable_sha256=initial_hash,final_trainable_sha256=final_hash,
        initial_checkpoint_files=original_files,checkpoint_files={p.name:digest(p) for p in saved.iterdir() if p.is_file()},
        inputs_frozen_sha256=digest(data/'FROZEN.json'),training_script_sha256=digest(__file__),
        full_masked_vs_restricted_projection_ce=dict(reference_sum=reference_value,restricted_sum=restricted_value,passed=True),
        gradient_loss_contract='sum-completion-CE / matched-source-update-token-count.v1',base_weights_frozen=True,
        load_and_precheck_seconds=load_seconds,training_seconds=time.perf_counter()-train_started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        selected_checkpoint='final predetermined epoch 3; no development selection',formal_run=False)
    (out/'training-report.json').write_text(json.dumps(report,indent=2));print('MATCHED_SFT_COMPLETE',args.arm,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['demo_only','demo_plus_correction'],required=True)
    for name in ['runtime','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
