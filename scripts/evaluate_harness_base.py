"""Run a local base checkpoint or explicit adapter on the same frozen environment.

Development diagnostic only. No teacher prefix, no scripted policy in measured runs.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend/src"))
from agentic.action_executor import TravelActionExecutor
from agentic.clock import frozen_reference_time
from agentic.harness import ARCHITECTURE_VERSION, HARNESS_REVISION
from agentic.loop import BoundedAgentLoop, PolicyAction
from agentic.policy_repair import SelfRepairingAgentPolicy
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from agentic.trajectory import EpisodeRecorder, EpisodeReplayVerifier
from agentic.policy_prompts import policy_prompt_payload
from agentic.reward import HierarchicalRewardEngine
from core.settings import settings
from evaluation.validator import VALIDATOR_VERSION
from schemas import ToolResult
from tools.tool_executor import ToolExecutor

MOMENT = datetime(2026, 9, 5, 0, 0, tzinfo=UTC)
VERSION = "harness-base-diagnostic24.v1"
JUDGING_VERSION = "entity-evidence-contract-v4"
from evaluation import research_fixture
FIXTURE_REVISION = research_fixture.FIXTURE_REVISION

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def build_cases():
    cities = [("上海",31.23,121.47),("杭州",30.25,120.15),("成都",30.66,104.06),("北京",39.90,116.40)]
    cases = []
    for family in ("basic", "query", "evidence", "recovery", "constraint", "clarification"):
        for variant, (city, lat, lng) in enumerate(cities):
            poi_name = f"{city}城市博物馆"
            slots = {"destination":city,"travel_days":1,"start_date":"2026-09-18","end_date":"2026-09-18","budget_range":1500,"interests":["历史","公园"],"must_visit":[poi_name]}
            request = f"9月18日在{city}玩1天，总预算1500元，想看历史和公园，必须去{poi_name}。请安排完整的一日行程，含午晚餐；日期与天数已确定。"
            pois = [{"id":f"p{i}","name":name,"category":"attraction","score":0.95-i*.04,"lat":lat+i*.002,"lng":lng+i*.002,"ticket_price":30,"open_time":"08:00","close_time":"21:00","duration_minutes":90,"tags":["历史","公园"]} for i,name in enumerate([poi_name,f"{city}滨河公园",f"{city}文化展览馆",f"{city}历史街区"])]
            expected = "plan"
            missing=[]
            if family in {"query","evidence"}:
                slots["information_needs"]=["opening_hours"]
                slots["current_info_queries"]=[f"{poi_name} 2026-09-18 开放时间"]
                request += f"请查证{poi_name}当天的开放时间再安排，不能把无关或过期网页当作依据。"
            if family == "recovery" and variant == 3:
                slots["information_needs"]=["weather"]
                request += "请同时检查当天的天气。"
            if family == "constraint":
                expected="grounded_stop"
                if variant < 2:
                    pois[0]["closed_dates"]=["2026-09-18"]
                    request += "必须当天进入博物馆参观，不接受外观打卡替代；日期和必去项不能自动更改。"
                else:
                    slots["budget_range"]=300
                    pois[0]["ticket_price"]=2000
                    request=request.replace("1500元","300元")+"预算是硬上限，不能擅自提高或取消必去项。"
            if family == "clarification" and variant < 3:
                expected="clarify"
                key=("destination","travel_days","origin")[variant]
                missing=[key]
                slots.pop(key,None)
                if key=="destination":
                    slots.pop("must_visit",None)
                    request="9月18日想出去玩一天，预算1500元，目的地还没确定。请帮我规划。"
                elif key=="travel_days":
                    slots.pop("end_date",None)
                    request=f"9月18日开始去{city}玩，预算1500元，待几天还没有决定，希望去{poi_name}。"
                else:
                    slots["transport_modes_requested"]=["train"]
                    request=f"9月18日坐火车去{city}玩一天，总预算1500元，必须去{poi_name}，也要安排到达前后的交通。"
            cases.append({"id":f"{family}-{variant+1:02d}","family":family,"variant":variant,"request":request,"slots":slots,"missing_slots":missing,"pois":pois,"expected":expected,"missing_field":missing[0] if missing else None,"reference_time":MOMENT.isoformat()})
    return cases

def load_cases(path=None):
    cases=json.loads(Path(path).read_text(encoding="utf-8")) if path else build_cases()
    if not isinstance(cases,list) or not cases or len({c["id"] for c in cases})!=len(cases):
        raise ValueError("Case file must contain a nonempty list of unique case IDs")
    return cases

class FrozenResearchExecutor(ToolExecutor):
    """Argument-sensitive fixtures for research only; matrix/solve/validate remain real."""
    def __init__(self, case):
        super().__init__()
        self.case=deepcopy(case);self.counts=Counter();self.calls=[];self.first_info_query=None
        for name in ("search_pois","get_poi_detail","get_weather","retrieve_city_knowledge","search_current_info","search_transport"):
            self._handlers[name]=self.research_handler(name)
        if case["family"]=="recovery" and case["variant"]==2:
            actual=self._handlers["get_route_matrix"]
            async def flaky_matrix(args):
                self.counts["matrix_fault_probe"]+=1
                if self.counts["matrix_fault_probe"]==1: raise TimeoutError("Frozen route provider timed out; a subsequent attempt is available")
                return await actual(args)
            self._handlers["get_route_matrix"]=flaky_matrix

    async def execute(self, tool_calls, guard_context=None):
        records=await super().execute(tool_calls, guard_context)
        for call,record in zip(tool_calls,records):
            self.calls.append({"call":deepcopy(call),"record":deepcopy(record)})
        return records

    def research_handler(self, name):
        async def handle(args):
            self.counts[name]+=1
            case=self.case; family=case["family"]; v=case["variant"]
            fault={0:"search_pois",1:"get_poi_detail",3:"get_weather"}.get(v)
            if family=="recovery" and name==fault and self.counts[name]==1:
                raise TimeoutError("Frozen provider request timed out; request may be retried")
            if name=="search_pois": data=research_fixture.search_pois(case,args)
            elif name=="get_poi_detail":
                data=research_fixture.detail(case,args["poi_name"])
            elif name=="retrieve_city_knowledge":
                data={"city":case["slots"].get("destination"),"summary":"测试资料：当地有历史展馆和城市公园。开放情况应以当前证据为准。","pois":[]}
            elif name=="get_weather":
                data=[{"date":args.get("date") or case["slots"].get("start_date"),"condition":"晴","temperature":"20-27℃"}]
            elif name=="search_current_info":
                query=str(args.get("query") or "")
                stale=False
                if family=="evidence":
                    if self.first_info_query is None:self.first_info_query=query
                    stale=query==self.first_info_query
                data=research_fixture.current_info(case,args,MOMENT,stale=stale)
            elif name=="search_transport":
                url="https://frozen.example.invalid/train"
                data={"queried_at":MOMENT.isoformat(),"results":[{"url":url,"title":"测试列车时刻","snippet":"G100 06:00出发 08:00到达"}],"legs":[{"direction":"inbound","date":"2026-09-18","selected_option":{"service_code":"G100","departure_time":"06:00","arrival_time":"08:00","source_url":url}}]}
            else:raise AssertionError(name)
            return ToolResult(data=data,data_source="built_in",confidence=1.0)
        return handle

class AuditedPolicy:
    def __init__(self, delegate, path):self.delegate=delegate;self.path=Path(path);self.calls=[]
    async def propose(self,context):
        record={"state":policy_prompt_payload(context)}
        try:
            action=await self.delegate.propose(context)
            record["action"]=action.model_dump(mode="json")
            return action
        except Exception as exc:
            record["error"]={"type":type(exc).__name__,"code":getattr(exc,"code",None),"message":str(exc)}
            raise
        finally:
            record["generation"]=self.delegate.last_generation_audit
            self.calls.append(record)
            with self.path.open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
            gen=record.get("generation") or {}
            print(json.dumps({"event":"model_call","case":self.path.parent.name,"call":len(self.calls),"action":(record.get("action") or {}).get("action"),"error":record.get("error"),"completion_tokens":gen.get("inference_metrics",{}).get("completion_tokens")},ensure_ascii=False),flush=True)

def initial(case):
    state=initialize_agent_ledger({"user_input":case["request"],"slots":case["slots"],"missing_slots":case["missing_slots"]},mode="agent")
    return AgentLedgerState(**state["agent_ledger"])

def judge(case, episode, calls):
    actions=[s.action.action for s in episode.steps]
    last=actions[-1] if actions else None
    artifacts=(episode.final_state or {}).get("artifacts",{})
    validations=[a["payload"] for a in artifacts.values() if a["artifact_type"]=="validation_report"]
    report=validations[-1] if validations else None
    submitted=last=="finish" and episode.termination_reason=="awaiting_user" and bool(report and report.get("hard_pass"))
    # Independently recheck all required evidence; an old hard_pass report alone
    # cannot establish coverage or freshness under the current contract.
    from agentic.react import ResearchSufficiencyVerifier
    if submitted:
        with frozen_reference_time(datetime.fromisoformat(case.get("reference_time", MOMENT.isoformat()))):
            evidence_report = ResearchSufficiencyVerifier().evaluate(AgentLedgerState(**episode.final_state))
        submitted = submitted and evidence_report.sufficient
    question=str(episode.steps[-1].action.arguments.get("question", "")) if last=="ask_user" else ""
    aliases={"destination":["哪里","城市","目的地","地点"],"travel_days":["几天","天数","多久"],"origin":["哪里","出发","出发地","出发城市"]}
    correct_question=last=="ask_user" and any(x in question for x in aliases.get(case["missing_field"],[]))
    failures=[f["code"] for f in (episode.final_state or {}).get("failures",[])]
    observed_constraint=bool(validations and not report.get("hard_pass")) or any(s.action.action=="solve_itinerary" and s.verification.get("error_code") for s in episode.steps)
    observed_constraint = observed_constraint or any(
        a["artifact_type"] == "constraint_evidence" and
        (a.get("payload", {}).get("feasibility") or {}).get("status") == "infeasible"
        for a in artifacts.values())
    # Merely asking a question after any solver rejection is not a grounded
    # resolution. Enforce the shared terminal contract (including unnecessary
    # questions and evidence-grounded abort/tradeoff requirements).
    terminal_gate = HierarchicalRewardEngine().score(episode).gate_status
    grounded_stop=last in {"ask_user","abort","propose_tradeoff"} and observed_constraint and terminal_gate == "passed"
    success={"plan":submitted,"clarify":correct_question,"grounded_stop":grounded_stop}[case["expected"]]
    metrics=[(r.get("generation") or {}).get("inference_metrics",{}) for r in calls]
    return {"case":case["id"],"family":case["family"],"expected":case["expected"],"criterion_passed":success,"submitted_valid_plan":submitted,"correct_clarification":correct_question,"grounded_stop":grounded_stop,"status":episode.status,"termination_reason":episode.termination_reason,"actions":actions,"failures":failures,"validation":report,"question":question,"model_calls":len(calls),"parse_or_schema_errors":sum("error" in r for r in calls),"prompt_tokens":sum(m.get("prompt_tokens",0) for m in metrics),"completion_tokens":sum(m.get("completion_tokens",0) for m in metrics),"inference_ms":sum(m.get("request_latency_ms",0) for m in metrics)}

async def fixture_check(cases,out):
    """Explicit infrastructure preflight; scripted actions are NEVER baseline measurements."""
    rows=[]
    for case in [cases[0],cases[16],cases[18]]:
        state=initial(case);backend=FrozenResearchExecutor(case);executor=TravelActionExecutor(backend)
        for name in ["search_pois","get_poi_detail","get_route_matrix","solve_itinerary","validate_itinerary"]:
            action=PolicyAction(action=name)
            result=await executor.execute(task=state.task_graph.tasks[0],action=action,ledger=state)
            for a in result.artifacts:state.artifacts[a.artifact_id]=a
            for f in result.facts:state.facts[f.fact_id]=f
        reports=[a.payload for a in state.artifacts.values() if a.artifact_type=="validation_report"]
        rows.append({"case":case["id"],"report":reports[-1] if reports else None,"tool_calls":backend.calls})
    dump(out/"fixture-check.json",rows)
    assert rows[0]["report"] and rows[0]["report"]["hard_pass"],"positive fixture cannot be solved by the real tools"
    assert all(not r["report"]["hard_pass"] for r in rows[1:]),"negative fixtures did not violate real constraints"
    print("FIXTURE_CHECK_PASSED",flush=True)

async def main(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    settings.agentic_guard_mode="enforce"
    all_cases=load_cases(getattr(args,"cases_file",None))
    if (out/"cases.json").exists():
        assert json.loads((out/"cases.json").read_text(encoding="utf-8")) == all_cases, "Case set changed; use a new output directory"
    else:
        dump(out/"cases.json",all_cases)
    if args.fixture_check:
        with frozen_reference_time(MOMENT):await fixture_check(all_cases,out)
        return
    from agentic.local_policy import LocalCheckpointAgentPolicy
    model_dir=Path(args.checkpoint)
    assert model_dir.is_dir(), "checkpoint directory required"
    is_adapter = (model_dir / "adapter_config.json").is_file()
    assert is_adapter == bool(getattr(args, "adapter", False)), "Use --adapter only for an adapter checkpoint"
    base_dir = Path(json.loads((model_dir / "adapter_config.json").read_text())["base_model_name_or_path"]) if is_adapter else model_dir
    assert base_dir.is_dir() and not (base_dir / "adapter_config.json").exists(), "local base weights required"
    files=[*base_dir.glob("*.safetensors"),base_dir/"config.json",base_dir/"tokenizer_config.json",base_dir/"tokenizer.json"]
    print("HASHING_BASE_CHECKPOINT",flush=True)
    manifest={"version":VERSION,"architecture":ARCHITECTURE_VERSION,"checkpoint":str(model_dir),"adapter_loaded":False,"weight_hashes":{p.name:sha(p) for p in files},"cases_sha256":sha(out/"cases.json"),"research":"frozen synthetic fixtures, not live providers","solver":"real TravelVRPSolver","validator":"real ItineraryValidator","parsing":"predeclared user-grounded slots; intent parser not evaluated","decoding":"native, greedy, thinking disabled, max_new_tokens=384","seed":42,"source_sha256":{str(p.relative_to(ROOT)):sha(p) for folder in ["backend/src/agentic","backend/src/tools","backend/src/vrp_solver_service"] for p in (ROOT/folder).rglob("*.py")},"script_sha256":sha(__file__)}
    manifest["harness_revision"] = HARNESS_REVISION
    manifest["adapter_loaded"] = is_adapter
    manifest["base_checkpoint"] = str(base_dir)
    manifest["adapter_hashes"] = {p.name: sha(p) for p in model_dir.iterdir() if p.is_file() and (p.suffix == ".safetensors" or p.name in {"adapter_config.json", "tokenizer.json", "tokenizer_config.json"})} if is_adapter else {}
    manifest["judging_version"] = JUDGING_VERSION
    manifest["fixture_revision"] = FIXTURE_REVISION
    manifest["source_sha256"].update({str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
        for p in (ROOT / "backend/src/evaluation").rglob("*.py")})
    if (out/"manifest.json").exists():
        previous = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        mismatched = [key for key, value in manifest.items() if previous.get(key) != value]
        assert not mismatched, f"Refusing to mix incompatible runs: {mismatched}; use a new output directory"
    dump(out/"manifest.json",manifest)
    started=time.perf_counter()
    policy=LocalCheckpointAgentPolicy(str(model_dir),max_new_tokens=384,seed=42,do_sample=False,load_in_4bit=False,structured_decoding="native")
    manifest["load_seconds"]=time.perf_counter()-started
    manifest["model_class"]=type(policy.model).__name__;manifest["dtype"]=str(next(policy.model.parameters()).dtype)
    import torch,transformers
    manifest["runtime"]={"torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name()}
    dump(out/"manifest.json",manifest)
    print("BASE_MODEL_LOADED",flush=True)
    selected=[c for c in all_cases if not args.case or c["id"] in args.case]
    summaries=[]
    for case in selected:
        target=out/case["id"];target.mkdir(exist_ok=True)
        if (target/"summary.json").exists():
            summaries.append(json.loads((target/"summary.json").read_text()));continue
        print(json.dumps({"event":"case_start","case":case["id"]},ensure_ascii=False),flush=True)
        policy.set_rollout_seed(42)
        audited=AuditedPolicy(policy,target/"model-calls.jsonl")
        with frozen_reference_time(MOMENT):
            state=initial(case);backend=FrozenResearchExecutor(case)
            recorder=EpisodeRecorder(state,environment_version=ARCHITECTURE_VERSION,validator_version=VALIDATOR_VERSION,policy_name="Qwen3-4B-SFT" if is_adapter else "Qwen3-4B-base",policy_version=VERSION)
            started=time.perf_counter()
            await BoundedAgentLoop().run(state,policy=SelfRepairingAgentPolicy(audited),executor=TravelActionExecutor(backend),recorder=recorder)
            episode=recorder.episode
            dump(target/"episode.json",episode.model_dump(mode="json"))
            dump(target/"tool-calls.json",backend.calls)
            replay_errors = EpisodeReplayVerifier().verify(episode)
            assert not replay_errors, replay_errors
            summary=judge(case,episode,audited.calls)
            summary["wall_seconds"]=time.perf_counter()-started
            summary["reward"]=HierarchicalRewardEngine().score(episode).model_dump(mode="json")
            dump(target/"summary.json",summary)
            summaries.append(summary);dump(out/"summary.json",summaries)
            print(json.dumps({"event":"case_done",**{k:summary[k] for k in ["case","criterion_passed","termination_reason","actions","wall_seconds"]}},ensure_ascii=False),flush=True)
    dump(out/"summary.json",summaries)
    print("EVALUATION_COMPLETE",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",default="/root/autodl-tmp/models/Qwen3-4B")
    parser.add_argument("--adapter",action="store_true",help="Explicitly evaluate a PEFT adapter with the same loop and decoding")
    parser.add_argument("--fixture-check",action="store_true")
    parser.add_argument("--case",action="append")
    parser.add_argument("--cases-file",type=Path)
    asyncio.run(main(parser.parse_args()))
