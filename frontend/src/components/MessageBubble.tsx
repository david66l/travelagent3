"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { RunStatus } from "@/stores/chatStore";
import { Clock, Activity, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  runStatus?: RunStatus;
}

function fmtDur(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s >= 1 ? `${m}m${s.toFixed(0)}s` : `${m}m`;
}

function fmtMs(ms: number): string {
  return fmtDur(ms / 1000);
}

function fmtStep(name: string): string {
  const map: Record<string, string> = {
    intent_node: "意图识别",
    collect_info_node: "收集信息",
    prepare_context_node: "准备上下文",
    poi_search_node: "搜索景点",
    weather_node: "查询天气",
    budget_init_node: "初始化预算",
    context_enrichment_node: "丰富上下文",
    planner_node: "规划行程",
    validation_node: "验证行程",
    route_node: "优化路线",
    budget_calc_node: "计算预算",
    apply_routes_node: "应用路线",
    proposal_node: "生成方案",
    update_prefs_node: "更新偏好",
    qa_node: "问答处理",
    confirm_node: "确认行程",
    save_memory_node: "保存记忆",
    format_output_node: "格式化输出",
    format_output: "格式化输出",
    ask_modification_node: "询问修改",
  };
  return map[name] || name;
}

export function MessageBubble({ role, content, runStatus }: MessageBubbleProps) {
  const isUser = role === "user";
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  return (
    <div className={cn("flex w-full flex-col", isUser ? "items-end" : "items-start")}>
      {/* Message bubble */}
      <div
        className={cn(
          "max-w-[560px] rounded-2xl px-3 py-2.5 text-[13px] leading-relaxed",
          isUser
            ? "bg-[#111111] text-white"
            : "bg-[#FFFFFFC2] text-[#111111E6] backdrop-blur-md"
        )}
        style={
          !isUser
            ? {
                border: "1px solid rgba(255,255,255,0.8)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
              }
            : {
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              }
        }
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : (
          <div className="prose prose-sm max-w-none text-inherit">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {/* Inline run stats for assistant messages */}
      {!isUser && runStatus && (
        <div className="mt-1.5 max-w-[560px]">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-2 rounded-lg bg-white/30 px-2 py-1 text-left backdrop-blur-sm transition-colors hover:bg-white/50"
          >
            <Clock className="h-3 w-3 text-[#111111]/40" />
            <span className="text-[10px] text-[#111111]/50">
              {fmtDur(runStatus.elapsed_seconds)}
            </span>
            <span className="text-[10px] text-[#111111]/30">|</span>
            <Activity className="h-3 w-3 text-[#111111]/40" />
            <span className="text-[10px] text-[#111111]/50">
              {runStatus.total_tokens.toLocaleString()} tokens
            </span>
            <span className="text-[10px] text-[#111111]/30">|</span>
            <span className="text-[10px] text-[#111111]/50">
              {runStatus.step_count} 步
            </span>
            {expanded ? (
              <ChevronUp className="h-3 w-3 text-[#111111]/30" />
            ) : (
              <ChevronDown className="h-3 w-3 text-[#111111]/30" />
            )}
          </button>

          {expanded && (
            <div className="mt-1.5 space-y-2 rounded-xl border border-white/40 bg-white/40 px-3 py-2.5 backdrop-blur-md">
              {/* Copy button */}
              <div className="flex items-center justify-end">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!runStatus) return;
                    const text = [
                      `运行时间: ${fmtDur(runStatus.elapsed_seconds)}`,
                      runStatus.ttft_seconds !== null && runStatus.ttft_seconds > 0
                        ? `TTFT: ${fmtDur(runStatus.ttft_seconds)}`
                        : null,
                      `总Token: ${runStatus.total_tokens.toLocaleString()}`,
                      `prompt: ${runStatus.prompt_tokens.toLocaleString()}`,
                      `completion: ${runStatus.completion_tokens.toLocaleString()}`,
                      `LLM调用: ${runStatus.llm_calls}次`,
                      `步骤: ${runStatus.completed_count}/${runStatus.step_count}`,
                      "",
                      "步骤明细:",
                      ...(runStatus.step_details || []).map((s) => {
                        const start = s.start_offset ?? 0;
                        const end = s.end_offset ?? start;
                        return `  ${fmtStep(s.name)}: ${fmtMs(s.duration_ms)} (+${fmtDur(start)} ~ +${fmtDur(end)})`;
                      }),
                    ]
                      .filter(Boolean)
                      .join("\n");
                    navigator.clipboard.writeText(text).then(() => {
                      setCopied(true);
                      setTimeout(() => setCopied(false), 1500);
                    });
                  }}
                  className="flex items-center gap-1 rounded-md bg-white/50 px-2 py-1 text-[10px] text-[#111111]/50 transition-colors hover:bg-white/80"
                >
                  {copied ? (
                    <>
                      <Check className="h-3 w-3 text-green-500" />
                      <span className="text-green-600">已复制</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>复制参数</span>
                    </>
                  )}
                </button>
              </div>

              {/* Stats grid */}
              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-xl bg-white/50 px-2 py-2 text-center">
                  <p className="text-[10px] text-[#111111]/40">运行时间</p>
                  <p className="text-sm font-semibold text-[#111111]">
                    {fmtDur(runStatus.elapsed_seconds)}
                  </p>
                </div>
                {runStatus.ttft_seconds !== null && runStatus.ttft_seconds > 0 && (
                  <div className="rounded-xl bg-white/50 px-2 py-2 text-center">
                    <p className="text-[10px] text-[#111111]/40">TTFT</p>
                    <p className="text-sm font-semibold text-[#111111]">
                      {fmtDur(runStatus.ttft_seconds)}
                    </p>
                  </div>
                )}
                <div className="rounded-xl bg-white/50 px-2 py-2 text-center">
                  <p className="text-[10px] text-[#111111]/40">总 Token</p>
                  <p className="text-sm font-semibold text-[#111111]">
                    {runStatus.total_tokens.toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Token breakdown */}
              {runStatus.total_tokens > 0 && (
                <div className="flex items-center gap-3 text-[11px] text-[#111111]/50">
                  <span>prompt: {runStatus.prompt_tokens.toLocaleString()}</span>
                  <span className="h-3 w-px bg-[#111111]/10" />
                  <span>completion: {runStatus.completion_tokens.toLocaleString()}</span>
                  <span className="h-3 w-px bg-[#111111]/10" />
                  <span>LLM 调用: {runStatus.llm_calls} 次</span>
                </div>
              )}

              {/* Steps timeline with mini gantt */}
              <div className="space-y-1">
                <p className="text-[10px] font-medium text-[#111111]/40">
                  执行步骤 ({runStatus.completed_count}/{runStatus.step_count})
                </p>
                <div className="max-h-52 overflow-y-auto space-y-1.5 scrollbar-thin pr-1">
                  {(runStatus.step_details || []).map((step, i) => {
                    const totalSpan = Math.max(runStatus.elapsed_seconds, 1);
                    const start = step.start_offset ?? 0;
                    const end = step.end_offset ?? start;
                    const dur = step.duration_ms;
                    const leftPct = Math.min(100, (start / totalSpan) * 100);
                    const widthPct = Math.max(
                      1,
                      Math.min(100 - leftPct, ((end - start) / totalSpan) * 100)
                    );
                    return (
                      <div key={`${step.name}-${i}`} className="space-y-0.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-green-100 text-[9px] text-green-600">
                              ✓
                            </span>
                            <span className="text-[11px] text-[#111111]/70">
                              {fmtStep(step.name)}
                            </span>
                          </div>
                          <span className="text-[10px] text-[#111111]/40">
                            {fmtMs(dur)}
                          </span>
                        </div>
                        {/* Mini timeline bar */}
                        <div className="flex items-center gap-1.5">
                          <span className="w-8 text-[9px] text-[#111111]/30 text-right">
                            +{fmtDur(start)}
                          </span>
                          <div className="flex-1 h-1.5 rounded-full bg-[#111111]/5 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-green-400/60"
                              style={{
                                marginLeft: `${leftPct}%`,
                                width: `${widthPct}%`,
                              }}
                            />
                          </div>
                          <span className="w-8 text-[9px] text-[#111111]/30">
                            ~+{fmtDur(end)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
