/** Map backend stage / graph node ids to user-facing status text. */

const GATHERING_STAGES = new Set([
  "gathering",
  "demand_parsed",
  "clarify",
]);

const PLANNING_STAGES = new Set([
  "planning",
  "running",
  "understand",
  "profile_recall",
  "memory_loaded",
  "retrieve",
  "weather_check",
  "rag_done",
  "plan",
  "planned",
  "tool_call",
  "tools_executed",
  "factcheck",
  "fact_check_done",
  "hallucination",
  "replan_local",
  "apply_single_change",
  "enriching",
  "confirmed",
  "data_collection",
  "draft_ready",
  "itinerary_final",
  "writing",
  "awaiting_booking",
]);

export type ActivityPhase = "idle" | "gathering" | "planning";

export function resolveActivityPhase(stage: string): ActivityPhase {
  if (GATHERING_STAGES.has(stage)) return "gathering";
  if (PLANNING_STAGES.has(stage)) return "planning";
  return "idle";
}

export function labelForStage(stage: string): string | null {
  if (GATHERING_STAGES.has(stage)) {
    return "让我思考一下…";
  }

  const planningLabels: Record<string, string> = {
    planning: "正在规划…",
    running: "正在规划…",
    understand: "正在规划…",
    profile_recall: "正在读取偏好…",
    retrieve: "正在收集景点信息…",
    weather_check: "正在查询天气…",
    rag_done: "正在整理候选景点…",
    plan: "正在安排行程…",
    planned: "正在优化行程…",
    tool_call: "正在查询实时信息…",
    tools_executed: "正在整合查询结果…",
    factcheck: "正在校验行程…",
    fact_check_done: "正在校验行程…",
    hallucination: "正在检查行程质量…",
    draft_ready: "行程草稿已生成",
    writing: "正在润色文案…",
    itinerary_final: "行程已优化",
    data_collection: "正在收集信息…",
    awaiting_booking: "等待确认预订…",
    enriching: "正在完善行程…",
    confirmed: "正在确认行程…",
  };

  return planningLabels[stage] ?? null;
}
