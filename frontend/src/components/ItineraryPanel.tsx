"use client";

import { useEffect, useMemo, useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import { DayCard } from "./DayCard";
import { CheckCircle2, Loader2 } from "lucide-react";

type StageKey =
  | "intent_ready"
  | "data_collection"
  | "draft_ready"
  | "itinerary_final"
  | "writing"
  | "completed";

interface StageInfo {
  key: StageKey;
  label: string;
  description: string;
}

const STAGES: StageInfo[] = [
  {
    key: "intent_ready",
    label: "意图识别",
    description: "解析目的地、天数、预算等需求",
  },
  {
    key: "data_collection",
    label: "数据收集",
    description: "查询景点、天气、价格等信息",
  },
  {
    key: "draft_ready",
    label: "草稿生成",
    description: "生成初步行程安排",
  },
  {
    key: "itinerary_final",
    label: "规则校验",
    description: "校验并修复行程约束",
  },
  {
    key: "writing",
    label: "文案润色",
    description: "生成自然语言行程方案",
  },
  {
    key: "completed",
    label: "完成",
    description: "行程方案已就绪",
  },
];

function normalizeStage(stage: string | null): StageKey | null {
  if (!stage) return null;
  if (stage === "running") return "intent_ready";
  if (STAGES.some((s) => s.key === stage)) return stage as StageKey;
  if (stage === "failed" || stage === "cancelled") return null;
  return null;
}

function StageProgress({ currentStage }: { currentStage: string | null }) {
  const activeStage = normalizeStage(currentStage);
  const activeIndex = useMemo(() => {
    if (!activeStage) return -1;
    return STAGES.findIndex((s) => s.key === activeStage);
  }, [activeStage]);

  if (activeIndex < 0) return null;

  return (
    <div className="rounded-2xl border border-white/60 bg-white/50 p-3 backdrop-blur-md">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-[#111111]/80">
          {STAGES[activeIndex]?.label}
        </span>
        <span className="text-[10px] text-[#111111]/50">
          {activeIndex + 1} / {STAGES.length}
        </span>
      </div>

      {/* Progress bar */}
      <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-[#111111]/10">
        <div
          className="h-full rounded-full bg-[#111111] transition-all duration-500 ease-out"
          style={{ width: `${((activeIndex + 1) / STAGES.length) * 100}%` }}
        />
      </div>

      {/* Stage steps */}
      <div className="space-y-2">
        {STAGES.map((stage, idx) => {
          const isCompleted = idx < activeIndex;
          const isActive = idx === activeIndex;
          return (
            <div key={stage.key} className="flex items-start gap-2">
              <div className="mt-0.5">
                {isCompleted ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : isActive ? (
                  <Loader2 className="h-4 w-4 animate-spin text-[#111111]" />
                ) : (
                  <div className="h-4 w-4 rounded-full border-2 border-[#111111]/20" />
                )}
              </div>
              <div className="flex-1">
                <p
                  className={`text-xs font-medium ${
                    isActive || isCompleted
                      ? "text-[#111111]"
                      : "text-[#111111]/40"
                  }`}
                >
                  {stage.label}
                </p>
                {isActive && (
                  <p className="text-[10px] text-[#111111]/60">
                    {stage.description}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ItineraryPanel() {
  const store = useChatStore();
  const { itinerary, currentTrip, currentStage, isLoading } = store;
  const [activeDay, setActiveDay] = useState(0);

  // Reset active day when itinerary changes or shrinks
  useEffect(() => {
    if (!itinerary || itinerary.length === 0) {
      setActiveDay(0);
      return;
    }
    if (activeDay >= itinerary.length) {
      setActiveDay(itinerary.length - 1);
    }
  }, [itinerary, activeDay]);

  const tripTitle = currentTrip ? `${currentTrip.title}` : "行程编排台";

  return (
    <div
      className="flex h-full flex-col gap-2.5 rounded-3xl p-3.5"
      style={{
        background: "rgba(255,255,255,0.66)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(255,255,255,0.79)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-[#111111]">{tripTitle}</h2>
        {itinerary && (
          <button
            onClick={() => store.setActiveView("export")}
            className="rounded-xl bg-[#111111] px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#333333]"
          >
            导出
          </button>
        )}
      </div>

      {/* Stage indicator */}
      {isLoading && currentStage && <StageProgress currentStage={currentStage} />}

      {!itinerary || itinerary.length === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <p className="text-sm text-[#111111]/30">行程将在这里展示</p>
          <p className="mt-1 text-xs text-[#111111]/20">开始聊天来生成行程</p>
        </div>
      ) : (
        <>
          {/* Day Tabs */}
          <div className="flex gap-2">
            {itinerary.map((day, idx) => (
              <button
                key={day.day_number}
                onClick={() => setActiveDay(idx)}
                className="rounded-full px-3 py-2 text-xs font-medium transition-colors"
                style={
                  idx === activeDay
                    ? { background: "#111111", color: "#FFFFFF" }
                    : { background: "rgba(255,255,255,0.78)", color: "#333333" }
                }
              >
                第 {day.day_number} 天
              </button>
            ))}
          </div>

          {/* Day Content */}
          <div className="flex-1 overflow-y-auto scrollbar-thin">
            {itinerary[activeDay] && <DayCard day={itinerary[activeDay]} />}
          </div>
        </>
      )}
    </div>
  );
}
