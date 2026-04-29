"use client";

import { useRef } from "react";
import { useChatStore, deriveBriefItinerary } from "@/stores/chatStore";

export function PreviewPanel() {
  const store = useChatStore();
  const { confirmedInfo, itinerary, activeBriefDay, pendingSuggestions, setActiveBriefDay } = store;
  const scrollRef = useRef<HTMLDivElement>(null);

  const briefItinerary = deriveBriefItinerary(itinerary);
  const hasData = confirmedInfo || (briefItinerary && briefItinerary.length > 0) || pendingSuggestions.length > 0;

  const scrollDays = (direction: "left" | "right") => {
    if (!scrollRef.current) return;
    const scrollAmount = 80;
    scrollRef.current.scrollBy({
      left: direction === "left" ? -scrollAmount : scrollAmount,
      behavior: "smooth",
    });
  };

  return (
    <div
      className="flex h-full flex-col gap-3 rounded-4xl p-4"
      style={{
        background: "rgba(255,255,255,0.64)",
        backdropFilter: "blur(30px)",
        WebkitBackdropFilter: "blur(30px)",
        border: "1px solid rgba(255,255,255,0.7)",
        boxShadow: "0 14px 32px rgba(0,0,0,0.12)",
      }}
    >
      {!hasData ? (
        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <p className="text-sm text-[#111111]/30">开始对话来收集行程信息</p>
          <p className="mt-1 text-xs text-[#111111]/20">已确认信息将在这里展示</p>
        </div>
      ) : (
        <>
          {/* 已确认信息 */}
          {confirmedInfo && (
            <div
              className="flex flex-col gap-2 rounded-xl p-3"
              style={{ background: "rgba(255,255,255,0.8)" }}
            >
              <h3 className="text-sm font-semibold text-[#111111]">已确认信息</h3>
              <div className="flex flex-col gap-1.5">
                {confirmedInfo.destination && (
                  <InfoRow label="目的地" value={confirmedInfo.destination} />
                )}
                {confirmedInfo.travel_dates && (
                  <InfoRow label="日期" value={confirmedInfo.travel_dates} />
                )}
                {confirmedInfo.travelers_count && (
                  <InfoRow label="人数" value={`${confirmedInfo.travelers_count} 人`} />
                )}
                {confirmedInfo.budget_range && (
                  <InfoRow label="预算" value={`¥${confirmedInfo.budget_range.toLocaleString()}`} />
                )}
                {confirmedInfo.travelers_type && (
                  <InfoRow label="类型" value={confirmedInfo.travelers_type} />
                )}
                {confirmedInfo.pace && (
                  <InfoRow label="节奏" value={confirmedInfo.pace} />
                )}
              </div>
            </div>
          )}

          {/* 行程概览 */}
          {briefItinerary && briefItinerary.length > 0 && (
            <div
              className="flex flex-col gap-2 rounded-xl p-3"
              style={{ background: "rgba(255,255,255,0.8)" }}
            >
              <h3 className="text-sm font-semibold text-[#111111]">行程概览</h3>

              {/* Day buttons with scroll */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => scrollDays("left")}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs text-[#111111]/50 transition-colors hover:bg-white/60"
                >
                  ◀
                </button>
                <div
                  ref={scrollRef}
                  className="flex gap-1.5 overflow-x-auto scrollbar-thin"
                  style={{ scrollbarWidth: "none" }}
                >
                  {briefItinerary.map((day) => (
                    <button
                      key={day.day_number}
                      onClick={() => setActiveBriefDay(day.day_number - 1)}
                      className="shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors"
                      style={
                        activeBriefDay === day.day_number - 1
                          ? { background: "#111111", color: "#FFFFFF" }
                          : { background: "rgba(255,255,255,0.6)", color: "#333333" }
                      }
                    >
                      DAY{day.day_number}
                    </button>
                  ))}
                </div>
                <button
                  onClick={() => scrollDays("right")}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs text-[#111111]/50 transition-colors hover:bg-white/60"
                >
                  ▶
                </button>
              </div>

              {/* Selected day highlights */}
              {briefItinerary[activeBriefDay] && (
                <div className="flex flex-col gap-1.5">
                  <p className="text-xs font-medium text-[#111111]/70">
                    {briefItinerary[activeBriefDay].theme}
                  </p>
                  {briefItinerary[activeBriefDay].highlights.map((h, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-[#111111]/30" />
                      <span className="text-xs text-[#111111]/80">{h}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 待确认建议 */}
          {pendingSuggestions.length > 0 && (
            <div
              className="flex flex-col gap-2 rounded-xl p-3"
              style={{ background: "rgba(255,255,255,0.8)" }}
            >
              <h3 className="text-sm font-semibold text-[#111111]">待确认建议</h3>
              <div className="flex flex-col gap-2">
                {pendingSuggestions.map((s) => (
                  <div
                    key={s.id}
                    className="flex items-start gap-2 rounded-lg px-2.5 py-2"
                    style={{ background: "rgba(255,255,255,0.5)" }}
                  >
                    <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#FF8400]" />
                    <span className="text-xs text-[#111111]/80">{s.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-[#111111]/50">{label}</span>
      <span className="text-xs font-medium text-[#111111]">{value}</span>
    </div>
  );
}
