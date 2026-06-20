"use client";

import { useRef } from "react";
import { useChatStore, deriveBriefItinerary } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

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
    <div className="glass-card flex h-full flex-col gap-3 rounded-4xl p-4">
      {!hasData ? (
        <div className="flex flex-1 flex-col items-center justify-center text-center">
          <p className="text-sm text-mute/60">开始对话来收集行程信息</p>
          <p className="mt-1 text-xs text-mute/40">已确认信息将在这里展示</p>
        </div>
      ) : (
        <>
          {/* 已确认信息 */}
          {confirmedInfo && (
            <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
              <h3 className="text-sm font-semibold text-ink">已确认信息</h3>
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
            <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
              <h3 className="text-sm font-semibold text-ink">行程概览</h3>

              {/* Day buttons with scroll */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => scrollDays("left")}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs text-mute transition-colors hover:bg-canvas"
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
                      className={cn(
                        "shrink-0 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                        activeBriefDay === day.day_number - 1
                          ? "bg-ink text-canvas"
                          : "bg-canvas text-body hover:bg-primary-pale"
                      )}
                    >
                      DAY{day.day_number}
                    </button>
                  ))}
                </div>
                <button
                  onClick={() => scrollDays("right")}
                  className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs text-mute transition-colors hover:bg-canvas"
                >
                  ▶
                </button>
              </div>

              {/* Selected day highlights */}
              {briefItinerary[activeBriefDay] && (
                <div className="flex flex-col gap-1.5">
                  <p className="text-xs font-medium text-body">
                    {briefItinerary[activeBriefDay].theme}
                  </p>
                  {briefItinerary[activeBriefDay].highlights.map((h, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="h-1 w-1 rounded-full bg-mute/60" />
                      <span className="text-xs text-body">{h}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 待确认建议 */}
          {pendingSuggestions.length > 0 && (
            <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
              <h3 className="text-sm font-semibold text-ink">待确认建议</h3>
              <div className="flex flex-col gap-2">
                {pendingSuggestions.map((s) => (
                  <div
                    key={s.id}
                    className="flex items-start gap-2 rounded-lg bg-canvas px-2.5 py-2"
                  >
                    <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-orange" />
                    <span className="text-xs text-body">{s.text}</span>
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
      <span className="text-xs text-mute">{label}</span>
      <span className="text-xs font-medium text-ink">{value}</span>
    </div>
  );
}
