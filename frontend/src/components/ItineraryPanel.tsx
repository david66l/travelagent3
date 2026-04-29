"use client";

import { useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import { DayCard } from "./DayCard";

export function ItineraryPanel() {
  const store = useChatStore();
  const { itinerary, currentTrip } = store;
  const [activeDay, setActiveDay] = useState(0);

  const tripTitle = currentTrip
    ? `${currentTrip.title}`
    : "行程编排台";

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
            {itinerary[activeDay] && (
              <DayCard day={itinerary[activeDay]} />
            )}
          </div>
        </>
      )}
    </div>
  );
}
