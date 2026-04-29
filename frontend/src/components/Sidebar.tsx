"use client";

import { useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

interface SidebarProps {
  onNewChat: () => void;
}

export function Sidebar({ onNewChat }: SidebarProps) {
  const store = useChatStore();
  const { chatSnapshots, trips, activeView, setActiveView, loadTrip, restoreChat, clearRunStatus } = store;

  const [dialogExpanded, setDialogExpanded] = useState(true);
  const [tripExpanded, setTripExpanded] = useState(true);

  // 找到最近行程（当前行程按钮用）
  const findNearestTrip = () => {
    if (trips.length === 0) return null;
    const now = new Date().toISOString().split("T")[0];
    // 按 startDate 排序，找最近的
    const sorted = [...trips].sort(
      (a, b) =>
        new Date(a.startDate).getTime() - new Date(b.startDate).getTime()
    );
    // 找 startDate >= today 的第一个，或最后一个已过期的
    const upcoming = sorted.find((t) => t.startDate >= now);
    return upcoming || sorted[sorted.length - 1];
  };

  const handleCurrentTrip = () => {
    const trip = findNearestTrip();
    if (trip) {
      loadTrip(trip.id);
      setActiveView("itinerary");
    } else {
      // 暂无行程时给出反馈
      alert("暂无行程，请先规划并确认一个行程。");
    }
  };

  return (
    <div className="glass-sidebar flex h-full w-[250px] flex-col gap-2.5 rounded-3xl p-3.5">
      {/* New Chat Button */}
      <button
        onClick={onNewChat}
        className="btn-primary-dark w-full text-center transition-colors"
      >
        新建对话
      </button>

      {/* Current Trip Button */}
      <button
        onClick={handleCurrentTrip}
        className={cn(
          "w-full rounded-[10px] px-2.5 py-2 text-[13px] font-medium transition-colors",
          activeView === "itinerary"
            ? "bg-[#111111] text-white"
            : "bg-[#111111] text-white hover:bg-[#333333]"
        )}
      >
        当前行程
      </button>

      {/* Dialog Group */}
      <div className="group-header flex flex-col gap-2 p-2">
        <button
          onClick={() => setDialogExpanded(!dialogExpanded)}
          className="flex items-center justify-between px-1 py-0.5"
        >
          <span className="text-sm font-semibold text-[#111111]/85">对话</span>
          <span
            className="text-sm font-semibold text-[#111111]/50 transition-transform"
            style={{ transform: dialogExpanded ? "rotate(0deg)" : "rotate(-90deg)" }}
          >
            ▾
          </span>
        </button>
        {dialogExpanded && (
          <div className="flex flex-col gap-1.5">
            {chatSnapshots.length === 0 ? (
              <p className="px-2 py-1.5 text-xs text-[#111111]/30">暂无对话</p>
            ) : (
              chatSnapshots.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => {
                    restoreChat(chat.id);
                    setActiveView("chat");
                  }}
                  className="w-full px-2.5 py-2 text-left text-[13px] font-medium transition-colors chat-item-inactive"
                >
                  {chat.title}
                </button>
              ))
            )}
          </div>
        )}
      </div>

      {/* Trip Group */}
      <div className="group-header flex flex-col gap-2 p-2">
        <button
          onClick={() => setTripExpanded(!tripExpanded)}
          className="flex items-center justify-between px-1 py-0.5"
        >
          <span className="text-sm font-semibold text-[#111111]/85">历史行程</span>
          <span
            className="text-sm font-semibold text-[#111111]/50 transition-transform"
            style={{ transform: tripExpanded ? "rotate(0deg)" : "rotate(-90deg)" }}
          >
            ▾
          </span>
        </button>
        {tripExpanded && (
          <div className="flex flex-col gap-1.5">
            {trips.length === 0 ? (
              <p className="px-2 py-1.5 text-xs text-[#111111]/30">暂无行程</p>
            ) : (
              trips.map((trip) => (
                <button
                  key={trip.id}
                  onClick={() => {
                    loadTrip(trip.id);
                    setActiveView("itinerary");
                  }}
                  className="flex w-full items-center justify-between px-2.5 py-2 text-left text-[13px] font-medium transition-colors chat-item-inactive"
                >
                  <span>{trip.title}</span>
                  <span className="text-[11px] text-[#111111]/40">
                    {trip.status === "active"
                      ? "(进行中)"
                      : trip.status === "upcoming"
                        ? "(即将出发)"
                        : "(已结束)"}
                  </span>
                </button>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
