"use client";

import { useState } from "react";
import type { Activity } from "@/stores/chatStore";

interface ActivityItemProps {
  activity: Activity;
  index: number;
  isLast: boolean;
  dayNumber: number;
  onModify?: (message: string) => void | Promise<void>;
  isLoading?: boolean;
}

const categoryLabels: Record<string, string> = {
  attraction: "景点",
  restaurant: "餐饮",
  hotel: "住宿",
  transport: "交通",
};

const categoryDotClasses: Record<string, string> = {
  attraction: "bg-accent-orange",
  restaurant: "bg-negative",
  hotel: "bg-accent-cyan",
  transport: "bg-positive",
};

const categoryBadgeClasses: Record<string, string> = {
  attraction: "bg-accent-orange/15 text-accent-orange",
  restaurant: "bg-negative/15 text-negative",
  hotel: "bg-accent-cyan/15 text-accent-cyan",
  transport: "bg-positive/15 text-positive",
};

export function ActivityItem({
  activity,
  index,
  isLast,
  dayNumber,
  onModify,
  isLoading = false,
}: ActivityItemProps) {
  const label = categoryLabels[activity.category] || activity.category;
  const dotClass = categoryDotClasses[activity.category] || "bg-mute";
  const badgeClass = categoryBadgeClasses[activity.category] || "bg-hairline-soft text-mute";
  const timeText = activity.start_time
    ? activity.end_time
      ? `${activity.start_time}-${activity.end_time}`
      : activity.start_time
    : "";

  const [replaceName, setReplaceName] = useState("");
  const [showReplaceInput, setShowReplaceInput] = useState(false);

  const handleReplace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!replaceName.trim() || isLoading || !onModify) return;
    await onModify(
      `把第 ${dayNumber} 天的 ${activity.poi_name} 替换为 ${replaceName.trim()}`
    );
    setReplaceName("");
    setShowReplaceInput(false);
  };

  const handleDelete = async () => {
    if (isLoading || !onModify) return;
    await onModify(`删除第 ${dayNumber} 天的 ${activity.poi_name}`);
  };

  const buttonBase =
    "rounded-lg border border-hairline bg-canvas-soft px-2 py-1 text-[10px] font-medium text-ink/70 transition-colors hover:bg-primary-pale hover:text-ink disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div className="flex gap-3">
      {/* Timeline left */}
      <div className="flex flex-col items-center">
        {/* Time dot */}
        <div className={`flex h-3 w-3 items-center justify-center rounded-full ${dotClass}`} />
        {/* Vertical line */}
        {!isLast && (
          <div className="mt-1 w-px flex-1 bg-hairline-soft" />
        )}
      </div>

      {/* Content right */}
      <div className="flex-1 pb-4">
        {/* Time */}
        {timeText && (
          <span className="text-[11px] font-medium text-mute/60">
            {timeText}
          </span>
        )}

        {/* Title row */}
        <div className="mt-0.5 flex items-center gap-2">
          <h4 className="text-[13px] font-medium text-ink">
            {activity.poi_name}
          </h4>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass}`}
          >
            {label}
          </span>
        </div>

        {/* Meta info */}
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-mute">
          {activity.duration_min > 0 && (
            <span>{activity.duration_min} 分钟</span>
          )}
          {activity.ticket_price !== undefined && activity.ticket_price > 0 && (
            <span>预算 ¥{activity.ticket_price.toLocaleString()}</span>
          )}
        </div>

        {/* Recommendation reason */}
        {activity.recommendation_reason && (
          <p className="mt-1 text-[11px] text-mute leading-relaxed">
            {activity.recommendation_reason}
          </p>
        )}

        {/* Tags */}
        {activity.tags.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {activity.tags.map((tag) => (
              <span
                key={tag}
                className="rounded bg-canvas-soft px-1.5 py-0.5 text-[10px] text-mute"
              >
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Per-activity actions */}
        {onModify && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {showReplaceInput ? (
              <form
                onSubmit={handleReplace}
                className="flex items-center gap-1"
              >
                <input
                  type="text"
                  value={replaceName}
                  onChange={(e) => setReplaceName(e.target.value)}
                  placeholder="新景点名称"
                  disabled={isLoading}
                  autoFocus
                  className="input-wise w-32 py-1 px-2 text-[11px]"
                />
                <button
                  type="submit"
                  disabled={isLoading || !replaceName.trim()}
                  className={buttonBase}
                >
                  确认
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowReplaceInput(false);
                    setReplaceName("");
                  }}
                  disabled={isLoading}
                  className={buttonBase}
                >
                  取消
                </button>
              </form>
            ) : (
              <button
                onClick={() => setShowReplaceInput(true)}
                disabled={isLoading}
                className={buttonBase}
              >
                替换景点
              </button>
            )}
            <button
              onClick={handleDelete}
              disabled={isLoading}
              className={buttonBase}
            >
              删除景点
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
