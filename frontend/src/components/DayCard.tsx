import type { DayPlan } from "@/stores/chatStore";
import { ActivityItem } from "./ActivityItem";

interface DayCardProps {
  day: DayPlan;
  onModify?: (message: string) => void | Promise<void>;
  isLoading?: boolean;
}

export function DayCard({ day, onModify, isLoading }: DayCardProps) {
  return (
    <div className="flex flex-col">
      {/* Day header */}
      <div className="mb-3 flex items-center gap-2">
        <span className="rounded-full bg-ink px-3 py-1 text-xs font-medium text-canvas">
          第 {day.day_number} 天
        </span>
        {day.date && (
          <span className="text-xs text-mute">{day.date}</span>
        )}
        {day.theme && (
          <span className="text-xs text-body">{day.theme}</span>
        )}
      </div>

      {/* Activities timeline */}
      <div className="flex flex-col">
        {day.activities.map((activity, idx) => (
          <ActivityItem
            key={idx}
            activity={activity}
            index={idx}
            isLast={idx === day.activities.length - 1}
            dayNumber={day.day_number}
            onModify={onModify}
            isLoading={isLoading}
          />
        ))}
      </div>

      {/* Day total cost */}
      {day.total_cost > 0 && (
        <div className="mt-2 text-right text-xs text-mute/60">
          当日花费: ¥{day.total_cost.toFixed(0)}
        </div>
      )}
    </div>
  );
}
