import type { Activity } from "@/stores/chatStore";

interface ActivityItemProps {
  activity: Activity;
  index: number;
  isLast: boolean;
}

const categoryLabels: Record<string, string> = {
  attraction: "景点",
  restaurant: "餐饮",
  hotel: "住宿",
  transport: "交通",
};

const categoryDotColors: Record<string, string> = {
  attraction: "#FF8400",
  restaurant: "#D93C15",
  hotel: "#2563EB",
  transport: "#059669",
};

export function ActivityItem({ activity, index, isLast }: ActivityItemProps) {
  const label = categoryLabels[activity.category] || activity.category;
  const dotColor = categoryDotColors[activity.category] || "#666666";
  const timeText = activity.start_time
    ? activity.end_time
      ? `${activity.start_time}-${activity.end_time}`
      : activity.start_time
    : "";

  return (
    <div className="flex gap-3">
      {/* Timeline left */}
      <div className="flex flex-col items-center">
        {/* Time dot */}
        <div
          className="flex h-3 w-3 items-center justify-center rounded-full"
          style={{ background: dotColor }}
        />
        {/* Vertical line */}
        {!isLast && (
          <div
            className="mt-1 w-px flex-1"
            style={{ background: "rgba(0,0,0,0.08)" }}
          />
        )}
      </div>

      {/* Content right */}
      <div className={`flex-1 pb-4 ${isLast ? "" : ""}`}>
        {/* Time */}
        {timeText && (
          <span className="text-[11px] font-medium text-[#111111]/40">
            {timeText}
          </span>
        )}

        {/* Title row */}
        <div className="mt-0.5 flex items-center gap-2">
          <h4 className="text-[13px] font-medium text-[#111111]">
            {activity.poi_name}
          </h4>
          <span
            className="rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{
              background: `${dotColor}15`,
              color: dotColor,
            }}
          >
            {label}
          </span>
        </div>

        {/* Meta info */}
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-[#111111]/50">
          {activity.duration_min > 0 && (
            <span>{activity.duration_min} 分钟</span>
          )}
          {activity.ticket_price !== undefined && activity.ticket_price > 0 && (
            <span>预算 ¥{activity.ticket_price.toLocaleString()}</span>
          )}
        </div>

        {/* Recommendation reason */}
        {activity.recommendation_reason && (
          <p className="mt-1 text-[11px] text-[#111111]/50 leading-relaxed">
            {activity.recommendation_reason}
          </p>
        )}

        {/* Tags */}
        {activity.tags.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {activity.tags.map((tag) => (
              <span
                key={tag}
                className="rounded bg-[#111111]/5 px-1.5 py-0.5 text-[10px] text-[#111111]/50"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
