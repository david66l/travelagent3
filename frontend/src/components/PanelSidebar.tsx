"use client";

import { useChatStore } from "@/stores/chatStore";

export function PanelSidebar() {
  const store = useChatStore();
  const { budgetPanel, preferencePanel } = store;

  const totalBudget = budgetPanel?.total_budget ?? 0;
  const spent = budgetPanel?.spent ?? 0;
  const remaining = budgetPanel?.remaining ?? (totalBudget - spent);
  const usagePercent = totalBudget > 0 ? (spent / totalBudget) * 100 : 0;

  const breakdownEntries = budgetPanel?.breakdown
    ? Object.entries(budgetPanel.breakdown)
    : [];
  const maxBreakdown =
    breakdownEntries.length > 0
      ? Math.max(...breakdownEntries.map(([, v]) => v))
      : 1;

  return (
    <div className="glass-card flex h-full flex-col gap-2.5 rounded-3xl p-3">
      {/* Budget Section */}
      <div className="flex flex-col gap-2.5">
        <h3 className="text-base font-semibold text-ink">预算预览</h3>

        {!budgetPanel ? (
          <p className="text-xs text-mute/60">暂无预算数据</p>
        ) : (
          <>
            {/* Summary Cards Row */}
            <div className="flex gap-2">
              <BudgetCard label="预算总额" value={`¥${totalBudget.toLocaleString()}`} />
              <BudgetCard label="已使用" value={`¥${spent.toLocaleString()}`} />
              <BudgetCard label="可用余额" value={`¥${remaining.toLocaleString()}`} />
            </div>

            {/* Progress Card */}
            <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-body">
                  预算使用率
                </span>
                <span className="text-xs font-bold text-ink">
                  {usagePercent.toFixed(1)}% 已用
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-hairline-soft">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{ width: `${Math.min(usagePercent, 100)}%` }}
                />
              </div>
            </div>

            {/* Category Breakdown */}
            {breakdownEntries.length > 0 && (
              <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
                <span className="text-xs font-semibold text-body">
                  分类支出
                </span>
                {breakdownEntries.map(([key, val]) => {
                  const pct = maxBreakdown > 0 ? (val / maxBreakdown) * 100 : 0;
                  return (
                    <div key={key} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-body">
                          {translateBudgetKey(key)}
                        </span>
                        <span className="text-xs font-medium text-ink">
                          ¥{val?.toLocaleString() ?? "0"}
                        </span>
                      </div>
                      <div className="h-2 w-full overflow-hidden rounded-full bg-hairline-soft">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{
                            width: `${pct}%`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>

      {/* Preference Section */}
      <div className="flex flex-col gap-2">
        <h3 className="text-base font-semibold text-ink">偏好</h3>

        {!preferencePanel ? (
          <p className="text-xs text-mute/60">暂无偏好数据</p>
        ) : (
          <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
            {preferencePanel.destination && (
              <PrefLine label="目的地" value={preferencePanel.destination} />
            )}
            {preferencePanel.travel_days && (
              <PrefLine
                label="天数"
                value={`${preferencePanel.travel_days} 天`}
              />
            )}
            {preferencePanel.travel_dates && (
              <PrefLine label="日期" value={preferencePanel.travel_dates} />
            )}
            {preferencePanel.travelers_count && (
              <PrefLine
                label="人数"
                value={`${preferencePanel.travelers_count} 人`}
              />
            )}
            {preferencePanel.travelers_type && (
              <PrefLine label="类型" value={preferencePanel.travelers_type} />
            )}
            {preferencePanel.budget_range && (
              <PrefLine
                label="预算"
                value={`¥${preferencePanel.budget_range.toLocaleString()}`}
              />
            )}
            {preferencePanel.pace && (
              <PrefLine label="节奏" value={translatePace(preferencePanel.pace)} />
            )}
            {preferencePanel.interests?.length > 0 && (
              <PrefLine
                label="兴趣"
                value={preferencePanel.interests.join("、")}
              />
            )}
            {preferencePanel.food_preferences?.length > 0 && (
              <PrefLine
                label="饮食"
                value={preferencePanel.food_preferences.join("、")}
              />
            )}
            {preferencePanel.special_requests?.length > 0 && (
              <PrefLine
                label="特殊要求"
                value={preferencePanel.special_requests.join("、")}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function BudgetCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-1 flex-col gap-1 rounded-xl bg-canvas-soft p-2.5">
      <span className="text-xs font-medium text-mute">{label}</span>
      <span className="text-lg font-bold text-ink">{value}</span>
    </div>
  );
}

function PrefLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2">
      <span className="shrink-0 text-xs font-medium text-mute">
        {label}:
      </span>
      <span className="text-[13px] text-ink">{value}</span>
    </div>
  );
}

function translateBudgetKey(key: string): string {
  const map: Record<string, string> = {
    accommodation: "住宿",
    meals: "餐饮",
    transport: "交通",
    tickets: "门票",
    shopping: "购物",
    buffer: "缓冲",
  };
  return map[key] || key;
}

function translatePace(pace?: string): string {
  const map: Record<string, string> = {
    relaxed: "轻松",
    moderate: "适中",
    intensive: "紧凑",
  };
  return map[pace || ""] || pace || "";
}
