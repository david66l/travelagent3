"use client";

import { useChatStore } from "@/stores/chatStore";

export function TopBar() {
  const store = useChatStore();
  const routing = store.policyRouting;

  return (
    <div className="glass-topbar flex h-[72px] items-center justify-between rounded-[22px] px-[18px] py-[14px]">
      {/* Logo */}
      <div className="flex items-center gap-2.5">
        <div className="h-5 w-5 rounded-full bg-primary" />
        <span className="text-xl font-semibold text-ink">旅行助手</span>
      </div>

      <div className="flex items-center gap-3">
        {routing && (
          <div
            className="hidden items-center gap-2 rounded-full border border-hairline bg-canvas-soft px-3 py-1.5 text-xs text-mute shadow-card sm:flex"
            title={`本轮生成 ${routing.completion_tokens} tokens，模型耗时 ${routing.request_latency_ms.toFixed(0)} ms`}
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            <span className="font-medium text-ink">智能路由</span>
            <span>4B {routing.route_counts.student}</span>
            <span>8B {routing.route_counts.teacher}</span>
            {routing.fallback_count > 0 && (
              <span className="text-amber-700">升级 {routing.fallback_count}</span>
            )}
          </div>
        )}

        {/* Avatar */}
        <button
          onClick={() => store.setActiveView("settings")}
          className="h-[38px] w-[38px] rounded-full border border-hairline bg-canvas shadow-card transition-transform hover:scale-105"
        />
      </div>
    </div>
  );
}
