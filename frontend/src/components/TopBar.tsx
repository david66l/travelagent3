"use client";

import { useChatStore } from "@/stores/chatStore";

export function TopBar() {
  const store = useChatStore();

  return (
    <div className="glass-topbar flex h-[72px] items-center justify-between rounded-[22px] px-[18px] py-[14px]">
      {/* Logo */}
      <div className="flex items-center gap-2.5">
        <div className="h-5 w-5 rounded-full bg-primary" />
        <span className="text-xl font-semibold text-ink">旅行助手</span>
      </div>

      {/* Avatar */}
      <button
        onClick={() => store.setActiveView("settings")}
        className="h-[38px] w-[38px] rounded-full border border-hairline bg-canvas shadow-card transition-transform hover:scale-105"
      />
    </div>
  );
}
