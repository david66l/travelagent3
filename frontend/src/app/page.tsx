"use client";

import { useEffect } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useWebSocket } from "@/hooks/useWebSocket";
import { TopBar } from "@/components/TopBar";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { ItineraryPanel } from "@/components/ItineraryPanel";
import { PanelSidebar } from "@/components/PanelSidebar";
import { PreviewPanel } from "@/components/PreviewPanel";
import { ExportCenter } from "@/components/ExportCenter";
import { SettingsPanel } from "@/components/SettingsPanel";
import { cn } from "@/lib/utils";

export default function Home() {
  const { sendMessage, reconnect } = useWebSocket();
  const store = useChatStore();
  const { activeView, activeTab } = store;

  // 定期刷新行程状态（upcoming → active → completed）
  useEffect(() => {
    store.refreshTripStatuses();
    const interval = setInterval(() => store.refreshTripStatuses(), 60000);
    return () => clearInterval(interval);
  }, []);

  const handleNewChat = () => {
    store.setActiveView("chat");
    store.clear();
    reconnect();
  };

  return (
    <main className="flex h-full flex-col gap-5 p-5">
      {/* TopBar */}
      <TopBar />

      {/* Mobile tab bar */}
      <div className="flex border-b border-white/30 bg-white/40 backdrop-blur-md md:hidden">
        {(["chat", "itinerary", "panels"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => store.setActiveTab(tab)}
            className={cn(
              "flex-1 px-4 py-2.5 text-sm font-medium",
              activeTab === tab
                ? "border-b-2 border-[#111111] text-[#111111]"
                : "text-[#111111]/50"
            )}
          >
            {tab === "chat" && "对话"}
            {tab === "itinerary" && "行程"}
            {tab === "panels" && "面板"}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-4 overflow-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden md:block md:h-full">
          <Sidebar onNewChat={handleNewChat} />
        </div>

        {/* Main Panel - Desktop */}
        <div className="hidden flex-1 gap-4 md:flex md:h-full">
          {activeView === "chat" && (
            <>
              <div className="flex-1">
                <ChatPanel sendMessage={sendMessage} />
              </div>
              <div className="w-[360px]">
                <PreviewPanel />
              </div>
            </>
          )}
          {activeView === "itinerary" && (
            <>
              <div className="flex-1">
                <ItineraryPanel />
              </div>
              <div className="w-[350px]">
                <PanelSidebar />
              </div>
            </>
          )}
          {activeView === "export" && (
            <div className="flex flex-1">
              <ExportCenter />
            </div>
          )}
          {activeView === "settings" && (
            <div className="flex flex-1">
              <SettingsPanel />
            </div>
          )}
        </div>

        {/* Mobile Panels */}
        <div className="flex w-full flex-col overflow-hidden md:hidden">
          {activeTab === "chat" && (
            <div className="flex flex-1 flex-col gap-2 overflow-hidden">
              <div className="h-[220px] shrink-0">
                <PreviewPanel />
              </div>
              <div className="min-h-0 flex-1">
                <ChatPanel sendMessage={sendMessage} />
              </div>
            </div>
          )}
          {activeTab === "itinerary" && (
            <ItineraryPanel />
          )}
          {activeTab === "panels" && (
            <PanelSidebar />
          )}
        </div>
      </div>
    </main>
  );
}
