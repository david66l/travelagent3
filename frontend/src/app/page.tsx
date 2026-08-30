"use client";

import { useEffect, useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import { useChat } from "@/hooks/useChat";
import { TopBar } from "@/components/TopBar";
import { Sidebar } from "@/components/Sidebar";
import { ChatPanel } from "@/components/ChatPanel";
import { ItineraryPanel } from "@/components/ItineraryPanel";
import { PanelSidebar } from "@/components/PanelSidebar";
import { PreviewPanel } from "@/components/PreviewPanel";
import { ExportCenter } from "@/components/ExportCenter";
import { BookingPanel } from "@/components/BookingPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { cn } from "@/lib/utils";

export default function Home() {
  const { sendMessage, sendAction, reconnect } = useChat();
  const store = useChatStore();
  const [isStartingNewChat, setIsStartingNewChat] = useState(false);
  const { activeView, activeTab, refreshTripStatuses } = store;
  const canModifyCurrentItinerary =
    !store.currentTrip ||
    (!!store.currentTrip.conversationId &&
      store.currentTrip.conversationId === store.sessionId);

  // 定期刷新行程状态（upcoming → active → completed）
  useEffect(() => {
    refreshTripStatuses();
    const interval = setInterval(() => refreshTripStatuses(), 60000);
    return () => clearInterval(interval);
  }, [refreshTripStatuses]);

  const handleNewChat = async () => {
    if (isStartingNewChat) return;
    setIsStartingNewChat(true);
    store.setActiveView("chat");
    try {
      await reconnect();
    } finally {
      setIsStartingNewChat(false);
    }
  };

  return (
    <main className="flex h-full flex-col gap-5 p-5">
      {/* TopBar */}
      <TopBar />

      {/* Mobile tab bar */}
      <div className="flex border-b border-hairline bg-canvas-soft/60 backdrop-blur-md md:hidden">
        {(["chat", "itinerary", "panels"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => store.setActiveTab(tab)}
            className={cn(
              "flex-1 px-4 py-2.5 text-sm font-medium",
              activeTab === tab
                ? "border-b-2 border-ink text-ink"
                : "text-mute"
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
          <Sidebar
            onNewChat={handleNewChat}
            isStartingNewChat={isStartingNewChat}
          />
        </div>

        {/* Main content. Stateful panels are mounted exactly once; responsive
            classes only change their placement and visibility. */}
        <div className="flex min-w-0 flex-1 flex-col gap-2 overflow-hidden md:h-full md:flex-row md:gap-4">
          {activeView === "chat" && (
            <>
              <div
                className={cn(
                  "order-2 min-h-0 flex-1 md:order-1 md:block",
                  activeTab === "chat" ? "block" : "hidden"
                )}
              >
                <ChatPanel sendMessage={sendMessage} sendAction={sendAction} />
              </div>
              <div
                className={cn(
                  "order-1 h-[220px] shrink-0 md:order-2 md:block md:h-auto md:w-[360px]",
                  activeTab === "chat" ? "block" : "hidden"
                )}
              >
                <PreviewPanel />
              </div>
            </>
          )}
          {activeView === "itinerary" && (
            <>
              <div className="flex-1">
                <ItineraryPanel
                  onModify={canModifyCurrentItinerary ? async (change) => {
                    store.addMessage({ role: "user", content: "修改行程草案", timestamp: Date.now() });
                    await sendAction("modify", { change });
                  } : undefined}
                  onConfirm={async () => { await sendAction("confirm"); }}
                  onRegenerate={async () => { await sendAction("reject"); }}
                />
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
          {activeView === "booking" && (
            <div className="flex flex-1">
              <BookingPanel />
            </div>
          )}
          {activeView === "settings" && (
            <div className="flex flex-1">
              <SettingsPanel />
            </div>
          )}
          {activeView === "chat" && activeTab === "itinerary" && (
            <div className="flex min-h-0 flex-1 md:hidden">
              <ItineraryPanel
                onModify={canModifyCurrentItinerary ? async (change) => {
                  store.addMessage({ role: "user", content: "修改行程草案", timestamp: Date.now() });
                  await sendAction("modify", { change });
                } : undefined}
                onConfirm={async () => { await sendAction("confirm"); }}
                onRegenerate={async () => { await sendAction("reject"); }}
              />
            </div>
          )}
          {activeView === "chat" && activeTab === "panels" && (
            <div className="flex min-h-0 flex-1 md:hidden">
              <PanelSidebar />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
