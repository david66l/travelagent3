"use client";

import { useState, useRef, useEffect } from "react";
import { useChatStore } from "@/stores/chatStore";
import { MessageBubble } from "./MessageBubble";
import { ThinkingBubble } from "./ThinkingBubble";
import { StreamingText } from "./StreamingText";
import { cn } from "@/lib/utils";

type SendStatus = "sent" | "queued" | "failed";

type ActionStatus = "sent" | "failed";

interface ChatPanelProps {
  sendMessage: (content: string) => SendStatus | Promise<SendStatus>;
  sendAction?: (
    action: "confirm" | "modify" | "reject" | "trip_event",
    payload?: { change?: unknown; external_event?: unknown }
  ) => ActionStatus | Promise<ActionStatus>;
}

const TRIP_EVENT_LABELS: Record<"closure" | "weather" | "delay", string> = {
  closure: "景点临时关闭",
  weather: "天气变化",
  delay: "行程延误",
};

export function ChatPanel({ sendMessage, sendAction }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const store = useChatStore();

  const sendDraftAction = async (action: "confirm" | "reject") => {
    if (!sendAction || store.isLoading) return;
    const label = action === "confirm" ? "确认行程" : "重新生成一版行程";
    store.addMessage({ role: "user", content: label, timestamp: Date.now() });
    store.setLoading(true);
    store.setActivityPhase("planning");
    store.setCurrentStage(
      action === "confirm" ? "正在查询实时信息并校验…" : "正在重新规划…"
    );
    const status = await sendAction(action);
    if (status === "failed") store.setLoading(false);
  };

  // In-trip disruption reporting → local replan (trip_event action).
  const [eventOpen, setEventOpen] = useState(false);
  const [evtType, setEvtType] = useState<"closure" | "weather" | "delay">("delay");
  const [evtPoi, setEvtPoi] = useState("");
  const [evtDetail, setEvtDetail] = useState("");

  const itineraryPois = (store.itinerary ?? []).flatMap((d) =>
    (d.activities ?? []).map((a) => a.poi_name).filter(Boolean)
  );
  const canReportEvent = !!sendAction && itineraryPois.length > 0;

  const handleTripEvent = async () => {
    if (!sendAction || store.isLoading) return;
    const externalEvent: Record<string, string> = {
      type: evtType,
      detail: evtDetail.trim(),
    };
    if (evtType === "closure" && evtPoi) externalEvent.poi = evtPoi;

    const summary = [
      `[行程突发] ${TRIP_EVENT_LABELS[evtType]}`,
      evtType === "closure" && evtPoi ? `· ${evtPoi}` : "",
      evtDetail.trim() ? `· ${evtDetail.trim()}` : "",
    ]
      .filter(Boolean)
      .join(" ");
    store.addMessage({ role: "user", content: summary, timestamp: Date.now() });
    store.setLoading(true);
    store.setCurrentStage("正在根据突发情况调整行程…");
    setEventOpen(false);
    setEvtDetail("");

    try {
      const status = await sendAction("trip_event", { external_event: externalEvent });
      if (status === "failed") {
        store.addMessage({
          role: "assistant",
          content: "上报突发情况失败，请重试。",
          timestamp: Date.now(),
        });
        store.setLoading(false);
      }
    } catch {
      store.setLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || store.isLoading) return;

    const content = input.trim();
    if (store.isStreaming) {
      store.stopStreaming();
      store.setStreamingContent("");
    }
    store.addMessage({
      role: "user",
      content,
      timestamp: Date.now(),
    });
    store.setLoading(true);
    store.setCurrentStage("让我思考一下…");
    store.setActivityPhase("gathering");
    setInput("");

    try {
      const status = await sendMessage(content);
      if (status === "failed") {
        store.addMessage({
          role: "assistant",
          content: "连接已断开，请刷新页面重试。",
          timestamp: Date.now(),
        });
        store.setLoading(false);
      }
    } catch {
      store.addMessage({
        role: "assistant",
        content: "发送失败，请检查网络后重试。",
        timestamp: Date.now(),
      });
      store.setLoading(false);
    }
  };

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [store.messages]);

  const isEmpty = store.messages.length === 0;

  return (
    <div className="glass-card flex h-full flex-col gap-3.5 rounded-4xl p-4">
      {/* Empty State Header */}
      {isEmpty && (
        <div className="flex flex-col gap-1.5">
          <h1 className="text-[38px] font-semibold leading-tight text-ink">
            几分钟内生成你的首个行程
          </h1>
          <p className="font-mono text-[13px] text-mute">
            告诉我目的地、日期和预算，我会在对话中实时生成行程。
          </p>
        </div>
      )}

      {/* Messages */}
      <div
        ref={scrollRef}
        data-testid="messages-container"
        className="flex-1 space-y-2.5 overflow-y-auto py-1 scrollbar-thin"
      >
        {isEmpty && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <p className="text-lg font-medium text-mute/60">欢迎</p>
            <p className="mt-1 text-sm text-mute/40">
              告诉我您想去哪里旅行？
            </p>
          </div>
        )}
        {store.messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} />
        ))}
        {store.isStreaming && (
          <div className="flex w-full flex-col items-start">
            <div className="glass-message-ai max-w-[560px] px-3 py-2.5 text-[13px] leading-relaxed">
              <StreamingText
                content={store.streamingContent}
                isStreaming={store.isStreaming}
              />
            </div>
          </div>
        )}
        {store.isLoading && !store.isStreaming && <ThinkingBubble />}
      </div>

      {/* In-trip disruption reporting → triggers a local replan */}
      {canReportEvent && (
        <div className="flex flex-col gap-2">
          {!eventOpen ? (
            <button
              data-testid="trip-event-toggle"
              onClick={() => setEventOpen(true)}
              disabled={store.isLoading}
              className="self-start rounded-2xl border border-hairline bg-canvas px-3 py-1.5 font-mono text-xs text-mute transition-colors hover:text-ink disabled:opacity-50"
            >
              🚨 行程中遇到突发情况
            </button>
          ) : (
            <div className="glass-card flex flex-col gap-2.5 rounded-[20px] p-3">
              <div className="flex items-center justify-between">
                <span className="text-[13px] font-semibold text-ink">上报突发情况，重新调整行程</span>
                <button
                  onClick={() => setEventOpen(false)}
                  className="font-mono text-xs text-mute hover:text-ink"
                >
                  取消
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {(Object.keys(TRIP_EVENT_LABELS) as Array<keyof typeof TRIP_EVENT_LABELS>).map(
                  (t) => (
                    <button
                      key={t}
                      onClick={() => setEvtType(t)}
                      className={cn(
                        "rounded-xl px-2.5 py-1 font-mono text-xs transition-colors",
                        evtType === t
                          ? "bg-primary text-ink"
                          : "border border-hairline bg-canvas text-mute hover:text-ink"
                      )}
                    >
                      {TRIP_EVENT_LABELS[t]}
                    </button>
                  )
                )}
              </div>
              {evtType === "closure" && (
                <select
                  data-testid="trip-event-poi"
                  value={evtPoi}
                  onChange={(e) => setEvtPoi(e.target.value)}
                  className="rounded-2xl border border-hairline bg-canvas px-3 py-2 font-mono text-xs text-ink outline-none"
                >
                  <option value="">选择受影响的景点…</option>
                  {Array.from(new Set(itineraryPois)).map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))}
                </select>
              )}
              <input
                type="text"
                data-testid="trip-event-detail"
                value={evtDetail}
                onChange={(e) => setEvtDetail(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleTripEvent()}
                placeholder={
                  evtType === "weather"
                    ? "例如：下午有雷阵雨"
                    : evtType === "delay"
                      ? "例如：航班延误 2 小时"
                      : "补充说明（可选）"
                }
                className="rounded-2xl border border-hairline bg-canvas px-3 py-2 font-mono text-xs text-ink placeholder:text-mute outline-none"
              />
              <button
                data-testid="trip-event-submit"
                onClick={handleTripEvent}
                disabled={store.isLoading || (evtType === "closure" && !evtPoi)}
                className={cn(
                  "self-end rounded-2xl px-4 py-2 text-xs font-semibold transition-all",
                  !store.isLoading && !(evtType === "closure" && !evtPoi)
                    ? "bg-primary text-ink shadow-panel hover:bg-primary-active"
                    : "cursor-not-allowed bg-hairline text-mute"
                )}
              >
                提交并调整
              </button>
            </div>
          )}
        </div>
      )}

      {store.waitingForConfirmation && sendAction && (
        <div className="glass-card flex items-center justify-between gap-3 rounded-[20px] p-3">
          <div>
            <p className="text-sm font-semibold text-ink">这是行程草案</p>
            <p className="text-xs text-mute">可在行程页替换或删除景点，满意后再确认。</p>
          </div>
          <div className="flex shrink-0 gap-2">
            <button
              data-testid="regenerate-itinerary"
              onClick={() => sendDraftAction("reject")}
              disabled={store.isLoading}
              className="rounded-xl border border-hairline px-3 py-2 text-xs text-mute hover:text-ink disabled:opacity-50"
            >
              重新生成
            </button>
            <button
              data-testid="confirm-itinerary"
              onClick={() => sendDraftAction("confirm")}
              disabled={store.isLoading}
              className="btn-primary-dark px-4 py-2 text-xs disabled:opacity-50"
            >
              确认并完善
            </button>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="glass-card flex items-center gap-2.5 rounded-[20px] p-2.5">
        <div className="flex flex-1 items-center rounded-2xl border border-hairline bg-canvas px-3.5 py-3">
          <input
            type="text"
            data-testid="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={store.waitingForConfirmation}
            placeholder={
              store.waitingForConfirmation
                ? "请先确认草案，或前往行程页进行修改"
                : "试试：'成都 4 天，预算 3000 元，喜欢火锅和历史文化'"
            }
            className="w-full bg-transparent font-mono text-sm text-ink placeholder:text-mute outline-none"
          />
        </div>
        <button
          data-testid="send-button"
          onClick={handleSend}
          disabled={!input.trim() || store.isLoading || store.waitingForConfirmation}
          className={cn(
            "flex items-center gap-1 rounded-2xl px-4 py-3 text-sm font-semibold transition-all",
            input.trim() && !store.isLoading
              ? "bg-primary text-ink shadow-panel hover:bg-primary-active"
              : "bg-hairline text-mute cursor-not-allowed"
          )}
        >
          <span>发送</span>
        </button>
      </div>
    </div>
  );
}
