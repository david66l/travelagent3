"use client";

import { useState, useRef, useEffect } from "react";
import { CheckCircle, XCircle } from "lucide-react";
import { useChatStore } from "@/stores/chatStore";
import { MessageBubble } from "./MessageBubble";
import { ThinkingBubble } from "./ThinkingBubble";
import { cn } from "@/lib/utils";

interface ChatPanelProps {
  sendMessage: (content: string) => boolean;
}

export function ChatPanel({ sendMessage }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const store = useChatStore();

  const handleSend = () => {
    if (!input.trim() || store.isLoading) return;

    // Clear previous run status before starting a new turn
    store.clearRunStatus();
    store.addMessage({
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    });
    store.setLoading(true);

    const ok = sendMessage(input.trim());
    if (!ok) {
      store.addMessage({
        role: "assistant",
        content: "连接已断开，请刷新页面重试。",
        timestamp: Date.now(),
      });
      store.setLoading(false);
    }
    setInput("");
  };

  const handleConfirm = () => {
    if (store.isLoading) return;
    store.clearRunStatus();
    store.addMessage({
      role: "user",
      content: "确认行程",
      timestamp: Date.now(),
    });
    store.setLoading(true);
    sendMessage("确认行程");
  };

  const handleModify = () => {
    if (store.isLoading) return;
    store.clearRunStatus();
    store.addMessage({
      role: "user",
      content: "继续修改行程",
      timestamp: Date.now(),
    });
    store.setLoading(true);
    sendMessage("继续修改行程");
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
          <h1 className="text-[38px] font-semibold leading-tight text-[#111111]">
            几分钟内生成你的首个行程
          </h1>
          <p className="font-mono text-[13px] text-[#111111A6]">
            告诉我目的地、日期和预算，我会在对话中实时生成行程。
          </p>
        </div>
      )}

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 space-y-2.5 overflow-y-auto py-1 scrollbar-thin"
      >
        {isEmpty && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <p className="text-lg font-medium text-[#111111]/30">欢迎</p>
            <p className="mt-1 text-sm text-[#111111]/20">
              告诉我您想去哪里旅行？
            </p>
          </div>
        )}
        {store.messages.map((msg, i) => (
          <MessageBubble key={i} role={msg.role} content={msg.content} runStatus={msg.runStatus} />
        ))}
        {store.isLoading && <ThinkingBubble />}
      </div>

      {/* Confirmation buttons */}
      {store.waitingForConfirmation && (
        <div className="rounded-2xl border border-white/60 bg-white/50 p-3 backdrop-blur-md">
          <p className="mb-2 text-xs text-[#111111]/60">对行程满意吗？</p>
          <div className="flex gap-2">
            <button
              onClick={handleConfirm}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-xl bg-[#111111] px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#333333]"
            >
              <CheckCircle className="h-4 w-4" />
              确认行程
            </button>
            <button
              onClick={handleModify}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-white/60 bg-white/60 px-4 py-2.5 text-sm font-medium text-[#111111] backdrop-blur-sm transition-colors hover:bg-white/80"
            >
              <XCircle className="h-4 w-4" />
              继续修改
            </button>
          </div>
        </div>
      )}

      {/* Input */}
      <div
        className="flex items-center gap-2.5 rounded-[20px] p-2.5"
        style={{
          background: "rgba(255,255,255,0.7)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: "1px solid rgba(255,255,255,0.8)",
        }}
      >
        <div
          className="flex flex-1 items-center rounded-2xl px-3.5 py-3"
          style={{
            background: "rgba(255,255,255,0.77)",
            border: "1px solid rgba(255,255,255,0.82)",
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="试试：'成都 4 天，预算 3000 元，喜欢火锅和历史文化'"
            className="w-full bg-transparent font-mono text-sm text-[#111111] placeholder:text-[#111111]/50 outline-none"
          />
        </div>
        <button
          onClick={handleSend}
          disabled={!input.trim() || store.isLoading}
          className={cn(
            "flex items-center gap-1 rounded-2xl px-4 py-3 text-sm font-semibold text-white transition-all",
            input.trim() && !store.isLoading
              ? "bg-[#111111] shadow-lg hover:bg-[#333333]"
              : "bg-[#111111]/30 cursor-not-allowed"
          )}
          style={
            input.trim() && !store.isLoading
              ? { boxShadow: "0 6px 14px rgba(0,0,0,0.2)" }
              : {}
          }
        >
          <span>发送</span>
        </button>
      </div>
    </div>
  );
}
