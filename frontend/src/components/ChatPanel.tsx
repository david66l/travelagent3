"use client";

import { useState, useRef, useEffect } from "react";
import { CheckCircle, XCircle } from "lucide-react";
import { useChatStore } from "@/stores/chatStore";
import { MessageBubble } from "./MessageBubble";
import { ThinkingBubble } from "./ThinkingBubble";
import { StreamingText } from "./StreamingText";
import { cn } from "@/lib/utils";

type SendStatus = "sent" | "queued" | "failed";

interface ChatPanelProps {
  sendMessage: (content: string) => SendStatus | Promise<SendStatus>;
}

export function ChatPanel({ sendMessage }: ChatPanelProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const store = useChatStore();

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

  const handleConfirm = async () => {
    if (store.isLoading) return;
    store.addMessage({
      role: "user",
      content: "确认行程",
      timestamp: Date.now(),
    });
    store.setLoading(true);
    store.setCurrentStage("正在规划…");
    store.setActivityPhase("planning");
    try {
      await sendMessage("确认行程");
    } catch {
      store.setLoading(false);
    }
  };

  const handleModify = async () => {
    if (store.isLoading) return;
    store.addMessage({
      role: "user",
      content: "继续修改行程",
      timestamp: Date.now(),
    });
    store.setLoading(true);
    store.setCurrentStage("正在规划…");
    store.setActivityPhase("planning");
    try {
      await sendMessage("继续修改行程");
    } catch {
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

      {/* Confirmation buttons */}
      {store.waitingForConfirmation && (
        <div className="glass-card p-3">
          <p className="mb-2 text-xs text-body">对行程满意吗？</p>
          <div className="flex gap-2">
            <button
              onClick={handleConfirm}
              className="btn-primary-dark flex flex-1 items-center justify-center gap-1.5"
            >
              <CheckCircle className="h-4 w-4" />
              确认行程
            </button>
            <button
              onClick={handleModify}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-xl border border-hairline bg-canvas px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-primary-pale"
            >
              <XCircle className="h-4 w-4" />
              继续修改
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
            placeholder="试试：'成都 4 天，预算 3000 元，喜欢火锅和历史文化'"
            className="w-full bg-transparent font-mono text-sm text-ink placeholder:text-mute outline-none"
          />
        </div>
        <button
          data-testid="send-button"
          onClick={handleSend}
          disabled={!input.trim() || store.isLoading}
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
