"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
}

export function MessageBubble({ role, content }: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div className={cn("flex w-full flex-col", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[560px] px-3 py-2.5 text-[13px] leading-relaxed",
          isUser ? "glass-message-user text-canvas" : "glass-message-ai text-ink"
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap">{content}</div>
        ) : (
          <div className="prose prose-sm max-w-none text-inherit">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
