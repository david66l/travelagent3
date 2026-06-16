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
          "max-w-[560px] rounded-2xl px-3 py-2.5 text-[13px] leading-relaxed",
          isUser
            ? "bg-[#111111] text-white"
            : "bg-[#FFFFFFC2] text-[#111111E6] backdrop-blur-md"
        )}
        style={
          !isUser
            ? {
                border: "1px solid rgba(255,255,255,0.8)",
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
              }
            : {
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
              }
        }
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
