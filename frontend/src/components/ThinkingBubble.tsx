"use client";

import { useState, useEffect, useRef } from "react";
import { useChatStore } from "@/stores/chatStore";
import { Loader2 } from "lucide-react";

function fmtDur(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s >= 1 ? `${m}m${s.toFixed(0)}s` : `${m}m`;
}

export function ThinkingBubble() {
  const store = useChatStore();
  const currentStage = store.currentStage;
  const [displayElapsed, setDisplayElapsed] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    if (!store.isLoading) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      startRef.current = null;
      setDisplayElapsed(0);
      return;
    }

    startRef.current = Date.now();
    setDisplayElapsed(0);
    intervalRef.current = setInterval(() => {
      if (startRef.current) {
        setDisplayElapsed((Date.now() - startRef.current) / 1000);
      }
    }, 200);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [store.isLoading]);

  const stageLabel = currentStage || "让我思考一下…";

  return (
    <div className="flex w-full flex-col items-start">
      <div className="glass-message-ai max-w-[560px] px-3 py-2.5 text-[13px] leading-relaxed">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-mute" />
          <span className="text-body">{stageLabel}</span>
          {displayElapsed > 0.5 && (
            <span className="font-mono text-[11px] text-mute/60">
              {fmtDur(displayElapsed)}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
