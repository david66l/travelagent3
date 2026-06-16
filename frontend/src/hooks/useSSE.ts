import { useCallback, useRef } from "react";
import {
  buildChatStreamUrl,
  getDeviceFingerprint,
  getStoredAccessToken,
} from "@/lib/api";
import { handleChatEvent } from "@/lib/chatEvents";
import { useChatStore } from "@/stores/chatStore";

function parseSSEPart(
  part: string,
  onData: (data: Record<string, unknown>) => void
): boolean {
  if (!part.trim() || part.startsWith(":")) return false;
  let dataStr = "";
  for (const line of part.split("\n")) {
    if (line.startsWith("data:")) dataStr = line.slice(5).trim();
  }
  if (!dataStr) return false;
  try {
    const data = JSON.parse(dataStr) as Record<string, unknown>;
    onData(data);
    return data.type === "done";
  } catch {
    console.warn("Invalid SSE JSON:", dataStr);
    return false;
  }
}

export function useSSE() {
  const abortRef = useRef<AbortController | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const lastEventIdRef = useRef(0);

  const disconnect = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    useChatStore.getState().setConnected(false);
  }, []);

  const connect = useCallback(
    async (
      conversationId: string,
      options?: { jobId?: string; lastEventId?: number }
    ) => {
      disconnect();

      const token = getStoredAccessToken();
      const fingerprint = getDeviceFingerprint();
      if (!token) {
        throw new Error("No access token — call ensureGuestSession first");
      }

      const url = buildChatStreamUrl(conversationId, {
        jobId: options?.jobId,
        lastEventId: options?.lastEventId ?? lastEventIdRef.current,
      });

      const controller = new AbortController();
      abortRef.current = controller;
      const store = useChatStore.getState();
      store.setConnected(true);

      const refs = { activeJobIdRef, lastEventIdRef };

      try {
        const res = await fetch(url, {
          headers: {
            Authorization: `Bearer ${token}`,
            "X-Device-Fingerprint": fingerprint,
            Accept: "text/event-stream",
          },
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`SSE open failed: ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";
          for (const part of parts) {
            const isDone = parseSSEPart(part, (data) =>
              handleChatEvent(data, refs)
            );
            if (isDone) return;
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.error("SSE error:", err);
          store.setLoading(false);
          store.addMessage({
            role: "assistant",
            content: "连接出错，请检查网络或刷新页面重试。",
            timestamp: Date.now(),
          });
        }
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        useChatStore.getState().setConnected(false);
      }
    },
    [disconnect]
  );

  return { connect, disconnect, activeJobIdRef, lastEventIdRef };
}
