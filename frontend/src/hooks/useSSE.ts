import { useCallback, useEffect, useRef } from "react";
import {
  buildChatStreamUrl,
  getDeviceFingerprint,
  getStoredAccessToken,
} from "@/lib/api";
import { handleChatEvent } from "@/lib/chatEvents";
import { useChatStore } from "@/stores/chatStore";

export type SSEEventType =
  | "stage"
  | "token"
  | "job"
  | "message"
  | "error"
  | "done";

export interface SSEMessage {
  type: SSEEventType;
  [key: string]: unknown;
}

export interface UseSSEOptions {
  onMessage?: (msg: SSEMessage) => void;
  onError?: (error: Error) => void;
}

interface ParseContext {
  activeJobIdRef: React.MutableRefObject<string | null>;
  lastEventIdRef: React.MutableRefObject<number>;
}

function isAuthError(status: number): boolean {
  return status === 401 || status === 403;
}

function parseRetryAfterMs(res: Response, body?: unknown): number {
  const header = res.headers.get("Retry-After");
  if (header) {
    const secs = parseInt(header, 10);
    if (!Number.isNaN(secs) && secs > 0) {
      return secs * 1000;
    }
  }
  if (body && typeof body === "object" && body !== null) {
    const err = (body as Record<string, unknown>).error;
    if (err && typeof err === "object") {
      const details = (err as Record<string, unknown>).details;
      if (details && typeof details === "object") {
        const retryAfter = (details as Record<string, unknown>).retry_after;
        if (typeof retryAfter === "number" && retryAfter > 0) {
          return retryAfter * 1000;
        }
      }
    }
  }
  return 30_000;
}

function computeReconnectDelay(attempt: number): number {
  // Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s.
  const base = Math.min(30, Math.pow(2, attempt));
  // Add jitter (±25%) to avoid thundering herd.
  const jitter = 0.75 + Math.random() * 0.5;
  return Math.round(base * jitter * 1000);
}

function parseSSEPart(
  part: string,
  onData: (data: Record<string, unknown>) => void
): boolean {
  if (!part.trim() || part.startsWith(":")) return false;

  const lines = part.split("\n");
  let dataStr = "";
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataStr = line.slice(5).trim();
    }
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

/**
 * Manages a single Server-Sent Events connection for the chat stream.
 *
 * Features:
 * - Automatic reconnect with exponential backoff (max 30s) and jitter.
 * - Resume from the last received event id (`last_event_id`).
 * - Forwards events to the chat store and optional callbacks.
 * - Detects 401/403 auth errors and stops infinite reconnection loops.
 */
export function useSSE(options: UseSSEOptions = {}) {
  const { onMessage, onError } = options;
  const abortRef = useRef<AbortController | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const lastEventIdRef = useRef(0);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectingRef = useRef<string | null>(null);
  const isMountedRef = useRef(true);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    useChatStore.getState().setConnected(false);
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      disconnect();
    };
  }, [disconnect]);

  const connect = useCallback(
    async (
      conversationId: string,
      connectOptions?: { jobId?: string; lastEventId?: number }
    ) => {
      const connectionKey = `${conversationId}:${connectOptions?.jobId ?? "session"}`;
      if (connectingRef.current === connectionKey) {
        return;
      }
      connectingRef.current = connectionKey;

      disconnect();

      if (!isMountedRef.current) {
        connectingRef.current = null;
        return;
      }

      const token = getStoredAccessToken();
      const fingerprint = getDeviceFingerprint();
      if (!token) {
        const err = new Error("No access token — call ensureGuestSession first");
        onError?.(err);
        throw err;
      }

      const lastEventId =
        connectOptions?.lastEventId ?? lastEventIdRef.current;
      const url = buildChatStreamUrl(conversationId, {
        jobId: connectOptions?.jobId,
        lastEventId,
      });

      const controller = new AbortController();
      abortRef.current = controller;
      const store = useChatStore.getState();
      store.setConnected(true);

      const refs: ParseContext = { activeJobIdRef, lastEventIdRef };
      let authFailed = false;
      let rateLimitedMs = 0;

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
          if (isAuthError(res.status)) {
            authFailed = true;
            // Token expired or revoked — do not auto-reconnect, let the app handle auth.
            store.setConnected(false);
            store.setLoading(false);
            const err = new Error(
              `SSE auth failed: ${res.status}. Please log in again.`
            );
            onError?.(err);
            throw err;
          }
          if (res.status === 429) {
            let body: unknown;
            try {
              body = await res.json();
            } catch {
              body = undefined;
            }
            rateLimitedMs = parseRetryAfterMs(res, body);
            throw new Error(`SSE open failed: 429`);
          }
          throw new Error(`SSE open failed: ${res.status}`);
        }

        // Reset reconnect counter on successful connection.
        reconnectAttemptRef.current = 0;

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
            const isDone = parseSSEPart(part, (data) => {
              handleChatEvent(data, refs);
              onMessage?.(data as SSEMessage);
            });
            if (isDone) {
              disconnect();
              return;
            }
          }
        }
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        if (error.name === "AbortError") {
          // Graceful disconnect — do not reconnect.
          return;
        }

        console.error("SSE error:", error);
        onError?.(error);
        store.setConnected(false);
        store.setLoading(false);

        if (isMountedRef.current && !authFailed) {
          // Retry unless this was an auth error. Respect server Retry-After on 429.
          const delay =
            rateLimitedMs > 0
              ? rateLimitedMs
              : computeReconnectDelay(reconnectAttemptRef.current);
          if (rateLimitedMs <= 0) {
            reconnectAttemptRef.current += 1;
          }
          reconnectTimerRef.current = setTimeout(() => {
            if (isMountedRef.current) {
              connect(conversationId, {
                jobId: activeJobIdRef.current ?? connectOptions?.jobId,
                lastEventId: lastEventIdRef.current,
              }).catch((retryErr) => {
                console.error("SSE reconnect failed:", retryErr);
              });
            }
          }, delay);
        }
        return;
      } finally {
        if (connectingRef.current === connectionKey) {
          connectingRef.current = null;
        }
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        useChatStore.getState().setConnected(false);
      }
    },
    [disconnect, onMessage, onError]
  );

  return { connect, disconnect, activeJobIdRef, lastEventIdRef };
}
