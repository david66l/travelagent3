import { useCallback, useEffect, useRef } from "react";
import {
  buildChatStreamUrl,
  getDeviceFingerprint,
  getStoredAccessToken,
} from "@/lib/api";
import { handleChatEvent, type EventRefs } from "@/lib/chatEvents";
import { emptyConversationRuntime, useChatStore } from "@/stores/chatStore";

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

interface ConnectionHandle {
  controller: AbortController;
  jobId: string | null;
  lastEventId: number;
  reconnectAttempt: number;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  /** Set once a 401/403 proves the token is unusable, to stop reconnect loops. */
  authFailed: boolean;
  /** True while a fetch for this conversation is in flight. */
  open: boolean;
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
 * Manages one SSE connection **per conversation**.
 *
 * The previous implementation held a single `abortRef` / `lastEventIdRef`, so
 * opening a stream for another conversation necessarily tore the previous one
 * down: a job running in a conversation the user had switched away from lost its
 * stream, its cursor, and any live progress. Connections are now keyed by
 * conversation id and are independent:
 *
 * - switching conversations no longer closes the stream being left behind;
 * - each conversation keeps its own job id, event cursor and reconnect backoff;
 * - `connectedConversationIds` is a set of conversations rather than one global
 *   boolean, so N open streams cannot clobber each other's connected flag.
 *
 * Every event is routed with the `conversation_id` the server now stamps on each
 * frame, so a background conversation's tokens land on that conversation's
 * runtime instead of the one on screen.
 */
export function useSSE(options: UseSSEOptions = {}) {
  const { onMessage, onError } = options;
  const connectionsRef = useRef<Map<string, ConnectionHandle>>(new Map());
  const isMountedRef = useRef(true);
  // Callbacks change identity between renders; read them through refs so
  // `open`/`close` stay stable and never re-create a connection needlessly.
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);
  onMessageRef.current = onMessage;
  onErrorRef.current = onError;

  const publishConnected = useCallback(() => {
    useChatStore
      .getState()
      .setConnectedIds([...connectionsRef.current.keys()]);
  }, []);

  const close = useCallback(
    (conversationId: string) => {
      const handle = connectionsRef.current.get(conversationId);
      if (!handle) return;
      if (handle.reconnectTimer) {
        clearTimeout(handle.reconnectTimer);
        handle.reconnectTimer = null;
      }
      handle.controller.abort();
      connectionsRef.current.delete(conversationId);
      publishConnected();
    },
    [publishConnected]
  );

  const closeAll = useCallback(() => {
    for (const conversationId of [...connectionsRef.current.keys()]) {
      close(conversationId);
    }
  }, [close]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      closeAll();
    };
  }, [closeAll]);

  /**
   * Open (or re-open) the stream for one conversation. Never touches any other
   * conversation's connection.
   */
  const open = useCallback(
    async (
      conversationId: string,
      openOptions?: {
        jobId?: string;
        lastEventId?: number;
        /** Deliberately re-attach even if this conversation already has a stream. */
        force?: boolean;
      }
    ) => {
      if (!conversationId) return;
      const existing = connectionsRef.current.get(conversationId);
      if (existing && !openOptions?.force) {
        // Same conversation, same job: the stream is already doing this work.
        // A different job means the caller wants to follow the new turn.
        const wantedJob = openOptions?.jobId ?? null;
        if (existing.jobId === wantedJob) return;
      }
      if (existing) close(conversationId);

      if (!isMountedRef.current) return;

      const token = getStoredAccessToken();
      const fingerprint = getDeviceFingerprint();
      if (!token) {
        const err = new Error("No access token — call ensureGuestSession first");
        onErrorRef.current?.(err);
        throw err;
      }

      const current = useChatStore.getState().runtimeFor(conversationId);
      const jobId = openOptions?.jobId ?? current.jobId ?? null;
      const lastEventId =
        openOptions?.lastEventId ?? (jobId ? current.lastEventId : 0);

      const handle: ConnectionHandle = {
        controller: new AbortController(),
        jobId,
        lastEventId,
        reconnectAttempt: existing?.reconnectAttempt ?? 0,
        reconnectTimer: null,
        authFailed: false,
        open: true,
      };
      connectionsRef.current.set(conversationId, handle);
      publishConnected();

      // Live events for a background conversation must land on that
      // conversation's runtime, never on the one being displayed.
      const refs: EventRefs = {
        activeJobIdRef: {
          get current() {
            return handle.jobId;
          },
          set current(value: string | null) {
            handle.jobId = value;
          },
        },
        lastEventIdRef: {
          get current() {
            return handle.lastEventId;
          },
          set current(value: number) {
            handle.lastEventId = value;
          },
        },
        conversationId,
      };

      const url = buildChatStreamUrl(conversationId, {
        jobId: jobId ?? undefined,
        lastEventId,
      });
      let rateLimitedMs = 0;

      const scheduleReconnect = () => {
        if (!isMountedRef.current) return;
        if (handle.authFailed) return;
        const delay =
          rateLimitedMs > 0
            ? rateLimitedMs
            : computeReconnectDelay(handle.reconnectAttempt);
        if (rateLimitedMs <= 0) handle.reconnectAttempt += 1;
        handle.reconnectTimer = setTimeout(() => {
          if (!isMountedRef.current) return;
          if (!connectionsRef.current.has(conversationId)) return;
          void open(conversationId, {
            jobId: handle.jobId ?? undefined,
            lastEventId: handle.lastEventId,
            force: true,
          }).catch((retryErr) => {
            console.error("SSE reconnect failed:", retryErr);
          });
        }, delay);
      };

      try {
        const res = await fetch(url, {
          headers: {
            Authorization: `Bearer ${token}`,
            "X-Device-Fingerprint": fingerprint,
            Accept: "text/event-stream",
          },
          signal: handle.controller.signal,
        });

        if (!res.ok || !res.body) {
          if (isAuthError(res.status)) {
            handle.authFailed = true;
            // Token expired or revoked — do not auto-reconnect, let the app handle auth.
            close(conversationId);
            const err = new Error(
              `SSE auth failed: ${res.status}. Please log in again.`
            );
            onErrorRef.current?.(err);
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

        handle.reconnectAttempt = 0;

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
              onMessageRef.current?.(data as SSEMessage);
            });
            if (isDone) {
              close(conversationId);
              return;
            }
          }
        }
        // Server closed the stream without `done` (timeout, restart). Reconnect
        // so a still-running job is not silently abandoned.
        scheduleReconnect();
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        if (error.name === "AbortError") {
          // Deliberate close — never reconnect.
          return;
        }
        console.error("SSE error:", error);
        onErrorRef.current?.(error);
        scheduleReconnect();
      } finally {
        handle.open = false;
        if (
          connectionsRef.current.get(conversationId)?.controller ===
          handle.controller
        ) {
          connectionsRef.current.delete(conversationId);
          publishConnected();
        }
      }
    },
    [close, publishConnected]
  );

  /** True when this conversation currently holds an open stream. */
  const isOpen = useCallback(
    (conversationId: string) => connectionsRef.current.has(conversationId),
    []
  );

  const jobIdFor = useCallback(
    (conversationId: string) =>
      connectionsRef.current.get(conversationId)?.jobId ?? null,
    []
  );

  return { open, close, closeAll, isOpen, jobIdFor };
}

/**
 * Seed a runtime entry for every conversation that still has an unfinished job.
 *
 * A page load starts with no streams and an empty runtime map, so a job that was
 * already running would be invisible. The server is the only authority on what is
 * still running (`GET /chat/stream` without a job id attaches to the newest
 * pending/running job), so the caller drives discovery through `open()` and this
 * helper only guarantees a runtime entry exists to write progress into.
 */
export function ensureRuntimeEntry(conversationId: string): void {
  const store = useChatStore.getState();
  if (store.runtimeByConversation[conversationId]) return;
  store.setRuntime(conversationId, emptyConversationRuntime());
}
