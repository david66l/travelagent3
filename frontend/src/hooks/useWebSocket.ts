import { useCallback, useEffect, useRef } from "react";
import { useChatStore, type ConfirmedInfo, type PreferencePanel } from "@/stores/chatStore";
import { generateSessionId } from "@/lib/utils";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

function profileToConfirmedInfo(profile: unknown): ConfirmedInfo | null {
  if (!profile || typeof profile !== "object") return null;
  const p = profile as Record<string, unknown>;
  const trip = (p.trip as Record<string, unknown>) || p;
  const personal = (p.personal as Record<string, unknown>) || {};
  const merged: ConfirmedInfo = {
    destination: (trip.destination as string) || undefined,
    travel_dates: (trip.travel_dates as string) || undefined,
    travelers_count: (trip.travelers_count as number) || undefined,
    budget_range: (trip.budget_range as number) || undefined,
    travelers_type: (trip.travelers_type as string) || undefined,
    pace: (personal.pace as string) || (trip.pace as string) || undefined,
  };
  const hasAny = Object.values(merged).some((v) => v !== undefined && v !== null);
  return hasAny ? merged : null;
}

function profileToPreferencePanel(profile: unknown): PreferencePanel | null {
  if (!profile || typeof profile !== "object") return null;
  const p = profile as Record<string, unknown>;
  const trip = (p.trip as Record<string, unknown>) || p;
  const personal = (p.personal as Record<string, unknown>) || {};
  const merged = { ...personal, ...trip };
  const panel: PreferencePanel = {
    destination: (merged.destination as string) || undefined,
    travel_days: (merged.travel_days as number) || undefined,
    travel_dates: (merged.travel_dates as string) || undefined,
    travelers_count: (merged.travelers_count as number) || undefined,
    travelers_type: (merged.travelers_type as string) || undefined,
    budget_range: (merged.budget_range as number) || undefined,
    pace: (merged.pace as string) || undefined,
    food_preferences: (merged.food_preferences as string[]) || [],
    interests: (merged.interests as string[]) || [],
    special_requests: (merged.special_requests as string[]) || [],
  };
  if (!panel.destination && !panel.travel_days) return null;
  return panel;
}

function applyProfileFromServer(profile: unknown) {
  const confirmed = profileToConfirmedInfo(profile);
  const preference = profileToPreferencePanel(profile);
  if (confirmed) {
    useChatStore.getState().setConfirmedInfo(confirmed);
  }
  if (preference) {
    useChatStore.getState().setPreferencePanel(preference);
  }
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const pendingMessagesRef = useRef<string[]>([]);
  const activeJobIdRef = useRef<string | null>(null);
  const lastEventIdRef = useRef<number>(0);
  const manualCloseRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const store = useChatStore();

  const sessionIdRef = useRef(store.sessionId);
  const prevSessionIdRef = useRef<string>("");

  useEffect(() => {
    sessionIdRef.current = store.sessionId;
  }, [store.sessionId]);

  const connect = useCallback((newSessionId?: string) => {
    const sessionId = newSessionId || sessionIdRef.current || generateSessionId();

    if (!sessionIdRef.current && !newSessionId) {
      store.setSessionId(sessionId);
    }

    if (wsRef.current) {
      manualCloseRef.current = true;
      wsRef.current.close();
      wsRef.current = null;
    }
    manualCloseRef.current = false;

    const url = `${WS_URL}/ws/chat/${sessionId}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      store.setConnected(true);
      const activeJobId = activeJobIdRef.current || useChatStore.getState().jobId;
      const state = useChatStore.getState();
      if (
        activeJobId &&
        state.isLoading &&
        !["completed", "failed", "cancelled"].includes(state.jobStatus || "")
      ) {
        ws.send(
          JSON.stringify({
            type: "subscribe",
            job_id: activeJobId,
            last_event_id: lastEventIdRef.current,
          })
        );
      }
      while (pendingMessagesRef.current.length > 0) {
        const msg = pendingMessagesRef.current.shift();
        if (msg) {
          ws.send(JSON.stringify({ content: msg, user_id: "anonymous" }));
        }
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch {
        console.warn("Received non-JSON WebSocket message:", event.data);
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) {
        return;
      }
      store.setConnected(false);
      wsRef.current = null;
      const state = useChatStore.getState();
      if (
        !manualCloseRef.current &&
        state.isLoading &&
        (activeJobIdRef.current || state.jobId)
      ) {
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current);
        }
        reconnectTimerRef.current = setTimeout(() => {
          reconnectTimerRef.current = null;
          connect(sessionId);
        }, 1000);
      }
    };

    ws.onerror = () => {
      store.setConnected(false);
      store.setLoading(false);
      store.setCurrentStage("连接出错");
      store.addMessage({
        role: "assistant",
        content: "连接出错，请检查网络或刷新页面重试。",
        timestamp: Date.now(),
      });
    };

    wsRef.current = ws;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const disconnect = useCallback(() => {
    manualCloseRef.current = true;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    store.setConnected(false);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const reconnect = useCallback(
    (options: { clearPending?: boolean } = {}) => {
      const newSessionId = generateSessionId();
      store.setSessionId(newSessionId);
      activeJobIdRef.current = null;
      lastEventIdRef.current = 0;
      if (options.clearPending !== false) {
        pendingMessagesRef.current = [];
      }
      connect(newSessionId);
    },
    [connect, store]
  );

  const sendMessage = useCallback(
    (content: string): "sent" | "queued" | "failed" => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        pendingMessagesRef.current.push(content);
        reconnect({ clearPending: false });
        return "queued";
      }
      try {
        wsRef.current.send(JSON.stringify({ content, user_id: "anonymous" }));
        return "sent";
      } catch (error) {
        console.error("Failed to send WebSocket message:", error);
        return "failed";
      }
    },
    [reconnect]
  );

  const handleMessage = (data: Record<string, unknown>) => {
    const type = data.type as string | undefined;

    if (type === "job_created") {
      store.setJobId(data.job_id as string);
      activeJobIdRef.current = data.job_id as string;
      lastEventIdRef.current = 0;
      store.setJobStatus("pending");
      store.setCurrentStage("等待处理");
      store.setLoading(true);
      store.setNeedsClarification(false);
      return;
    }

    if (type === "needs_clarification") {
      const questions = (data.questions as string[]) || [];
      const text =
        questions.length > 0
          ? questions.join("\n")
          : "请补充一下目的地和出行天数，我好继续规划。";
      store.setNeedsClarification(true);
      store.setLoading(false);
      store.setCurrentStage(null);
      applyProfileFromServer(data.profile);
      store.addMessage({
        role: "assistant",
        content: text,
        timestamp: Date.now(),
      });
      return;
    }

    if (type === "state_restored") {
      applyProfileFromServer(data.profile);
      return;
    }

    if (type === "revision_created") {
      applyProfileFromServer(data.profile);
      return;
    }

    if (type === "stage" || data.stage) {
      if (typeof data.event_id === "number") {
        lastEventIdRef.current = Math.max(lastEventIdRef.current, data.event_id);
      }
      const stage = data.stage as string;
      store.setJobStatus(stage);

      if (stage === "running") {
        store.setCurrentStage("正在规划...");
      } else if (stage === "draft_ready") {
        store.setCurrentStage("行程草稿已生成");
        const payload = data.payload as Record<string, unknown> | undefined;
        if (payload?.itinerary_draft) {
          store.setItinerary(payload.itinerary_draft as Parameters<typeof store.setItinerary>[0]);
        }
      } else if (stage === "itinerary_final") {
        store.setCurrentStage("行程已优化");
        const payload = data.payload as Record<string, unknown> | undefined;
        if (payload?.itinerary_final) {
          store.setItinerary(payload.itinerary_final as Parameters<typeof store.setItinerary>[0]);
        }
      } else if (stage === "writing") {
        store.setCurrentStage("正在润色文案...");
      } else if (stage === "completed") {
        activeJobIdRef.current = null;
        lastEventIdRef.current = 0;
        store.setCurrentStage("完成");
        store.setLoading(false);
        store.setNeedsClarification(false);
        const payload = data.payload as Record<string, unknown> | undefined;
        if (payload?.proposal_text) {
          store.addMessage({
            role: "assistant",
            content: payload.proposal_text as string,
            timestamp: Date.now(),
          });
        }
        if (payload?.itinerary || payload?.itinerary_final) {
          store.setItinerary(
            (payload.itinerary_final || payload.itinerary) as Parameters<
              typeof store.setItinerary
            >[0]
          );
        }
        store.saveChatSnapshot();
      } else if (stage === "failed" || stage === "cancelled") {
        activeJobIdRef.current = null;
        lastEventIdRef.current = 0;
        store.setCurrentStage(stage === "failed" ? "处理失败" : "已取消");
        store.setLoading(false);
        store.addMessage({
          role: "assistant",
          content:
            stage === "failed"
              ? `错误: ${(data.error as string) || "处理失败"}`
              : "行程规划已取消",
          timestamp: Date.now(),
        });
      }
      return;
    }

    if (type === "error") {
      store.addMessage({
        role: "assistant",
        content: `错误: ${(data.error as string) || "未知错误"}`,
        timestamp: Date.now(),
      });
      store.setLoading(false);
      store.setCurrentStage(null);
      return;
    }
  };

  useEffect(() => {
    connect();
    return () => disconnect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const prev = prevSessionIdRef.current;
    const curr = store.sessionId;
    if (prev && curr && prev !== curr && wsRef.current) {
      connect(curr);
    }
    prevSessionIdRef.current = curr;
  }, [store.sessionId, connect]);

  return { sendMessage, connect, disconnect, reconnect };
}
