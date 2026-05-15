import { useCallback, useEffect, useRef } from "react";
import { useChatStore } from "@/stores/chatStore";
import { generateSessionId } from "@/lib/utils";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const pendingMessagesRef = useRef<string[]>([]);
  const store = useChatStore();

  // 用 ref 避免 useCallback 因 store.sessionId 变化而频繁重建
  const sessionIdRef = useRef(store.sessionId);
  sessionIdRef.current = store.sessionId;

  const prevSessionIdRef = useRef<string>("");

  const connect = useCallback((newSessionId?: string) => {
    const sessionId = newSessionId || sessionIdRef.current || generateSessionId();

    // 只有在没有传入新 sessionId 且 store 中也没有时才设置
    if (!sessionIdRef.current && !newSessionId) {
      store.setSessionId(sessionId);
    }

    // 关闭旧连接
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const url = `${WS_URL}/ws/chat/${sessionId}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      store.setConnected(true);
      // Flush pending messages
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
        // ignore non-JSON
      }
    };

    ws.onclose = () => {
      store.setConnected(false);
      wsRef.current = null;
    };

    ws.onerror = () => {
      store.setConnected(false);
    };

    wsRef.current = ws;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    store.setConnected(false);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const reconnect = useCallback(() => {
    const newSessionId = generateSessionId();
    store.setSessionId(newSessionId);
    pendingMessagesRef.current = []; // clear old session's queued messages
    connect(newSessionId);
  }, [connect, store]);

  const sendMessage = useCallback(
    (content: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        // Queue message and reconnect; it will be sent on onopen
        pendingMessagesRef.current.push(content);
        reconnect();
        return true;
      }
      wsRef.current.send(
        JSON.stringify({ content, user_id: "anonymous" })
      );
      return true;
    },
    [reconnect]
  );

  const handleMessage = (data: any) => {
    // Phase 2: job-based event handling
    if (data.type === "job_created") {
      store.setJobId(data.job_id);
      store.setJobStatus("pending");
      store.setCurrentStage("等待处理");
      store.setLoading(true);
      return;
    }

    if (data.type === "stage" || data.stage) {
      const stage = data.stage;
      store.setJobStatus(stage);

      if (stage === "running") {
        store.setCurrentStage("正在规划...");
      } else if (stage === "draft_ready") {
        store.setCurrentStage("行程草稿已生成");
        if (data.payload?.itinerary_draft) {
          store.setItinerary(data.payload.itinerary_draft);
        }
      } else if (stage === "itinerary_final") {
        store.setCurrentStage("行程已优化");
        if (data.payload?.itinerary_final) {
          store.setItinerary(data.payload.itinerary_final);
        }
      } else if (stage === "writing") {
        store.setCurrentStage("正在润色文案...");
      } else if (stage === "completed") {
        store.setCurrentStage("完成");
        store.setLoading(false);
        if (data.payload?.proposal_text) {
          const currentRunStatus = useChatStore.getState().runStatus;
          store.addMessage({
            role: "assistant",
            content: data.payload.proposal_text,
            timestamp: Date.now(),
            runStatus: currentRunStatus ?? undefined,
          });
        }
        if (data.payload?.itinerary) {
          store.setItinerary(data.payload.itinerary);
        }
        store.saveChatSnapshot();
      } else if (stage === "failed" || stage === "cancelled") {
        store.setCurrentStage(stage === "failed" ? "处理失败" : "已取消");
        store.setLoading(false);
        store.addMessage({
          role: "assistant",
          content: stage === "failed"
            ? `错误: ${data.error || "处理失败"}`
            : "行程规划已取消",
          timestamp: Date.now(),
        });
      }
      return;
    }

    if (data.type === "error") {
      store.addMessage({
        role: "assistant",
        content: `错误: ${data.error || "未知错误"}`,
        timestamp: Date.now(),
      });
      store.setLoading(false);
      store.clearRunStatus();
      return;
    }

    if (data.type === "run_status") {
      store.setRunStatus(data);
      return;
    }

    if (data.type === "message") {
      // === 核心：解析后端返回的结构化数据 ===
      store.setIntent(data.intent || null);
      store.setItinerary(data.itinerary || null);
      store.setBudgetPanel(data.budget_panel || null);
      store.setPreferencePanel(data.preference_panel || null);
      store.setValidationResult(data.validation_result || null);
      store.setNeedsClarification(data.needs_clarification || false);
      store.setWaitingForConfirmation(data.waiting_for_confirmation || false);

      // 已确认信息（新增）
      // 优先用 confirmed_info，若为空则回退到 preference_panel
      const cInfo = data.confirmed_info || {};
      const pref = data.preference_panel || {};
      const mergedInfo = {
        destination: cInfo.destination || pref.destination || undefined,
        travel_dates: cInfo.travel_dates || pref.travel_dates || undefined,
        startDate: cInfo.startDate || pref.startDate || undefined,
        endDate: cInfo.endDate || pref.endDate || undefined,
        travelers_count: cInfo.travelers_count || pref.travelers_count || undefined,
        budget_range: cInfo.budget_range || pref.budget_range || undefined,
        travelers_type: cInfo.travelers_type || pref.travelers_type || undefined,
        pace: cInfo.pace || pref.pace || undefined,
      };
      const hasAny = Object.values(mergedInfo).some((v) => v !== undefined);
      store.setConfirmedInfo(hasAny ? mergedInfo : null);

      // 待确认建议（新增）
      const suggestions = (data.pending_suggestions || []).map(
        (text: string, idx: number) => ({
          id: `sugg-${idx}`,
          text,
        })
      );
      store.setPendingSuggestions(suggestions);

      // 重置 activeBriefDay 当行程变化时
      if (data.itinerary && data.itinerary.length > 0) {
        store.setActiveBriefDay(0);
      }

      if (data.assistant_message) {
        // 用 getState() 获取当前 runStatus，避免闭包捕获旧值
        const currentRunStatus = useChatStore.getState().runStatus;
        store.addMessage({
          role: "assistant",
          content: data.assistant_message,
          timestamp: Date.now(),
          runStatus: currentRunStatus ?? undefined,
        });
      }

      // 如果行程已确认，保存到本地 trips 列表
      if (data.itinerary_status === "confirmed") {
        store.confirmCurrentItinerary();
      }

      // 保存对话快照到左侧列表
      store.saveChatSnapshot();
      store.setLoading(false);
      // Keep runStatus visible (completed) — user can click to see full stats
      // It will be cleared only when a new user message is sent
    }
  };

  // mount 时建立初始连接
  useEffect(() => {
    connect();
    return () => disconnect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // sessionId 从非空变为另一个非空值时（如 restoreChat），自动重连
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
