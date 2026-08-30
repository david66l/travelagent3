import { useCallback, useEffect, useRef } from "react";
import {
  createConversation,
  ensureGuestSession,
  postChatAction,
  postChatMessage,
} from "@/lib/api";
import { useSSE } from "@/hooks/useSSE";
import { useChatStore } from "@/stores/chatStore";

export function useChat() {
  const store = useChatStore();
  const { connect, disconnect, activeJobIdRef, lastEventIdRef } = useSSE();
  const authRef = useRef<{ token: string; fingerprint: string } | null>(null);
  const prevSessionIdRef = useRef<string>("");
  // Every conversation switch advances the epoch. Async bootstrap/auth/create
  // work from an older epoch is then ignored instead of reconnecting or posting
  // to the conversation the user has just left.
  const conversationEpochRef = useRef(0);

  const ensureAuth = useCallback(async () => {
    if (!authRef.current) {
      authRef.current = await ensureGuestSession();
    }
    return authRef.current;
  }, []);

  const openStream = useCallback(
    async (conversationId: string, requestedJobId?: string) => {
      const state = useChatStore.getState();
      const jobId =
        requestedJobId ||
        activeJobIdRef.current ||
        (state.isLoading && state.jobId ? state.jobId : undefined);
      await connect(conversationId, {
        jobId,
        lastEventId: requestedJobId ? 0 : lastEventIdRef.current,
      });
    },
    [connect, activeJobIdRef, lastEventIdRef]
  );

  const bootstrap = useCallback(async () => {
    const epoch = conversationEpochRef.current;
    const auth = await ensureAuth();
    if (epoch !== conversationEpochRef.current) return "";
    let conversationId = useChatStore.getState().sessionId;
    if (!conversationId) {
      conversationId = await createConversation(auth.token, auth.fingerprint);
      if (epoch !== conversationEpochRef.current) return "";
      store.setSessionId(conversationId);
    }
    void openStream(conversationId).catch((err) =>
      console.error("Chat stream failed:", err)
    );
    return conversationId;
  }, [ensureAuth, openStream, store, conversationEpochRef]);

  useEffect(() => {
    bootstrap().catch((err) => console.error("Chat bootstrap failed:", err));
    return () => disconnect();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const prev = prevSessionIdRef.current;
    const curr = store.sessionId;
    if (prev && curr && prev !== curr) {
      activeJobIdRef.current = null;
      lastEventIdRef.current = 0;
      openStream(curr).catch((err) =>
        console.error("SSE reconnect failed:", err)
      );
    }
    prevSessionIdRef.current = curr;
  }, [store.sessionId, openStream, activeJobIdRef, lastEventIdRef]);

  const sendMessage = useCallback(
    async (content: string): Promise<"sent" | "queued" | "failed"> => {
      try {
        const epoch = conversationEpochRef.current;
        const auth = await ensureAuth();
        if (epoch !== conversationEpochRef.current) return "failed";
        let conversationId = useChatStore.getState().sessionId;
        if (!conversationId) {
          conversationId = await createConversation(
            auth.token,
            auth.fingerprint
          );
          if (epoch !== conversationEpochRef.current) return "failed";
          store.setSessionId(conversationId);
        }
        if (
          epoch !== conversationEpochRef.current ||
          useChatStore.getState().sessionId !== conversationId
        ) {
          return "failed";
        }
        if (!useChatStore.getState().isConnected) {
          void openStream(conversationId).catch((err) =>
            console.error("Chat stream failed:", err)
          );
        }
        const jobId = await postChatMessage(
          auth.token,
          auth.fingerprint,
          conversationId,
          content,
          crypto.randomUUID()
        );
        store.setJobId(jobId);
        store.setJobStatus("pending");
        store.setLoading(true);
        activeJobIdRef.current = jobId;
        lastEventIdRef.current = 0;
        void openStream(conversationId, jobId).catch((err) =>
          console.error("Chat job stream failed:", err)
        );
        return "sent";
      } catch (err) {
        console.error("sendMessage failed:", err);
        return "failed";
      }
    },
    [
      ensureAuth,
      openStream,
      store,
      conversationEpochRef,
      activeJobIdRef,
      lastEventIdRef,
    ]
  );

  const sendAction = useCallback(
    async (
      action: "confirm" | "modify" | "reject" | "trip_event",
      payload?: { change?: unknown; external_event?: unknown; approval?: unknown }
    ): Promise<"sent" | "failed"> => {
      try {
        const epoch = conversationEpochRef.current;
        const auth = await ensureAuth();
        if (epoch !== conversationEpochRef.current) return "failed";
        const conversationId = useChatStore.getState().sessionId;
        if (!conversationId) return "failed";
        useChatStore.getState().setWaitingForConfirmation(false);
        useChatStore.getState().setLoading(true);
        if (!useChatStore.getState().isConnected) {
          void openStream(conversationId).catch((err) =>
            console.error("Chat stream failed:", err)
          );
        }
        if (
          epoch !== conversationEpochRef.current ||
          useChatStore.getState().sessionId !== conversationId
        ) {
          return "failed";
        }
        const approval = useChatStore.getState().pendingApproval;
        const jobId = await postChatAction(
          auth.token,
          auth.fingerprint,
          conversationId,
          action,
          action === "trip_event" ? payload : { ...payload, approval },
          crypto.randomUUID()
        );
        useChatStore.getState().setJobId(jobId);
        activeJobIdRef.current = jobId;
        lastEventIdRef.current = 0;
        void openStream(conversationId, jobId).catch((err) =>
          console.error("Chat action stream failed:", err)
        );
        return "sent";
      } catch (err) {
        console.error("sendAction failed:", err);
        useChatStore.getState().setLoading(false);
        return "failed";
      }
    },
    [
      ensureAuth,
      openStream,
      conversationEpochRef,
      activeJobIdRef,
      lastEventIdRef,
    ]
  );

  const reconnect = useCallback(async () => {
    const epoch = conversationEpochRef.current + 1;
    conversationEpochRef.current = epoch;
    disconnect();
    activeJobIdRef.current = null;
    lastEventIdRef.current = 0;
    store.clear();
    store.setLoading(true);
    store.setCurrentStage("正在创建新对话…");
    try {
      const auth = await ensureAuth();
      if (epoch !== conversationEpochRef.current) return;
      const conversationId = await createConversation(
        auth.token,
        auth.fingerprint
      );
      if (epoch !== conversationEpochRef.current) return;
      store.setSessionId(conversationId);
      void openStream(conversationId).catch((err) => {
        console.error("SSE reconnect failed:", err);
        if (epoch === conversationEpochRef.current) {
          useChatStore.getState().setLoading(false);
        }
      });
    } finally {
      if (epoch === conversationEpochRef.current) {
        useChatStore.getState().setLoading(false);
        useChatStore.getState().setCurrentStage(null);
      }
    }
  }, [
    disconnect,
    ensureAuth,
    openStream,
    store,
    activeJobIdRef,
    lastEventIdRef,
    conversationEpochRef,
  ]);

  return { sendMessage, sendAction, reconnect, disconnect };
}
