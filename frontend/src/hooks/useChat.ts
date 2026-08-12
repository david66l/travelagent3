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

  const ensureAuth = useCallback(async () => {
    if (!authRef.current) {
      authRef.current = await ensureGuestSession();
    }
    return authRef.current;
  }, []);

  const openStream = useCallback(
    async (conversationId: string) => {
      const state = useChatStore.getState();
      const jobId =
        activeJobIdRef.current ||
        (state.isLoading && state.jobId ? state.jobId : undefined);
      await connect(conversationId, {
        jobId,
        lastEventId: lastEventIdRef.current,
      });
    },
    [connect, activeJobIdRef, lastEventIdRef]
  );

  const bootstrap = useCallback(async () => {
    const auth = await ensureAuth();
    let conversationId = useChatStore.getState().sessionId;
    if (!conversationId) {
      conversationId = await createConversation(auth.token, auth.fingerprint);
      store.setSessionId(conversationId);
    }
    await openStream(conversationId);
    return conversationId;
  }, [ensureAuth, openStream, store]);

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
        const auth = await ensureAuth();
        let conversationId = useChatStore.getState().sessionId;
        if (!conversationId) {
          conversationId = await createConversation(
            auth.token,
            auth.fingerprint
          );
          store.setSessionId(conversationId);
        }
        if (!useChatStore.getState().isConnected) {
          await openStream(conversationId);
        }
        await postChatMessage(
          auth.token,
          auth.fingerprint,
          conversationId,
          content
        );
        return "sent";
      } catch (err) {
        console.error("sendMessage failed:", err);
        return "failed";
      }
    },
    [ensureAuth, openStream, store]
  );

  const sendAction = useCallback(
    async (
      action: "confirm" | "modify" | "reject" | "trip_event",
      payload?: { change?: unknown; external_event?: unknown }
    ): Promise<"sent" | "failed"> => {
      try {
        const auth = await ensureAuth();
        const conversationId = useChatStore.getState().sessionId;
        if (!conversationId) return "failed";
        useChatStore.getState().setWaitingForConfirmation(false);
        useChatStore.getState().setLoading(true);
        if (!useChatStore.getState().isConnected) {
          await openStream(conversationId);
        }
        await postChatAction(
          auth.token,
          auth.fingerprint,
          conversationId,
          action,
          payload
        );
        return "sent";
      } catch (err) {
        console.error("sendAction failed:", err);
        useChatStore.getState().setLoading(false);
        return "failed";
      }
    },
    [ensureAuth, openStream]
  );

  const reconnect = useCallback(async () => {
    const auth = await ensureAuth();
    activeJobIdRef.current = null;
    lastEventIdRef.current = 0;
    store.clear();
    const conversationId = await createConversation(
      auth.token,
      auth.fingerprint
    );
    store.setSessionId(conversationId);
    await openStream(conversationId);
  }, [ensureAuth, openStream, store, activeJobIdRef, lastEventIdRef]);

  return { sendMessage, sendAction, reconnect, disconnect };
}
