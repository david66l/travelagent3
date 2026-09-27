import { useCallback, useEffect, useRef } from "react";
import {
  createConversation,
  ensureGuestSession,
  postChatAction,
  postChatMessage,
} from "@/lib/api";
import { useSSE } from "@/hooks/useSSE";
import { useChatStore } from "@/stores/chatStore";

interface SubmitKey {
  signature: string;
  key: string;
}

/**
 * One logical submit carries one Idempotency-Key.
 *
 * The backend resolves a key to the job it already created and replays it
 * (202 with the *same* job_id, `idempotent_replay: true`) without re-persisting
 * the user message. That is what makes a retry safe when the first attempt
 * reached the server but its response never came back — the exact case where a
 * fresh key buys a second, unattended job in the same conversation.
 *
 * Two rules make it work:
 *  - the key survives a failed attempt, so the retry can present it again;
 *  - the key changes as soon as the request body would change, because reusing
 *    it for a different body is a 409 (`IDEMPOTENCY_KEY_CONFLICT`).
 *
 * Hence the signature: same request identity → same key, anything else → new key.
 */
function idempotencyKeyFor(
  ref: { current: SubmitKey | null },
  signature: string
): { key: string; retried: boolean } {
  if (ref.current?.signature === signature) {
    return { key: ref.current.key, retried: true };
  }
  const key = crypto.randomUUID();
  ref.current = { signature, key };
  return { key, retried: false };
}

function clearSubmitKey(ref: { current: SubmitKey | null }, key: string): void {
  if (ref.current?.key === key) {
    ref.current = null;
  }
}

/**
 * Owns the chat connection for the page.
 *
 * Switching conversations is an *operation*, not a reaction to `sessionId`
 * changing: every operation that can move the user to another conversation takes
 * the next epoch from `beginConversationOperation` before its first `await`, and
 * re-checks that epoch after each await. Work that started under an older epoch
 * can therefore never post, open a stream, or write job state on top of the
 * conversation the user has since moved to.
 *
 * Two invariants follow from that:
 *  - `sessionId` is written by `adoptConversation` (switch / bootstrap / new
 *    chat) and emptied by `leaveConversation` — nowhere else. Store actions such
 *    as `restoreChat` / `loadTrip` only move data.
 *  - A stream is opened in exactly two roles: adopting the active conversation,
 *    and following a job that was just accepted. Nothing subscribes because a
 *    value changed, so there is no "passive follower" path to race with.
 *
 * The epoch lives in a ref, so instantiate this hook once per page
 * (`app/page.tsx`) and pass the callbacks down. A second `useChat()` would get
 * its own epoch counter and its own SSE refs, and every guard below would
 * silently stop meaning anything.
 */
export function useChat() {
  const store = useChatStore();
  // Connections are per conversation: `open` never touches another
  // conversation's stream, and `close` targets exactly one.
  const { open, close, closeAll, isOpen } = useSSE();
  const authRef = useRef<{ token: string; fingerprint: string } | null>(null);
  const conversationEpochRef = useRef(0);
  // The submit the user can still retry: kept on failure, dropped on success.
  const submitKeyRef = useRef<SubmitKey | null>(null);

  const ensureAuth = useCallback(async () => {
    if (!authRef.current) {
      authRef.current = await ensureGuestSession();
    }
    return authRef.current;
  }, []);

  const openStream = useCallback(
    async (conversationId: string, requestedJobId?: string) => {
      const state = useChatStore.getState();
      const runtime = state.runtimeFor(conversationId);
      const jobId =
        requestedJobId ||
        runtime.jobId ||
        (conversationId === state.sessionId && state.isLoading
          ? state.jobId ?? undefined
          : undefined);
      await open(conversationId, {
        jobId: jobId ?? undefined,
        // A brand-new job starts its own event log, so its cursor starts at 0.
        // Re-attaching an existing job resumes from where this conversation left
        // off — the cursor is per conversation, never shared.
        lastEventId: requestedJobId ? 0 : runtime.lastEventId,
      });
    },
    [open]
  );

  /**
   * Number a new conversation operation.
   *
   * This deliberately no longer tears down the stream of the conversation being
   * left, and it must not clear that conversation's buffered prose either: the
   * whole point of per-conversation runtimes is that a job keeps streaming and
   * keeps its accumulated text while the user is elsewhere. Nothing bleeds onto
   * the conversation being opened because `setRuntime` only mirrors onto the
   * top-level fields for the *active* conversation, and `adoptConversation`
   * projects the target's own runtime immediately after this returns.
   */
  const beginConversationOperation = useCallback(() => {
    const epoch = conversationEpochRef.current + 1;
    conversationEpochRef.current = epoch;
    return epoch;
  }, []);

  /**
   * Make `conversationId` the active conversation and open the stream that
   * follows it. This is the only writer of `sessionId`, so every caller must
   * already own the current epoch and pass it in.
   */
  const adoptConversation = useCallback(
    async (epoch: number, conversationId: string) => {
      if (epoch !== conversationEpochRef.current) return;
      store.setSessionId(conversationId);
      // Project this conversation's own runtime onto the top-level fields. Order
      // matters: `sessionId` is written first, so `setRuntime` now sees this
      // conversation as the active one and mirrors its stage / buffered prose /
      // loading flag — which is what makes switching back show the progress the
      // background stream accumulated instead of the previous conversation's.
      const runtime = useChatStore.getState().runtimeFor(conversationId);
      useChatStore.getState().setRuntime(conversationId, runtime);
      // The transcript just swapped in may already carry this conversation's
      // in-flight reply: `switchConversation` folds the live prose into the
      // departing conversation's snapshot, so a buffer that came back from a
      // reconnect is a *replay* of text already committed. Rendering it again as
      // a streaming bubble would show the reply twice, so drop the buffer — while
      // keeping `isStreaming`, which is what tells the user the job is still
      // running and lets the next fresh token resume the bubble.
      const restored = useChatStore.getState().messages;
      const lastMessage = restored[restored.length - 1];
      const buffered = runtime.streamingContent.trim();
      if (buffered && lastMessage?.role === "assistant") {
        const committed = lastMessage.content.trim();
        // Either the same text, or a replay that covers only part of it (the
        // reconnect stopped mid-reply), which is likewise already on screen.
        if (committed === buffered || committed.includes(buffered)) {
          useChatStore
            .getState()
            .setRuntime(conversationId, { streamingContent: "" });
        }
      }
      try {
        // The stream URL is built from the token in storage, so auth has to be
        // ready before the stream is opened — a switch can beat bootstrap's
        // first guest-session call on a cold start.
        await ensureAuth();
      } catch (err) {
        console.error("Chat auth failed:", err);
        return;
      }
      if (epoch !== conversationEpochRef.current) return;
      // Deliberately not awaited: `open` resolves when the stream *ends*, so
      // awaiting it here would block the caller for the life of the conversation.
      void openStream(conversationId).catch((err) =>
        console.error("Chat stream failed:", err)
      );
    },
    [ensureAuth, openStream, store, conversationEpochRef]
  );

  /**
   * The one entry point for switching conversations. `apply` performs the local
   * state swap for the target (a snapshot or a trip); it must not touch
   * `sessionId`, which this operation owns.
   */
  const switchConversation = useCallback(
    async (targetId: string | null, apply?: () => void) => {
      const currentId = useChatStore.getState().sessionId;
      if (!targetId || targetId === currentId) {
        // Re-opening the conversation already on screen (or a legacy record with
        // no conversation of its own) only needs the data swap: tearing the
        // stream down would replay the conversation from its first event.
        apply?.();
        return;
      }
      const epoch = beginConversationOperation();
      // Persist what the departing conversation looked like *before* the target
      // swaps the transcript in. Without this the message the user just sent
      // (and any prose the server has already streamed for it) is discarded by
      // the target's `restoreChat`, and switching back shows an empty thread.
      // `sessionId` still names the departing conversation here, which is what
      // makes the snapshot land under the right id.
      useChatStore.getState().snapshotConversation(currentId);
      apply?.();
      await adoptConversation(epoch, targetId);
    },
    [beginConversationOperation, adoptConversation]
  );

  /**
   * Leave the active conversation: number the operation, close **its** stream,
   * clear the state. Other conversations keep streaming.
   *
   * Returns the new epoch so `reconnect` can continue the same numbered
   * operation instead of starting a second one.
   */
  const leaveConversation = useCallback((): number => {
    const leaving = useChatStore.getState().sessionId;
    if (leaving) close(leaving);
    const epoch = beginConversationOperation();
    store.clear();
    return epoch;
  }, [beginConversationOperation, close, store]);

  /**
   * A failed submit does not prove the server never saw it. A lost response
   * leaves a job running that this client has no id for and no stream on — and
   * nothing lists jobs, so the only way to find it is to ask the stream for
   * whatever is running: `GET /chat/stream` without a `job_id` attaches to the
   * newest pending/running job of the conversation.
   */
  const resumeAfterFailedSubmit = useCallback(async () => {
    const conversationId = useChatStore.getState().sessionId;
    if (!conversationId) return;
    // Drop the cursor with the job id: the event ids belong to a different job's
    // log, and the server applies `last_event_id` to whichever job it discovers.
    useChatStore
      .getState()
      .setRuntime(conversationId, { jobId: null, lastEventId: 0 });
    try {
      await open(conversationId, { lastEventId: 0, force: true });
    } catch (err) {
      console.error("Resuming the conversation stream failed:", err);
    }
  }, [open]);

  const bootstrap = useCallback(async () => {
    const epoch = conversationEpochRef.current;
    const auth = await ensureAuth();
    if (epoch !== conversationEpochRef.current) return "";
    let conversationId = useChatStore.getState().sessionId;
    if (!conversationId) {
      conversationId = await createConversation(auth.token, auth.fingerprint);
      if (epoch !== conversationEpochRef.current) return "";
    }
    // Resuming the stored conversation on load is not a switch — epoch 0 is
    // already the current operation — so it is adopted under that same number.
    await adoptConversation(epoch, conversationId);
    return conversationId;
  }, [ensureAuth, adoptConversation, conversationEpochRef]);

  useEffect(() => {
    bootstrap().catch((err) => console.error("Chat bootstrap failed:", err));
    return () => closeAll();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
          // First message of the session: adopt the conversation so this
          // `sessionId` write goes through the numbered path like every other.
          await adoptConversation(epoch, conversationId);
        }
        if (
          epoch !== conversationEpochRef.current ||
          useChatStore.getState().sessionId !== conversationId
        ) {
          return "failed";
        }
        if (!isOpen(conversationId)) {
          void openStream(conversationId).catch((err) =>
            console.error("Chat stream failed:", err)
          );
        }
        // Same conversation + same text = the same logical submit, so a retry
        // after a lost response replays the accepted job instead of adding one.
        const submit = idempotencyKeyFor(
          submitKeyRef,
          JSON.stringify([conversationId, "chat", content])
        );
        if (submit.retried) {
          console.info(
            "Retrying an unconfirmed submit with the same Idempotency-Key:",
            submit.key
          );
        }
        const jobId = await postChatMessage(
          auth.token,
          auth.fingerprint,
          conversationId,
          content,
          submit.key
        );
        // The turn is ours now, so the next identical message is a new submit.
        clearSubmitKey(submitKeyRef, submit.key);
        // The accepted job belongs to `conversationId` whether or not the user is
        // still looking at it. `setRuntime` writes the top-level fields only when
        // that conversation is the active one, so this single path covers both:
        // it starts the turn on screen, or it keeps a conversation the user has
        // already left streaming in the background.
        useChatStore.getState().setRuntime(conversationId, {
          jobId,
          lastEventId: 0,
          jobStatus: "pending",
          currentStage: "正在规划…",
          activityPhase: "planning",
          isLoading: true,
          needsClarification: false,
        });
        void openStream(conversationId, jobId).catch((err) =>
          console.error("Chat job stream failed:", err)
        );
        return "sent";
      } catch (err) {
        console.error("sendMessage failed:", err);
        void resumeAfterFailedSubmit();
        return "failed";
      }
    },
    [
      ensureAuth,
      openStream,
      adoptConversation,
      resumeAfterFailedSubmit,
      isOpen,
      conversationEpochRef,
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
        if (!isOpen(conversationId)) {
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
        const actionPayload =
          action === "trip_event" ? payload : { ...payload, approval };
        // The body also carries the action and its payload, so they are part of
        // the submit's identity — a different payload must not reuse the key.
        const submit = idempotencyKeyFor(
          submitKeyRef,
          JSON.stringify([
            conversationId,
            action,
            actionPayload?.change ?? null,
            actionPayload?.external_event ?? null,
            actionPayload?.approval ?? null,
          ])
        );
        if (submit.retried) {
          console.info(
            "Retrying an unconfirmed submit with the same Idempotency-Key:",
            submit.key
          );
        }
        const jobId = await postChatAction(
          auth.token,
          auth.fingerprint,
          conversationId,
          action,
          actionPayload,
          submit.key
        );
        clearSubmitKey(submitKeyRef, submit.key);
        // Same attribution rule as `sendMessage`: the job is recorded against its
        // own conversation, and only mirrored to the top-level state when the
        // user is still looking at it.
        useChatStore.getState().setRuntime(conversationId, {
          jobId,
          lastEventId: 0,
          jobStatus: "pending",
          isLoading: true,
        });
        void openStream(conversationId, jobId).catch((err) =>
          console.error("Chat action stream failed:", err)
        );
        return "sent";
      } catch (err) {
        console.error("sendAction failed:", err);
        useChatStore.getState().setLoading(false);
        void resumeAfterFailedSubmit();
        return "failed";
      }
    },
    [
      ensureAuth,
      openStream,
      resumeAfterFailedSubmit,
      isOpen,
      conversationEpochRef,
    ]
  );

  const reconnect = useCallback(async () => {
    // "New chat" is one operation: clear, create, adopt. `adoptConversation`
    // owns the `sessionId` write and the single subscription, so nothing has to
    // fire again when that write lands.
    const epoch = leaveConversation();
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
      await adoptConversation(epoch, conversationId);
    } catch (err) {
      console.error("Starting a new conversation failed:", err);
    } finally {
      if (epoch === conversationEpochRef.current) {
        useChatStore.getState().setLoading(false);
        useChatStore.getState().setCurrentStage(null);
      }
    }
  }, [
    leaveConversation,
    ensureAuth,
    adoptConversation,
    store,
    conversationEpochRef,
  ]);

  return {
    sendMessage,
    sendAction,
    switchConversation,
    reconnect,
    leaveConversation,
    /** Close every conversation's stream (page teardown / explicit reset). */
    disconnect: closeAll,
  };
}
