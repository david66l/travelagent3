import {
  deriveItineraryBudget,
  useChatStore,
  type ConfirmedInfo,
  type Message,
  type PreferencePanel,
  type PolicyRoutingSummary,
  type PendingApproval,
} from "@/stores/chatStore";
import { labelForStage, resolveActivityPhase } from "@/lib/stageLabels";

/**
 * Merge the server's `recent_messages` into the locally held transcript.
 *
 * Why not replace: `recent_messages` is truncated server-side (last 10), so
 * assigning it over `local` would discard older history the client still has
 * (a freshly reloaded page keeps a longer window in its snapshots).
 *
 * Why not a plain "skip contents that already appear": a user can legitimately
 * send the same text twice, and collapsing those would delete a real message.
 * Instead we look for the longest suffix of `server` that matches a suffix of
 * `local` and append only what lies beyond it. That is exact for the case this
 * exists for — the client is behind the server because a reply was produced
 * while the user was looking at another conversation — and it cannot silently
 * drop a repeated message, because a repeated *older* message is never part of
 * the matched suffix unless everything after it also matches in order.
 *
 * Known residual limitation: `Message` carries no stable id, so a misalignment
 * (e.g. a local optimistic message the server never accepted) prevents the
 * suffix match and the merge appends nothing. Distinguishing "missing reply"
 * from "diverged transcript" needs a server-assigned id — a backend contract
 * change, deliberately out of scope here. Appending nothing is the safe side:
 * the alternative loses or duplicates user-visible content.
 */
export function mergeRecentMessages(local: Message[], server: Message[]): Message[] {
  if (server.length === 0) return local;
  if (local.length === 0) return server;

  const same = (a: Message, b: Message): boolean =>
    a.role === b.role && a.content === b.content;

  // Longest suffix of `server` that equals a suffix of `local`.
  let matched = 0;
  const maxOverlap = Math.min(local.length, server.length);
  for (let size = maxOverlap; size > 0; size -= 1) {
    let allEqual = true;
    for (let index = 0; index < size; index += 1) {
      if (
        !same(
          local[local.length - size + index],
          server[server.length - size + index]
        )
      ) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) {
      matched = size;
      break;
    }
  }

  return matched === server.length ? local : [...local, ...server.slice(matched)];
}

function splitTravelDates(value: unknown): {
  travel_dates?: string;
  startDate?: string;
  endDate?: string;
} {
  if (typeof value !== "string" || !value.trim()) return {};
  const travelDates = value.trim();
  const dates = travelDates.match(/\d{4}-\d{2}-\d{2}/g) || [];
  return {
    travel_dates: travelDates,
    startDate: dates[0],
    endDate: dates[1] || dates[0],
  };
}

export function profileToConfirmedInfo(profile: unknown): ConfirmedInfo | null {
  if (!profile || typeof profile !== "object") return null;
  const p = profile as Record<string, unknown>;
  const trip = (p.trip as Record<string, unknown>) || p;
  const personal = (p.personal as Record<string, unknown>) || {};
  const dateRange = splitTravelDates(trip.travel_dates);
  const merged: ConfirmedInfo = {
    destination: (trip.destination as string) || undefined,
    ...dateRange,
    travelers_count: (trip.travelers_count as number) || undefined,
    budget_range: (trip.budget_range as number) || undefined,
    travelers_type: (trip.travelers_type as string) || undefined,
    pace: (personal.pace as string) || (trip.pace as string) || undefined,
  };
  const hasAny = Object.values(merged).some((v) => v !== undefined && v !== null);
  return hasAny ? merged : null;
}

export function profileToPreferencePanel(profile: unknown): PreferencePanel | null {
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

export function applyProfileFromServer(
  profile: unknown,
  store: ChatStoreApi = useChatStore.getState()
) {
  const confirmed = profileToConfirmedInfo(profile);
  const preference = profileToPreferencePanel(profile);
  if (confirmed) {
    store.setConfirmedInfo(confirmed);
  }
  if (preference) {
    store.setPreferencePanel(preference);
  }
}

export type EventRefs = {
  activeJobIdRef: { current: string | null };
  lastEventIdRef: { current: number };
  /**
   * The conversation this event belongs to. The server stamps every frame with
   * it, so an event can be attributed even when the user has already switched to
   * a different conversation. Absent only in tests that call
   * `handleChatEvent` directly, where the single-conversation behaviour applies.
   */
  conversationId?: string;
};

type ChatStoreApi = ReturnType<typeof useChatStore.getState>;

/**
 * The store slice an event may write to.
 *
 * Writing every event straight onto the global store only works while exactly
 * one conversation can be in flight. Now that several can stream at once, an
 * event carries its own conversation and has to be attributed: progress from a
 * background conversation belongs in `runtimeByConversation`, never on the
 * conversation the user is looking at.
 *
 * Rather than branch at ~95 call sites, the whole handler runs against this
 * facade. Streaming-state methods are mapped onto the runtime entry for
 * `conversationId`; transcript-writing methods become no-ops for a background
 * conversation, because its durable messages arrive again from the server
 * (`state_restored` / durable replay) when the user switches back — writing them
 * into the active transcript is exactly the corruption this prevents.
 */
function scopedStore(conversationId: string | undefined): ChatStoreApi {
  const global = useChatStore.getState();
  if (!conversationId || conversationId === global.sessionId) {
    return global;
  }

  const runtime = global.runtimeFor(conversationId);
  const patch = (values: Parameters<typeof global.setRuntime>[1]) =>
    global.setRuntime(conversationId, values);
  const noop = () => undefined;

  return {
    ...global,
    // --- conversation-scoped reads ---
    isLoading: runtime.isLoading,
    isStreaming: runtime.isStreaming,
    streamingContent: runtime.streamingContent,
    needsClarification: runtime.needsClarification,
    waitingForConfirmation: runtime.waitingForConfirmation,
    currentStage: runtime.currentStage,
    jobStatus: runtime.jobStatus,
    activityPhase: runtime.activityPhase,
    // --- conversation-scoped writes ---
    setLoading: (value: boolean) => patch({ isLoading: value }),
    setNeedsClarification: (value: boolean) => patch({ needsClarification: value }),
    setWaitingForConfirmation: (value: boolean) =>
      patch({ waitingForConfirmation: value }),
    setCurrentStage: (value: string | null) => patch({ currentStage: value }),
    setJobStatus: (value: string | null) => patch({ jobStatus: value }),
    setActivityPhase: (value: typeof runtime.activityPhase) =>
      patch({ activityPhase: value }),
    setStreamingContent: (value: string) => patch({ streamingContent: value }),
    startStreaming: () => patch({ isStreaming: true }),
    stopStreaming: () => patch({ isStreaming: false }),
    // Appending is an atomic store action: several frames are parsed in one tick
    // and a value computed from this facade's snapshot would drop all but the
    // last chunk.
    appendStreamingContent: (chunk: string) =>
      global.appendStreamingChunk(conversationId, chunk),
    setJobId: (value: string | null) => patch({ jobId: value }),
    // --- transcript writes never apply to a background conversation ---
    addMessage: noop,
    setItinerary: noop,
    setPendingApproval: noop,
    setBudgetPanel: noop,
    setPolicyRouting: noop,
    setOutputUrls: noop,
    saveChatSnapshot: noop,
    confirmCurrentItinerary: noop,
  } as unknown as ChatStoreApi;
}

/**
 * Persist the coarse progress of a background conversation after a handler ran,
 * so the sidebar and any runtime indicator can show that it is still working.
 */
function mirrorBackgroundProgress(
  conversationId: string,
  store: ChatStoreApi
): void {
  const global = useChatStore.getState();
  if (conversationId === global.sessionId) return;
  global.setRuntime(conversationId, {
    isLoading: store.isLoading,
    isStreaming: store.isStreaming,
    currentStage: store.currentStage,
    jobStatus: store.jobStatus,
    activityPhase: store.activityPhase,
  });
}

/** Commit assistant prose once — skip if the latest bubble already matches. */
function commitAssistantProse(content: string, store: ChatStoreApi) {
  const text = content.trim();
  if (!text) return;
  const last = store.messages[store.messages.length - 1];
  if (last?.role === "assistant" && last.content.trim() === text) {
    return;
  }
  store.addMessage({
    role: "assistant",
    content: text,
    timestamp: Date.now(),
  });
}

function finalizeStreamingToMessage(store: ChatStoreApi) {
  if (!store.isStreaming && !store.streamingContent.trim()) {
    return;
  }
  const text = store.streamingContent.trim();
  if (text) {
    commitAssistantProse(text, store);
  }
  store.stopStreaming();
  store.setStreamingContent("");
}

function commitConfirmedItinerary(store: ChatStoreApi) {
  if (!store.itinerary?.length || store.waitingForConfirmation) return;
  store.confirmCurrentItinerary();
}

function applyServerBudget(raw: unknown, store: ChatStoreApi) {
  if (!raw || typeof raw !== "object") return;
  const budget = raw as Record<string, unknown>;
  if (typeof budget.total !== "number") return;
  const totalBudget = store.confirmedInfo?.budget_range ?? undefined;
  const spent = budget.total;
  const breakdown = Object.fromEntries(
    Object.entries(budget).filter(
      ([key, value]) =>
        typeof value === "number" && !["total", "travelers_count"].includes(key)
    )
  ) as Record<string, number>;
  store.setBudgetPanel({
    total_budget: totalBudget,
    spent,
    remaining: totalBudget === undefined ? undefined : totalBudget - spent,
    breakdown,
    status:
      totalBudget === undefined
        ? "estimate"
        : spent <= totalBudget
          ? "within_budget"
          : "over_budget",
  });
}

function applyPolicyRouting(raw: unknown, store: ChatStoreApi) {
  if (!raw || typeof raw !== "object") return;
  const summary = raw as Partial<PolicyRoutingSummary>;
  if (
    summary.schema_version !== "agent-policy-routing-summary.v1" ||
    !summary.route_counts ||
    !Array.isArray(summary.decisions)
  ) {
    return;
  }
  store.setPolicyRouting(summary as PolicyRoutingSummary);
}

export function handleChatEvent(
  data: Record<string, unknown>,
  refs: EventRefs
) {
  // The frame names its conversation; fall back to the stream that delivered it
  // so a missing field can never silently attribute progress to the wrong one.
  const conversationId =
    (typeof data.conversation_id === "string" && data.conversation_id) ||
    refs.conversationId;
  const store = scopedStore(conversationId);
  const type = data.type as string | undefined;

  // Event ids belong to the durable PlanningJob event log, regardless of the
  // public event type. Persist the cursor for reconnect before handling it.
  if (typeof data.event_id === "number") {
    refs.lastEventIdRef.current = Math.max(
      refs.lastEventIdRef.current,
      data.event_id
    );
  }

  if (type === "job_created") {
    store.setJobId(data.job_id as string);
    refs.activeJobIdRef.current = data.job_id as string;
    refs.lastEventIdRef.current = 0;
    store.setJobStatus("pending");
    store.setCurrentStage("正在规划…");
    store.setActivityPhase("planning");
    store.setLoading(true);
    store.setNeedsClarification(false);
    return;
  }

  if (type === "intent_ready") {
    const content =
      (data.content as string) || "意图识别已完成，接下来将进行大致的规划。";
    applyProfileFromServer(data.profile, store);
    store.addMessage({
      role: "assistant",
      content,
      timestamp: Date.now(),
    });
    store.setActivityPhase("planning");
    store.setCurrentStage("正在规划…");
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
    store.setActivityPhase("idle");
    applyProfileFromServer(data.profile, store);
    store.addMessage({
      role: "assistant",
      content: text,
      timestamp: Date.now(),
    });
    return;
  }

  if (type === "message" && data.role === "assistant") {
    const content = (data.content as string) || "";
    const itinerary = data.itinerary as Parameters<typeof store.setItinerary>[0] | undefined;
    const outputUrls = {
      pdf: (data.output_pdf_url as string) || undefined,
      excel: (data.output_excel_url as string) || undefined,
      map: (data.output_map_url as string) || undefined,
    };
    store.setLoading(false);
    store.setCurrentStage("完成");
    store.setActivityPhase("idle");
    store.setNeedsClarification(false);
    store.setWaitingForConfirmation(false);
    store.setPendingApproval(null);
    applyProfileFromServer(data.profile, store);
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    applyServerBudget(data.budget_breakdown, store);
    applyPolicyRouting(data.agent_policy_routing, store);
    store.setOutputUrls(outputUrls);
    finalizeStreamingToMessage(store);
    if (content) {
      commitAssistantProse(content, store);
    }
    commitConfirmedItinerary(store);
    store.saveChatSnapshot();
    return;
  }

  if (type === "awaiting_confirm") {
    const itinerary = data.itinerary as Parameters<typeof store.setItinerary>[0] | undefined;
    store.setLoading(false);
    store.setCurrentStage("待确认");
    store.setActivityPhase("idle");
    store.setNeedsClarification(false);
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    applyPolicyRouting(data.agent_policy_routing, store);
    // Ensure streamed / partial prose is persisted before we drop the buffer.
    finalizeStreamingToMessage(store);
    store.setWaitingForConfirmation(true);
    // Only replace the stored approval when the event actually carries one.
    // Coercing a missing key to null erased the approval the user must echo
    // back, so confirmation always failed with APPROVAL_REQUIRED.
    if ("pending_approval" in data) {
      store.setPendingApproval(
        (data.pending_approval as PendingApproval | undefined) || null
      );
    }
    store.saveChatSnapshot();
    return;
  }

  if (type === "partial") {
    const payload = data.payload as Record<string, unknown> | undefined;
    const content = (payload?.content as string) || "";
    const itinerary = payload?.itinerary as Parameters<typeof store.setItinerary>[0] | undefined;
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    applyPolicyRouting(payload?.agent_policy_routing, store);
    if (content) {
      // A full `content` payload supersedes the live token buffer. Drop the
      // buffer (without committing it) and commit the consolidated prose once
      // so the itinerary text is never appended twice.
      const streamed = useChatStore.getState().streamingContent.trim();
      const finalContent = streamed.length > content.length ? streamed : content;
      store.stopStreaming();
      store.setStreamingContent("");
      commitAssistantProse(finalContent, store);
    } else if (store.isStreaming) {
      finalizeStreamingToMessage(store);
    }
    const outputUrls = {
      pdf: (payload?.output_pdf_url as string) || undefined,
      excel: (payload?.output_excel_url as string) || undefined,
      map: (payload?.output_map_url as string) || undefined,
    };
    if (outputUrls.pdf || outputUrls.excel || outputUrls.map) {
      store.setOutputUrls(outputUrls);
    }
    return;
  }

  if (type === "token") {
    const chunk = (data.chunk as string) || "";
    if (chunk) {
      if (!store.isStreaming) {
        store.startStreaming();
      }
      store.appendStreamingContent(chunk);
    }
    return;
  }

  if (type === "final") {
    const payload = data.payload as Record<string, unknown> | undefined;
    const content = (payload?.content as string) || "";
    const itinerary = payload?.itinerary as Parameters<typeof store.setItinerary>[0] | undefined;
    const outputUrls = {
      pdf: (payload?.output_pdf_url as string) || undefined,
      excel: (payload?.output_excel_url as string) || undefined,
      map: (payload?.output_map_url as string) || undefined,
    };
    store.setLoading(false);
    store.setCurrentStage("完成");
    store.setActivityPhase("idle");
    store.setWaitingForConfirmation(false);
    store.setPendingApproval(null);
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    applyPolicyRouting(payload?.agent_policy_routing, store);
    store.setOutputUrls(outputUrls);
    finalizeStreamingToMessage(store);
    if (content) {
      commitAssistantProse(content, store);
    }
    commitConfirmedItinerary(store);
    store.saveChatSnapshot();
    return;
  }

  if (type === "state_restored") {
    applyProfileFromServer(data.profile, store);
    applyPolicyRouting(data.agent_policy_routing, store);
    const phase = (data.phase as string) || "gathering";
    const itinerary = data.itinerary as Parameters<typeof store.setItinerary>[0] | undefined;
    const recentMessages = Array.isArray(data.recent_messages)
      ? data.recent_messages
          .filter(
            (message): message is Record<string, unknown> =>
              !!message &&
              typeof message === "object" &&
              (message.role === "user" || message.role === "assistant") &&
              typeof message.content === "string"
          )
          .map((message) => ({
            role: message.role as "user" | "assistant",
            content: message.content as string,
            timestamp:
              typeof message.ts === "number" ? message.ts * 1000 : Date.now(),
          }))
      : [];
    useChatStore.setState((current) => ({
      // The transcript may already hold this conversation's earlier history
      // (restored from a snapshot when the user switched back). Merge the
      // server's window into it so a reply produced while the user was looking
      // at another conversation is not discarded — see mergeRecentMessages.
      messages: mergeRecentMessages(current.messages, recentMessages),
      itinerary: itinerary?.length ? itinerary : current.itinerary,
      waitingForConfirmation: phase === "awaiting_confirm",
      pendingApproval:
        (data.pending_approval as PendingApproval | undefined) ??
        current.pendingApproval,
      isLoading: phase === "planning",
      currentStage:
        phase === "awaiting_confirm"
          ? "待确认"
          : phase === "completed"
            ? "完成"
            : current.currentStage,
      activityPhase: phase === "planning" ? "planning" : "idle",
    }));
    if (data.budget_breakdown) {
      applyServerBudget(data.budget_breakdown, store);
    } else if (itinerary?.length) {
      const current = useChatStore.getState();
      current.setBudgetPanel(
        deriveItineraryBudget(
          itinerary,
          current.confirmedInfo?.budget_range ??
            current.preferencePanel?.budget_range
        )
      );
    }
    if (phase === "completed") {
      commitConfirmedItinerary(store);
    }
    return;
  }

  if (type === "revision_created") {
    applyProfileFromServer(data.profile, store);
    return;
  }

  if (type === "stage" || data.stage) {
    const stagePayload = data.payload as Record<string, unknown> | undefined;
    applyPolicyRouting(stagePayload?.agent_policy_routing, store);
    const stage = data.stage as string;
    store.setJobStatus(stage);

    const phase = resolveActivityPhase(stage);
    if (phase !== "idle") {
      store.setActivityPhase(phase);
    }

    const stageLabel = labelForStage(stage);
    if (stageLabel) {
      store.setLoading(true);
      store.setCurrentStage(stageLabel);
    }

    if (stage === "draft_ready" || stage === "planned") {
      const payload = data.payload as Record<string, unknown> | undefined;
      // The plan node emits the solved itinerary under `itinerary` (not
      // `itinerary_draft`); render it immediately so the user sees the plan as
      // soon as solving finishes, instead of waiting out the prose polish.
      const draft = payload?.itinerary_draft ?? payload?.itinerary;
      if (draft) {
        store.setItinerary(draft as Parameters<typeof store.setItinerary>[0]);
      }
    } else if (stage === "itinerary_final") {
      if (!stageLabel) {
        store.setCurrentStage("行程已优化");
      }
      const payload = data.payload as Record<string, unknown> | undefined;
      if (payload?.itinerary_final) {
        store.setItinerary(
          payload.itinerary_final as Parameters<typeof store.setItinerary>[0]
        );
      }
    } else if (stage === "writing") {
      if (!stageLabel) {
        store.setCurrentStage("正在生成行程方案…");
      }
      // Do NOT start streaming here: starting the caret before any token
      // arrives shows an empty blinking cursor while the model warms up. The
      // `token` handler starts streaming on the first real chunk instead.
    } else if (stage === "completed") {
      refs.activeJobIdRef.current = null;
      refs.lastEventIdRef.current = 0;
      store.setCurrentStage("完成");
      store.setActivityPhase("idle");
      store.setLoading(false);
      store.setNeedsClarification(false);
      const payload = data.payload as Record<string, unknown> | undefined;
      // Prefer the finalized proposal text when available; fall back to the
      // streaming buffer for cases where the backend did not send a proposal.
      const finalText =
        (payload?.proposal_text as string) || store.streamingContent;
      finalizeStreamingToMessage(store);
      if (finalText) {
        commitAssistantProse(finalText, store);
      }
      if (payload?.itinerary || payload?.itinerary_final) {
        store.setItinerary(
          (payload.itinerary_final || payload.itinerary) as Parameters<
            typeof store.setItinerary
          >[0]
        );
      }
      commitConfirmedItinerary(store);
      store.saveChatSnapshot();
    } else if (stage === "failed" || stage === "cancelled") {
      refs.activeJobIdRef.current = null;
      refs.lastEventIdRef.current = 0;
      store.setActivityPhase("idle");
      store.setCurrentStage(stage === "failed" ? "处理失败" : "已取消");
      store.setLoading(false);
      if (store.isStreaming) {
        store.addMessage({
          role: "assistant",
          content: store.streamingContent || "行程规划已中断",
          timestamp: Date.now(),
        });
        store.stopStreaming();
        store.setStreamingContent("");
      } else {
        store.addMessage({
          role: "assistant",
          content:
            stage === "failed"
              ? `错误: ${(data.error as string) || "处理失败"}`
              : "行程规划已取消",
          timestamp: Date.now(),
        });
      }
    }
    return;
  }

  if (type === "error") {
    if (store.isStreaming) {
      store.addMessage({
        role: "assistant",
        content: store.streamingContent || `错误: ${(data.error as string) || "未知错误"}`,
        timestamp: Date.now(),
      });
      store.stopStreaming();
      store.setStreamingContent("");
    } else {
      store.addMessage({
        role: "assistant",
        content: `错误: ${(data.error as string) || "未知错误"}`,
        timestamp: Date.now(),
      });
    }
    store.setLoading(false);
    store.setCurrentStage(null);
    store.setActivityPhase("idle");
    return;
  }

  if (type === "done") {
    if (store.isStreaming) {
      store.addMessage({
        role: "assistant",
        content: store.streamingContent,
        timestamp: Date.now(),
      });
      store.stopStreaming();
      store.setStreamingContent("");
    }
    store.setConnected(false);
  }
}
