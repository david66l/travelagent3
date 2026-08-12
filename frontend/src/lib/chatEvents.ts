import {
  deriveItineraryBudget,
  useChatStore,
  type ConfirmedInfo,
  type PreferencePanel,
} from "@/stores/chatStore";
import { labelForStage, resolveActivityPhase } from "@/lib/stageLabels";

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

export function applyProfileFromServer(profile: unknown) {
  const confirmed = profileToConfirmedInfo(profile);
  const preference = profileToPreferencePanel(profile);
  if (confirmed) {
    useChatStore.getState().setConfirmedInfo(confirmed);
  }
  if (preference) {
    useChatStore.getState().setPreferencePanel(preference);
  }
}

type EventRefs = {
  activeJobIdRef: { current: string | null };
  lastEventIdRef: { current: number };
};

/** Commit assistant prose once — skip if the latest bubble already matches. */
function commitAssistantProse(content: string) {
  const text = content.trim();
  if (!text) return;
  const store = useChatStore.getState();
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

function finalizeStreamingToMessage() {
  const store = useChatStore.getState();
  if (!store.isStreaming && !store.streamingContent.trim()) {
    return;
  }
  const text = store.streamingContent.trim();
  if (text) {
    commitAssistantProse(text);
  }
  store.stopStreaming();
  store.setStreamingContent("");
}

function commitConfirmedItinerary() {
  const store = useChatStore.getState();
  if (!store.itinerary?.length || store.waitingForConfirmation) return;
  store.confirmCurrentItinerary();
}

function applyServerBudget(raw: unknown) {
  if (!raw || typeof raw !== "object") return;
  const budget = raw as Record<string, unknown>;
  if (typeof budget.total !== "number") return;
  const store = useChatStore.getState();
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

export function handleChatEvent(
  data: Record<string, unknown>,
  refs: EventRefs
) {
  const store = useChatStore.getState();
  const type = data.type as string | undefined;

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
    applyProfileFromServer(data.profile);
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
    applyProfileFromServer(data.profile);
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
    applyProfileFromServer(data.profile);
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    applyServerBudget(data.budget_breakdown);
    store.setOutputUrls(outputUrls);
    finalizeStreamingToMessage();
    if (content) {
      commitAssistantProse(content);
    }
    commitConfirmedItinerary();
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
    // Ensure streamed / partial prose is persisted before we drop the buffer.
    finalizeStreamingToMessage();
    store.setWaitingForConfirmation(true);
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
    if (content) {
      // A full `content` payload supersedes the live token buffer. Drop the
      // buffer (without committing it) and commit the consolidated prose once
      // so the itinerary text is never appended twice.
      const streamed = useChatStore.getState().streamingContent.trim();
      const finalContent = streamed.length > content.length ? streamed : content;
      store.stopStreaming();
      store.setStreamingContent("");
      commitAssistantProse(finalContent);
    } else if (store.isStreaming) {
      finalizeStreamingToMessage();
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
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    store.setOutputUrls(outputUrls);
    finalizeStreamingToMessage();
    if (content) {
      commitAssistantProse(content);
    }
    commitConfirmedItinerary();
    store.saveChatSnapshot();
    return;
  }

  if (type === "state_restored") {
    applyProfileFromServer(data.profile);
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
      messages: current.messages.length > 0 ? current.messages : recentMessages,
      itinerary: itinerary?.length ? itinerary : current.itinerary,
      waitingForConfirmation: phase === "awaiting_confirm",
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
      applyServerBudget(data.budget_breakdown);
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
      commitConfirmedItinerary();
    }
    return;
  }

  if (type === "revision_created") {
    applyProfileFromServer(data.profile);
    return;
  }

  if (type === "stage" || data.stage) {
    if (typeof data.event_id === "number") {
      refs.lastEventIdRef.current = Math.max(
        refs.lastEventIdRef.current,
        data.event_id
      );
    }
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
      finalizeStreamingToMessage();
      if (finalText) {
        commitAssistantProse(finalText);
      }
      if (payload?.itinerary || payload?.itinerary_final) {
        store.setItinerary(
          (payload.itinerary_final || payload.itinerary) as Parameters<
            typeof store.setItinerary
          >[0]
        );
      }
      commitConfirmedItinerary();
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
