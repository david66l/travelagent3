import {
  useChatStore,
  type ConfirmedInfo,
  type PreferencePanel,
} from "@/stores/chatStore";
import { labelForStage, resolveActivityPhase } from "@/lib/stageLabels";

export function profileToConfirmedInfo(profile: unknown): ConfirmedInfo | null {
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
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    store.setOutputUrls(outputUrls);
    if (content) {
      store.addMessage({
        role: "assistant",
        content,
        timestamp: Date.now(),
      });
    }
    store.saveChatSnapshot();
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
    if (itinerary) {
      store.setItinerary(itinerary);
    }
    store.setOutputUrls(outputUrls);
    if (content) {
      store.addMessage({
        role: "assistant",
        content,
        timestamp: Date.now(),
      });
    }
    store.saveChatSnapshot();
    return;
  }

  if (type === "state_restored" || type === "revision_created") {
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

    if (stage === "draft_ready") {
      const payload = data.payload as Record<string, unknown> | undefined;
      if (payload?.itinerary_draft) {
        store.setItinerary(
          payload.itinerary_draft as Parameters<typeof store.setItinerary>[0]
        );
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
        store.setCurrentStage("正在润色文案...");
      }
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
      const finalText = (payload?.proposal_text as string) || store.streamingContent;
      if (finalText) {
        store.addMessage({
          role: "assistant",
          content: finalText,
          timestamp: Date.now(),
        });
      }
      if (store.isStreaming) {
        store.stopStreaming();
        store.setStreamingContent("");
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
