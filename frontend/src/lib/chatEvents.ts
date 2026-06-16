import {
  useChatStore,
  type ConfirmedInfo,
  type PreferencePanel,
} from "@/stores/chatStore";

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

    if (stage === "running") {
      store.setCurrentStage("正在规划...");
    } else if (stage === "draft_ready") {
      store.setCurrentStage("行程草稿已生成");
      const payload = data.payload as Record<string, unknown> | undefined;
      if (payload?.itinerary_draft) {
        store.setItinerary(
          payload.itinerary_draft as Parameters<typeof store.setItinerary>[0]
        );
      }
    } else if (stage === "itinerary_final") {
      store.setCurrentStage("行程已优化");
      const payload = data.payload as Record<string, unknown> | undefined;
      if (payload?.itinerary_final) {
        store.setItinerary(
          payload.itinerary_final as Parameters<typeof store.setItinerary>[0]
        );
      }
    } else if (stage === "writing") {
      store.setCurrentStage("正在润色文案...");
    } else if (stage === "completed") {
      refs.activeJobIdRef.current = null;
      refs.lastEventIdRef.current = 0;
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
      refs.activeJobIdRef.current = null;
      refs.lastEventIdRef.current = 0;
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

  if (type === "done") {
    store.setConnected(false);
  }
}
