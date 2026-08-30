import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface Activity {
  poi_name: string;
  category: string;
  start_time?: string;
  end_time?: string;
  duration_min: number;
  ticket_price?: number;
  recommendation_reason: string;
  tags: string[];
}

export interface DayPlan {
  day_number: number;
  date?: string;
  theme?: string;
  activities: Activity[];
  total_cost: number;
}

export interface BudgetPanel {
  total_budget?: number;
  spent: number;
  remaining?: number;
  breakdown: Record<string, number>;
  status: string;
}

export interface PreferencePanel {
  destination?: string;
  travel_days?: number;
  travel_dates?: string;
  travelers_count?: number;
  travelers_type?: string;
  budget_range?: number;
  food_preferences: string[];
  interests: string[];
  pace?: string;
  special_requests: string[];
}

export interface ValidationResult {
  passed: boolean;
  scores: Record<string, number>;
  total_score: number;
  critical_failures: string[];
  improvement_suggestions: string[];
}

export interface ChatHistoryItem {
  id: string;
  title: string;
  date: string;
}

export interface ConfirmedInfo {
  destination?: string;
  travel_dates?: string;
  startDate?: string;
  endDate?: string;
  travelers_count?: number;
  budget_range?: number;
  travelers_type?: string;
  pace?: string;
}

export interface PendingSuggestion {
  id: string;
  text: string;
}

export interface TripRecord {
  id: string;
  conversationId: string;
  title: string;
  destination: string;
  dates: string;
  startDate: string;
  endDate: string;
  status: "upcoming" | "active" | "completed";
  createdAt: number;
  itinerary: DayPlan[];
  preferencePanel: PreferencePanel;
  budgetPanel: BudgetPanel;
}

export interface BriefDayPlan {
  day_number: number;
  theme: string;
  highlights: string[];
}

// === 对话快照 ===
export interface OutputUrls {
  pdf?: string;
  excel?: string;
  map?: string;
}

export interface PendingApproval {
  schema_version: string;
  approval_id: string;
  goal_version: number;
  plan_version: number;
  itinerary_hash: string;
  issued_at: string;
  expires_at: string;
  action_scope: string[];
}

export interface PolicyRoutingDecision {
  step_index: number;
  task_id: string;
  action: string;
  requested_target: "student" | "teacher";
  executed_target: "student" | "teacher";
  family: "clarification" | "search" | "recovery" | "tradeoff" | "complex";
  reason: string;
  fallback_used: boolean;
  fallback_error_code?: string | null;
  model?: string | null;
  completion_tokens: number;
  request_latency_ms: number;
}

export interface PolicyRoutingSummary {
  schema_version: "agent-policy-routing-summary.v1";
  decisions: PolicyRoutingDecision[];
  route_counts: { student: number; teacher: number };
  family_counts: Record<string, number>;
  fallback_count: number;
  completion_tokens: number;
  request_latency_ms: number;
}

export interface ChatSnapshot {
  id: string;
  title: string;
  date: string;
  messages: Message[];
  confirmedInfo: ConfirmedInfo | null;
  itinerary: DayPlan[] | null;
  preferencePanel: PreferencePanel | null;
  budgetPanel: BudgetPanel | null;
  pendingSuggestions: PendingSuggestion[];
}

export interface ChatState {
  sessionId: string;
  messages: Message[];
  isConnected: boolean;
  isLoading: boolean;
  itinerary: DayPlan[] | null;
  budgetPanel: BudgetPanel | null;
  preferencePanel: PreferencePanel | null;
  validationResult: ValidationResult | null;
  intent: string | null;
  needsClarification: boolean;
  waitingForConfirmation: boolean;
  pendingApproval: PendingApproval | null;
  activeTab: "chat" | "itinerary" | "panels";
  activeView: "chat" | "itinerary" | "export" | "booking" | "settings";

  chatHistory: ChatHistoryItem[];
  chatSnapshots: ChatSnapshot[];  // 新增：完整对话快照

  confirmedInfo: ConfirmedInfo | null;
  activeBriefDay: number;
  pendingSuggestions: PendingSuggestion[];
  trips: TripRecord[];
  currentTrip: TripRecord | null;

  // Job-based planning state
  jobId: string | null;
  currentStage: string | null;
  jobStatus: string | null;
  activityPhase: "idle" | "gathering" | "planning";

  // Exported artifact URLs from the graph runtime
  outputUrls: OutputUrls | null;
  policyRouting: PolicyRoutingSummary | null;

  // Streaming text state
  streamingContent: string;
  isStreaming: boolean;

  setSessionId: (id: string) => void;
  addMessage: (msg: Message) => void;
  setConnected: (v: boolean) => void;
  setLoading: (v: boolean) => void;
  setItinerary: (v: DayPlan[] | null) => void;
  setBudgetPanel: (v: BudgetPanel | null) => void;
  setPreferencePanel: (v: PreferencePanel | null) => void;
  setValidationResult: (v: ValidationResult | null) => void;
  setIntent: (v: string | null) => void;
  setNeedsClarification: (v: boolean) => void;
  setWaitingForConfirmation: (v: boolean) => void;
  setPendingApproval: (v: PendingApproval | null) => void;
  setActiveTab: (v: "chat" | "itinerary" | "panels") => void;
  setActiveView: (v: "chat" | "itinerary" | "export" | "booking" | "settings") => void;

  setConfirmedInfo: (v: ConfirmedInfo | null) => void;
  setActiveBriefDay: (v: number) => void;
  setPendingSuggestions: (v: PendingSuggestion[]) => void;
  confirmCurrentItinerary: () => void;
  loadTrip: (tripId: string) => void;

  // 新增方法
  saveChatSnapshot: () => void;
  restoreChat: (snapshotId: string) => void;
  refreshTripStatuses: () => void;

  // Job state setters
  setJobId: (id: string | null) => void;
  setCurrentStage: (stage: string | null) => void;
  setJobStatus: (status: string | null) => void;
  setActivityPhase: (phase: "idle" | "gathering" | "planning") => void;
  setOutputUrls: (urls: OutputUrls | null) => void;
  setPolicyRouting: (summary: PolicyRoutingSummary | null) => void;

  // Streaming text setters
  appendStreamingContent: (chunk: string) => void;
  setStreamingContent: (content: string) => void;
  startStreaming: () => void;
  stopStreaming: () => void;

  clear: () => void;
}

export function deriveBriefItinerary(
  itinerary: DayPlan[] | null
): BriefDayPlan[] | null {
  if (!itinerary || itinerary.length === 0) return null;
  return itinerary.map((day) => ({
    day_number: day.day_number,
    theme: day.theme || `第 ${day.day_number} 天`,
    highlights: day.activities.slice(0, 3).map((a) => a.poi_name),
  }));
}

export function deriveItineraryBudget(
  itinerary: DayPlan[],
  totalBudget?: number
): BudgetPanel {
  const spent = itinerary.reduce(
    (total, day) => total + (Number(day.total_cost) || 0),
    0
  );
  return {
    total_budget: totalBudget,
    spent,
    remaining: totalBudget === undefined ? undefined : totalBudget - spent,
    breakdown: { itinerary: spent },
    status:
      totalBudget === undefined
        ? "estimate"
        : spent <= totalBudget
          ? "within_budget"
          : "over_budget",
  };
}

function normalizeTrip(
  trip: TripRecord,
  snapshots: ChatSnapshot[]
): TripRecord {
  const totalBudget = trip.budgetPanel.total_budget ?? trip.preferencePanel.budget_range;
  const itineraryCost = trip.itinerary.reduce(
    (total, day) => total + (Number(day.total_cost) || 0),
    0
  );
  const hasLegacyEmptyBudget =
    trip.budgetPanel.spent === 0 &&
    Object.keys(trip.budgetPanel.breakdown || {}).length === 0 &&
    (itineraryCost > 0 || totalBudget !== undefined);
  const matchedSnapshot = snapshots.find(
    (snapshot) =>
      snapshot.confirmedInfo?.destination === trip.destination &&
      snapshot.confirmedInfo?.travel_dates === trip.dates
  );
  const normalized = {
    ...trip,
    conversationId: trip.conversationId || matchedSnapshot?.id || "",
  };
  if (!hasLegacyEmptyBudget) return normalized;
  return {
    ...normalized,
    budgetPanel: {
      total_budget: totalBudget,
      spent: itineraryCost,
      remaining:
        totalBudget === undefined ? undefined : totalBudget - itineraryCost,
      breakdown: { itinerary: itineraryCost },
      status:
        totalBudget === undefined
          ? "estimate"
          : itineraryCost <= totalBudget
            ? "within_budget"
            : "over_budget",
    },
  };
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessionId: "",
      messages: [],
      isConnected: false,
      isLoading: false,
      itinerary: null,
      budgetPanel: null,
      preferencePanel: null,
      validationResult: null,
      intent: null,
      needsClarification: false,
      waitingForConfirmation: false,
      pendingApproval: null,
      activeTab: "chat",
      activeView: "chat",

      chatHistory: [],
      chatSnapshots: [],

      confirmedInfo: null,
      activeBriefDay: 0,
      pendingSuggestions: [],
      trips: [],
      currentTrip: null,

      jobId: null,
      currentStage: null,
      jobStatus: null,
      activityPhase: "idle",
      outputUrls: null,
      policyRouting: null,

      streamingContent: "",
      isStreaming: false,

      setSessionId: (id) => set({ sessionId: id }),

      addMessage: (msg) =>
        set((state) => {
          const newMessages = [...state.messages, msg];
          let newChatHistory = state.chatHistory;
          // Only create a new chat history entry when this is the first user message
          // AND we don't already have a history entry for this session
          if (
            msg.role === "user" &&
            state.messages.length === 0 &&
            state.chatHistory.length === 0
          ) {
            const title =
              msg.content.slice(0, 15) + (msg.content.length > 15 ? "..." : "");
            const chatId = state.sessionId || `chat-${Date.now()}`;
            newChatHistory = [
              {
                id: chatId,
                title,
                date: new Date().toISOString().split("T")[0],
              },
            ];
          }
          return { messages: newMessages, chatHistory: newChatHistory };
        }),

      setConnected: (v) => set({ isConnected: v }),
      setLoading: (v) => set({ isLoading: v }),
      setItinerary: (v) =>
        set((state) => {
          let newActiveBriefDay = state.activeBriefDay;
          if (v && v.length > 0 && newActiveBriefDay >= v.length) {
            newActiveBriefDay = v.length - 1;
          } else if (!v || v.length === 0) {
            newActiveBriefDay = 0;
          }
          return {
            itinerary: v,
            activeBriefDay: newActiveBriefDay,
            // The itinerary is the source of truth while a draft is being
            // edited. A later completed booking event may replace this with
            // its wider flight/hotel projection.
            budgetPanel: v?.length
              ? deriveItineraryBudget(
                  v,
                  state.confirmedInfo?.budget_range ??
                    state.preferencePanel?.budget_range
                )
              : state.budgetPanel,
          };
        }),
      setBudgetPanel: (v) => set({ budgetPanel: v }),
      setPreferencePanel: (v) => set({ preferencePanel: v }),
      setValidationResult: (v) => set({ validationResult: v }),
      setIntent: (v) => set({ intent: v }),
      setNeedsClarification: (v) => set({ needsClarification: v }),
      setWaitingForConfirmation: (v) => set({ waitingForConfirmation: v }),
      setPendingApproval: (v) => set({ pendingApproval: v }),
      setActiveTab: (v) => set({ activeTab: v }),
      setActiveView: (v) => set({ activeView: v }),

      setConfirmedInfo: (v) => set({ confirmedInfo: v }),
      setActiveBriefDay: (v) => set({ activeBriefDay: v }),
      setPendingSuggestions: (v) => set({ pendingSuggestions: v }),

      confirmCurrentItinerary: () => {
        const state = get();
        if (!state.itinerary || state.itinerary.length === 0) return;

        const destination = state.confirmedInfo?.destination || "";
        const startDate = state.confirmedInfo?.startDate || "";
        const endDate = state.confirmedInfo?.endDate || "";
        const totalBudget = state.confirmedInfo?.budget_range;
        const budgetPanel =
          state.budgetPanel || deriveItineraryBudget(state.itinerary, totalBudget);

        // 防重：检查是否已存在相同目的地和日期的行程
        const duplicate = state.trips.find(
          (t) =>
            t.destination === destination &&
            t.startDate === startDate &&
            t.endDate === endDate
        );
        if (duplicate) {
          const updatedTrip: TripRecord = {
            ...duplicate,
            conversationId: state.sessionId || duplicate.conversationId,
            itinerary: state.itinerary,
            preferencePanel: state.preferencePanel || duplicate.preferencePanel,
            budgetPanel,
          };
          set((current) => ({
            trips: current.trips.map((trip) =>
              trip.id === duplicate.id ? updatedTrip : trip
            ),
            currentTrip: updatedTrip,
          }));
          return;
        }

        const trip: TripRecord = {
          id: `trip-${Date.now()}`,
          conversationId: state.sessionId,
          title: destination
            ? `${destination}${state.itinerary.length}日游`
            : `行程 ${state.trips.length + 1}`,
          destination,
          dates: state.confirmedInfo?.travel_dates || "",
          startDate,
          endDate,
          status: "upcoming",
          createdAt: Date.now(),
          itinerary: state.itinerary,
          preferencePanel: state.preferencePanel || {
            food_preferences: [],
            interests: [],
            special_requests: [],
          },
          budgetPanel,
        };

        set((s) => ({
          trips: [trip, ...s.trips],
          currentTrip: trip,
        }));
      },

      loadTrip: (tripId) => {
        const state = get();
        const trip = state.trips.find((t) => t.id === tripId);
        if (!trip) return;

        set({
          sessionId: trip.conversationId || state.sessionId,
          currentTrip: trip,
          itinerary: trip.itinerary,
          preferencePanel: trip.preferencePanel,
          budgetPanel: trip.budgetPanel,
          confirmedInfo: {
            destination: trip.destination,
            travel_dates: trip.dates,
            startDate: trip.startDate,
            endDate: trip.endDate,
          },
          isLoading: false,
          isConnected: false,
        });
      },

      refreshTripStatuses: () => {
        const today = new Date().toISOString().split("T")[0];
        set((state) => ({
          trips: state.trips.map((trip) => {
            if (trip.status === "completed") return trip;
            if (trip.endDate && trip.endDate < today) {
              return { ...trip, status: "completed" as const };
            }
            if (trip.startDate && trip.startDate <= today && trip.endDate && trip.endDate >= today) {
              return { ...trip, status: "active" as const };
            }
            return { ...trip, status: "upcoming" as const };
          }),
        }));
      },

      setJobId: (id) => set({ jobId: id }),
      setCurrentStage: (stage) => set({ currentStage: stage }),
      setJobStatus: (status) => set({ jobStatus: status }),
      setActivityPhase: (phase) => set({ activityPhase: phase }),
      setOutputUrls: (urls) => set({ outputUrls: urls }),
      setPolicyRouting: (summary) => set({ policyRouting: summary }),

      appendStreamingContent: (chunk) =>
        set((state) => ({
          streamingContent: state.streamingContent + chunk,
        })),
      setStreamingContent: (content) => set({ streamingContent: content }),
      startStreaming: () => set({ isStreaming: true, streamingContent: "" }),
      stopStreaming: () => set({ isStreaming: false }),

      // 保存当前对话快照
      saveChatSnapshot: () => {
        const state = get();
        if (state.messages.length === 0) return;

        const snapshotId = state.sessionId || `chat-${Date.now()}`;
        const title = state.chatHistory[0]?.title || "未命名对话";

        const snapshot: ChatSnapshot = {
          id: snapshotId,
          title,
          date: new Date().toISOString().split("T")[0],
          messages: [...state.messages],
          confirmedInfo: state.confirmedInfo,
          itinerary: state.itinerary,
          preferencePanel: state.preferencePanel,
          budgetPanel: state.budgetPanel,
          pendingSuggestions: [...state.pendingSuggestions],
        };

        set((s) => {
          const existing = s.chatSnapshots.findIndex((cs) => cs.id === snapshotId);
          let newSnapshots;
          if (existing >= 0) {
            newSnapshots = [...s.chatSnapshots];
            newSnapshots[existing] = snapshot;
          } else {
            newSnapshots = [snapshot, ...s.chatSnapshots];
          }
          return { chatSnapshots: newSnapshots };
        });
      },

      // 恢复对话快照
      restoreChat: (snapshotId) => {
        const state = get();
        const snapshot = state.chatSnapshots.find((s) => s.id === snapshotId);
        if (!snapshot) return;

        set({
          sessionId: snapshot.id,
          messages: snapshot.messages,
          confirmedInfo: snapshot.confirmedInfo,
          itinerary: snapshot.itinerary,
          preferencePanel: snapshot.preferencePanel,
          budgetPanel: snapshot.budgetPanel,
          pendingSuggestions: snapshot.pendingSuggestions,
          activeBriefDay: 0,
          chatHistory: [
            {
              id: snapshot.id,
              title: snapshot.title,
              date: snapshot.date,
            },
          ],
          isLoading: false,
        });
      },

      clear: () => {
        const state = get();
        // 先保存当前对话
        if (state.messages.length > 0) {
          state.saveChatSnapshot();
        }
        set({
          sessionId: "",
          messages: [],
          itinerary: null,
          budgetPanel: null,
          preferencePanel: null,
          validationResult: null,
          intent: null,
          needsClarification: false,
          waitingForConfirmation: false,
          pendingApproval: null,
          confirmedInfo: null,
          activeBriefDay: 0,
          pendingSuggestions: [],
          currentTrip: null,
          chatHistory: [],
          isLoading: false,
          jobId: null,
          currentStage: null,
          jobStatus: null,
          activityPhase: "idle",
          outputUrls: null,
          policyRouting: null,
          streamingContent: "",
          isStreaming: false,
        });
      },
    }),
    {
      name: "travel-agent-chat-storage",
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<ChatState> | undefined;
        const snapshots = persisted?.chatSnapshots ?? [];
        const sessionId = persisted?.sessionId ?? "";
        const activeSnapshot = snapshots.find((snapshot) => snapshot.id === sessionId);
        const trips = (persisted?.trips ?? []).map((trip) =>
          normalizeTrip(trip, snapshots)
        );
        const persistedCurrentTrip = persisted?.currentTrip;
        const currentTrip = persistedCurrentTrip
          ? trips.find((trip) => trip.id === persistedCurrentTrip.id) ??
            normalizeTrip(persistedCurrentTrip, snapshots)
          : null;
        return {
          ...currentState,
          sessionId,
          messages: activeSnapshot?.messages ?? [],
          itinerary: currentTrip?.itinerary ?? activeSnapshot?.itinerary ?? null,
          confirmedInfo: activeSnapshot?.confirmedInfo ?? null,
          preferencePanel:
            currentTrip?.preferencePanel ?? activeSnapshot?.preferencePanel ?? null,
          budgetPanel: currentTrip?.budgetPanel ?? activeSnapshot?.budgetPanel ?? null,
          pendingSuggestions: activeSnapshot?.pendingSuggestions ?? [],
          waitingForConfirmation: persisted?.waitingForConfirmation ?? false,
          pendingApproval: persisted?.pendingApproval ?? null,
          activeView: persisted?.activeView ?? "chat",
          currentTrip,
          chatSnapshots: snapshots,
          trips,
        };
      },
      partialize: (state) => ({
        sessionId: state.sessionId,
        waitingForConfirmation: state.waitingForConfirmation,
        pendingApproval: state.pendingApproval,
        activeView: state.activeView,
        currentTrip: state.currentTrip,
        chatSnapshots: state.chatSnapshots,
        trips: state.trips,
      }),
    }
  )
);
