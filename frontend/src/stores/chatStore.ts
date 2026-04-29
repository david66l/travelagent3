import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  runStatus?: RunStatus;
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

// === AI 运行状态 ===
export interface StepDetail {
  name: string;
  status: string;
  duration_ms: number;
  start_offset: number;
  end_offset: number | null;
}

export interface RunStatus {
  status: "running" | "completed" | "error";
  current_step: string | null;
  completed_steps: string[];
  step_details: StepDetail[];
  elapsed_seconds: number;
  ttft_seconds: number | null;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  llm_calls: number;
  step_count: number;
  completed_count: number;
}

// === 新增：对话快照 ===
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
  activeTab: "chat" | "itinerary" | "panels";
  activeView: "chat" | "itinerary" | "export" | "settings";

  chatHistory: ChatHistoryItem[];
  chatSnapshots: ChatSnapshot[];  // 新增：完整对话快照

  confirmedInfo: ConfirmedInfo | null;
  activeBriefDay: number;
  pendingSuggestions: PendingSuggestion[];
  trips: TripRecord[];
  currentTrip: TripRecord | null;

  runStatus: RunStatus | null;

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
  setActiveTab: (v: "chat" | "itinerary" | "panels") => void;
  setActiveView: (v: "chat" | "itinerary" | "export" | "settings") => void;

  setConfirmedInfo: (v: ConfirmedInfo | null) => void;
  setActiveBriefDay: (v: number) => void;
  setPendingSuggestions: (v: PendingSuggestion[]) => void;
  confirmCurrentItinerary: () => void;
  loadTrip: (tripId: string) => void;

  // 新增方法
  saveChatSnapshot: () => void;
  restoreChat: (snapshotId: string) => void;
  refreshTripStatuses: () => void;

  setRunStatus: (v: RunStatus | null) => void;
  clearRunStatus: () => void;

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
      activeTab: "chat",
      activeView: "chat",

      chatHistory: [],
      chatSnapshots: [],

      confirmedInfo: null,
      activeBriefDay: 0,
      pendingSuggestions: [],
      trips: [],
      currentTrip: null,

      runStatus: null,

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
          return { itinerary: v, activeBriefDay: newActiveBriefDay };
        }),
      setBudgetPanel: (v) => set({ budgetPanel: v }),
      setPreferencePanel: (v) => set({ preferencePanel: v }),
      setValidationResult: (v) => set({ validationResult: v }),
      setIntent: (v) => set({ intent: v }),
      setNeedsClarification: (v) => set({ needsClarification: v }),
      setWaitingForConfirmation: (v) => set({ waitingForConfirmation: v }),
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

        // 防重：检查是否已存在相同目的地和日期的行程
        const duplicate = state.trips.find(
          (t) =>
            t.destination === destination &&
            t.startDate === startDate &&
            t.endDate === endDate
        );
        if (duplicate) {
          // 已存在则只设为当前行程，不再新建
          set({ currentTrip: duplicate });
          return;
        }

        const trip: TripRecord = {
          id: `trip-${Date.now()}`,
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
          budgetPanel: state.budgetPanel || {
            spent: 0,
            breakdown: {},
            status: "within_budget",
          },
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
          runStatus: null,
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

      setRunStatus: (v) => set({ runStatus: v }),
      clearRunStatus: () => set({ runStatus: null }),

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
          runStatus: null,
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
          confirmedInfo: null,
          activeBriefDay: 0,
          pendingSuggestions: [],
          currentTrip: null,
          chatHistory: [],
          isLoading: false,
          runStatus: null,
        });
      },
    }),
    {
      name: "travel-agent-chat-storage",
      partialize: (state) => ({
        chatSnapshots: state.chatSnapshots,
        trips: state.trips,
      }),
    }
  )
);
