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

/**
 * In-flight state for ONE conversation.
 *
 * Every field here used to be a single top-level value on the store. That shape
 * can only describe the conversation the user is looking at, so a job running in
 * a conversation the user switched away from had nowhere to live: its job id and
 * event cursor were dropped and its progress was unrecoverable. These values are
 * therefore keyed by conversation id, while the top-level mirrors keep the active
 * conversation readable for existing components.
 */
export interface ConversationRuntime {
  jobId: string | null;
  /** Cursor into this job's durable event log. Never shared across jobs. */
  lastEventId: number;
  currentStage: string | null;
  jobStatus: string | null;
  activityPhase: "idle" | "gathering" | "planning";
  streamingContent: string;
  isStreaming: boolean;
  isLoading: boolean;
  needsClarification: boolean;
  waitingForConfirmation: boolean;
  /** Last few streamed prose chunks, so a background conversation can show a preview. */
  recentTokens: string[];
  /**
   * Event cursor at which this conversation's live buffer was last folded into
   * its durable transcript (0 = never). This is what makes the buffer
   * replay-safe: a reconnect re-delivers frames from the cursor, and prose that
   * was already committed must not be committed twice — nor silently dropped
   * because its buffer was cleared.
   */
  foldedAtEventId: number;
}

export function emptyConversationRuntime(): ConversationRuntime {
  return {
    jobId: null,
    lastEventId: 0,
    currentStage: null,
    jobStatus: null,
    activityPhase: "idle",
    streamingContent: "",
    isStreaming: false,
    isLoading: false,
    needsClarification: false,
    waitingForConfirmation: false,
    recentTokens: [],
    foldedAtEventId: 0,
  };
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

  /**
   * Per-conversation in-flight state. The top-level `jobId` / `currentStage` /
   * `isLoading` / `streamingContent` above are the projection of
   * `runtimeByConversation[activeConversationId]` and stay in sync with it.
   */
  runtimeByConversation: Record<string, ConversationRuntime>;
  /** Conversations that currently hold an open SSE stream. */
  connectedConversationIds: string[];

  // Exported artifact URLs from the graph runtime
  outputUrls: OutputUrls | null;
  policyRouting: PolicyRoutingSummary | null;

  // Streaming text state
  streamingContent: string;
  isStreaming: boolean;

  /**
   * Write the active conversation id. Only `useChat().adoptConversation` calls
   * this — a bare write here changes which conversation is displayed without
   * moving the stream, which is exactly the split the epoch guards exist to
   * prevent.
   */
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
  /**
   * Apply a saved trip to the store. Pure data operation: it deliberately does
   * not write `sessionId`, so the caller must move the conversation itself via
   * `useChat().switchConversation(trip.conversationId || null, () => loadTrip(id))`.
   */
  loadTrip: (tripId: string) => void;

  // 新增方法
  /**
   * Persist the active conversation's transcript under its own id. `snapshotId`
   * lets a caller snapshot the conversation it is *leaving* right before the
   * target's `restoreChat` replaces `messages`: at that instant `sessionId`
   * still names the departing conversation, so the id is passed explicitly
   * rather than inferred late.
   */
  saveChatSnapshot: (snapshotId?: string) => void;
  /**
   * Fold a conversation's pending live prose into its own snapshot before the
   * user leaves it, so switching back shows the same transcript the user was
   * looking at instead of dropping the in-flight reply.
   *
   * This is the deliberate counterpart to `scopedStore`, which refuses
   * transcript writes for a *background* conversation: prose may only enter a
   * transcript while that conversation is the one on screen, so the fold
   * happens here — at the instant of leaving — rather than on arrival.
   *
   * Returns `true` when it stored something.
   */
  snapshotConversation: (conversationId: string) => boolean;
  /**
   * Apply a saved conversation snapshot to the store. Pure data operation, same
   * contract as `loadTrip`: it does not write `sessionId` and does not open a
   * stream, because a conversation switch is a numbered operation owned by
   * `useChat().switchConversation`.
   */
  restoreChat: (snapshotId: string) => void;
  refreshTripStatuses: () => void;

  // Job state setters
  setJobId: (id: string | null) => void;
  setCurrentStage: (stage: string | null) => void;
  setJobStatus: (status: string | null) => void;
  setActivityPhase: (phase: "idle" | "gathering" | "planning") => void;
  setOutputUrls: (urls: OutputUrls | null) => void;
  setPolicyRouting: (summary: PolicyRoutingSummary | null) => void;

  /**
   * Merge a partial runtime into one conversation, and mirror it onto the
   * top-level in-flight fields **only when that conversation is the active one**.
   * Background conversations keep updating their own entry, so a job running in
   * a conversation the user left neither loses its state nor leaks it onto the
   * conversation being displayed.
   */
  setRuntime: (
    conversationId: string,
    patch: Partial<ConversationRuntime>
  ) => void;
  /**
   * Append a streamed chunk to one conversation's buffer.
   *
   * Must be its own action: computing `previous + chunk` from a snapshot taken
   * when the caller was constructed loses every chunk but the last, because
   * several SSE frames are parsed in the same tick and all read the same base.
   * The concatenation therefore happens inside the store update.
   */
  appendStreamingChunk: (conversationId: string, chunk: string) => void;
  /** Drop a finished runtime entry; never removes the active conversation. */
  retireRuntime: (conversationId: string) => void;
  runtimeFor: (conversationId: string) => ConversationRuntime;
  setConnectedIds: (ids: string[]) => void;
  /**
   * Append to the active conversation's streamed buffer.
   *
   * Kept alongside `appendStreamingChunk` because callers outside
   * `chatEvents` (and the scoped facade) address the active conversation
   * without naming it. It writes through to that conversation's runtime so the
   * two views of the same buffer cannot drift apart.
   */
  appendStreamingContent: (chunk: string) => void;

  // Streaming text setters
  setStreamingContent: (content: string) => void;
  startStreaming: () => void;
  stopStreaming: () => void;

  /**
   * Drop the active conversation's state (and keep its snapshot). This empties
   * `sessionId`, so it belongs to a numbered conversation operation: always call
   * it through `useChat().leaveConversation()`, which also closes the stream.
   */
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

      runtimeByConversation: {},
      connectedConversationIds: [],

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

      // Pure data operation — see the note on `loadTrip` in ChatState: it must
      // not touch `sessionId` or `isConnected`, because moving the active
      // conversation (and its stream) is a numbered operation owned by
      // `useChat().switchConversation`.
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

      setRuntime: (conversationId, patch) =>
        set((state) => {
          if (!conversationId) return {};
          const previous =
            state.runtimeByConversation[conversationId] ??
            emptyConversationRuntime();
          const next: ConversationRuntime = { ...previous, ...patch };
          const runtimeByConversation = {
            ...state.runtimeByConversation,
            [conversationId]: next,
          };
          // Only the active conversation owns the top-level mirror. Without this
          // guard a background job's stage/tokens would overwrite what the user
          // is currently looking at.
          if (state.sessionId !== conversationId) {
            return { runtimeByConversation };
          }
          return {
            runtimeByConversation,
            jobId: next.jobId,
            currentStage: next.currentStage,
            jobStatus: next.jobStatus,
            activityPhase: next.activityPhase,
            streamingContent: next.streamingContent,
            isStreaming: next.isStreaming,
            isLoading: next.isLoading,
            needsClarification: next.needsClarification,
            waitingForConfirmation: next.waitingForConfirmation,
          };
        }),

      appendStreamingChunk: (conversationId, chunk) =>
        set((state) => {
          if (!conversationId) return {};
          const previous =
            state.runtimeByConversation[conversationId] ??
            emptyConversationRuntime();
          const cursor = previous.lastEventId;
          // The buffer was committed at this very cursor and is empty now, so
          // this chunk is a replay of prose the transcript already holds:
          // rebuild the buffer from it instead of appending, or the committed
          // reply would be followed by a duplicated tail.
          const isReplay =
            previous.streamingContent === "" &&
            previous.foldedAtEventId > 0 &&
            cursor <= previous.foldedAtEventId;
          const next: ConversationRuntime = {
            ...previous,
            isStreaming: true,
            streamingContent: isReplay
              ? chunk
              : previous.streamingContent + chunk,
            // Bounded preview so a background conversation can show recent text
            // without retaining the whole stream twice.
            recentTokens: [...previous.recentTokens, chunk].slice(-40),
            // The buffer is live prose again, so it is no longer a replay of
            // what the transcript already holds.
            foldedAtEventId: isReplay ? 0 : previous.foldedAtEventId,
          };
          const runtimeByConversation = {
            ...state.runtimeByConversation,
            [conversationId]: next,
          };
          // The active conversation must land in BOTH places: the runtime entry
          // (so switching away and back preserves the buffered prose) and the
          // top-level mirror (which is what the rendered panel reads while that
          // conversation is the active one).
          if (state.sessionId !== conversationId) {
            return { runtimeByConversation };
          }
          return {
            runtimeByConversation,
            isStreaming: true,
            streamingContent: next.streamingContent,
          };
        }),

      retireRuntime: (conversationId) =>
        set((state) => {
          if (!(conversationId in state.runtimeByConversation)) return {};
          // Never drop the active conversation's entry: the UI reads its
          // isLoading/streaming flags from the mirror and needs them to settle.
          if (state.sessionId === conversationId) {
            return {
              runtimeByConversation: {
                ...state.runtimeByConversation,
                [conversationId]: emptyConversationRuntime(),
              },
            };
          }
          const runtimeByConversation = { ...state.runtimeByConversation };
          delete runtimeByConversation[conversationId];
          return { runtimeByConversation };
        }),

      runtimeFor: (conversationId) =>
        get().runtimeByConversation[conversationId] ??
        emptyConversationRuntime(),

      setConnectedIds: (ids) =>
        set((state) => ({
          connectedConversationIds: ids,
          isConnected: ids.includes(state.sessionId),
        })),

      appendStreamingContent: (chunk) =>
        get().appendStreamingChunk(get().sessionId, chunk),
      setStreamingContent: (content) => set({ streamingContent: content }),
      startStreaming: () => set({ isStreaming: true, streamingContent: "" }),
      stopStreaming: () => set({ isStreaming: false }),

      // 保存当前对话快照
      saveChatSnapshot: (explicitSnapshotId) => {
        const state = get();
        const snapshotId =
          explicitSnapshotId || state.sessionId || `chat-${Date.now()}`;

        // A conversation's live prose is part of what the user saw, but it has
        // not reached `messages` yet (only a `finalize` frame does that, and for
        // a background conversation transcript writes are refused). Fold it in
        // here so a snapshot taken while leaving keeps the reply on screen.
        const pendingProse =
          state.runtimeByConversation[snapshotId]?.streamingContent.trim() ?? "";
        const messages = [...state.messages];
        const last = messages[messages.length - 1];
        const alreadyCommitted =
          !!last &&
          last.role === "assistant" &&
          (last.content.trim() === pendingProse ||
            (!!pendingProse && last.content.includes(pendingProse)));
        if (pendingProse && !alreadyCommitted) {
          messages.push({
            role: "assistant",
            content: pendingProse,
            timestamp: Date.now(),
          });
        }

        // Nothing worth persisting: the snapshot list must not gain empty
        // entries for conversations the user merely visited.
        if (messages.length === 0) return;

        // The title belongs to the conversation, not to the transcript that
        // happens to be loaded: when this snapshots a conversation being left,
        // `chatHistory` still names the outgoing one, so reusing it would
        // relabel the target's sidebar entry with the wrong conversation.
        // Existing snapshots keep their own title; only a first-time save needs
        // a name, and there `chatHistory` really does describe this transcript.
        const existingSnapshot = state.chatSnapshots.find(
          (cs) => cs.id === snapshotId
        );
        const title =
          existingSnapshot?.title || state.chatHistory[0]?.title || "未命名对话";

        const snapshot: ChatSnapshot = {
          id: snapshotId,
          title,
          date: new Date().toISOString().split("T")[0],
          messages,
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
          // The prose now lives in the transcript, so drop it from the live
          // buffer: otherwise switching back would render the same reply twice —
          // once as a committed bubble and once as the streaming bubble, which
          // reads the top-level mirror that `adoptConversation` projects from
          // this same runtime entry.
          const runtime = s.runtimeByConversation[snapshotId];
          if (pendingProse && runtime) {
            return {
              chatSnapshots: newSnapshots,
              runtimeByConversation: {
                ...s.runtimeByConversation,
                [snapshotId]: {
                  ...runtime,
                  streamingContent: "",
                  // Remember where this prose was committed so a reconnect that
                  // re-delivers frames from this cursor rebuilds the buffer
                  // rather than appending a second copy of it.
                  foldedAtEventId: runtime.lastEventId,
                },
              },
            };
          }
          return { chatSnapshots: newSnapshots };
        });
      },

      snapshotConversation: (conversationId) => {
        if (!conversationId) return false;
        // The transcript in `messages` always belongs to the active
        // conversation, and this is only ever called for the conversation being
        // left — so `messages` is the right transcript and the runtime entry
        // under this id is the right prose buffer.
        get().saveChatSnapshot(conversationId);
        return true;
      },

      // Restore a saved snapshot
      restoreChat: (snapshotId) => {        const state = get();
        const snapshot = state.chatSnapshots.find((s) => s.id === snapshotId);
        if (!snapshot) return;

        set({
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
        // Background conversations keep their runtimes: they may still be
        // streaming, and `leaveConversation` is what closes their streams.
        const runtimeByConversation = { ...state.runtimeByConversation };
        if (state.sessionId) delete runtimeByConversation[state.sessionId];
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
          runtimeByConversation,
          isConnected: state.connectedConversationIds.length > 0,
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

/**
 * Expose the store for end-to-end tests only.
 *
 * The multi-conversation contract is about *per-conversation runtime state*
 * (each conversation's cursor, stage and streamed tokens), which the DOM cannot
 * show for a conversation that is not on screen. Exposing the store in
 * development lets the e2e suite assert those internals directly instead of
 * inferring them from the rendered conversation.
 *
 * Guarded by `NODE_ENV` so it never ships in a production build.
 */
if (process.env.NODE_ENV !== "production") {
  (globalThis as unknown as { __chatStore?: typeof useChatStore }).__chatStore =
    useChatStore;
}
