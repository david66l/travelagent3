// Use same-origin API calls. In production the Go gateway owns /api; when the
// Next.js dev/server port is opened directly, next.config.js proxies /api to the
// internal backend. This keeps browser traffic on the gateway instead of
// bypassing its auth, rate-limit and circuit-breaker middleware.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

export type ItineraryChange = {
  action:
    | "remove"
    | "replace"
    | "add"
    | "reorder"
    | "set_budget"
    | "set_pace"
    | "change_days";
  day_number?: number;
  poi_id?: string;
  new_poi?: Record<string, unknown>;
  order?: string[];
  value?: number | string;
  delta?: number;
};

const TOKEN_KEY = "ta_access_token";
const REFRESH_KEY = "ta_refresh_token";
const FINGERPRINT_KEY = "ta_device_fp";

export function getApiBaseUrl(): string {
  return API_URL;
}

export function getDeviceFingerprint(): string {
  if (typeof window === "undefined") return "server";
  let fp = localStorage.getItem(FINGERPRINT_KEY);
  if (!fp) {
    fp = crypto.randomUUID();
    localStorage.setItem(FINGERPRINT_KEY, fp);
  }
  return fp;
}

export function getStoredAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function getStoredRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_KEY);
}

export function setStoredTokens(accessToken: string, refreshToken: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_KEY, refreshToken);
}

export function clearStoredTokens(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

/**
 * True if a stored JWT is still safely usable. Guest tokens expire after 30 min,
 * but the token persists in localStorage — reusing an expired one made every
 * request 401 and a page refresh could not self-heal. We decode the `exp` claim
 * and require >30s of remaining life so a token cannot expire mid-request.
 */
export function isTokenValid(token: string | null): token is string {
  if (!token) return false;
  try {
    const payload = token.split(".")[1];
    if (!payload) return false;
    const claims = JSON.parse(
      atob(payload.replace(/-/g, "+").replace(/_/g, "/"))
    );
    if (typeof claims.exp !== "number") return true; // no expiry → assume usable
    return claims.exp * 1000 > Date.now() + 30_000;
  } catch {
    return false;
  }
}

export function authHeaders(token: string, fingerprint: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "X-Device-Fingerprint": fingerprint,
    "Content-Type": "application/json",
  };
}

export interface AuthTokenPayload {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  expires_in?: number;
  role: string;
}

export async function ensureGuestSession(): Promise<{
  token: string;
  fingerprint: string;
  role: string;
}> {
  const fingerprint = getDeviceFingerprint();
  const stored = getStoredAccessToken();
  // Reuse only a still-valid token; an expired one is dropped so we mint a fresh
  // guest session instead of looping on 401s.
  if (isTokenValid(stored)) {
    return { token: stored, fingerprint, role: "guest" };
  }
  clearStoredTokens();

  const res = await fetch(`${API_URL}/api/v1/auth/guest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_fingerprint: fingerprint }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.message || `Guest auth failed: ${res.status}`);
  }
  const json = await res.json();
  const data = json.data as AuthTokenPayload | undefined;
  if (!data?.access_token) {
    throw new Error("Guest auth response missing access_token");
  }
  localStorage.setItem(TOKEN_KEY, data.access_token);
  return { token: data.access_token, fingerprint, role: data.role || "guest" };
}

export async function loginUser(credentials: {
  email: string;
  password: string;
}): Promise<AuthTokenPayload> {
  const res = await fetch(`${API_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });

  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(json.message || `Login failed: ${res.status}`);
  }

  const data = json.data as AuthTokenPayload | undefined;
  if (!data?.access_token) {
    throw new Error("Login response missing access_token");
  }

  setStoredTokens(data.access_token, data.refresh_token || "");
  return data;
}

export async function createConversation(
  token: string,
  fingerprint: string,
  title = "新对话"
): Promise<string> {
  let res = await fetch(`${API_URL}/api/v1/conversations`, {
    method: "POST",
    headers: authHeaders(token, fingerprint),
    body: JSON.stringify({ title }),
  });
  // Self-heal a rejected token (expired / secret rotated): drop it, mint a fresh
  // guest session, and retry once before surfacing an error to the user.
  if (res.status === 401) {
    clearStoredTokens();
    const fresh = await ensureGuestSession();
    res = await fetch(`${API_URL}/api/v1/conversations`, {
      method: "POST",
      headers: authHeaders(fresh.token, fresh.fingerprint),
      body: JSON.stringify({ title }),
    });
  }
  if (!res.ok) {
    throw new Error(`Create conversation failed: ${res.status}`);
  }
  const json = await res.json();
  const id = json.data?.id as string;
  if (!id) {
    throw new Error("Create conversation response missing id");
  }
  return id;
}

export async function postChatMessage(
  token: string,
  fingerprint: string,
  conversationId: string,
  content: string,
  idempotencyKey: string
): Promise<string> {
  const res = await fetch(`${API_URL}/api/v1/chat/message`, {
    method: "POST",
    headers: {
      ...authHeaders(token, fingerprint),
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      content,
      stream: true,
    }),
  });
  if (!res.ok && res.status !== 202) {
    const text = await res.text();
    throw new Error(`Chat message failed: ${res.status} ${text}`);
  }
  const json = await res.json();
  const jobId = json.data?.job_id as string | undefined;
  if (!jobId) {
    throw new Error("Chat message response missing job_id");
  }
  return jobId;
}

export async function postChatAction(
  token: string,
  fingerprint: string,
  conversationId: string,
  action: "confirm" | "modify" | "reject" | "trip_event",
  payload?: { change?: unknown; external_event?: unknown; approval?: unknown },
  idempotencyKey: string = crypto.randomUUID()
): Promise<string> {
  const res = await fetch(`${API_URL}/api/v1/chat/message`, {
    method: "POST",
    headers: {
      ...authHeaders(token, fingerprint),
      "Idempotency-Key": idempotencyKey,
    },
    body: JSON.stringify({
      conversation_id: conversationId,
      content: "",
      stream: true,
      action,
      change: payload?.change,
      external_event: payload?.external_event,
      approval: payload?.approval,
    }),
  });
  if (!res.ok && res.status !== 202) {
    const text = await res.text();
    throw new Error(`Chat action failed: ${res.status} ${text}`);
  }
  const json = await res.json();
  const jobId = json.data?.job_id as string | undefined;
  if (!jobId) {
    throw new Error("Chat action response missing job_id");
  }
  return jobId;
}

// --- Booking (mock backend, source="mock") -------------------------------

export interface FlightResult {
  flight_no: string;
  departure: string;
  arrival: string;
  duration: string;
  price: number;
  airline: string;
}
export interface HotelResult {
  name: string;
  district: string;
  price_per_night: number;
  rating: number;
  has_breakfast: boolean;
  has_parking: boolean;
  distance_to_center: string;
}
export interface TicketResult {
  poi_name: string;
  ticket_price: number;
  available: boolean;
  need_reservation: boolean;
}
export interface ReserveResult {
  restaurant: string;
  reservation_id: string;
  status: string;
}

async function bookingPost<T>(path: string, body: unknown): Promise<T> {
  const auth = await ensureGuestSession();
  const res = await fetch(`${API_URL}/api/v1/bookings${path}`, {
    method: "POST",
    headers: authHeaders(auth.token, auth.fingerprint),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Booking ${path} failed: ${res.status}`);
  const json = await res.json();
  return json.data as T;
}

export const searchFlights = (origin: string, dest: string, date: string) =>
  bookingPost<{ flights: FlightResult[] }>("/flights/search", { origin, dest, date });

export const searchHotels = (
  city: string,
  checkin: string,
  checkout: string,
  budgetPerNight?: number
) =>
  bookingPost<{ hotels: HotelResult[] }>("/hotels/search", {
    city,
    checkin,
    checkout,
    guests: 1,
    budget_per_night: budgetPerNight,
  });

export const checkTicket = (poiName: string, date: string) =>
  bookingPost<TicketResult>("/attractions/tickets", { poi_name: poiName, date });

export const reserveRestaurant = (
  restaurant: string,
  date: string,
  time: string,
  persons: number
) =>
  bookingPost<ReserveResult>("/restaurants/reserve", { restaurant, date, time, persons });

// --- Account & profile (memory) ------------------------------------------

export interface MeResponse {
  id: string;
  email?: string | null;
  phone?: string | null;
  role: string;
  created_at: string;
}
export interface ProfileResponse {
  user_id: string;
  personal?: Record<string, unknown> | null;
  preferences?: Record<string, unknown> | null;
  frequent_destinations?: unknown[] | null;
  updated_at?: string;
}

async function authedGet<T>(path: string): Promise<T> {
  const auth = await ensureGuestSession();
  const res = await fetch(`${API_URL}/api/v1${path}`, {
    headers: authHeaders(auth.token, auth.fingerprint),
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return (await res.json()).data as T;
}

export const getMe = () => authedGet<MeResponse>("/users/me");
export const getMyProfile = () => authedGet<ProfileResponse | null>("/users/me/profile");

export async function updateMyProfile(body: {
  personal?: Record<string, unknown>;
  preferences?: Record<string, unknown>;
  frequent_destinations?: unknown[];
}): Promise<ProfileResponse> {
  const auth = await ensureGuestSession();
  const res = await fetch(`${API_URL}/api/v1/users/me/profile`, {
    method: "PUT",
    headers: authHeaders(auth.token, auth.fingerprint),
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Update profile failed: ${res.status}`);
  return (await res.json()).data as ProfileResponse;
}

/** Revoke the session server-side and clear the stored token. */
export async function logoutUser(): Promise<void> {
  const token = getStoredAccessToken();
  if (token) {
    try {
      await fetch(`${API_URL}/api/v1/auth/logout`, {
        method: "POST",
        headers: authHeaders(token, getDeviceFingerprint()),
      });
    } catch {
      /* best-effort — clear locally regardless */
    }
  }
  clearStoredTokens();
}

export function buildChatStreamUrl(
  conversationId: string,
  params: { lastEventId?: number; jobId?: string; timeout?: number } = {}
): string {
  const search = new URLSearchParams({ conversation_id: conversationId });
  if (params.lastEventId) {
    search.set("last_event_id", String(params.lastEventId));
  }
  if (params.jobId) {
    search.set("job_id", params.jobId);
  }
  if (params.timeout) {
    search.set("timeout", String(params.timeout));
  }
  return `${API_URL}/api/v1/chat/stream?${search.toString()}`;
}
