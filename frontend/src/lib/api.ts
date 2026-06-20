const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  if (stored) {
    return { token: stored, fingerprint, role: "guest" };
  }

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
  const res = await fetch(`${API_URL}/api/v1/conversations`, {
    method: "POST",
    headers: authHeaders(token, fingerprint),
    body: JSON.stringify({ title }),
  });
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
  content: string
): Promise<void> {
  const res = await fetch(`${API_URL}/api/v1/chat/message`, {
    method: "POST",
    headers: authHeaders(token, fingerprint),
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
