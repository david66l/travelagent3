const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TOKEN_KEY = "ta_access_token";
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

export function authHeaders(token: string, fingerprint: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "X-Device-Fingerprint": fingerprint,
    "Content-Type": "application/json",
  };
}

export async function ensureGuestSession(): Promise<{
  token: string;
  fingerprint: string;
}> {
  const fingerprint = getDeviceFingerprint();
  const stored = getStoredAccessToken();
  if (stored) {
    return { token: stored, fingerprint };
  }

  const res = await fetch(`${API_URL}/api/v1/auth/guest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_fingerprint: fingerprint }),
  });
  if (!res.ok) {
    throw new Error(`Guest auth failed: ${res.status}`);
  }
  const json = await res.json();
  const token = json.data?.access_token as string;
  if (!token) {
    throw new Error("Guest auth response missing access_token");
  }
  localStorage.setItem(TOKEN_KEY, token);
  return { token, fingerprint };
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
