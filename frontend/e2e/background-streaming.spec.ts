/**
 * Background conversations stream in parallel.
 *
 * The single-connection design could only follow the conversation on screen: a
 * job accepted for a conversation the user had switched away from lost its
 * stream, its cursor, and all live progress, and its events had nowhere to land.
 * These tests pin the replacement contract:
 *
 *  - a background conversation keeps its own stream while another is displayed;
 *  - its tokens/history update *its* runtime, never the visible conversation;
 *  - leaving a conversation persists it, and switching back shows exactly what
 *    that conversation accumulated — once, not twice.
 */
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const STORAGE_KEY = "travel-agent-chat-storage";
const REPLY = "BG-ONLY-REPLY: 后台会话的回复";
const STAGE = "BG-ONLY-STAGE";

function savedConversation(id: string, title: string) {
  return {
    id,
    title,
    date: "2026-01-01",
    messages: [],
    confirmedInfo: null,
    itinerary: null,
    preferencePanel: null,
    budgetPanel: null,
    pendingSuggestions: [],
  };
}

/** Seed two conversations so the sidebar can switch between them. */
async function seedConversations(page: Page) {
  await page.addInitScript(
    ([key, value]: [string, string]) => {
      window.localStorage.setItem(key, value);
    },
    [
      STORAGE_KEY,
      JSON.stringify({
        state: {
          sessionId: "conversation-a",
          waitingForConfirmation: false,
          pendingApproval: null,
          activeView: "chat",
          currentTrip: null,
          chatSnapshots: [
            savedConversation("conversation-a", "对话 A"),
            savedConversation("conversation-b", "对话 B"),
          ],
          trips: [],
        },
        version: 0,
      }),
    ] as [string, string]
  );
}

async function mockGuestAuth(page: Page) {
  await page.route("**/api/v1/auth/guest", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: { access_token: "test-token", refresh_token: "test-refresh" },
      }),
    });
  });
}

/** The job submit always succeeds and is always B's job. */
async function mockAcceptedSubmit(page: Page) {
  await page.route("**/api/v1/chat/message", async (route) => {
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ data: { job_id: "job-b" } }),
    });
  });
}

/**
 * Serve B's stream with the given frames (only B's stream carries traffic, so
 * any contamination of A is unambiguous). A's stream answers and stays silent.
 */
async function mockStreams(
  page: Page,
  seen: Array<{ conversationId: string | null; jobId: string | null }>,
  bFrames: string[]
) {
  await page.route("**/api/v1/chat/stream?**", async (route) => {
    const url = new URL(route.request().url());
    const conversationId = url.searchParams.get("conversation_id");
    const jobId = url.searchParams.get("job_id");
    seen.push({ conversationId, jobId });

    if (conversationId !== "conversation-b" || !jobId) {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: ": connected\n\n",
      });
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: bFrames.join(""),
    });
  });
}

function backgroundFrame(payload: Record<string, unknown>) {
  return `event: message\ndata: ${JSON.stringify({
    conversation_id: "conversation-b",
    job_id: "job-b",
    ...payload,
  })}\n\n`;
}

interface StoreView {
  sessionId: string;
  globalStreamingContent: string;
  messageContents: string[];
  a: {
    streamingContent: string;
    recentTokens: string[];
    jobId: string | null;
    isStreaming: boolean;
    messages: string[];
  };
  b: {
    streamingContent: string;
    recentTokens: string[];
    jobId: string | null;
    isStreaming: boolean;
    messages: string[];
  };
}

/** Read both conversations straight out of the store. */
function readRuntime(page: Page): Promise<StoreView | null> {
  return page.evaluate(() => {
    const store = (
      window as unknown as {
        __chatStore?: {
          getState: () => {
            sessionId: string;
            streamingContent: string;
            messages: Array<{ role: string; content: string }>;
            runtimeByConversation: Record<
              string,
              {
                streamingContent: string;
                recentTokens: string[];
                jobId: string | null;
                isStreaming: boolean;
              }
            >;
            chatSnapshots: Array<{
              id: string;
              messages: Array<{ role: string; content: string }>;
            }>;
          };
        };
      }
    ).__chatStore;
    if (!store) return null;
    const state = store.getState();
    // A conversation's durable transcript is its snapshot's messages; the
    // top-level `messages` belongs to whoever is on screen.
    const transcriptOf = (id: string) =>
      (
        state.chatSnapshots.find((snapshot) => snapshot.id === id)?.messages ?? []
      ).map((m) => `${m.role}:${m.content}`);
    return {
      sessionId: state.sessionId,
      globalStreamingContent: state.streamingContent,
      messageContents: state.messages.map((m) => `${m.role}:${m.content}`),
      a: {
        ...(state.runtimeByConversation["conversation-a"] ??
          {
            streamingContent: "",
            recentTokens: [],
            jobId: null,
            isStreaming: false,
          }),
        messages: transcriptOf("conversation-a"),
      },
      b: {
        ...(state.runtimeByConversation["conversation-b"] ??
          {
            streamingContent: "",
            recentTokens: [],
            jobId: null,
            isStreaming: false,
          }),
        messages: transcriptOf("conversation-b"),
      },
    };
  });
}

/** Type a message in the conversation on screen and submit it. */
async function sendInVisibleConversation(page: Page, text: string) {
  const input = page.locator('[data-testid="chat-input"]:visible');
  await input.fill(text);
  await page.locator('[data-testid="send-button"]:visible').click();
}

async function switchTo(page: Page, title: string) {
  await page.getByRole("button", { name: title, exact: true }).click();
}

function occurrences(haystack: string, needle: string) {
  return haystack.split(needle).length - 1;
}

/**
 * Count the message-surface elements whose text contains `text`.
 *
 * A duplicate is two *elements* carrying the reply (a committed bubble plus a
 * live streaming bubble), which `innerText` cannot express: it collapses the
 * partly-streamed bubble.
 */
function bubbleCount(page: Page, text: string) {
  return page
    .locator(
      '[data-testid="messages-container"] .glass-message-ai, ' +
        '[data-testid="messages-container"] .glass-message-user'
    )
    .evaluateAll(
      (nodes, needle) =>
        nodes.filter((node) => (node.textContent ?? "").includes(needle)).length,
      text
    );
}

test("后台会话的 token 实时推进，且不污染当前显示的会话", async ({ page }) => {
  const seen: Array<{ conversationId: string | null; jobId: string | null }> = [];

  await mockGuestAuth(page);
  await mockAcceptedSubmit(page);
  // B's stream delivers prose and a stage frame, then closes. A's stays silent.
  await mockStreams(page, seen, [
    backgroundFrame({ type: "stage", stage: "writing", payload: {} }),
    backgroundFrame({ type: "token", chunk: REPLY }),
    backgroundFrame({ type: "stage", stage: STAGE, payload: {} }),
  ]);
  await seedConversations(page);

  await page.goto("/");
  await expect
    .poll(() => seen.at(-1)?.conversationId, { timeout: 10_000 })
    .toBe("conversation-a");

  // Send in B, then leave for A while the job is running.
  await switchTo(page, "对话 B");
  await sendInVisibleConversation(page, "在 B 里发一条消息");
  await switchTo(page, "对话 A");

  // B's job stream must exist and must stay open while A is displayed.
  await expect
    .poll(
      () =>
        seen.filter(
          (request) =>
            request.conversationId === "conversation-b" &&
            request.jobId === "job-b"
        ).length,
      { timeout: 10_000 }
    )
    .toBeGreaterThan(0);

  await expect
    .poll(async () => (await readRuntime(page))?.b?.recentTokens?.join("") ?? "", {
      timeout: 10_000,
    })
    .toContain(REPLY);

  const runtime = await readRuntime(page);
  expect(runtime).not.toBeNull();
  // The user is still on A (the job was accepted for B).
  expect(runtime?.sessionId).toBe("conversation-a");
  // B's job and its streamed prose live on B, and nowhere else.
  expect(runtime?.b?.jobId).toBe("job-b");
  expect(runtime?.b?.isStreaming).toBe(true);
  expect(runtime?.b?.recentTokens?.join("") ?? "").toContain(REPLY);
  // A's runtime never saw B's tokens.
  expect(runtime?.a?.streamingContent ?? "").not.toContain(REPLY);
  // And the live buffer was not mirrored onto the active conversation.
  expect(runtime?.globalStreamingContent ?? "").not.toContain(REPLY);
  // The displayed transcript must not contain B's reply either.
  expect((runtime?.messageContents ?? []).join("|")).not.toContain(REPLY);
  // A's transcript likewise holds nothing from B.
  expect(runtime?.a?.messages?.join("|") ?? "").not.toContain(REPLY);
  // The visible panel shows no trace of B's traffic.
  await expect(
    page.locator('[data-testid="messages-container"]')
  ).not.toContainText(REPLY);

  // Switching back shows B — and B's own transcript, including the message the
  // user sent before leaving: leaving must persist the departing conversation
  // instead of letting the target's `restoreChat` erase it.
  await switchTo(page, "对话 B");
  await expect
    .poll(async () => (await readRuntime(page))?.sessionId, { timeout: 10_000 })
    .toBe("conversation-b");
  await expect(
    page.locator('[data-testid="messages-container"]')
  ).toContainText("在 B 里发一条消息", { timeout: 10_000 });
  // B's own reply is present too — the prose the stream accumulated while the
  // user was on A, not something resurrected from A.
  await expect(
    page.locator('[data-testid="messages-container"]')
  ).toContainText(REPLY, { timeout: 10_000 });
  // …exactly once: the reply was folded into B's transcript when the user left,
  // and B's stream is still open, so a re-rendered live buffer would show it a
  // second time as a streaming bubble.
  await expect.poll(() => bubbleCount(page, REPLY)).toBe(1);
});

test("切回仍在流式输出的会话时，回复只显示一次", async ({ page }) => {
  const seen: Array<{ conversationId: string | null; jobId: string | null }> = [];

  await mockGuestAuth(page);
  await mockAcceptedSubmit(page);
  // B streams a single token and then ends the response without a `done` frame,
  // which is how a timeout or a server restart looks: the client reconnects at
  // the same cursor and the server re-delivers that token. The reply is already
  // folded into B's transcript by then, so a naive client shows it twice.
  await mockStreams(page, seen, [backgroundFrame({ type: "token", chunk: REPLY })]);
  await seedConversations(page);

  await page.goto("/");
  await expect
    .poll(() => seen.at(-1)?.conversationId, { timeout: 10_000 })
    .toBe("conversation-a");

  await switchTo(page, "对话 B");
  await sendInVisibleConversation(page, "在 B 里发一条消息");
  // Wait for the live buffer to hold the reply *before* leaving, so the fold is
  // guaranteed to have something to persist. (Waiting for it afterwards would
  // race the click that clears it.)
  await expect
    .poll(async () => (await readRuntime(page))?.b?.streamingContent ?? "", {
      timeout: 10_000,
    })
    .toContain(REPLY);

  await switchTo(page, "对话 A");
  // Give the ended stream time to reconnect at least once, so the replayed token
  // is absorbed while the user is away from B.
  await page.waitForTimeout(1500);
  await switchTo(page, "对话 B");
  expect(
    seen.filter((request) => request.conversationId === "conversation-b").length
  ).toBeGreaterThan(1);

  await expect(
    page.locator('[data-testid="messages-container"]')
  ).toContainText(REPLY, { timeout: 10_000 });
  // The reply is in B's durable transcript…
  const runtime = await readRuntime(page);
  expect(runtime?.sessionId).toBe("conversation-b");
  expect(runtime?.b?.messages?.join("|") ?? "").toContain(REPLY);
  // …and on screen once, not once as a committed message plus once as a live
  // streaming bubble re-reading the folded buffer.
  await expect.poll(() => bubbleCount(page, REPLY)).toBe(1);
  // Leaving B and returning with its stream still open must not have dropped
  // the job's in-flight state.
  expect(runtime?.b?.isStreaming).toBe(true);
  expect(runtime?.b?.jobId).toBe("job-b");
});
