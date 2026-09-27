import { expect, test } from "@playwright/test";

const STORAGE_KEY = "travel-agent-chat-storage";

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

async function mockGuestAuth(page: import("@playwright/test").Page) {
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


test("新建对话只向新会话发送一次消息", async ({ page }) => {
  const createdConversationIds: string[] = [];
  const postedConversationIds: string[] = [];
  let nextId = 1;

  await mockGuestAuth(page);
  await page.route("**/api/v1/conversations", async (route) => {
    const id = `conversation-${nextId++}`;
    createdConversationIds.push(id);
    // Make creation slow enough to expose a click/create/send race.
    await new Promise((resolve) => setTimeout(resolve, 150));
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ data: { id } }),
    });
  });
  await page.route("**/api/v1/chat/stream?**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": connected\n\n",
    });
  });
  await page.route("**/api/v1/chat/message", async (route) => {
    const payload = route.request().postDataJSON() as { conversation_id: string };
    postedConversationIds.push(payload.conversation_id);
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ data: { accepted: true } }),
    });
  });

  await page.goto("/");
  await expect.poll(() => createdConversationIds.length).toBe(1);

  await page.getByRole("button", { name: "新建对话", exact: true }).click();
  await expect(page.getByRole("button", { name: "正在新建…" })).toBeDisabled();
  await expect.poll(() => createdConversationIds.length).toBe(2);
  await expect(page.getByRole("button", { name: "新建对话", exact: true })).toBeEnabled();

  const input = page.locator('[data-testid="chat-input"]:visible');
  await input.fill("南京三日历史文化游");
  await page.locator('[data-testid="send-button"]:visible').click();

  await expect.poll(() => postedConversationIds).toEqual(["conversation-2"]);
});

test("切换对话后，在途的发送不会把流拉回旧会话", async ({ page }) => {
  const streamRequests: Array<{
    conversationId: string | null;
    jobId: string | null;
  }> = [];
  const postedConversationIds: string[] = [];
  let releaseAcceptedMessage: () => void = () => {};
  const acceptedMessageGate = new Promise<void>((resolve) => {
    releaseAcceptedMessage = resolve;
  });

  // Two saved conversations, with the first one already active.
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

  await mockGuestAuth(page);
  await page.route("**/api/v1/chat/stream?**", async (route) => {
    const url = new URL(route.request().url());
    streamRequests.push({
      conversationId: url.searchParams.get("conversation_id"),
      jobId: url.searchParams.get("job_id"),
    });
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": connected\n\n",
    });
  });
  await page.route("**/api/v1/chat/message", async (route) => {
    const payload = route.request().postDataJSON() as {
      conversation_id: string;
    };
    postedConversationIds.push(payload.conversation_id);
    // Hold the POST open so the user can leave the conversation mid-flight.
    await acceptedMessageGate;
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ data: { job_id: "job-for-b" } }),
    });
  });

  await page.goto("/");
  // Startup adopts the stored conversation and opens exactly its stream.
  await expect
    .poll(() => streamRequests.at(-1)?.conversationId)
    .toBe("conversation-a");

  await page.getByRole("button", { name: "对话 B", exact: true }).click();
  await expect
    .poll(() => streamRequests.at(-1)?.conversationId)
    .toBe("conversation-b");

  const input = page.locator('[data-testid="chat-input"]:visible');
  await input.fill("在 B 会话里发一条消息");
  await page.locator('[data-testid="send-button"]:visible').click();
  await expect.poll(() => postedConversationIds).toEqual(["conversation-b"]);

  // Leave for A while the message POST is still in flight.
  await page.getByRole("button", { name: "对话 A", exact: true }).click();
  await expect
    .poll(() => streamRequests.at(-1)?.conversationId)
    .toBe("conversation-a");

  releaseAcceptedMessage();
  await page.waitForTimeout(500);

  // The accepted message belongs to B, and B is now a *background* conversation.
  // Requirements conflict here, so the contract is explicit:
  //  - B's job must keep streaming, even though the user is looking at A
  //    (background progress is the whole point of per-conversation streams), and
  //  - that stream must not hijack the one the user is watching: A's stream is
  //    still open and the displayed conversation is still A.
  const jobStream = streamRequests.find((request) => request.jobId !== null);
  expect(jobStream?.conversationId).toBe("conversation-b");
  expect(jobStream?.jobId).toBe("job-for-b");

  // Both conversations hold a stream at the same time — this is what the single
  // `useSSE` connection used to make impossible.
  const openConversations = new Set(
    streamRequests.map((request) => request.conversationId)
  );
  expect(openConversations.has("conversation-a")).toBe(true);
  expect(openConversations.has("conversation-b")).toBe(true);

  // The stream opened for B's job is the *last* request, and A's stream was not
  // re-opened to follow B's job (which would have pulled the view to B).
  expect(streamRequests.at(-1)?.conversationId).toBe("conversation-b");
  expect(
    streamRequests.filter(
      (request) => request.conversationId === "conversation-a" && request.jobId
    )
  ).toEqual([]);

  // The message still went to B only.
  expect(postedConversationIds).toEqual(["conversation-b"]);
  // And the user is still looking at A.
  await expect(
    page.getByRole("button", { name: "对话 A", exact: true })
  ).toBeVisible();
});
