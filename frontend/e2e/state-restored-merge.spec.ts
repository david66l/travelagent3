/**
 * 回归测试：切回一个「在别处跑完」的会话时，服务端的 recent_messages 必须显示出来。
 *
 * 背景：`restoreChat` 切会话时会用本地快照填满 `messages`，而 state_restored
 * 分支曾经只在「本地一条消息都没有」时才采用服务端的 recent_messages，
 * 于是「用户在别的会话等待期间产生的回复」被丢弃，只能刷新页面才看得到。
 * 修复见 `lib/chatEvents.ts` 的 `mergeRecentMessages`。
 */
import { expect, test, type Page } from "@playwright/test";

const STORAGE_KEY = "travel-agent-chat-storage";

function savedConversation(
  id: string,
  title: string,
  messages: Array<{ role: "user" | "assistant"; content: string; timestamp: number }>
) {
  return {
    id,
    title,
    date: "2026-01-01",
    messages,
    confirmedInfo: null,
    itinerary: null,
    preferencePanel: null,
    budgetPanel: null,
    pendingSuggestions: [],
  };
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

/** Fulfil every stream request with one state_restored frame carrying `messages`. */
async function mockStreamWithRecentMessages(page: Page, messages: unknown[]) {
  await page.route("**/api/v1/chat/stream?**", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "Cache-Control": "no-cache" },
      body: [
        "event: message",
        `data: ${JSON.stringify({
          type: "state_restored",
          phase: "completed",
          profile: {},
          recent_messages: messages,
        })}`,
        "",
        "",
      ].join("\n"),
    });
  });
}

async function seedTwoConversations(page: Page, conversationAMessages: unknown[]) {
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
            savedConversation("conversation-a", "对话 A", conversationAMessages as never),
            savedConversation("conversation-b", "对话 B", []),
          ],
          trips: [],
        },
        version: 0,
      }),
    ] as [string, string]
  );
}

test("切回在别处跑完的会话时，应看到服务端带来的回复", async ({ page }) => {
  const localOnlyMessage = "快照里已有的旧消息";
  const serverReply = "服务器生成的回复：上海两日行程已排好";

  await mockGuestAuth(page);
  await mockStreamWithRecentMessages(page, [
    { role: "user", content: "帮我排上海两天", ts: 1700000000 },
    { role: "assistant", content: serverReply, ts: 1700000001 },
  ]);
  await seedTwoConversations(page, [
    { role: "user", content: localOnlyMessage, timestamp: 1700000000000 },
  ]);

  await page.goto("/");

  // 切走再切回：模拟「用户在别处等待期间，这个会话的回复产生了」。
  await page.getByRole("button", { name: "对话 B", exact: true }).click();
  await page.waitForTimeout(400);
  await page.getByRole("button", { name: "对话 A", exact: true }).click();

  const container = page.locator('[data-testid="messages-container"]');

  // 本地更早的历史必须保留（服务端只给最近 10 条，不能整体替换）。
  await expect(container).toContainText(localOnlyMessage, { timeout: 8000 });

  // 服务端带来的回复必须显示出来 —— 这就是本缺陷。
  await expect(container).toContainText(serverReply, { timeout: 8000 });
});

test("服务端消息与本地重叠时不产生重复", async ({ page }) => {
  const sharedUser = "帮我排上海两天";
  const sharedAnswer = "好的，我先查一下上海的景点";

  await mockGuestAuth(page);
  // 服务端窗口与本地快照完全一致 → 不应追加任何内容。
  await mockStreamWithRecentMessages(page, [
    { role: "user", content: sharedUser, ts: 1700000000 },
    { role: "assistant", content: sharedAnswer, ts: 1700000001 },
  ]);
  await seedTwoConversations(page, [
    { role: "user", content: sharedUser, timestamp: 1700000000000 },
    { role: "assistant", content: sharedAnswer, timestamp: 1700000001000 },
  ]);

  await page.goto("/");
  await page.getByRole("button", { name: "对话 B", exact: true }).click();
  await page.waitForTimeout(400);
  await page.getByRole("button", { name: "对话 A", exact: true }).click();

  const container = page.locator('[data-testid="messages-container"]');
  await expect(container).toContainText(sharedAnswer, { timeout: 8000 });
  await page.waitForTimeout(500);

  // 每条内容只应出现一次。
  const text = await container.innerText();
  const occurrences = text.split(sharedAnswer).length - 1;
  expect(occurrences).toBe(1);
  const userOccurrences = text.split(sharedUser).length - 1;
  expect(userOccurrences).toBe(1);
});
