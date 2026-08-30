import { expect, test } from "@playwright/test";


test("新建对话只向新会话发送一次消息", async ({ page }) => {
  const createdConversationIds: string[] = [];
  const postedConversationIds: string[] = [];
  let nextId = 1;

  await page.route("**/api/v1/auth/guest", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: { access_token: "test-token", refresh_token: "test-refresh" },
      }),
    });
  });
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
