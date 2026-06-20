import { test, expect } from "@playwright/test";

test.describe("主流程", () => {
  test("用户可以输入需求并发送消息", async ({ page }) => {
    await page.goto("/");

    // 1. 初始欢迎态（heading 在桌面端/移动端同时存在，取第一个）
    await expect(
      page.getByRole("heading", { name: "几分钟内生成你的首个行程" }).first()
    ).toBeVisible();

    // 2. 输入旅行需求（使用可见输入框，兼容桌面/移动端布局）
    const input = page.locator('[data-testid="chat-input"]:visible');
    await expect(input).toBeVisible();
    await input.fill("北京 3 天，预算 5000 元");

    // 3. 等待发送按钮变为可用并点击
    const sendButton = page.locator('[data-testid="send-button"]:visible');
    await expect(sendButton).toBeEnabled();
    await sendButton.click();

    // 4. 用户消息出现在对话列表
    const messages = page.locator('[data-testid="messages-container"]:visible');
    await expect(messages.getByText("北京 3 天，预算 5000 元")).toBeVisible();
  });
});
