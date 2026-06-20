import { test, expect } from "@playwright/test";

test.describe("真实场景端到端", () => {
  test("成都 4 天行程可在对话与预览面板中生成并展示", async ({ page }) => {
    test.setTimeout(120_000);

    page.on("console", (msg) => {
      if (msg.type() === "error" || msg.type() === "warning") {
        console.log(`[${msg.type()}]`, msg.text());
      }
    });

    await page.goto("/");

    // 1. 初始欢迎态
    await expect(
      page.getByRole("heading", { name: "几分钟内生成你的首个行程" }).first()
    ).toBeVisible();

    // 2. 输入需求并发送
    const input = page.locator('[data-testid="chat-input"]:visible');
    await expect(input).toBeVisible();
    const query = "成都 4 天，预算 3000 元，喜欢火锅和历史文化";
    await input.fill(query);

    const sendButton = page.locator('[data-testid="send-button"]:visible');
    await expect(sendButton).toBeEnabled();
    await sendButton.click();

    // 3. 用户消息出现
    const messages = page.locator('[data-testid="messages-container"]:visible');
    await expect(messages.getByText(query)).toBeVisible();

    // 4. 等待 AI 生成完整行程回复（聊天消息里会出现 "第1天"）
    await expect(messages.getByText(/第1天/)).toBeVisible({ timeout: 90_000 });

    // 5. 右侧预览面板出现 4 天概览
    await expect(page.getByRole("button", { name: "DAY1" })).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByRole("button", { name: "DAY2" })).toBeVisible();
    await expect(page.getByRole("button", { name: "DAY3" })).toBeVisible();
    await expect(page.getByRole("button", { name: "DAY4" })).toBeVisible();

    // 6. 截图留档
    await page.screenshot({
      path: "test-results/real-scenario.png",
      fullPage: false,
    });
  });
});
