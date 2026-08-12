import { test, expect } from "@playwright/test";

test.describe("真实场景端到端", () => {
  test("完整需求可以生成、确认并保存为当前行程", async ({ page }) => {
    test.setTimeout(180_000);

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
    const start = new Date();
    start.setDate(start.getDate() + 30);
    const end = new Date(start);
    end.setDate(end.getDate() + 2);
    const formatDate = (date: Date) => date.toISOString().slice(0, 10);
    const query = `我和朋友两个人从上海出发，${formatDate(start)}到${formatDate(end)}去北京玩3天，不带老人和小孩，总预算5000元，喜欢历史文化和美食`;
    await input.fill(query);

    const sendButton = page.locator('[data-testid="send-button"]:visible');
    await expect(sendButton).toBeEnabled();
    await sendButton.click();

    // 3. 用户消息出现
    const messages = page.locator('[data-testid="messages-container"]:visible');
    await expect(messages.getByText(query)).toBeVisible();

    // 4. 等待草案和人工确认门出现
    await expect(page.getByTestId("confirm-itinerary")).toBeVisible({
      timeout: 120_000,
    });

    // 5. 右侧预览面板出现 3 天概览
    await expect(page.getByRole("button", { name: "DAY1" })).toBeVisible({
      timeout: 10_000,
    });
    await expect(page.getByRole("button", { name: "DAY2" })).toBeVisible();
    await expect(page.getByRole("button", { name: "DAY3" })).toBeVisible();

    // 6. 后端确认成功后，行程才进入“当前/历史行程”
    await page.getByTestId("confirm-itinerary").click();
    await expect(page.getByText("确认行程")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByText("北京3日游")).toBeVisible({ timeout: 10_000 });
    const currentTrip = page.getByRole("button", { name: "当前行程" });
    await expect(currentTrip).toBeEnabled();
    await currentTrip.click();
    await expect(page.getByRole("button", { name: "导出" })).toBeVisible();
    await expect(page.getByText("¥5,000").first()).toBeVisible();
    await expect(page.getByText("¥0", { exact: true })).toHaveCount(0);

    // 7. 截图留档
    await page.screenshot({
      path: "test-results/real-scenario.png",
      fullPage: false,
    });
  });
});
