import { test, expect } from "@playwright/test";

interface Scenario {
  name: string;
  query: string;
  expectedDays: number;
}

const SCENARIOS: Scenario[] = [
  {
    name: "shanghai",
    query: "我2026年7月1日从北京出发，2人去上海玩3天，预算每人8000元，亲子游，喜欢迪士尼和博物馆",
    expectedDays: 3,
  },
];

test.describe("多场景前端端到端", () => {
  for (const s of SCENARIOS) {
    test(`${s.name}: ${s.query}`, async ({ page }) => {
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
      await input.fill(s.query);

      const sendButton = page.locator('[data-testid="send-button"]:visible');
      await expect(sendButton).toBeEnabled();
      await sendButton.click();

      // 3. 用户消息出现
      const messages = page.locator('[data-testid="messages-container"]:visible');
      await expect(messages.getByText(s.query)).toBeVisible();

      // 4. 等待 AI 生成完整行程回复
      await expect(messages.getByText(/第1天/)).toBeVisible({ timeout: 120_000 });

      // 5. 右侧预览面板出现对应天数
      for (let day = 1; day <= s.expectedDays; day++) {
        await expect(page.getByRole("button", { name: `DAY${day}` })).toBeVisible({
          timeout: 10_000,
        });
      }

      // 6. 截图留档
      await page.screenshot({
        path: `test-results/multi-scenario-${s.name}.png`,
        fullPage: false,
      });
    });
  }
});
