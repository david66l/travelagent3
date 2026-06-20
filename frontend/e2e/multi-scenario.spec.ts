import { test, expect } from "@playwright/test";

interface Scenario {
  name: string;
  query: string;
  expectedDays: number;
}

const SCENARIOS: Scenario[] = [
  { name: "chengdu", query: "成都 4 天，预算 3000 元，喜欢火锅和历史文化", expectedDays: 4 },
  { name: "shanghai", query: "上海 3 天亲子游，喜欢迪士尼和博物馆", expectedDays: 3 },
  { name: "beijing", query: "北京 4 天历史文化游", expectedDays: 4 },
  { name: "guangzhou", query: "广州 2 天美食游，喜欢早茶和粤菜", expectedDays: 2 },
  { name: "hangzhou", query: "杭州 3 天西湖自然风光游", expectedDays: 3 },
  { name: "xian4", query: "西安 4 天历史古迹游", expectedDays: 4 },
  { name: "xian7", query: "西安 7 天深度历史古迹游", expectedDays: 7 },
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
