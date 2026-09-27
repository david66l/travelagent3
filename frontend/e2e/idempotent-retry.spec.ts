import { expect, test, type Page, type Route } from "@playwright/test";

interface StreamRequest {
  conversationId: string | null;
  jobId: string | null;
}

interface SubmitRecord {
  key: string | null;
  content: string;
}

interface Harness {
  submits: SubmitRecord[];
  streams: StreamRequest[];
}

/**
 * Installs the minimum backend the send path needs and records every message
 * submission (with its Idempotency-Key) and every stream subscription.
 *
 * `respondToSubmit` decides what the Nth submission does, which is how the
 * "request reached the server but the response never came back" case is
 * reproduced: the fetch fails after the request was already sent.
 */
async function installHarness(
  page: Page,
  options: {
    respondToSubmit: (attempt: number, route: Route) => Promise<void>;
    /**
     * Leave the SSE response pending instead of completing it. The client then
     * believes it is connected, so nothing but a deliberate re-subscribe can
     * produce another stream request.
     */
    holdStreamOpen?: boolean;
  }
): Promise<Harness> {
  const harness: Harness = { submits: [], streams: [] };

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
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ data: { id: "conversation-1" } }),
    });
  });
  await page.route("**/api/v1/chat/stream?**", async (route) => {
    const url = new URL(route.request().url());
    harness.streams.push({
      conversationId: url.searchParams.get("conversation_id"),
      jobId: url.searchParams.get("job_id"),
    });
    if (options.holdStreamOpen) return;
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: ": connected\n\n",
    });
  });
  await page.route("**/api/v1/chat/message", async (route) => {
    const body = route.request().postDataJSON() as { content?: string };
    harness.submits.push({
      key: route.request().headers()["idempotency-key"] ?? null,
      content: body.content ?? "",
    });
    await options.respondToSubmit(harness.submits.length, route);
  });

  return harness;
}

function fulfillAccepted(route: Route, jobId: string) {
  return route.fulfill({
    status: 202,
    contentType: "application/json",
    body: JSON.stringify({ data: { job_id: jobId } }),
  });
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

async function sendDraft(page: Page, text: string) {
  const input = page.locator('[data-testid="chat-input"]:visible');
  await input.fill(text);
  await page.locator('[data-testid="send-button"]:visible').click();
}

test("重发同样内容复用同一个幂等键，并接回可能已在跑的任务", async ({ page }) => {
  const harness = await installHarness(page, {
    holdStreamOpen: true,
    respondToSubmit: async (attempt, route) => {
      if (attempt === 1) {
        // The request went out, the response never came back.
        await route.abort("connectionfailed");
        return;
      }
      await fulfillAccepted(route, "job-1");
    },
  });

  await page.goto("/");
  await expect.poll(() => harness.streams.length).toBe(1);

  await sendDraft(page, "南京三日历史文化游");
  await expect.poll(() => harness.submits.length).toBe(1);

  // Retrying with identical content is the same logical submit.
  await sendDraft(page, "南京三日历史文化游");
  await expect.poll(() => harness.submits.length).toBe(2);

  expect.soft(harness.submits[0].key).toMatch(UUID_RE);
  expect
    .soft(harness.submits[1].key, "retry reuses the key so the job is replayed")
    .toBe(harness.submits[0].key);
  // Between the failure and the retry, a failed submit must ask the stream for
  // whatever is still running rather than assume the request never arrived. The
  // first stream is still open, so this can only be a deliberate re-subscribe,
  // and it must be the *second* subscription — before the retry's job stream.
  expect
    .soft(harness.streams[1]?.jobId, "the resume lets the server pick the job")
    .toBeNull();
  await expect
    .poll(() => harness.streams.at(-1)?.jobId, {
      message: "the accepted job is the one that gets followed",
    })
    .toBe("job-1");
});

test("发送失败后草稿回到输入框，重试不必重打一遍", async ({ page }) => {
  const harness = await installHarness(page, {
    respondToSubmit: async (_attempt, route) => {
      await route.abort("connectionfailed");
    },
  });

  await page.goto("/");
  await sendDraft(page, "南京三日历史文化游");
  await expect.poll(() => harness.submits.length).toBe(1);

  await expect(page.locator('[data-testid="chat-input"]:visible')).toHaveValue(
    "南京三日历史文化游"
  );
});

test("内容变了就是新的一次提交，换新的幂等键", async ({ page }) => {
  const harness = await installHarness(page, {
    respondToSubmit: async (_attempt, route) => {
      await route.abort("connectionfailed");
    },
  });

  await page.goto("/");

  await sendDraft(page, "南京三日历史文化游");
  await expect.poll(() => harness.submits.length).toBe(1);

  // Editing the draft before retrying changes the request body; reusing the key
  // there is a 409 (IDEMPOTENCY_KEY_CONFLICT) on the backend.
  await sendDraft(page, "北京五日历史文化游");
  await expect.poll(() => harness.submits.length).toBe(2);

  expect(harness.submits[1].key).toMatch(UUID_RE);
  expect(harness.submits[1].key).not.toBe(harness.submits[0].key);
});
