/**
 * k6 control-plane load test for asynchronous chat acceptance (M6 §13.3).
 *
 * This deliberately measures the HTTP/PostgreSQL/Redis admission path only.
 * Model execution is handled by workers and has a separate episode latency
 * budget; mixing the two would hide queue admission regressions behind model
 * provider latency.
 *
 * Usage:
 *   k6 run scripts/load/k6_sse.js
 *   k6 run -e BASE_URL=http://localhost:8000 -e VUS=50 scripts/load/k6_sse.js
 */
import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  discardResponseBodies: false,
  scenarios: {
    chat_load: {
      executor: 'constant-vus',
      vus: Number(__ENV.VUS || 20),
      duration: __ENV.DURATION || '2m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
    'http_req_duration{name:chat_acceptance}': ['p(95)<600', 'p(99)<1000'],
  },
};

function requestId() {
  return `${Date.now()}-${__VU}-${__ITER}-${Math.random().toString(16).slice(2)}`;
}

export default function () {
  const fp = requestId();
  const guestRes = http.post(
    `${BASE_URL}/api/v1/auth/guest`,
    JSON.stringify({ device_fingerprint: fp }),
    { headers: { 'Content-Type': 'application/json' }, tags: { name: 'guest_auth' } },
  );
  check(guestRes, { 'guest token': (r) => r.status === 200 });

  const token = guestRes.json('data.access_token');
  if (!token) {
    sleep(1);
    return;
  }

  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
    'X-Device-Fingerprint': fp,
  };

  const convRes = http.post(
    `${BASE_URL}/api/v1/conversations`,
    JSON.stringify({ title: 'k6-load' }),
    { headers, tags: { name: 'create_conversation' } },
  );
  check(convRes, { 'create conversation': (r) => r.status === 200 || r.status === 201 });

  const conversationId = convRes.json('data.id');
  if (!conversationId) {
    sleep(1);
    return;
  }

  const msgRes = http.post(
    `${BASE_URL}/api/v1/chat/message`,
    JSON.stringify({
      conversation_id: conversationId,
      content: '北京三日游推荐',
      stream: true,
    }),
    {
      headers: { ...headers, 'Idempotency-Key': requestId() },
      timeout: '5s',
      tags: { name: 'chat_acceptance' },
    },
  );
  check(msgRes, { 'post message': (r) => r.status === 200 || r.status === 202 });

  sleep(1);
}
