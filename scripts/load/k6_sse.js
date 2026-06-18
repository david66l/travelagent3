/**
 * k6 load test for chat API (M6 §13.3)
 *
 * Usage:
 *   k6 run scripts/load/k6_sse.js
 *   k6 run -e BASE_URL=http://localhost:8000 -e VUS=50 scripts/load/k6_sse.js
 */
import http from 'k6/http';
import { check, sleep } from 'k6';
import { uuidv4 } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  scenarios: {
    chat_load: {
      executor: 'constant-vus',
      vus: Number(__ENV.VUS || 20),
      duration: __ENV.DURATION || '2m',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.1'],
    http_req_duration: ['p(99)<5000'],
  },
};

export default function () {
  const fp = uuidv4();
  const guestRes = http.post(
    `${BASE_URL}/api/v1/auth/guest`,
    JSON.stringify({ device_fingerprint: fp }),
    { headers: { 'Content-Type': 'application/json' } },
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
    { headers },
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
    { headers },
    { timeout: '30s' },
  );
  check(msgRes, { 'post message': (r) => r.status === 200 || r.status === 202 });

  sleep(1);
}
