// Created by: TEAM-107 | 2025-10-18
// k6 stress testing script - find breaking point
// Gradually increase load until system fails

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const latency = new Trend('latency');

export const options = {
  stages: [
    // Start with 100 users
    { duration: '2m', target: 100 },
    // Increase to 500
    { duration: '2m', target: 500 },
    // Increase to 1000
    { duration: '2m', target: 1000 },
    // Increase to 2000
    { duration: '2m', target: 2000 },
    // Increase to 3000
    { duration: '2m', target: 3000 },
    // Increase to 5000 (breaking point)
    { duration: '2m', target: 5000 },
    // Hold at 5000
    { duration: '5m', target: 5000 },
    // Ramp down
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    // Allow higher error rates for stress testing
    'errors': ['rate<0.05'],
    // Relaxed latency requirements
    'http_req_duration': ['p(95)<2000'],
  },
};

const BASE_URL = __ENV.QUEEN_URL || 'http://localhost:8080';

export default function () {
  const payload = JSON.stringify({
    model: 'tinyllama-q4',
    prompt: 'Test prompt for stress testing',
    max_tokens: 50,
    temperature: 0.7,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test-token-123',
    },
    timeout: '60s',
  };

  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/v2/tasks`, payload, params);
  const duration = Date.now() - startTime;

  latency.add(duration);

  const success = check(response, {
    'status is 200 or 503': (r) => r.status === 200 || r.status === 503,
  });

  errorRate.add(success ? 0 : 1);

  // Minimal sleep for maximum load
  sleep(0.1);
}

export function setup() {
  console.log('🔥 Starting stress test - finding breaking point');
  console.log(`Target: ${BASE_URL}`);
  console.log('Will ramp up to 5000 concurrent users');
  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log('');
  console.log('🏁 Stress test completed');
  console.log(`Duration: ${data.startTime} to ${new Date().toISOString()}`);
}
