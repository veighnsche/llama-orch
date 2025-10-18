// Created by: TEAM-107 | 2025-10-18
// k6 spike testing script - sudden traffic spikes
// Test system recovery from sudden load increases

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    // Normal load
    { duration: '1m', target: 100 },
    // Sudden spike to 2000 users
    { duration: '10s', target: 2000 },
    // Hold spike
    { duration: '1m', target: 2000 },
    // Drop back to normal
    { duration: '10s', target: 100 },
    // Hold normal
    { duration: '1m', target: 100 },
    // Another spike
    { duration: '10s', target: 3000 },
    // Hold spike
    { duration: '1m', target: 3000 },
    // Ramp down
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    'errors': ['rate<0.02'],
    'http_req_duration': ['p(95)<1000'],
  },
};

const BASE_URL = __ENV.QUEEN_URL || 'http://localhost:8080';

export default function () {
  const payload = JSON.stringify({
    model: 'tinyllama-q4',
    prompt: 'Spike test prompt',
    max_tokens: 50,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test-token-123',
    },
    timeout: '30s',
  };

  const response = http.post(`${BASE_URL}/v2/tasks`, payload, params);

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
  });

  errorRate.add(success ? 0 : 1);

  sleep(Math.random() * 2);
}

export function setup() {
  console.log('⚡ Starting spike test');
  console.log('Testing system recovery from sudden traffic spikes');
  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log('');
  console.log('🏁 Spike test completed');
}
