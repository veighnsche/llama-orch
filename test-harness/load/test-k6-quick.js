// Created by: TEAM-107 | 2025-10-18
// Quick k6 test to validate scripts work
// Runs for 30 seconds with 10 concurrent users

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '10s', target: 10 },  // Ramp up to 10 users
    { duration: '10s', target: 10 },  // Stay at 10 users
    { duration: '10s', target: 0 },   // Ramp down
  ],
  thresholds: {
    'errors': ['rate<0.1'],  // Less than 10% errors
    'http_req_duration': ['p(95)<1000'],  // 95% under 1 second
  },
};

const BASE_URL = __ENV.QUEEN_URL || 'http://localhost:8080';

export default function () {
  const payload = JSON.stringify({
    model: 'tinyllama-q4',
    prompt: 'Test prompt for k6 validation',
    max_tokens: 50,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test-token',
    },
  };

  const response = http.post(`${BASE_URL}/v2/tasks`, payload, params);

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'has response body': (r) => r.body && r.body.length > 0,
  });

  errorRate.add(success ? 0 : 1);

  sleep(0.5);
}

export function setup() {
  console.log('🧪 Quick k6 validation test');
  console.log(`Target: ${BASE_URL}`);
  
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`Service not healthy: ${healthResponse.status}`);
  }
  
  console.log('✅ Service health check passed');
  return { startTime: new Date().toISOString() };
}

export function teardown(data) {
  console.log('✅ Quick test completed');
  console.log(`Started at: ${data.startTime}`);
}
