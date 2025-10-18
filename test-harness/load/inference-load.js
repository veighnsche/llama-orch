// Created by: TEAM-107 | 2025-10-18
// k6 load testing script for inference endpoints
// Target: 1000+ concurrent requests with < 500ms p95 latency

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const inferenceLatency = new Trend('inference_latency');
const tokensGenerated = new Counter('tokens_generated');

// Test configuration
export const options = {
  stages: [
    // Ramp up to 100 users over 1 minute
    { duration: '1m', target: 100 },
    // Ramp up to 500 users over 2 minutes
    { duration: '2m', target: 500 },
    // Ramp up to 1000 users over 2 minutes
    { duration: '2m', target: 1000 },
    // Stay at 1000 users for 10 minutes (sustained load)
    { duration: '10m', target: 1000 },
    // Ramp down to 0 over 1 minute
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    // Error rate must be less than 1%
    'errors': ['rate<0.01'],
    // 95th percentile latency must be less than 500ms
    'http_req_duration': ['p(95)<500'],
    // 99th percentile latency must be less than 1000ms
    'http_req_duration': ['p(99)<1000'],
    // Success rate must be at least 99%
    'http_req_failed': ['rate<0.01'],
  },
};

// Base URL from environment or default
const BASE_URL = __ENV.QUEEN_URL || 'http://localhost:8080';

// Test data - various inference requests
const testPrompts = [
  'What is the capital of France?',
  'Explain quantum computing in simple terms.',
  'Write a haiku about programming.',
  'What are the benefits of renewable energy?',
  'How does photosynthesis work?',
  'Describe the water cycle.',
  'What is machine learning?',
  'Explain the theory of relativity.',
  'What causes seasons on Earth?',
  'How do vaccines work?',
];

export default function () {
  // Select random prompt
  const prompt = testPrompts[Math.floor(Math.random() * testPrompts.length)];
  
  // Prepare inference request
  const payload = JSON.stringify({
    model: 'tinyllama-q4',
    prompt: prompt,
    max_tokens: 100,
    temperature: 0.7,
    stream: false,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer test-token-123',
    },
    timeout: '30s',
  };

  // Send inference request
  const startTime = Date.now();
  const response = http.post(`${BASE_URL}/v2/tasks`, payload, params);
  const duration = Date.now() - startTime;

  // Record metrics
  inferenceLatency.add(duration);

  // Check response
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'has response body': (r) => r.body && r.body.length > 0,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  if (!success) {
    errorRate.add(1);
    console.error(`Request failed: ${response.status} - ${response.body}`);
  } else {
    errorRate.add(0);
    
    // Try to parse response and count tokens
    try {
      const data = JSON.parse(response.body);
      if (data.tokens) {
        tokensGenerated.add(data.tokens.length);
      }
    } catch (e) {
      // Ignore parse errors
    }
  }

  // Random think time between 0.5-2 seconds
  sleep(Math.random() * 1.5 + 0.5);
}

// Setup function - runs once at start
export function setup() {
  console.log('🚀 Starting load test');
  console.log(`Target: ${BASE_URL}`);
  console.log('Configuration:');
  console.log('  - Max concurrent users: 1000');
  console.log('  - Sustained load duration: 10 minutes');
  console.log('  - Target p95 latency: < 500ms');
  console.log('  - Target error rate: < 1%');
  console.log('');
  
  // Health check
  const healthResponse = http.get(`${BASE_URL}/health`);
  if (healthResponse.status !== 200) {
    throw new Error(`Service not healthy: ${healthResponse.status}`);
  }
  
  console.log('✅ Service health check passed');
  return { startTime: new Date().toISOString() };
}

// Teardown function - runs once at end
export function teardown(data) {
  console.log('');
  console.log('🏁 Load test completed');
  console.log(`Started at: ${data.startTime}`);
  console.log(`Ended at: ${new Date().toISOString()}`);
}
