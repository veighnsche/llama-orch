# Customer Onboarding — First Customer Success

**Date**: 2025-10-09  
**Status**: Ready to Execute

---

## Onboarding Flow

### Phase 1: Trial Signup (Day 0)

**Trigger:** Customer clicks trial signup link

**Automated:**
1. Create customer account
2. Generate API key
3. Send welcome email
4. Schedule onboarding call

**Welcome Email:**
```
Subject: Welcome to llama-orch! 🎉

Hi [Name],

Welcome to llama-orch! Here's everything you need to get started:

🔑 Your API Key
[API_KEY]

📚 Documentation
https://docs.llama-orch.eu

💬 Support
hello@llama-orch.eu

Quick Start:
1. Install SDK: npm install @llama-orch/sdk
2. Set API key: export LLORCH_API_KEY=[key]
3. Submit first job: [code example]

I've scheduled our onboarding call for [date/time].

Looking forward to helping you get started!

Best,
Vince
```

---

### Phase 2: Onboarding Call (Day 1)

**Duration:** 30 minutes

**Agenda:**

**1. Introduction (5 min)**
- Welcome and thank you
- Confirm their use case
- Set expectations

**2. API Key Setup (5 min)**
```bash
# Verify API key works
export LLORCH_API_KEY=[their_key]

# Test connection
curl -H "Authorization: Bearer $LLORCH_API_KEY" \
  https://api.llama-orch.eu/health

# Should return: {"status":"ok"}
```

**3. First Inference (10 min)**
```typescript
// Install SDK
npm install @llama-orch/sdk

// Create client
import { Client } from '@llama-orch/sdk'

const client = new Client(process.env.LLORCH_API_KEY)

// Submit job
const result = await client.inference({
  model: 'llama-3.1-8b',
  prompt: 'Hello world',
  max_tokens: 100
})

console.log(result.text)
```

**4. Audit Log Demo (5 min)**
```bash
# Show audit log
curl -H "Authorization: Bearer $LLORCH_API_KEY" \
  https://api.llama-orch.eu/audit/events

# Show GDPR export
curl -H "Authorization: Bearer $LLORCH_API_KEY" \
  "https://api.llama-orch.eu/gdpr/export?user_id=[customer_id]"
```

**5. Q&A and Next Steps (5 min)**
- Answer questions
- Schedule check-in (Day 7)
- Provide support contact

**Checklist:**
- [ ] API key verified
- [ ] First inference successful
- [ ] Audit log viewed
- [ ] Questions answered
- [ ] Check-in scheduled

---

### Phase 3: First Week Support (Days 2-7)

**Day 2: Check-in Email**
```
Subject: How's day 2 going?

Hi [Name],

Just checking in after our onboarding call yesterday.

Have you been able to integrate llama-orch into your application?

Any issues or questions?

I'm here to help!

Best,
Vince
```

**Day 4: Mid-week Check**
```
Subject: Quick question

Hi [Name],

How's the integration going?

I wanted to make sure you're not blocked on anything.

Available for a quick call if you need help.

Best,
Vince
```

**Day 7: Week 1 Review**
```
Subject: Week 1 review - How did it go?

Hi [Name],

You've been using llama-orch for a week now!

How's it going? I'd love to hear:
1. What's working well?
2. What's frustrating?
3. What features are missing?

Let's schedule a quick call to discuss.

Best,
Vince
```

---

### Phase 4: Trial to Paid Conversion (Days 8-14)

**Day 8: Value Reminder**
```
Subject: You're halfway through your trial

Hi [Name],

You're halfway through your 14-day trial!

So far you've:
- Processed [X] tokens
- Submitted [Y] jobs
- Generated [Z] audit events

All with full GDPR compliance. 🇪🇺

How's it meeting your needs?

Best,
Vince
```

**Day 11: Conversion Prep**
```
Subject: Trial ending soon - Let's chat

Hi [Name],

Your trial ends in 3 days.

I'd love to schedule a quick call to:
1. Review your experience
2. Answer any questions
3. Discuss next steps

Available this week?

Best,
Vince
```

**Day 13: Conversion Offer**
```
Subject: Special offer for [Company]

Hi [Name],

Your trial ends tomorrow. I hope it's been valuable!

I'd like to offer you a special deal:

Professional Plan
- Regular: €299/mo
- Your price: €199/mo for first 3 months

Includes:
- 2M tokens/month
- Full audit trails
- GDPR endpoints
- Priority support

Ready to continue?

Best,
Vince
```

**Day 14: Final Reminder**
```
Subject: Trial ends today - Don't lose access

Hi [Name],

Your trial ends today at midnight.

To continue using llama-orch:
1. Choose a plan: https://llama-orch.eu/pricing
2. Enter payment details
3. Keep your API key (no changes needed)

Questions? Just reply to this email.

Best,
Vince
```

---

## Support Playbook

### Common Issues

**Issue 1: API Key Not Working**
```
Symptoms:
- 401 Unauthorized errors
- "Invalid API key" messages

Solution:
1. Verify API key format (should start with "llorch_")
2. Check Authorization header: "Bearer [key]"
3. Verify key hasn't expired
4. Regenerate key if needed

Code example:
curl -H "Authorization: Bearer llorch_abc123..." \
  https://api.llama-orch.eu/health
```

**Issue 2: Slow Inference**
```
Symptoms:
- Requests taking > 10 seconds
- Timeouts

Solution:
1. Check worker availability
2. Verify model is loaded
3. Check network latency
4. Consider smaller model or fewer tokens

Debugging:
curl -H "Authorization: Bearer [key]" \
  https://api.llama-orch.eu/workers/status
```

**Issue 3: Audit Log Not Showing**
```
Symptoms:
- Empty audit log
- Missing events

Solution:
1. Verify EU audit mode is enabled
2. Check customer_id in query
3. Verify time range
4. Check permissions

Code example:
curl -H "Authorization: Bearer [key]" \
  "https://api.llama-orch.eu/audit/events?customer_id=[id]&start_time=2025-10-01"
```

**Issue 4: GDPR Export Failing**
```
Symptoms:
- 500 errors on /gdpr/export
- Incomplete data

Solution:
1. Verify customer_id is correct
2. Check data exists for customer
3. Verify permissions
4. Check audit log size (may timeout if huge)

Code example:
curl -H "Authorization: Bearer [key]" \
  "https://api.llama-orch.eu/gdpr/export?user_id=[id]" \
  > export.json
```

---

## Documentation

### Quick Start Guide

```markdown
# Quick Start

## 1. Get Your API Key

Sign up at https://llama-orch.eu/trial

You'll receive an email with your API key.

## 2. Install SDK

### Node.js
npm install @llama-orch/sdk

### Python
pip install llama-orch-sdk

### Rust
cargo add llama-orch-sdk

## 3. Submit Your First Job

### Node.js
```typescript
import { Client } from '@llama-orch/sdk'

const client = new Client(process.env.LLORCH_API_KEY)

const result = await client.inference({
  model: 'llama-3.1-8b',
  prompt: 'Write a haiku about Rust',
  max_tokens: 100
})

console.log(result.text)
```

### Python
```python
from llama_orch import Client

client = Client(os.environ['LLORCH_API_KEY'])

result = client.inference(
    model='llama-3.1-8b',
    prompt='Write a haiku about Rust',
    max_tokens=100
)

print(result.text)
```

### Rust
```rust
use llama_orch::Client;

let client = Client::new(env::var("LLORCH_API_KEY")?);

let result = client.inference(InferenceRequest {
    model: "llama-3.1-8b".to_string(),
    prompt: "Write a haiku about Rust".to_string(),
    max_tokens: 100,
}).await?;

println!("{}", result.text);
```

## 4. Stream Tokens

### Node.js
```typescript
const stream = await client.inferenceStream({
  model: 'llama-3.1-8b',
  prompt: 'Write a story'
})

for await (const token of stream) {
  process.stdout.write(token.text)
}
```

## 5. View Audit Log

```typescript
const events = await client.auditLog({
  start_time: '2025-10-01',
  end_time: '2025-10-09'
})

console.log(events)
```

## 6. GDPR Export

```typescript
const data = await client.gdprExport({
  user_id: 'customer-123'
})

console.log(data)
```

## Support

Questions? Email hello@llama-orch.eu
```

---

### API Reference

```markdown
# API Reference

## Authentication

All requests require an API key in the Authorization header:

```
Authorization: Bearer llorch_abc123...
```

## Endpoints

### POST /v2/tasks

Submit an inference job.

**Request:**
```json
{
  "model": "llama-3.1-8b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "job_id": "job-abc123",
  "status": "queued"
}
```

### GET /v2/tasks/{job_id}

Get job status.

**Response:**
```json
{
  "job_id": "job-abc123",
  "status": "completed",
  "result": {
    "text": "Hello! How can I help you?",
    "tokens": 8
  }
}
```

### GET /audit/events

Query audit log.

**Parameters:**
- `customer_id` (required)
- `start_time` (optional, ISO 8601)
- `end_time` (optional, ISO 8601)
- `event_type` (optional)

**Response:**
```json
{
  "events": [
    {
      "timestamp": "2025-10-09T12:00:00Z",
      "event_type": "task_submitted",
      "customer_id": "customer-123",
      "job_id": "job-abc123",
      "model": "llama-3.1-8b"
    }
  ]
}
```

### GET /gdpr/export

Export all customer data.

**Parameters:**
- `user_id` (required)

**Response:**
```json
{
  "user_id": "customer-123",
  "jobs": [...],
  "audit_events": [...],
  "created_at": "2025-10-09T12:00:00Z"
}
```

### POST /gdpr/delete

Delete customer data.

**Request:**
```json
{
  "user_id": "customer-123",
  "reason": "user_request"
}
```

**Response:**
```json
{
  "user_id": "customer-123",
  "jobs_deleted": 42,
  "status": "deleted"
}
```

## Rate Limits

- 100 requests/minute (Starter)
- 1000 requests/minute (Professional)
- Custom (Enterprise)

## Errors

**401 Unauthorized:**
Invalid or missing API key.

**429 Too Many Requests:**
Rate limit exceeded.

**503 Service Unavailable:**
No workers available for requested model.
```

---

## Success Metrics

### Onboarding Success
- [ ] API key verified within 24 hours
- [ ] First inference within 48 hours
- [ ] Audit log viewed within 72 hours
- [ ] Check-in call completed within 7 days

### Trial Success
- [ ] 10+ inferences during trial
- [ ] Audit log queried at least once
- [ ] GDPR export tested
- [ ] Positive feedback on check-in call

### Conversion Success
- [ ] Trial extended if needed
- [ ] Pricing discussed
- [ ] Contract signed
- [ ] Payment processed
- [ ] Customer happy

---

## Escalation Path

### Level 1: Email Support
**Response time:** < 4 hours  
**Handled by:** Vince  
**Scope:** Common issues, questions

### Level 2: Video Call
**Response time:** < 24 hours  
**Handled by:** Vince  
**Scope:** Complex issues, integration help

### Level 3: Custom Development
**Response time:** < 48 hours  
**Handled by:** Vince  
**Scope:** Missing features, custom requirements

---

## Customer Success Checklist

### Week 1
- [ ] Onboarding call completed
- [ ] API key working
- [ ] First inference successful
- [ ] Documentation reviewed
- [ ] Support contact established

### Week 2
- [ ] Integration progressing
- [ ] Audit log tested
- [ ] GDPR features tested
- [ ] Feedback collected
- [ ] Conversion discussed

### Week 3
- [ ] Trial converted to paid
- [ ] Payment processed
- [ ] Production deployment
- [ ] Monitoring setup
- [ ] Success metrics defined

### Week 4
- [ ] Production stable
- [ ] Customer happy
- [ ] Referral requested
- [ ] Case study discussed
- [ ] Testimonial collected

---

**Version**: 1.0  
**Status**: Ready to Execute  
**Last Updated**: 2025-10-09
