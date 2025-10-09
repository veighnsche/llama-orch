# Week 3: Marketing — Days 15-21

**Goal**: Landing page + first customer outreach  
**Deliverable**: 10 qualified leads

---

## Day 15 (Monday): Landing Page Structure

### Morning Session (09:00-13:00)

**Task 1: Create landing page project (1 hour)**
```bash
cd /home/vince/Projects/llama-orch/frontend
mkdir landing-page
cd landing-page

npm create vite@latest . -- --template vue-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Task 2: Hero section (2 hours)**
```vue
<!-- src/components/Hero.vue -->
<template>
  <section class="bg-gradient-to-br from-blue-600 to-blue-800 text-white">
    <div class="max-w-7xl mx-auto px-4 py-20">
      <div class="text-center">
        <h1 class="text-5xl font-bold mb-6">
          EU-Native LLM Inference<br/>
          with Full Audit Trails
        </h1>
        <p class="text-xl mb-8 text-blue-100">
          Run GPT-4 class models with GDPR compliance built-in.<br/>
          EU-only data residency. Immutable audit logs. No US servers.
        </p>
        <div class="flex gap-4 justify-center">
          <a 
            href="#pricing" 
            class="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold hover:bg-blue-50"
          >
            Start Free Trial
          </a>
          <a 
            href="#features" 
            class="border-2 border-white px-8 py-4 rounded-lg font-semibold hover:bg-blue-700"
          >
            View Features
          </a>
        </div>
      </div>
    </div>
  </section>
</template>
```

**Task 3: Features section (1 hour)**
```vue
<!-- src/components/Features.vue -->
<template>
  <section id="features" class="py-20 bg-white">
    <div class="max-w-7xl mx-auto px-4">
      <h2 class="text-4xl font-bold text-center mb-12">
        Why Choose llama-orch?
      </h2>
      
      <div class="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
        <div class="text-center">
          <div class="text-4xl mb-4">🇪🇺</div>
          <h3 class="text-xl font-semibold mb-2">GDPR Compliant</h3>
          <p class="text-gray-600">
            Built for EU regulations from day one. Full audit trails for every inference.
          </p>
        </div>
        
        <div class="text-center">
          <div class="text-4xl mb-4">🔒</div>
          <h3 class="text-xl font-semibold mb-2">EU-Only Data</h3>
          <p class="text-gray-600">
            Your data never leaves the EU. Guaranteed data sovereignty.
          </p>
        </div>
        
        <div class="text-center">
          <div class="text-4xl mb-4">📊</div>
          <h3 class="text-xl font-semibold mb-2">Immutable Logs</h3>
          <p class="text-gray-600">
            Every request logged. Compliance-ready audit trails included.
          </p>
        </div>
        
        <div class="text-center">
          <div class="text-4xl mb-4">💰</div>
          <h3 class="text-xl font-semibold mb-2">Transparent Pricing</h3>
          <p class="text-gray-600">
            Simple usage-based pricing. No hidden fees. Cancel anytime.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
```

### Afternoon Session (14:00-18:00)

**Task 4: Use cases section (2 hours)**
```vue
<!-- src/components/UseCases.vue -->
<template>
  <section class="py-20 bg-gray-50">
    <div class="max-w-7xl mx-auto px-4">
      <h2 class="text-4xl font-bold text-center mb-12">
        Built for EU Businesses
      </h2>
      
      <div class="grid md:grid-cols-3 gap-8">
        <div class="bg-white p-8 rounded-lg shadow">
          <h3 class="text-2xl font-semibold mb-4">🏥 Healthcare</h3>
          <p class="text-gray-600 mb-4">
            Process patient data with full GDPR compliance. Immutable audit trails for regulatory reporting.
          </p>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>✓ Patient data stays in EU</li>
            <li>✓ Full audit trail</li>
            <li>✓ GDPR-compliant by default</li>
          </ul>
        </div>
        
        <div class="bg-white p-8 rounded-lg shadow">
          <h3 class="text-2xl font-semibold mb-4">💼 Finance</h3>
          <p class="text-gray-600 mb-4">
            Analyze financial data with confidence. Every inference logged for compliance.
          </p>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>✓ Transaction data protected</li>
            <li>✓ Regulatory reporting ready</li>
            <li>✓ Immutable audit logs</li>
          </ul>
        </div>
        
        <div class="bg-white p-8 rounded-lg shadow">
          <h3 class="text-2xl font-semibold mb-4">🚀 Startups</h3>
          <p class="text-gray-600 mb-4">
            Build EU-compliant AI products from day one. No compliance headaches.
          </p>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>✓ Compliance without complexity</li>
            <li>✓ Pay-as-you-grow pricing</li>
            <li>✓ Scale with confidence</li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>
```

**Task 5: Pricing section (2 hours)**
```vue
<!-- src/components/Pricing.vue -->
<template>
  <section id="pricing" class="py-20 bg-white">
    <div class="max-w-7xl mx-auto px-4">
      <h2 class="text-4xl font-bold text-center mb-4">
        Simple, Transparent Pricing
      </h2>
      <p class="text-center text-gray-600 mb-12">
        Start free. Scale as you grow. Cancel anytime.
      </p>
      
      <div class="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        <div class="border-2 rounded-lg p-8">
          <h3 class="text-2xl font-semibold mb-2">Starter</h3>
          <div class="text-4xl font-bold mb-4">
            €99<span class="text-lg text-gray-600">/mo</span>
          </div>
          <ul class="space-y-3 mb-8">
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>500K tokens included</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>EU-only inference</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Basic audit logs</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Email support</span>
            </li>
          </ul>
          <button class="w-full bg-gray-200 text-gray-800 px-6 py-3 rounded-lg font-semibold hover:bg-gray-300">
            Start Free Trial
          </button>
        </div>
        
        <div class="border-4 border-blue-600 rounded-lg p-8 relative">
          <div class="absolute -top-4 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-4 py-1 rounded-full text-sm font-semibold">
            Most Popular
          </div>
          <h3 class="text-2xl font-semibold mb-2">Professional</h3>
          <div class="text-4xl font-bold mb-4">
            €299<span class="text-lg text-gray-600">/mo</span>
          </div>
          <ul class="space-y-3 mb-8">
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>2M tokens included</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Full audit trails</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>GDPR endpoints</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Priority support</span>
            </li>
          </ul>
          <button class="w-full bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700">
            Start Free Trial
          </button>
        </div>
        
        <div class="border-2 rounded-lg p-8">
          <h3 class="text-2xl font-semibold mb-2">Enterprise</h3>
          <div class="text-4xl font-bold mb-4">
            Custom
          </div>
          <ul class="space-y-3 mb-8">
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Unlimited tokens</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Dedicated instances</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Custom SLAs</span>
            </li>
            <li class="flex items-start">
              <span class="text-green-600 mr-2">✓</span>
              <span>Dedicated support</span>
            </li>
          </ul>
          <button class="w-full bg-gray-200 text-gray-800 px-6 py-3 rounded-lg font-semibold hover:bg-gray-300">
            Contact Sales
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
```

**Day 15 Deliverable**: Landing page structure complete

---

## Day 16 (Tuesday): Landing Page Content

### Full Day Session (09:00-18:00)

**Task 1: Write compelling copy (3 hours)**
```vue
<!-- Update Hero.vue -->
<template>
  <section class="bg-gradient-to-br from-blue-600 to-blue-800 text-white">
    <div class="max-w-7xl mx-auto px-4 py-20">
      <div class="max-w-4xl mx-auto text-center">
        <div class="inline-block bg-blue-500 px-4 py-2 rounded-full text-sm font-semibold mb-6">
          🇪🇺 Built for EU Compliance
        </div>
        
        <h1 class="text-6xl font-bold mb-6 leading-tight">
          Stop Worrying About<br/>
          GDPR Compliance
        </h1>
        
        <p class="text-2xl mb-8 text-blue-100 leading-relaxed">
          Run powerful LLM inference with built-in GDPR compliance.<br/>
          EU-only data residency. Full audit trails. Zero compliance headaches.
        </p>
        
        <div class="flex gap-4 justify-center mb-12">
          <a 
            href="#pricing" 
            class="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold text-lg hover:bg-blue-50 transition"
          >
            Start Free Trial →
          </a>
          <a 
            href="#demo" 
            class="border-2 border-white px-8 py-4 rounded-lg font-semibold text-lg hover:bg-blue-700 transition"
          >
            Watch Demo
          </a>
        </div>
        
        <div class="flex gap-8 justify-center text-sm text-blue-200">
          <div>✓ 14-day free trial</div>
          <div>✓ No credit card required</div>
          <div>✓ Cancel anytime</div>
        </div>
      </div>
    </div>
  </section>
</template>
```

**Task 2: Add social proof (2 hours)**
```vue
<!-- src/components/SocialProof.vue -->
<template>
  <section class="py-12 bg-gray-50 border-y">
    <div class="max-w-7xl mx-auto px-4">
      <p class="text-center text-gray-600 mb-8">
        Trusted by EU businesses for GDPR-compliant AI
      </p>
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-8 items-center opacity-60">
        <!-- Placeholder logos -->
        <div class="text-center font-semibold text-gray-400">Company A</div>
        <div class="text-center font-semibold text-gray-400">Company B</div>
        <div class="text-center font-semibold text-gray-400">Company C</div>
        <div class="text-center font-semibold text-gray-400">Company D</div>
      </div>
    </div>
  </section>
</template>
```

**Task 3: Add FAQ section (2 hours)**
```vue
<!-- src/components/FAQ.vue -->
<template>
  <section class="py-20 bg-white">
    <div class="max-w-3xl mx-auto px-4">
      <h2 class="text-4xl font-bold text-center mb-12">
        Frequently Asked Questions
      </h2>
      
      <div class="space-y-6">
        <details class="group">
          <summary class="flex justify-between items-center cursor-pointer p-4 bg-gray-50 rounded-lg font-semibold">
            <span>Is my data really EU-only?</span>
            <span class="transform group-open:rotate-180 transition">▼</span>
          </summary>
          <div class="p-4 text-gray-600">
            Yes. All inference happens on EU-based GPUs. Your data never leaves the European Union. We verify worker locations and enforce EU-only routing when audit mode is enabled.
          </div>
        </details>
        
        <details class="group">
          <summary class="flex justify-between items-center cursor-pointer p-4 bg-gray-50 rounded-lg font-semibold">
            <span>What about GDPR compliance?</span>
            <span class="transform group-open:rotate-180 transition">▼</span>
          </summary>
          <div class="p-4 text-gray-600">
            We provide full GDPR compliance out of the box: immutable audit logs, data export endpoints, data deletion on request, and consent tracking. Everything you need for regulatory compliance.
          </div>
        </details>
        
        <details class="group">
          <summary class="flex justify-between items-center cursor-pointer p-4 bg-gray-50 rounded-lg font-semibold">
            <span>How does pricing work?</span>
            <span class="transform group-open:rotate-180 transition">▼</span>
          </summary>
          <div class="p-4 text-gray-600">
            Simple usage-based pricing. Pay for what you use. Each plan includes a token allowance, and you can add more as needed. No hidden fees, no surprises.
          </div>
        </details>
        
        <details class="group">
          <summary class="flex justify-between items-center cursor-pointer p-4 bg-gray-50 rounded-lg font-semibold">
            <span>Can I try before I buy?</span>
            <span class="transform group-open:rotate-180 transition">▼</span>
          </summary>
          <div class="p-4 text-gray-600">
            Absolutely! We offer a 14-day free trial with no credit card required. Test our service with real workloads before committing.
          </div>
        </details>
        
        <details class="group">
          <summary class="flex justify-between items-center cursor-pointer p-4 bg-gray-50 rounded-lg font-semibold">
            <span>What models do you support?</span>
            <span class="transform group-open:rotate-180 transition">▼</span>
          </summary>
          <div class="p-4 text-gray-600">
            We support all major open-source models: Llama 3, Qwen, Phi-3, Mistral, and more. Need a specific model? Contact us and we'll add it.
          </div>
        </details>
      </div>
    </div>
  </section>
</template>
```

**Task 4: Add CTA section (1 hour)**
```vue
<!-- src/components/CTA.vue -->
<template>
  <section class="py-20 bg-blue-600 text-white">
    <div class="max-w-4xl mx-auto px-4 text-center">
      <h2 class="text-4xl font-bold mb-6">
        Ready to Get Started?
      </h2>
      <p class="text-xl mb-8 text-blue-100">
        Join EU businesses using llama-orch for GDPR-compliant AI.
      </p>
      <div class="flex gap-4 justify-center">
        <a 
          href="mailto:hello@llama-orch.eu?subject=Free Trial Request" 
          class="bg-white text-blue-600 px-8 py-4 rounded-lg font-semibold text-lg hover:bg-blue-50 transition"
        >
          Start Free Trial →
        </a>
        <a 
          href="mailto:sales@llama-orch.eu?subject=Demo Request" 
          class="border-2 border-white px-8 py-4 rounded-lg font-semibold text-lg hover:bg-blue-700 transition"
        >
          Schedule Demo
        </a>
      </div>
      <p class="mt-6 text-sm text-blue-200">
        Questions? Email us at hello@llama-orch.eu
      </p>
    </div>
  </section>
</template>
```

**Day 16 Deliverable**: Compelling copy and content

---

## Day 17 (Wednesday): Landing Page Polish

### Morning Session (09:00-13:00)

**Task 1: Responsive design (2 hours)**
- Test on mobile
- Fix layout issues
- Improve touch targets

**Task 2: SEO basics (1 hour)**
```vue
<!-- src/App.vue -->
<script setup lang="ts">
import { useHead } from '@vueuse/head'

useHead({
  title: 'llama-orch - EU-Native LLM Inference with GDPR Compliance',
  meta: [
    {
      name: 'description',
      content: 'Run powerful LLM inference with built-in GDPR compliance. EU-only data residency, full audit trails, transparent pricing.'
    },
    {
      property: 'og:title',
      content: 'llama-orch - EU-Native LLM Inference'
    },
    {
      property: 'og:description',
      content: 'GDPR-compliant LLM inference for EU businesses'
    },
    {
      name: 'keywords',
      content: 'GDPR, LLM, inference, EU, compliance, audit, AI'
    }
  ]
})
</script>
```

### Afternoon Session (14:00-18:00)

**Task 3: Analytics (1 hour)**
```html
<!-- index.html -->
<script defer data-domain="llama-orch.eu" src="https://plausible.io/js/script.js"></script>
```

**Task 4: Deploy landing page (2 hours)**
```bash
# Build
npm run build

# Deploy to Netlify/Vercel
# Or serve from orchestratord
```

**Task 5: Final polish (1 hour)**
- Fix typos
- Improve colors
- Add animations
- Test all links

**Day 17 Deliverable**: Professional landing page live

---

## Day 18 (Thursday): Outreach Prep

### Morning Session (09:00-13:00)

**Task 1: Identify 50 target companies (2 hours)**

**EU Healthcare:**
- 10 digital health startups
- 5 telemedicine companies
- 5 health tech companies

**EU Finance:**
- 10 fintech startups
- 5 banking tech companies
- 5 insurance tech companies

**EU Startups:**
- 10 AI startups
- 5 SaaS companies

**Task 2: Write outreach email template (1 hour)**
```
Subject: EU-compliant LLM inference for [Company]

Hi [Name],

I noticed [Company] is building [product] for EU customers.

Quick question: How are you handling GDPR compliance for LLM inference?

We built llama-orch specifically for EU businesses that need:
- EU-only data residency (guaranteed)
- Full audit trails (compliance-ready)
- Transparent pricing (no surprises)

Would a 15-min demo be useful?

Best,
Vince
llama-orch.eu
```

### Afternoon Session (14:00-18:00)

**Task 3: Create demo video (2 hours)**
```
Script:
1. Show landing page (10 sec)
2. Submit inference job (20 sec)
3. Show audit log (20 sec)
4. Show GDPR export (20 sec)
5. Show pricing (10 sec)

Total: 80 seconds

Record with Loom
```

**Task 4: Prepare pitch deck (2 hours)**
```
Slide 1: Problem
- EU businesses need GDPR-compliant LLM inference
- Current options: US-based (not compliant) or expensive

Slide 2: Solution
- llama-orch: EU-native LLM inference
- Built-in GDPR compliance
- Transparent pricing

Slide 3: How It Works
- Submit job via API
- Inference on EU-only GPUs
- Full audit trail included

Slide 4: Pricing
- Starter: €99/mo
- Professional: €299/mo
- Enterprise: Custom

Slide 5: Next Steps
- 14-day free trial
- Schedule demo
- Contact: hello@llama-orch.eu
```

**Day 18 Deliverable**: Outreach materials ready

---

## Day 19 (Friday): Outreach Start

### Morning Session (09:00-13:00)

**Task 1: Send 10 emails (2 hours)**
- Personalize each email
- Reference their product
- Keep it short
- Clear CTA

**Task 2: Post on Hacker News (1 hour)**
```
Title: Show HN: llama-orch – EU-Native LLM Inference with GDPR Compliance

Body:
Hi HN,

I built llama-orch to solve a problem I had: running LLM inference for EU customers without GDPR headaches.

Key features:
- EU-only data residency (guaranteed)
- Full audit trails (compliance-ready)
- Simple pricing (€99-299/mo)

Built with Rust + Vue. Open to feedback!

Demo: https://llama-orch.eu
```

### Afternoon Session (14:00-18:00)

**Task 3: LinkedIn outreach (2 hours)**
- Connect with 20 prospects
- Personalized messages
- Share demo video

**Task 4: Twitter thread (1 hour)**
```
1/ I spent the last month building llama-orch: EU-native LLM inference with GDPR compliance built-in.

Here's why EU businesses need this 🧵

2/ Problem: Most LLM APIs are US-based. For EU businesses, this means:
- GDPR compliance headaches
- Data sovereignty concerns
- Audit trail gaps

3/ Solution: llama-orch runs inference on EU-only GPUs with:
- Guaranteed data residency
- Immutable audit logs
- GDPR endpoints (export, delete)

4/ Pricing is simple:
- Starter: €99/mo (500K tokens)
- Pro: €299/mo (2M tokens)
- Enterprise: Custom

No hidden fees. Cancel anytime.

5/ Built with Rust for performance and Vue for the UI.

Open to feedback! Demo: https://llama-orch.eu

6/ If you're building for EU customers and need compliant LLM inference, let's chat: hello@llama-orch.eu
```

**Task 5: Track responses (1 hour)**
- Create spreadsheet
- Track opens/replies
- Schedule follow-ups

**Day 19 Deliverable**: 10 contacts made, public launch

---

## Days 20-21 (Weekend): Content Creation

### Saturday

**Task 1: Write blog post (3 hours)**
```markdown
# The Complete Guide to GDPR-Compliant LLM Inference

## Introduction

If you're building AI products for EU customers, GDPR compliance isn't optional.

Here's everything you need to know about running LLM inference while staying compliant.

## The Challenge

Most LLM APIs are US-based:
- OpenAI: US servers
- Anthropic: US servers
- Google: Multi-region (includes US)

For EU businesses, this creates problems:
1. Data sovereignty concerns
2. Audit trail gaps
3. Compliance uncertainty

## The Solution

EU-native LLM inference with:
1. EU-only data residency
2. Full audit trails
3. GDPR endpoints

## How llama-orch Works

[Technical details]

## Pricing Comparison

[Comparison table]

## Conclusion

GDPR compliance doesn't have to be hard.

Try llama-orch free for 14 days: https://llama-orch.eu
```

**Task 2: Create comparison table (2 hours)**
```markdown
| Feature | llama-orch | OpenAI | Anthropic |
|---------|------------|--------|-----------|
| EU-only data | ✅ Yes | ❌ No | ❌ No |
| Audit trails | ✅ Included | ❌ Extra | ❌ No |
| GDPR endpoints | ✅ Built-in | ❌ Manual | ❌ Manual |
| Pricing | €99-299/mo | $$$$ | $$$$ |
```

**Task 3: Record demo video (1 hour)**
- Screen recording
- Voiceover
- Upload to YouTube

### Sunday

**Task 1: Prepare case study template (2 hours)**
```markdown
# Case Study: [Company] Achieves GDPR Compliance with llama-orch

## Challenge

[Company] needed to process [data type] with LLMs while maintaining GDPR compliance.

## Solution

Switched to llama-orch for:
- EU-only inference
- Full audit trails
- Transparent pricing

## Results

- 100% GDPR compliant
- €X saved per month
- Zero compliance incidents

## Testimonial

"[Quote from customer]"
```

**Task 2: Create email follow-up sequence (2 hours)**
```
Day 0: Initial outreach
Day 3: Follow-up (if no response)
Day 7: Final follow-up (if no response)
Day 14: Re-engage (if interested but no action)
```

**Task 3: Rest and prepare for Week 4 (2 hours)**
- Review leads
- Plan demo calls
- Prepare for closing

---

## Week 3 Success Criteria

- [ ] Landing page live and professional
- [ ] 10 outreach emails sent
- [ ] HN post published
- [ ] LinkedIn outreach (20 connections)
- [ ] Twitter thread posted
- [ ] Demo video recorded
- [ ] Blog post written
- [ ] 3+ demo calls scheduled
- [ ] 1+ interested customer

---

**Version**: 1.0  
**Status**: EXECUTE  
**Last Updated**: 2025-10-09
