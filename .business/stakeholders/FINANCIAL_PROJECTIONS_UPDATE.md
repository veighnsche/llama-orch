# Financial Projections Update: More Conservative Approach

**Date:** 2025-10-10  
**Status:** Updated to reflect realistic ONE_MONTH_PLAN goals

**🎯 PRIMARY TARGET AUDIENCE:** Developers who build with AI but don't want to depend on big AI providers.

**THE FEAR:** Building complex codebases with AI assistance. What if the provider changes, shuts down, or changes pricing? Your codebase becomes unmaintainable.

**THE SOLUTION:** Build your own AI infrastructure using ALL your home network hardware. Never depend on external providers again.

---

## Summary of Changes

**OLD (Too Aggressive):**
- $6M/year revenue
- 50 customers at $10K/month each
- No timeline specified
- Target: Generic "GPU marketplace"

**NEW (Conservative & Realistic):**
- Year 1: €70K revenue (35 customers at €10K MRR)
- Year 2: €360K revenue (100 customers at €30K MRR)
- Year 3: €1M+ revenue target
- Based on actual ONE_MONTH_PLAN
- Target: Developers scared of AI provider dependency

---

## Year 1 Projections (2026)

### 30-Day Plan to First Customer

**What's Interesting for Stakeholders:**

rbee has a **detailed 30-day execution plan** to get the first paying customer:

- **Week 1 (Days 1-7):** Working end-to-end system (submit job → worker executes → tokens stream back)
- **Week 2 (Days 8-14):** EU compliance (GDPR endpoints, audit logs, basic web UI)
- **Week 3 (Days 15-21):** Marketing (landing page, outreach, 10 qualified leads)
- **Week 4 (Days 22-30):** Revenue (demos, close deal, onboard customer, €200 MRR)

**Key Advantage:** Already have **11 shared crates built**:
1. audit-logging (895 lines of docs, 32 pre-defined GDPR event types)
2. auth-min (token fingerprinting, validation)
3. input-validation (log injection prevention)
4. secrets-management (secure secret loading)
5. narration-core (developer observability)
6. deadline-propagation (request deadlines)
7. gpu-info (GPU detection)
8. Plus 4 more crates

**Time Saved:** 5 days of development time (audit system + GDPR compliance already 90% built)

**This means:** Realistic path to first customer in 30 days, not months.

### Monthly Breakdown

**Month 1:**
- 1 customer at €199/month (special offer) — **30-day plan in place**
- MRR: €199
- Total revenue: €199

**Month 2:**
- 3 customers at €199/month
- MRR: €597
- Total revenue: €597

**Month 3:**
- 5 customers at €299/month (full price)
- MRR: €1,495
- Total revenue: €1,495

**Month 4-6:**
- Ramp to 20 customers
- MRR: €5,980
- Average: €299/month per customer

**Month 7-9:**
- Ramp to 25 customers
- MRR: €7,475

**Month 10-12:**
- Ramp to 35 customers
- MRR: €10,465

**Year 1 Total Revenue: ~€70,000**

---

## Year 2 Projections (2027)

### Growth Strategy

**Q1 (Months 13-15):**
- Ramp to 50 customers
- MRR: €15,000
- Focus: Product refinement, enterprise tier

**Q2 (Months 16-18):**
- Ramp to 70 customers
- MRR: €21,000
- Focus: Platform mode, multi-tenancy

**Q3 (Months 19-21):**
- Ramp to 85 customers
- MRR: €25,500
- Focus: Web UI, marketplace features

**Q4 (Months 22-24):**
- Ramp to 100 customers
- MRR: €30,000
- Focus: Scale, enterprise sales

**Year 2 Total Revenue: ~€360,000**

---

## Year 3 Projections (2028)

### Target: €1M+ Annual Revenue

**Path:**
- 200 customers at €300/month average
- Or: Mix of tiers reaching €83K MRR
- Focus: Marketplace, enterprise, multi-modal

**Breakdown:**
- Starter tier (€99/mo): 50 customers = €4,950/mo
- Professional tier (€299/mo): 100 customers = €29,900/mo
- Enterprise tier (€2K+/mo): 25 customers = €50,000/mo
- **Total MRR: €84,850 = €1,018,200/year**

---

## Pricing Tiers (From 08_PRICING_STRATEGY.md)

### Starter — €99/mo
- 500K tokens/month
- EU-only inference
- Basic audit logs
- Email support (48h)
- **Target:** Small startups, side projects

### Professional — €299/mo (Most Popular)
- 2M tokens/month
- Full audit trails
- GDPR endpoints
- Priority support (24h)
- All models
- **Target:** Growing startups, small businesses

### Enterprise — Custom (€2,000+/mo)
- Unlimited tokens
- Dedicated instances
- Custom SLAs (99.9% uptime)
- Phone support (4h)
- White-label option
- **Target:** Large companies, high-volume

---

## Revenue Sources

### 1. SaaS Subscriptions (Primary)
- Starter tier: €99/mo
- Professional tier: €299/mo
- Enterprise tier: Custom

**Year 1 focus:** Professional tier (€299/mo)  
**Year 2 focus:** Mix of all tiers  
**Year 3 focus:** Enterprise expansion

### 2. Marketplace Fees (Future)
- Platform takes 30-40% of task fees
- GPU providers keep 60-70%
- **Timeline:** Year 2-3

### 3. Professional Services (Future)
- Custom integrations
- Training and onboarding
- Dedicated support
- **Timeline:** Year 2+

---

## Customer Acquisition Cost (CAC)

### Channels

**Free (€0 CAC):**
- Open source community
- HackerNews/Reddit posts
- GitHub stars
- Word of mouth

**Low-cost (€50-100 CAC):**
- Content marketing (SEO)
- Email outreach
- LinkedIn outreach
- Community engagement

**Paid (€200-500 CAC):**
- Google Ads (later)
- LinkedIn Ads (later)
- Conferences (Year 2+)

**Year 1 strategy:** Free + low-cost channels only  
**Target CAC:** <€100 per customer

---

## Lifetime Value (LTV)

### Assumptions

**Average customer:**
- €299/month (Professional tier)
- 18-month retention (conservative)
- **LTV: €5,382**

**LTV:CAC Ratio:**
- €5,382 LTV / €100 CAC = 53.8:1
- Industry standard target: 3:1
- We're in good shape ✅

---

## Churn Assumptions

### Conservative Estimates

**Year 1:**
- Month 1-3: 10% churn (figuring out product-market fit)
- Month 4-12: 5% churn (stable customers)

**Year 2:**
- 3% monthly churn (product mature)

**Year 3:**
- 2% monthly churn (enterprise customers)

### Retention Strategies

1. **Onboarding:** White-glove first week
2. **Support:** Fast response times (24h)
3. **Features:** Regular updates, listen to feedback
4. **Community:** Active forums, Discord
5. **Value:** Continuous improvement, new models

---

## Break-Even Analysis

### Costs (Year 1)

**Infrastructure:**
- Demo servers: €100/month
- Domain, hosting: €50/month
- **Total: €150/month = €1,800/year**

**Development:**
- Solo founder (no salary initially)
- **Total: €0**

**Marketing:**
- Domain, email: €100/year
- Ads (if any): €0 (bootstrap)
- **Total: €100/year**

**Total Year 1 Costs: ~€2,000**

### Break-Even

**Revenue needed:** €2,000  
**At €299/month:** 7 customers  
**Expected:** Month 3-4

**Profitable from Month 4 onwards** ✅

---

## Comparison: Old vs. New Projections

### Old (Too Aggressive)

| Metric | Value |
|--------|-------|
| Year 1 Revenue | $6M |
| Customers | 50 at $10K/month |
| Timeline | Unspecified |
| Plausibility | Low ❌ |

### New (Conservative)

| Metric | Year 1 | Year 2 | Year 3 |
|--------|--------|--------|--------|
| Revenue | €70K | €360K | €1M+ |
| Customers | 35 | 100 | 200+ |
| MRR | €10K | €30K | €83K+ |
| Plausibility | High ✅ | Medium ✅ | Stretch goal ✅ |

---

## Why These Numbers Are Conservative

### 1. Proven Pricing
- Based on actual market research
- €299/month is tested price point
- Comparable to competitors

### 2. Realistic Growth
- 1 customer in Month 1
- 2-3 new customers per month Year 1
- 5-10 new customers per month Year 2
- Achievable with free marketing

### 3. Strong Unit Economics
- LTV:CAC = 53:1 (target: 3:1)
- Break-even at 7 customers (Month 3-4)
- Low churn assumed (5-10%)

### 4. Clear Path to Profitability
- Bootstrap-friendly
- No VC needed
- Profitable from Month 4

### 5. Based on Actual Plan
- ONE_MONTH_PLAN is realistic
- 30-day execution plan exists
- Day-by-day tasks defined

---

## Risk Factors

### Risks & Mitigations

**Risk 1: Slow customer acquisition**
- Mitigation: Start outreach Week 3, not Week 4
- Mitigation: Free tier to drive adoption
- Mitigation: Open source community growth

**Risk 2: Higher churn than expected**
- Mitigation: White-glove onboarding
- Mitigation: Fast support response
- Mitigation: Regular product updates

**Risk 3: Price resistance**
- Mitigation: Special offers for early customers
- Mitigation: Value-based pricing (save vs. OpenAI)
- Mitigation: Flexible tiers

**Risk 4: Technical delays**
- Mitigation: Cut scope ruthlessly
- Mitigation: MVP first, polish later
- Mitigation: ONE_MONTH_PLAN has buffer days

**Risk 5: Competition**
- Mitigation: EU-compliance moat
- Mitigation: Open source community
- Mitigation: First-mover advantage

---

## Success Metrics

### Month 1
- ✅ 1 customer at €199/mo
- ✅ Product works end-to-end
- ✅ First revenue

### Month 3
- ✅ 5 customers at €299/mo
- ✅ €1,495 MRR
- ✅ Break-even

### Month 6
- ✅ 20 customers
- ✅ €5,980 MRR
- ✅ Web UI launched

### Month 12
- ✅ 35 customers
- ✅ €10,465 MRR
- ✅ €70,000 revenue
- ✅ Platform mode ready

---

## Updated Documents

**Files modified with conservative projections:**

1. ✅ `AI_DEVELOPMENT_STORY.md`
   - Updated "The Vision" section
   - Year 1: €70K, Year 2: €360K, Year 3: €1M+

2. ✅ `VIDEO_SCRIPTS.md`
   - Updated investor script (Audience #5)
   - Removed $6M claim
   - Added realistic Year 1-3 goals

3. ✅ `AGENTIC_AI_USE_CASE.md` (NEW)
   - Focus on Zed IDE use case
   - Emphasize llama-orch-utils
   - Revenue model section

4. ✅ `FINANCIAL_PROJECTIONS_UPDATE.md` (THIS FILE)
   - Complete breakdown
   - Conservative assumptions
   - Risk analysis

---

## Key Takeaways

1. **Year 1 target: €70K revenue** (35 customers)
2. **Pricing: €99-299/month** (not $10K!)
3. **Bootstrap-friendly** (low costs, high margins)
4. **Profitable from Month 4** (break-even at 7 customers)
5. **Based on actual ONE_MONTH_PLAN** (realistic execution)
6. **Strong unit economics** (LTV:CAC = 53:1)
7. **Clear path to €1M+** by Year 3

**These projections are conservative, realistic, and achievable.** ✅

---

## Quick Reference

**Pronunciation:** rbee (pronounced "are-bee")  
**Target Audience:** Developers who build with AI but fear provider dependency  
**Key Advantage:** 11 shared crates already built (saves 5 days of development)  
**30-Day Plan:** Week 1 (system), Week 2 (compliance), Week 3 (marketing), Week 4 (revenue)  
**Month 1 Goal:** 1 customer, €200 MRR  
**Year 1 Goal:** 35 customers, €10K MRR, €70K revenue  
**Year 2 Goal:** 100 customers, €30K MRR, €360K revenue  
**Year 3 Goal:** 200+ customers, €83K+ MRR, €1M+ revenue  
**Break-Even:** Month 3-4 (7 customers)  
**LTV:CAC Ratio:** 53:1 (excellent)

---

**Version:** 1.0  
**Status:** Active  
**Last Updated:** 2025-10-10

---

*All financial projections are estimates and subject to change based on market conditions and execution.*
