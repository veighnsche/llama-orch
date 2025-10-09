# One Month Master Plan — Ship or Die

**Date**: 2025-10-09  
**Deadline**: 2025-11-09 (30 days)  
**Goal**: Working product generating revenue

---

## The Reality

**You have:** 30 days, solid foundation, clear vision  
**You need:** Working product, first customers, revenue  
**You will:** Ship ruthlessly, cut scope aggressively, focus on money

---

## The Strategy

### Week 1: Foundation (Days 1-7)
**Goal:** Working end-to-end system  
**Deliverable:** Submit job → worker executes → tokens stream back

### Week 2: EU Compliance (Days 8-14)
**Goal:** EU audit toggle + basic web UI  
**Deliverable:** GDPR-compliant inference with audit logs

### Week 3: Marketing (Days 15-21)
**Goal:** Landing page + first customer outreach  
**Deliverable:** 10 qualified leads

### Week 4: Revenue (Days 22-30)
**Goal:** First paying customer  
**Deliverable:** $200 MRR (1 customer)

---

## Daily Breakdown

### Days 1-7: Foundation Week

**Day 1 (Monday): rbees-orcd**
- [ ] Create binary skeleton
- [ ] HTTP server (axum)
- [ ] In-memory job queue
- [ ] Worker registry
- [ ] Job submission endpoint
- [ ] Worker registration endpoint
- **Test:** Submit job via curl

**Day 2 (Tuesday): rbees-pool**
- [ ] Create CLI binary
- [ ] Model download command (hf CLI wrapper)
- [ ] Worker spawn command
- [ ] Model list command
- **Test:** Download tinyllama, spawn worker

**Day 3 (Wednesday): rbees-ctl**
- [ ] Create CLI binary
- [ ] SSH command forwarding
- [ ] Job submission command
- [ ] Pool status command
- **Test:** Command pool via SSH

**Day 4 (Thursday): Integration**
- [ ] Worker registration in rbees-workerd
- [ ] Job dispatch from rbees-orcd
- [ ] End-to-end flow testing
- **Test:** Full flow works

**Day 5 (Friday): Polish**
- [ ] Error handling
- [ ] Logging
- [ ] Progress indicators
- [ ] README
- **Test:** Clean slate to working system

**Day 6-7 (Weekend): Buffer**
- [ ] Fix critical bugs
- [ ] Improve DX
- [ ] Prepare for Week 2

**Deliverable:** Working orchestrator + CLIs

---

### Days 8-14: EU Compliance Week

**Day 8 (Monday): Audit Toggle**
- [ ] LLORCH_EU_AUDIT environment variable
- [ ] Conditional audit logging
- [ ] audit-logging crate integration
- **Test:** Toggle on/off works

**Day 9 (Tuesday): GDPR Endpoints**
- [ ] GET /gdpr/export endpoint
- [ ] POST /gdpr/delete endpoint
- [ ] POST /gdpr/consent endpoint
- **Test:** Export/delete data works

**Day 10 (Wednesday): Data Residency**
- [ ] Worker region tagging
- [ ] EU-only worker filtering
- [ ] Validation in job submission
- **Test:** Non-EU workers rejected

**Day 11 (Thursday): Web UI Start**
- [ ] Vue 3 + Vite setup
- [ ] Job submission form
- [ ] Token streaming display
- **Test:** Submit job via UI

**Day 12 (Friday): Web UI Finish**
- [ ] Audit log viewer
- [ ] Job history
- [ ] Basic styling (TailwindCSS)
- **Test:** Full UI flow works

**Day 13-14 (Weekend): Polish**
- [ ] Fix UI bugs
- [ ] Improve styling
- [ ] Add loading states
- [ ] Error messages

**Deliverable:** EU-compliant system with web UI

---

### Days 15-21: Marketing Week

**Day 15 (Monday): Landing Page Structure**
- [ ] Hero section
- [ ] Features section
- [ ] Pricing section
- [ ] CTA section
- **Test:** Page loads, looks decent

**Day 16 (Tuesday): Landing Page Content**
- [ ] Write copy
- [ ] Add testimonials (mock)
- [ ] Add use cases
- [ ] Add FAQ
- **Test:** Content is compelling

**Day 17 (Wednesday): Landing Page Polish**
- [ ] Styling (TailwindCSS)
- [ ] Responsive design
- [ ] SEO basics (meta tags)
- [ ] Analytics (Plausible/Simple Analytics)
- **Test:** Page looks professional

**Day 18 (Thursday): Outreach Prep**
- [ ] Identify 50 target companies
- [ ] Write outreach email template
- [ ] Create demo video (Loom)
- [ ] Prepare pitch deck (5 slides)
- **Test:** Materials ready

**Day 19 (Friday): Outreach Start**
- [ ] Send 10 emails
- [ ] Post on HN/Reddit
- [ ] LinkedIn outreach
- [ ] Twitter thread
- **Test:** 10 contacts made

**Day 20-21 (Weekend): Content**
- [ ] Write blog post (EU compliance guide)
- [ ] Create comparison table
- [ ] Record demo video
- [ ] Prepare case study template

**Deliverable:** Marketing site + 10 qualified leads

---

### Days 22-30: Revenue Week

**Day 22 (Monday): Follow-ups**
- [ ] Follow up with leads
- [ ] Schedule demos
- [ ] Answer questions
- **Goal:** 3 demo calls scheduled

**Day 23 (Tuesday): Demos**
- [ ] Run 3 demo calls
- [ ] Collect feedback
- [ ] Send proposals
- **Goal:** 1 interested customer

**Day 24 (Wednesday): Close**
- [ ] Negotiate pricing
- [ ] Send contract
- [ ] Setup payment (Stripe)
- **Goal:** First contract signed

**Day 25 (Thursday): Onboarding**
- [ ] Setup customer account
- [ ] API key generation
- [ ] Initial configuration
- [ ] First test inference
- **Goal:** Customer running

**Day 26 (Friday): Support**
- [ ] Fix customer issues
- [ ] Improve docs
- [ ] Add missing features
- **Goal:** Customer happy

**Day 27-28 (Weekend): Scale Prep**
- [ ] Improve onboarding
- [ ] Automate setup
- [ ] Prepare for customer 2

**Day 29 (Monday): More Outreach**
- [ ] Send 20 more emails
- [ ] Post success story
- [ ] Referral request
- **Goal:** 5 more leads

**Day 30 (Tuesday): Reflection**
- [ ] Measure results
- [ ] Plan Month 2
- [ ] Celebrate first customer

**Deliverable:** $200 MRR (1 paying customer)

---

## What We're Building

### Binaries (3)

1. **rbees-orcd** (Daemon)
   - HTTP server :8080
   - In-memory job queue
   - Worker registry
   - Job dispatch
   - SSE streaming

2. **rbees-pool** (CLI)
   - Model downloads
   - Worker spawn
   - Git operations

3. **rbees-ctl** (CLI)
   - SSH forwarding
   - Job submission
   - Pool commands

### Features (Minimal)

**Week 1:**
- Job submission
- Worker dispatch
- Token streaming

**Week 2:**
- EU audit toggle
- GDPR endpoints
- Web UI

**Week 3:**
- Landing page
- Outreach materials

**Week 4:**
- Customer onboarding
- Support

### What We're NOT Building

❌ pool-managerd daemon (use rbees-pool + SSH)  
❌ SQLite persistence (in-memory is fine)  
❌ Multi-tenancy (single customer per instance)  
❌ Marketplace (M5 is too far)  
❌ Fancy frontend (basic is fine)  
❌ Mobile apps (web only)  
❌ Advanced features (fine-tuning, custom models)

---

## Success Metrics

### Week 1
- [ ] End-to-end flow works
- [ ] Can submit job via CLI
- [ ] Tokens stream back
- [ ] README complete

### Week 2
- [ ] EU audit toggle works
- [ ] GDPR endpoints functional
- [ ] Web UI deployed
- [ ] Can submit job via UI

### Week 3
- [ ] Landing page live
- [ ] 10 outreach emails sent
- [ ] 3 demo calls scheduled
- [ ] 1 interested customer

### Week 4
- [ ] First contract signed
- [ ] Customer onboarded
- [ ] First inference running
- [ ] $200 MRR

---

## Risk Mitigation

### Risk: Technical delays
**Mitigation:** Cut scope ruthlessly, ship broken but working

### Risk: No customers
**Mitigation:** Outreach starts Week 3, not Week 4

### Risk: Customer needs features
**Mitigation:** Build fast, iterate with customer

### Risk: Burnout
**Mitigation:** Sleep, eat, exercise. Sustainable pace.

---

## The Non-Negotiables

1. **Ship every week** - Something must go live
2. **Talk to customers** - Start Week 3, not Week 4
3. **Cut scope** - If it doesn't make money, don't build it
4. **Focus** - One thing at a time, finish before starting next
5. **Revenue** - Everything serves the goal of first customer

---

## Daily Routine

**Morning (4 hours):**
- Focus work (coding, building)
- No distractions
- Ship something

**Afternoon (4 hours):**
- Testing, polish
- Documentation
- Customer work (Week 3+)

**Evening:**
- Plan tomorrow
- Reflect on progress
- Rest

**Weekend:**
- Buffer for delays
- Polish and improve
- Prepare for next week

---

## The Plan Files

Each section has detailed implementation plan:

- `01_WEEK_1_FOUNDATION.md` - Days 1-7 detailed tasks
- `02_WEEK_2_COMPLIANCE.md` - Days 8-14 detailed tasks
- `03_WEEK_3_MARKETING.md` - Days 15-21 detailed tasks
- `04_WEEK_4_REVENUE.md` - Days 22-30 detailed tasks
- `05_TECHNICAL_SPECS.md` - Technical implementation details
- `06_MARKETING_MATERIALS.md` - Copy, positioning, outreach
- `07_CUSTOMER_ONBOARDING.md` - Onboarding process
- `08_PRICING_STRATEGY.md` - Pricing and packaging

---

## The Promise

**In 30 days you will have:**
- Working product (rbees-orcd + CLIs + web UI)
- EU compliance (audit logs, GDPR endpoints)
- Marketing site (landing page, demo)
- First customer ($200 MRR)

**This is doable. This is necessary. Let's ship.**

---

**Version**: 1.0  
**Status**: EXECUTE  
**Last Updated**: 2025-10-09

---

**START TOMORROW. SHIP IN 30 DAYS. GET PAID.**
