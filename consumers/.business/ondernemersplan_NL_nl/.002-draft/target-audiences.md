# Target Audiences (Marketing Website)

## Overview

The marketing site is for **business and adoption**, not for technical documentation.  
Its purpose is to explain **why llama-orch and the plumbing services matter** to different audiences, and guide them to the right entry point (OSS, Public Tap, Private Tap, or custom toolkit work).

---

## Audience Profiles

### 1. Hobbyists & Independent Developers

- **Who they are:**  
  - Solo developers, open-source contributors, AI tinkerers without GPU homelabs.  
- **What they want:**  
  - Quick access to OSS tools.  
  - A low-barrier test bench (Public Tap credits).  
  - Community-driven transparency.  
- **What to show them:**  
  - “Start free with OSS” → GitHub.  
  - Public Tap prepaid credits (easy, no invoices).  
  - Blog/tutorials with practical setups.  

---

### 2. IT Teams (SMEs / Mid-sized Companies)

- **Who they are:**  
  - In-house IT departments in agencies, SMEs, or larger companies.  
- **What they want:**  
  - Reliable private infra without hiring AI/DevOps staff.  
  - Predictable billing (prepaid credits or GPU-hour packs).  
  - Transparency & control for compliance and procurement.  
- **What to show them:**  
  - Private Tap pricing table.  
  - Proof-first logs, Prometheus metrics.  
  - Case studies: “Deploy OSS models privately with zero lock-in.”  

---

### 3. Agencies & Consultancies

- **Who they are:**  
  - Digital agencies, software consultancies, IT resellers.  
- **What they want:**  
  - A flexible backend they can resell to clients.  
  - Fast prototyping (Public Tap).  
  - Private Tap for longer-term clients.  
- **What to show them:**  
  - Clear partner/reseller messaging.  
  - “Start on Public Tap, upgrade to Private Tap.”  
  - OSS transparency as a sales advantage (no vendor black box).  

---

### 4. Compliance-Sensitive Organizations

- **Who they are:**  
  - Enterprises, finance, healthcare, gov-related orgs.  
- **What they want:**  
  - Data privacy, EU hosting, AI Act alignment.  
  - Private LLMs instead of OpenAI black-box.  
  - Logs and documentation for audits.  
- **What to show them:**  
  - Dedicated Private Tap section.  
  - EU AI Act compliance story (transparency, logs, docs).  
  - Example pipeline reports / audit bundles.  

---

### 5. Small Businesses (Custom Toolkit Buyers)

- **Who they are:**  
  - Small businesses who don’t have dev capacity but need a custom AI tool.  
- **What they want:**  
  - One-off custom applet development (paid extra).  
  - Long shelf life, high robustness.  
- **What to show them:**  
  - “Custom Toolkit Development” as an **extra**, not the main product.  
  - High price, high value, but clearly optional.  

---

### 6. Investors / Funders (e.g., Qredits)

- **Who they are:**  
  - Business funders who need to evaluate feasibility.  
- **What they want:**  
  - Clear revenue model.  
  - Concrete pricing (Public Tap credits, Private Tap GPU/hour).  
  - Evidence of market need.  
- **What to show them:**  
  - Front-page revenue explanation (OSS → Public Tap → Private Tap).  
  - Market validation (AI adoption, EU AI Act, private LLMs).  
  - Simple, no-hype business framing (“AI plumbing is infrastructure”).  

---

## Summary

The marketing site must clearly segment:  

- **OSS** for hobbyists and developers.  
- **Public Tap** for hobbyists, agencies, early IT teams.  
- **Private Tap** for IT teams, agencies, compliance-sensitive orgs.  
- **Custom Toolkit (extra)** for small businesses.  
- **Feasibility pages** for investors/funders.  

Each audience gets its own **deeper dive page** linked from the front-page overview.  
Technical documentation will live at `docs.[brandname].com`, not on the marketing site.

## Audience Layers & Overlaps

Each target audience category develops across **four abstraction layers**.  

- **Layer 1:** Broad and cooperative — audiences overlap widely; categories are aware of each other and share concerns.  
- **Layer 2:** Semi-broad — smaller overlapping groups with stronger common ground; categories begin to show distinct preferences.  
- **Layer 3:** Deep — categories are mostly distinct, but still reference each other where workflows cross.  
- **Layer 4:** Specialized — categories stand fully apart, each with unique needs and jargon.  

---

### Layer 1 · Broad Cooperation

Overlaps: wide, almost everyone shares some common ground.  

- Hobbyists, IT Teams, Agencies → all care about *trying tools quickly* (OSS + Public Tap).  
- IT Teams, Compliance, Funders → all care about *predictability and governance*.  
- Agencies, Small Business → both care about *fast prototypes that can be shown to clients*.  
- Everyone overlaps on *the idea of plumbing as infrastructure*.  

---

### Layer 2 · Smaller Overlapping Groups

Overlaps shrink, groups have more in common with each other than with the whole.  

- **Hobbyists + Agencies:** Quick prototyping, Public Tap is their shared entry point.  
- **IT Teams + Compliance:** Private Tap as their shared solution (control + governance).  
- **Small Business + Agencies:** Toolkit development as a shared interest.  
- **IT Teams + Funders:** Both want evidence of ROI, predictable costs, pricing clarity.  

---

### Layer 3 · Deep but with Incidental Cross-Refs

Categories focus on their own needs, but acknowledge incidental links.  

- Hobbyists → start disappearing into OSS/docs, but sometimes referenced by Agencies for prototypes.  
- IT Teams → deep in Private Tap pricing & ops, but occasionally reference Compliance governance.  
- Compliance → focused on logs/docs bundles, with light cross-links to IT Teams for infra context.  
- Agencies → deep in reseller flows, but lightly link to Small Business case studies.  
- Funders → deep in competitor/USP analysis, occasionally referencing IT Teams revenue impact.  

---

### Layer 4 · Fully Specialized

Each category becomes independent.  

- **Hobbyists:** End up in docs portal (sdk/utils/orchestrator).  
- **IT Teams:** SLA packs, uptime guarantees, monitoring dashboards.  
- **Agencies:** Reseller terms, partner kits, white-label options.  
- **Compliance:** AI Act alignment packs, audit reports.  
- **Small Businesses:** Custom toolkit proposals and case studies.  
- **Funders:** Financial models, competitor docs, revenue streams.  

At this stage, audiences no longer overlap — each has its own endpoint.
