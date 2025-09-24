# Mesh Site Architecture

## Purpose

This document defines the website’s architecture as a **mesh** rather than a tree.  
Each **category** corresponds to a target audience.  
Each category has **4 abstraction layers**, moving from broad + cross-linked to deep + audience-specific.  
By **Layer 4**, each audience has self-routed into their own funnel without being forced.

---

## Categories (Audiences)

1. Hobbyists & Developers  
2. IT Teams (SMEs)  
3. Agencies & Consultancies  
4. Compliance-Sensitive Orgs (Finance, Healthcare, Gov)  
5. Small Businesses (Custom Toolkit Buyers)  
6. Investors / Funders  

---

## Layers of Abstraction

### Layer 1 — Broad & Cooperative

- Pages are wide in scope and show cooperation between categories.  
- Messaging: *“AI plumbing fits many needs.”*  
- Heavy cross-links: hobbyists → agencies, IT → compliance, etc.  
- Example: Front-page hero with all services (OSS, Public Tap, Private Tap).  

### Layer 2 — Semi-Broad

- Landing pages tailored to each audience but still cross-linked.  
- Example: *Agencies page* shows how agencies work with IT teams, and how Public Tap connects to Private Tap.  
- Still more width than depth.  

### Layer 3 — Deep with Light Cross-Refs

- Pages narrow in focus, mostly addressing one audience.  
- Still reference adjacent categories lightly (“For compliance, see proof library”).  
- Example: IT Teams page shows Private Tap pricing, logs, monitoring.  

### Layer 4 — Fully Specialized

- Audience-specific, final content.  
- Developers → SDK & docs (on separate docs subdomain).  
- IT Teams → SLA, uptime guarantees, monitoring proof.  
- Agencies → partner kits, reseller agreements.  
- Compliance → governance checklists, AI Act references.  
- Small Biz → case studies, custom toolkit offers.  
- Funders → revenue model, competitor maps, feasibility docs.  

---

## Routing Principles

- Users self-route: no forced funnels.  
- **Layer 1** encourages exploration.  
- **Layer 2** narrows but still connects.  
- **Layer 3** is audience-specific with rare cross-refs.  
- **Layer 4** is fully segmented: each audience in its own domain of expertise.  
- **Docs** live on `docs.[brand].com`, but reachable from Dev/Hobbyist category Layer 4.  

---

## Visual Overview (Conceptual)

```

\[Front Page]
↓
┌─────────────┬──────────────┬───────────────┬──────────────┬──────────────┬──────────────┐
\| Hobbyists   | IT Teams      | Agencies      | Compliance   | Small Biz    | Funders      | (Layer 1)
└─────┬───────┴──────┬────────┴──────┬────────┴──────┬───────┴──────┬──────┴────────────┘
↓              ↓               ↓               ↓               ↓                  ↓
\[Layer 2: semi-broad, cross-links remain]
↓              ↓               ↓               ↓               ↓                  ↓
\[Layer 3: deep, only light cross-refs]
↓              ↓               ↓               ↓               ↓                  ↓
\[Layer 4: final funnels → Docs / SLA / Toolkit / Biz models]

```

Love that. Here’s a clean Mermaid diagram you can drop straight into a `.md` file. I’d name the file **`site-architecture-mesh.mmd.md`** so it’s obvious it contains Mermaid.

---

````markdown
# Mesh Site Architecture (Mermaid)

> Each category (audience) has 4 abstraction layers.  
> L1 = widest + most cross-links → L4 = deepest + audience-specific (Dev = docs, etc.).

## High-level mesh (front page routes into 6 categories at Layer 1)

```mermaid
flowchart TB
  subgraph L0["Layer 0 · Front Page (Everyone)"]
    FP["Front Page<br/>Hero · USP · OSS · Public Tap · Private Tap"]
  end

  subgraph L1["Layer 1 · Broad & Cooperative (Width > Depth)"]
    D1["Hobbyists & Developers · L1"]
    IT1["IT Teams · L1"]
    A1["Agencies & Consultancies · L1"]
    C1["Compliance-sensitive Orgs · L1"]
    SB1["Small Business · L1"]
    F1["Investors/Funders · L1"]
  end

  FP --> D1 & IT1 & A1 & C1 & SB1 & F1

  %% Cross-links at Layer 1 (wide mesh)
  D1 --- IT1
  IT1 --- A1
  A1 --- C1
  C1 --- F1
  SB1 --- IT1
  D1 --- A1
````

## Per-category depth (each has its own 4 layers, with tapering cross-links)

```mermaid
flowchart LR
  %% DEVELOPERS
  subgraph DEV["Hobbyists & Developers"]
    D1["L1: Dev overview (OSS + Public Tap)"] --> D2["L2: Dev landing (quickstart, examples)"]
    D2 --> D3["L3: Deep dev (patterns, pipelines)"]
    D3 --> D4["L4: Docs portal (sdk/utils/orch) @ docs.brand.com"]
    %% Dev cross links (taper)
    D1 -.-> IT1["IT Teams · L1"]
    D2 -.-> IT2["IT Teams · L2 (Private Tap intro)"]
    D3 -.-> PR1["Proof Library · L3"]
  end

  %% IT TEAMS
  subgraph IT["IT Teams (SMEs)"]
    I1["L1: IT overview (why private tap)"] --> I2["L2: Use-cases & models"]
    I2 --> I3["L3: Pricing, SLA, metrics"]
    I3 --> I4["L4: Contracting pack (SLA, SLOs)"]
    I1 -.-> A1["Agencies · L1"]
    I2 -.-> C2["Compliance · L2 (governance refs)"]
    I3 -.-> PR1
  end

  %% AGENCIES
  subgraph AG["Agencies & Consultancies"]
    A1A["L1: Agencies overview (resell flow)"] --> A2["L2: Start on Public Tap → upgrade"]
    A2 --> A3["L3: White-label, partner kits"]
    A3 --> A4["L4: Reseller terms & margins"]
    A1A -.-> I2
    A2 -.-> D2
  end

  %% COMPLIANCE
  subgraph COM["Compliance-sensitive Orgs"]
    C1A["L1: Compliance overview (private LLM need)"] --> C2A["L2: Controls & deployment options"]
    C2A --> C3["L3: Audit bundle, logs, docs"]
    C3 --> C4["L4: Governance pack (AI Act alignment)"]
    C1A -.-> I1
    C2A -.-> PR1
  end

  %% SMALL BUSINESS
  subgraph SMB["Small Business (Custom Toolkit · Extra)"]
    S1["L1: What we build (examples)"] --> S2["L2: Eligibility (fits the toolkit?)"]
    S2 --> S3["L3: Case studies, scope"]
    S3 --> S4["L4: Proposal template & pricing signals"]
    S1 -.-> A1
    S2 -.-> I2
  end

  %% FUNDERS
  subgraph FUN["Investors / Funders"]
    F1A["L1: Business overview"] --> F2["L2: Revenue model (Public/Private)"]
    F2 --> F3["L3: Competitors & USP"]
    F3 --> F4["L4: Financials pack (unit economics)"]
    F1A -.-> I3
    F2 -.-> PR1
  end

  %% SHARED PROOF/RESOURCES
  subgraph SHARED["Shared Trust Content"]
    PR1["Proof Library · L3 (metrics, logs, SSE transcripts)"]
    CS1["Case Studies · L3"]
    USP["USP · L3"]
    COMP["Competitors · L3"]
  end

  %% Some shared references
  I3 --- PR1
  A3 --- CS1
  F3 --- COMP
  C3 --- PR1
```

## Legend

- **Layer 1**: Broad, cross-link heavy pages that explain cooperation between audiences.
- **Layer 2**: Audience landing pages—still wide, but starting depth.
- **Layer 3**: Deep audience content; only light cross-references to shared proof pages.
- **Layer 4**: Final audience funnels:

  - Dev → Docs portal (`docs.brand.com`)
  - IT → SLA/contract pack
  - Agencies → partner/reseller kit
  - Compliance → governance pack
  - Small Biz → proposal template
  - Funders → financials pack
