# Mesh Site Architecture

> Status: Future proposal — not implemented in Draft 2 (see `index.md` §C).

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

# Mesh Site Diagram

```mermaid
flowchart TB

  %% Layer 0
  L0["Front Page · Broad USP + OSS + Public Tap + Private Tap"]:::layer0

  %% Layer 1 — Big Groups
  subgraph L1["Layer 1 · Big Category Groups (broad, cooperative)"]
    MAKERS["Makers · Hobbyists + Agencies"]:::layer1
    OPERATORS["Operators · IT Teams + Compliance"]:::layer1
    BACKERS["Backers · Small Biz + Funders"]:::layer1
  end

  L0 --> MAKERS
  L0 --> OPERATORS
  L0 --> BACKERS

  %% Layer 2 — Smaller Overlap Groups
  subgraph L2["Layer 2 · Smaller Overlaps (narrower coalitions)"]
    M1["Fast Prototypers · Hobbyists + Agencies"]:::layer2
    O1["Infra Managers · IT + Compliance"]:::layer2
    B1["ROI Seekers · IT + Funders"]:::layer2
    B2["Custom Builders · Small Biz + Agencies"]:::layer2
  end

  MAKERS --> M1
  OPERATORS --> O1
  BACKERS --> B1
  BACKERS --> B2
  OPERATORS --> B1

  %% Layer 3 — Narrow Bundles
  subgraph L3["Layer 3 · Narrow Bundles (mostly audience-specific, light cross-links)"]
    DEV_DEEP["OSS / Public Tap Deep Dive"]:::layer3
    IT_DEEP["Private Tap Pricing & Ops"]:::layer3
    AGENCY_DEEP["Reseller / Partner Kits"]:::layer3
    COMPLIANCE_DEEP["Audit & Governance Bundles"]:::layer3
    FUNDING_DEEP["Business Model & Competitors"]:::layer3
    SMB_DEEP["Toolkit Case Studies"]:::layer3
  end

  M1 --> DEV_DEEP
  M1 --> AGENCY_DEEP
  O1 --> IT_DEEP
  O1 --> COMPLIANCE_DEEP
  B1 --> FUNDING_DEEP
  B2 --> SMB_DEEP
  AGENCY_DEEP -.-> SMB_DEEP
  IT_DEEP -.-> FUNDING_DEEP
  COMPLIANCE_DEEP -.-> IT_DEEP

  %% Layer 4 — Unique Endpoints
  subgraph L4["Layer 4 · Final Specialized Pages (audience endpoints)"]
    DEV_DOCS["Developers · Docs Portal"]:::layer4
    IT_SLA["IT Teams · SLA / Contract Pack"]:::layer4
    AGENCY_PARTNER["Agencies · Reseller Agreements"]:::layer4
    COMPLIANCE_PACK["Compliance · Governance Pack"]:::layer4
    SMB_PROPOSAL["Small Biz · Toolkit Proposal Template"]:::layer4
    FUNDERS_FIN["Funders · Financial Model"]:::layer4
  end

  DEV_DEEP --> DEV_DOCS
  IT_DEEP --> IT_SLA
  AGENCY_DEEP --> AGENCY_PARTNER
  COMPLIANCE_DEEP --> COMPLIANCE_PACK
  SMB_DEEP --> SMB_PROPOSAL
  FUNDING_DEEP --> FUNDERS_FIN

  %% Styles
  classDef layer0 fill:#fdf6e3,stroke:#657b83,stroke-width:2px;
  classDef layer1 fill:#eee8d5,stroke:#586e75;
  classDef layer2 fill:#93a1a1,stroke:#073642,color:#fff;
  classDef layer3 fill:#268bd2,stroke:#002b36,color:#fff;
  classDef layer4 fill:#2aa198,stroke:#002b36,color:#fff;
```

# 🔑 How to read it

- **Front Page (L0):** one universal entry point.  
- **Layer 1:** broad umbrellas (Makers, Operators, Backers).  
- **Layer 2:** smaller overlap coalitions.  
- **Layer 3:** narrowed bundles (deep dives).  
- **Layer 4:** final audience endpoints (unique, jargon-heavy, no more overlaps).  
