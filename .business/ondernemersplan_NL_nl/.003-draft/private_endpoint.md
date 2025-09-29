# Private Endpoint — Product & Licensing Strategy (Draft)

## Core direction

We know that **95% of AI initiatives fail**. Most teams experiment but struggle to integrate AI into their real growth, support, or productivity workflows. Our platform solves this by combining **open-source developer tools** with a **private AI endpoint** that takes successful prototypes into production.

## Layered product stack

* **Core orchestration (GPL)**

  * `llama-orch` is GPL-licensed.
  * Purpose: enforce contributions back to the orchestration layer, protect infra moat, ensure no silent forks outpace us.
* **Developer toolkit (permissive)**

  * `llama-orch-utils` + applets are Apache/MIT.
  * Purpose: maximize adoption, frictionless embedding into CRMs, helpdesks, and internal tools.
  * Includes ready-made applets for growth, support, and productivity use cases.
* **Control plane (closed)**

  * Endpoint management, autoscaling, quotas, billing, observability.
  * Purpose: SaaS lock-in and monetization via private endpoints.

## Business model

* **“Toolkit free, endpoint paid.”**
* Developers build freely with permissive SDK + applets.
* When they move to production, the smoothest path is our hosted private endpoint.
* Revenue comes from **subscriptions per endpoint**, with bundled quotas and overage protection.
* Efficient backend resource pooling raises utilization and margins while maintaining strict tenant isolation.

## Positioning

* *“We help the 5% of AI projects that succeed get to production.”*
* *“Permissive applets for adoption, private endpoints for monetization.”*
* *“Your private AI endpoint, ready in minutes—backed by efficiency, isolation, and enterprise-grade controls.”*

## Strategic implications

* OSS spreads the adoption surface, ensuring lock-in at the SDK/app level.
* SaaS captures predictable recurring revenue (subscription + quota).
* GPL core prevents competitive enclosures, while permissive utils ensure maximum developer integration.
* Differentiation: not raw GPU resale, but **developer success + endpoint deployment**.

---

Internally you’d still describe the architecture as **shared GPU pools with isolation**. Externally, everything stays framed as **“your private endpoint”**.
