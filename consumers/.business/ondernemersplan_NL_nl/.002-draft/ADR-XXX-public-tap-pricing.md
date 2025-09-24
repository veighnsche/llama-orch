# draft-002/PublicTap-Pricing.md

## Status

DRAFT

## Overview

The Public Tap is billed using **prepaid, non-refundable credits**.  
Customers purchase credit packs in euros (€). Credits are deducted automatically based on API usage.

---

## Pricing Model

- **Unit of measure:** 1M tokens (input + output combined).  
- **Baseline rate:** €1.20 per 1M tokens.  
  - Slightly above hyperscaler costs (OpenAI, Together.ai) to reflect solo-operator overhead, OSS transparency, and EU locality.  

---

## Credit Packs

- **Starter Pack** – €50 → ~41M tokens  
  - Target: hobbyists, early prototypes.  
- **Builder Pack** – €200 → ~166M tokens  
  - Target: agencies, small IT teams testing multiple flows.  
- **Pro Pack** – €500 → ~416M tokens  
  - Target: established teams, longer pilots.  

---

## Shelf Life

- Credits are valid for **12 months** from purchase.  
- Credits are **non-refundable** (see ADR-XXX).  

---

## Transparency

- Balance is always visible through dashboard or API:  
  - Example: `GET /v1/account` → `{ "credits_remaining": 12345678 }`  
- Service halts when balance reaches 0.  

---

## Positioning

- **Predictable:** no invoices, no surprise bills.  
- **Accessible:** low entry with €50 starter pack.  
- **Fair:** credits last 12 months, plenty of time to use them.  
- **Open:** all usage on top of OSS models, fully inspectable.  

---

## Next Steps

- Decide on exact pack sizes after GPU cost benchmarking (RunPod/LambdaLabs rates + margin).  
- Implement dashboard/API balance endpoint.  
- Draft ToS language for credit validity and non-refundability.  
