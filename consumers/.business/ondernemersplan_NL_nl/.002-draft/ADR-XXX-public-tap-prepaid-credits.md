# ADR-XXX: Public Tap Prepaid Credits (Non-Refundable)

## Status

DRAFT

## Context

The **Public Tap** is the shared, metered API offering on top of curated OSS AI models.  
As a solo operator, cash flow predictability and billing simplicity are critical.  
Traditional postpaid, metered billing (per token, invoiced monthly) introduces risks:

- Cash flow gaps.
- Potential disputes/refunds.
- Unexpected GPU bills without guaranteed customer payment.

Industry peers (Hugging Face Inference, RunPod, LambdaLabs) already use **prepaid, non-refundable credit systems** for API usage and GPU rental. This pattern is familiar to target customers (developers, IT teams, agencies).

## Decision

- **Public Tap usage will be prepaid.**
- Customers purchase credits (e.g. €50, €200, €500 packs).
- Credits are **non-refundable** once purchased.
- Credits have a long shelf life (e.g. 12 months) to maximize fairness.
- Usage decrements balance directly; when credits reach 0, service halts until more are purchased.
- Balance visibility will be provided via dashboard or simple API call.

## Rationale

- **Predictable cash flow** for the operator.
- **No credit risk** (upfront payment).
- **Transparency** for customers (no surprise bills).
- **Alignment with industry practices** — prepaid infra is common and accepted.
- **Simplicity** — no need to integrate complex metered billing systems initially.

## Consequences

- Customers cannot request refunds for unused credits (must be clearly stated in ToS).  
- Exceptional goodwill refunds remain possible if failures occur due to operator error.  
- Positioning must emphasize **control and predictability** to frame non-refundability positively.  
- A lightweight accounting/dashboard system must be implemented to track balances accurately.

## Alternatives Considered

- **Postpaid metering (per token, invoice):** rejected due to cash flow and credit risks.  
- **Refundable prepaid packs:** rejected to avoid administrative overhead and disputes.  
- **Subscription plans (flat monthly):** rejected for early stage; prepaid aligns better with unpredictable demand.

## Next Steps

- Draft Terms of Service snippet covering non-refundability and credit shelf life.  
- Define credit pack pricing and token conversion.  
- Implement credit balance tracking in Public Tap API.  
