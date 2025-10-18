# StartupsPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/startups` (Startups & Small Teams)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Startups page showcasing rbee for startups and small teams building AI products.

**Target Audience:** Startup founders, small dev teams, bootstrapped companies

**Key Message:** Build AI products without burning cash on API fees. Own your infrastructure from day one.

---

## 🔄 Key Template Adaptations for Startups

### Hero
- ✅ `HeroTemplate` with cost savings visualization
- ✅ `ProvidersHero` adapted (earnings → savings calculator)

### Problem: API Cost Spiral
- ✅ `ProblemTemplate` - Escalating costs, rate limits, vendor lock-in

### Solution: Own Your Stack
- ✅ `SolutionTemplate` - Cost control, scalability, independence

### ROI Calculator
- ✅ `ProvidersEarnings` adapted - Calculate savings (API costs vs self-hosted)
  - Input: Team size, API usage
  - Output: Monthly/yearly savings

### Growth Stages
- ✅ `EnterpriseHowItWorks` adapted - Startup growth roadmap (MVP → Scale → Enterprise)
- ✅ `PricingTemplate` adapted - Growth tiers (Solo → Team → Scale)

### Startup Scenarios
- ✅ `UseCasesTemplate` - Different startup types (B2B SaaS, consumer app, AI-first)

### Technical Stack
- ✅ `TechnicalTemplate` - Architecture for startups

### Comparison
- ✅ `ComparisonTemplate` - rbee vs API providers (cost, control, scalability)

---

## 📐 Proposed Structure

```tsx
<StartupsPage>
  <HeroTemplate /> {/* Cost savings focus */}
  <EmailCapture /> {/* "Start Free Trial" */}
  <ProblemTemplate /> {/* API cost spiral */}
  <SolutionTemplate /> {/* Own your stack */}
  <ProvidersEarnings /> {/* ROI calculator */}
  <EnterpriseHowItWorks /> {/* Growth roadmap */}
  <UseCasesTemplate /> {/* Startup scenarios */}
  <ComparisonTemplate /> {/* vs API providers */}
  <TechnicalTemplate /> {/* Tech stack */}
  <TestimonialsTemplate /> {/* Founder stories */}
  <FAQTemplate /> {/* Startup FAQs */}
  <CTATemplate /> {/* "Start Building" */}
</StartupsPage>
```

---

## 💰 ROI Calculator Adaptation

**Adapt `ProvidersEarnings` to calculate savings:**

```tsx
// Instead of: GPU × availability × rate = earnings
// Use: API cost - self-hosted cost = savings

calculatorProps = {
  title: "Calculate Your Savings",
  inputs: [
    { type: "select", label: "Team Size", options: ["1-5", "6-15", "16-50"] },
    { type: "slider", label: "API Requests/Month", min: 10000, max: 10000000 },
    { type: "select", label: "Current Provider", options: ["OpenAI", "Anthropic", "Both"] }
  ],
  calculation: (teamSize, requests, provider) => {
    const apiCost = calculateAPIcost(requests, provider)
    const selfHostedCost = calculateSelfHostedCost(teamSize)
    return apiCost - selfHostedCost
  },
  output: {
    monthly: "$X,XXX saved/month",
    yearly: "$XX,XXX saved/year",
    breakdown: "Detailed cost comparison"
  }
}
```

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `StartupsPageProps.tsx`
- [ ] Adapt `ProvidersEarnings` for ROI calculation
- [ ] Write startup-focused copy (emphasize cost savings, independence, scalability)
- [ ] Create `StartupsPage.tsx`
- [ ] Test and document

---

**Key Message:** Show the math. Startups care about ROI. Use the calculator to prove the value prop.
