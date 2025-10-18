# DevOpsPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/devops` (DevOps & Production)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the DevOps page showcasing rbee for production AI infrastructure and operations.

**Target Audience:** DevOps engineers, SREs, platform engineers, infrastructure teams

**Key Message:** Production-ready orchestration. SSH-first lifecycle. No orphaned workers. Clean shutdowns.

---

## 🔄 Key Template Adaptations for DevOps

### Hero
- ✅ `EnterpriseHero` adapted - Deployment console showing worker status

### Problem: Production Chaos
- ✅ `ProblemTemplate` - Orphaned processes, VRAM leaks, manual cleanup, no observability

### Solution: Production-Grade Orchestration
- ✅ `SolutionTemplate` - Cascading shutdown, process isolation, health checks

### Deployment Process
- ✅ `EnterpriseHowItWorks` - Deployment steps (provision → configure → deploy → monitor)

### Operational Features
- ✅ `EnterpriseSecurity` adapted - 6 ops features (SSH lifecycle, health checks, metrics, logging, etc.)

### Error Handling
- ✅ `ErrorHandlingTemplate` - Resilience and recovery

### Real-Time Monitoring
- ✅ `RealTimeProgress` - Live worker status, metrics streaming

### Infrastructure Options
- ✅ `ComparisonTemplate` - Deployment options (on-prem, cloud, hybrid)

### SLAs & Guarantees
- ✅ `EnterpriseCompliance` adapted - Operational guarantees (uptime, response time, support)

---

## 📐 Proposed Structure

```tsx
<DevOpsPage>
  <EnterpriseHero /> {/* Deployment console */}
  <EmailCapture /> {/* "Get Deployment Guide" */}
  <ProblemTemplate /> {/* Production chaos */}
  <SolutionTemplate /> {/* Production-grade */}
  <EnterpriseHowItWorks /> {/* Deployment process */}
  <EnterpriseSecurity /> {/* Ops features */}
  <ErrorHandlingTemplate /> {/* Error handling */}
  <RealTimeProgress /> {/* Monitoring */}
  <ComparisonTemplate /> {/* Deployment options */}
  <EnterpriseCompliance /> {/* SLAs */}
  <FAQTemplate /> {/* DevOps FAQs */}
  <CTATemplate /> {/* "Deploy Now" */}
</DevOpsPage>
```

---

## ✅ Implementation Checklist

- [ ] Read TEMPLATE_CATALOG.md
- [ ] Create `DevOpsPageProps.tsx`
- [ ] Adapt templates for DevOps context
- [ ] Write ops-focused copy (emphasize reliability, observability, clean lifecycle)
- [ ] Create `DevOpsPage.tsx`
- [ ] Test and document

---

**Key Message:** Show them the operational guarantees. DevOps cares about reliability, observability, and clean lifecycle management.
