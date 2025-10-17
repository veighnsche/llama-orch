# HomelabPage Development Guide

**Developer Assignment:** [Your Name Here]  
**Page:** `/homelab` (Homelab & Self-Hosting)  
**Status:** 🔴 Not Started  
**Last Updated:** Oct 17, 2025

---

## 🎯 Mission

Build the Homelab page showcasing rbee for homelab enthusiasts, self-hosters, and tinkerers.

**Target Audience:** Homelab enthusiasts, self-hosters, hardware tinkerers, privacy advocates

**Key Message:** Self-hosted LLMs across all your machines. Turn idle hardware into productive AI infrastructure.

---

## 📋 Required Reading

1. ✅ **TEMPLATE_CATALOG.md** - Template inventory with reusability analysis
2. ✅ **CONSOLIDATION_OPPORTUNITIES.md** - Consolidation guidelines
3. ✅ **CHECKLIST.md** (this directory) - Content requirements
4. ✅ **ResearchPage/PAGE_DEVELOPMENT_GUIDE.md** - Reference example

---

## 🔄 Template Reusability for Homelab

### Hero
- ✅ `HeroTemplate` with custom aside (network topology)
- ✅ `HomeHero` adapted (terminal showing multi-machine setup)
- ✅ `EnterpriseHero` adapted (console showing node status)

**Recommendation:** `HeroTemplate` with network topology visualization

### Problem: Homelab Complexity
- ✅ `ProblemTemplate` - Pain points (scattered GPUs, manual setup, no orchestration)

### Solution: Unified Orchestration
- ✅ `SolutionTemplate` - Feature cards + topology diagram

### Multi-Machine Setup
- ✅ `HowItWorks` - Step-by-step setup guide
- ✅ `CrossNodeOrchestration` - Multi-node visualization

**Recommendation:** Both - `HowItWorks` for steps, `CrossNodeOrchestration` for visualization

### Hardware Support
- ✅ `MultiBackendGpuTemplate` - CUDA, Metal, CPU support
- ✅ `AdditionalFeaturesGrid` - Hardware compatibility cards

**Recommendation:** `MultiBackendGpuTemplate`

### Power Cost Calculator
- ✅ `ProvidersEarnings` adapted - Calculate power costs (GPUs × electricity rate × hours)

**Recommendation:** `ProvidersEarnings` with power cost calculation

### Homelab Scenarios
- ✅ `UseCasesTemplate` - Different homelab setups (single PC, multi-node, hybrid)

### Security & Privacy
- ✅ `SecurityIsolation` - Process isolation, network security
- ✅ `EnterpriseSecurity` - Security features grid

**Recommendation:** `SecurityIsolation`

### CTA
- ✅ `EmailCapture` - "Get Homelab Setup Guide"
- ✅ `CTATemplate` - Final CTA

---

## 📐 Proposed Structure

```tsx
<HomelabPage>
  <HeroTemplate /> {/* Network topology aside */}
  <EmailCapture /> {/* Setup guide */}
  <ProblemTemplate /> {/* Homelab complexity */}
  <SolutionTemplate /> {/* Unified orchestration */}
  <HowItWorks /> {/* Setup steps */}
  <CrossNodeOrchestration /> {/* Multi-node viz */}
  <MultiBackendGpuTemplate /> {/* Hardware support */}
  <ProvidersEarnings /> {/* Power cost calculator */}
  <UseCasesTemplate /> {/* Homelab scenarios */}
  <SecurityIsolation /> {/* Security features */}
  <FAQTemplate /> {/* Homelab FAQs */}
  <CTATemplate /> {/* Final CTA */}
</HomelabPage>
```

**Total templates:** 11 existing templates, zero new templates needed

---

## ✅ Implementation Checklist

### Phase 1: Setup (30 min)
- [ ] Read all required documentation
- [ ] Review existing page props files
- [ ] Create `HomelabPageProps.tsx`

### Phase 2: Props (2-3 hours)
- [ ] Define all container and template props
- [ ] Adapt `ProvidersEarnings` for power cost calculation
- [ ] Create network topology visualization for hero aside

### Phase 3: Component (1 hour)
- [ ] Create `HomelabPage.tsx`
- [ ] Import and compose templates

### Phase 4: Content (2-3 hours)
- [ ] Write homelab-focused copy for all sections
- [ ] Create setup guide steps
- [ ] Write homelab scenario cards
- [ ] Write homelab-specific FAQs

### Phase 5: Testing (1 hour)
- [ ] Test in Storybook
- [ ] Test responsive, dark mode, accessibility

### Phase 6: Documentation (30 min)
- [ ] Update CHECKLIST.md
- [ ] Document adaptations

---

## 🎨 Homelab-Specific Considerations

**Tone:** Enthusiast-friendly, technical but approachable, emphasize control and privacy

**Visuals:** Network diagrams, hardware illustrations, terminal outputs

**Key Messages:**
- Use all your hardware (gaming PCs, old workstations, Macs)
- Zero cloud dependencies
- Complete privacy
- SSH-first lifecycle
- No orphaned processes

---

**Remember:** Homelab users love technical details. Show them the architecture, the SSH commands, the network topology. Use `HowItWorks` extensively!
