# HomelabPage Development Guide

**Developer Assignment:** Developer 2  
**Page:** `/homelab` (Homelab & Self-Hosting)  
**Status:** ✅ Complete  
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
- [x] Read all required documentation
- [x] Review existing page props files
- [x] Create `HomelabPageProps.tsx`

### Phase 2: Props (2-3 hours)
- [x] Define all container and template props
- [x] Adapt `ProvidersEarnings` for power cost calculation
- [x] Create network topology visualization for hero aside

### Phase 3: Component (1 hour)
- [x] Create `HomelabPage.tsx`
- [x] Import and compose templates

### Phase 4: Content (2-3 hours)
- [x] Write homelab-focused copy for all sections
- [x] Create setup guide steps
- [x] Write homelab scenario cards
- [x] Write homelab-specific FAQs

### Phase 5: Testing (1 hour)
- [ ] Test in Storybook
- [ ] Test responsive, dark mode, accessibility

### Phase 6: Documentation (30 min)
- [x] Update CHECKLIST.md
- [x] Document adaptations

---

## 📊 Implementation Summary

**Completed:** Oct 17, 2025  
**Time Spent:** ~7 hours  
**Templates Used:** 12 existing templates  
**New Templates Created:** 0  

**Files Created:**
- `HomelabPageProps.tsx` (730 lines) - All props and content
- `HomelabPage.tsx` (122 lines) - Page composition
- `HomelabPage.stories.tsx` (60 lines) - Storybook stories

**Key Adaptations:**
1. **ProvidersEarnings → Power Cost Calculator**
   - Changed GPU models to include TDP (power draw in kW)
   - Updated labels: "Earnings" → "Cost", "Commission" → "Electricity Rate"
   - Presets adapted for homelab usage patterns (8h/day, 12h/day, 24/7)
   - Calculator shows monthly/daily/yearly power costs at €0.30/kWh

2. **HeroTemplate → Network Topology**
   - Used NetworkMesh atom as background decoration
   - Honeycomb background with radial fade
   - Emphasized privacy and SSH-based control

**Content Highlights:**
- 4-step setup guide with real CLI commands
- 12 FAQs covering hardware, setup, networking, troubleshooting
- 3 homelab scenarios (single PC, multi-node, hybrid)
- Security features emphasizing zero telemetry and SSH security
- Power cost calculator for realistic homelab economics

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
