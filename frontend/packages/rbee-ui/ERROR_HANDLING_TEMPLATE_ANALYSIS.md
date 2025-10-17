# ErrorHandlingTemplate Analysis & Consolidation Opportunities

**Created:** October 17, 2025  
**Template:** `ErrorHandlingTemplate`  
**Current Usage:** 3 pages (FeaturesPage, DevOpsPage, SecurityPage)

## Template Structure

The ErrorHandlingTemplate is a **highly structured, three-section template**:

```tsx
<ErrorHandlingTemplate
  statusKPIs={[...]}           // 1. Top: 3-column KPI grid
  terminalContent={...}         // 2. Middle: Terminal window with timeline
  terminalFooter={...}          // 3. Bottom: Expandable playbook with categories
  playbookCategories={[...]}
/>
```

### Section Breakdown

1. **Status KPIs** (3-column grid)
   - Uses `StatusKPI` molecule
   - Icon + Label + Value
   - Color variants: `chart-3`, `primary`, `chart-2`

2. **Terminal Window** (timeline/log viewer)
   - Uses `TerminalWindow` molecule
   - Shows chronological events with timestamps
   - Optional footer text
   - Title: "error timeline — retries & jitter"

3. **Playbook** (expandable categories)
   - Uses `PlaybookHeader` + `PlaybookItem` molecules
   - Categories with severity indicators
   - Nested checks with action links
   - Expand/collapse all functionality

## Templates That Could Be Replaced

### ✅ **1. RealTimeProgress** - STRONG CANDIDATE

**Current Structure:**
```
- Card with IconCardHeader + TerminalWindow
- 3-column StatusKPI grid (with progress bars)
- Timeline steps
```

**Overlap with ErrorHandlingTemplate:**
- ✅ Uses StatusKPI (same molecule)
- ✅ Uses TerminalWindow (same molecule)
- ✅ 3-column grid layout
- ✅ Timeline/chronological events
- ❌ Has progress bars (extra feature)
- ❌ Has timeline steps (different from playbook)

**Consolidation Strategy:**
Make ErrorHandlingTemplate more flexible:
- Add optional `progressBars` prop to StatusKPIs
- Replace playbook with optional `timeline` or `playbook` sections
- **Estimated savings:** 150-200 lines

**Verdict:** **HIGH PRIORITY** - 70% structural overlap

---

### ✅ **2. TechnicalTemplate** - MODERATE CANDIDATE

**Current Structure:**
```
- Title section
- Terminal window with code
- Grid of feature cards
```

**Overlap with ErrorHandlingTemplate:**
- ✅ Uses TerminalWindow
- ✅ Grid layout below terminal
- ❌ No StatusKPIs
- ❌ Uses feature cards instead of playbook

**Consolidation Strategy:**
- Make StatusKPIs optional
- Replace playbook with generic `bottomSection` slot
- **Estimated savings:** 80-100 lines

**Verdict:** **MEDIUM PRIORITY** - 50% structural overlap

---

### ✅ **3. MultiBackendGpuTemplate** - MODERATE CANDIDATE

**Current Structure:**
```
- Card with IconCardHeader
- TerminalWindow (detection console)
- 3-column microcard grid
```

**Overlap with ErrorHandlingTemplate:**
- ✅ Uses TerminalWindow
- ✅ 3-column grid layout
- ❌ No StatusKPIs (uses custom microcards)
- ❌ No playbook section

**Consolidation Strategy:**
- Make StatusKPIs optional or allow custom grid content
- Make playbook section optional
- **Estimated savings:** 60-80 lines

**Verdict:** **MEDIUM PRIORITY** - 45% structural overlap

---

### ⚠️ **4. CrossNodeOrchestration** - WEAK CANDIDATE

**Current Structure:**
```
- Card with IconCardHeader + TerminalWindow
- 3-column benefits grid
- Separator + architecture diagram
```

**Overlap with ErrorHandlingTemplate:**
- ✅ Uses TerminalWindow
- ✅ 3-column grid
- ❌ No StatusKPIs
- ❌ Has architecture diagram section

**Consolidation Strategy:**
- Would require too many optional sections
- Better to keep separate
- **Estimated savings:** 20-30 lines (not worth it)

**Verdict:** **LOW PRIORITY** - 30% overlap, too specialized

---

## Templates Using Similar Molecules (Not Replaceable)

### ❌ **HowItWorks** - NOT REPLACEABLE
- Uses TerminalWindow but in step-by-step format
- Completely different structure (numbered steps)
- No KPIs or playbook concept

### ❌ **DevelopersHero** - NOT REPLACEABLE
- Uses TerminalWindow as hero aside
- Part of hero template pattern
- No grid or playbook structure

### ❌ **HomeHero** - NOT REPLACEABLE
- Uses TerminalWindow with floating KPI
- Hero template pattern
- Different purpose entirely

### ❌ **FeaturesTabs** - NOT REPLACEABLE
- Uses TerminalWindow inside tabs
- Tab-based navigation
- Completely different interaction model

---

## Recommended Consolidation Plan

### Phase 1: Make ErrorHandlingTemplate More Flexible (2-3 days)

**Goal:** Create a generalized "Technical Showcase Template" that can replace multiple templates

**Changes:**
```tsx
export interface TechnicalShowcaseTemplateProps {
  // Top section (optional)
  statusKPIs?: StatusKPIData[]
  showProgressBars?: boolean
  
  // Middle section (required)
  terminalTitle?: string
  terminalContent: ReactNode
  terminalFooter?: ReactNode
  
  // Bottom section (flexible)
  bottomSection: {
    type: 'playbook' | 'timeline' | 'grid' | 'custom'
    content: PlaybookCategory[] | TimelineStep[] | ReactNode
  }
  
  className?: string
}
```

**Benefits:**
- Replaces ErrorHandlingTemplate, RealTimeProgress, TechnicalTemplate
- Reduces 3 templates → 1 flexible template
- **Estimated savings:** 300-400 lines

### Phase 2: Consolidate RealTimeProgress (1 day)

**Action:**
- Migrate RealTimeProgress usages to new TechnicalShowcaseTemplate
- Update DevOpsPage, FeaturesPage
- Delete RealTimeProgress template

**Estimated savings:** 150-200 lines

### Phase 3: Consolidate TechnicalTemplate (1 day)

**Action:**
- Migrate TechnicalTemplate usages to TechnicalShowcaseTemplate
- Update SecurityPage, other pages
- Delete TechnicalTemplate

**Estimated savings:** 80-100 lines

---

## Summary

### Templates Replaceable by ErrorHandlingTemplate

| Template | Overlap | Priority | Lines Saved | Effort |
|----------|---------|----------|-------------|--------|
| **RealTimeProgress** | 70% | HIGH | 150-200 | 1 day |
| **TechnicalTemplate** | 50% | MEDIUM | 80-100 | 1 day |
| **MultiBackendGpuTemplate** | 45% | MEDIUM | 60-80 | 1 day |
| **CrossNodeOrchestration** | 30% | LOW | 20-30 | Not worth it |

### Total Impact

- **Templates consolidated:** 3 → 1
- **Lines of code removed:** 290-380 lines
- **Effort:** 3-4 days
- **Maintenance reduction:** 66% fewer technical showcase templates

### Recommendation

**Rename ErrorHandlingTemplate → TechnicalShowcaseTemplate** and make it the canonical pattern for:
- Error handling & resilience (current use)
- Real-time monitoring & progress (RealTimeProgress)
- Technical architecture demos (TechnicalTemplate)
- Any "KPIs + Terminal + Detailed Info" pattern

This creates a **reusable pattern** for technical content that combines:
1. Metrics/KPIs at the top
2. Live/example output in the middle (terminal)
3. Detailed breakdown at the bottom (playbook/timeline/grid)

---

## Implementation Notes

### Breaking Changes
- RealTimeProgress users need migration
- TechnicalTemplate users need migration
- Update all page imports

### Non-Breaking Approach
1. Create TechnicalShowcaseTemplate as new component
2. Make ErrorHandlingTemplate a wrapper around TechnicalShowcaseTemplate
3. Gradually migrate pages
4. Deprecate old templates
5. Remove after migration complete

### Testing Strategy
- Visual regression tests for all migrated pages
- Storybook stories for all variants
- Accessibility testing (keyboard navigation, ARIA)
