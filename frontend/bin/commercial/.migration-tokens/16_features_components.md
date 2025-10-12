# Work Unit 16: Features Page Components

**Priority:** LOW  
**Directory:** `/components/features/`

**Files:**
- `features-hero.tsx`
- `core-features-tabs.tsx`
- `cross-node-orchestration.tsx`
- `intelligent-model-management.tsx`
- `multi-backend-gpu.tsx`
- `real-time-progress.tsx`
- `error-handling.tsx`
- `security-isolation.tsx`
- `additional-features-grid.tsx`

---

## Migration Strategy

The features page components are technical deep-dives into rbee's capabilities. Apply the same token replacements with a focus on technical clarity.

### Common Patterns

| Current Pattern | Token Replacement |
|----------------|-------------------|
| `bg-slate-50` | `bg-secondary` |
| `bg-white` | `bg-background` or `bg-card` |
| `bg-slate-900` | `bg-background` (dark mode) or `bg-card` |
| `text-slate-900` | `text-foreground` or `text-card-foreground` |
| `text-slate-600` | `text-muted-foreground` |
| `text-slate-300` | `text-muted-foreground` |
| `text-amber-500/600` | `text-primary` |
| `bg-amber-500/600` | `bg-primary` |
| `text-blue-400/500/600` | `text-chart-2` |
| `bg-blue-100` | `bg-chart-2/10` |
| `text-green-400/500/600` | `text-chart-3` |
| `bg-green-100` | `bg-chart-3/10` |
| `text-purple-400/500` | `text-chart-4` |
| `text-red-400/500` | `text-destructive` |
| `border-slate-200` | `border-border` |

### Component-Specific Notes

#### features-hero.tsx
- Minimal hero → `bg-background`
- Heading → `text-foreground`
- Description → `text-muted-foreground`

#### core-features-tabs.tsx
- Similar to main `features-section.tsx`
- Tab content → `bg-card border-border`
- Code blocks → `bg-muted` with syntax highlighting
- Callout boxes → Use semantic colors

#### cross-node-orchestration.tsx
- Network diagrams → Use `bg-card` for nodes
- Connection lines → `border-border`
- Active nodes → `bg-primary`
- Inactive nodes → `bg-muted`
- Data flow indicators → `text-chart-2` (blue)

#### intelligent-model-management.tsx
- Model cards → `bg-card border-border`
- Download progress → `bg-primary` for fill
- Status indicators:
  - Ready → `text-chart-3` (green)
  - Loading → `text-chart-2` (blue)
  - Error → `text-destructive` (red)

#### multi-backend-gpu.tsx
- GPU cards → `bg-card border-border`
- Backend badges:
  - CUDA → `bg-chart-3/10 text-chart-3` (green)
  - Metal → `bg-chart-2/10 text-chart-2` (blue)
  - CPU → `bg-muted text-muted-foreground`
- Utilization bars → `bg-primary` for fill

#### real-time-progress.tsx
- Progress timeline → `bg-card border-border`
- Event indicators:
  - Success → `text-chart-3` (green)
  - In Progress → `text-chart-2` (blue)
  - Pending → `text-muted-foreground`
- SSE stream → `bg-muted` with syntax highlighting

#### error-handling.tsx
- Error cards → `bg-destructive/10 border-destructive/20`
- Error messages → `text-destructive`
- Recovery steps → `text-chart-3` (green)
- Code blocks → `bg-muted`

#### security-isolation.tsx
- Security features → `bg-chart-2/10 border-chart-2/20` (blue)
- Isolation diagram → Use `bg-card` for containers
- Sandbox indicators → `text-chart-2`
- Threat indicators → `text-destructive`

#### additional-features-grid.tsx
- Feature cards → `bg-card border-border`
- Icon backgrounds → Use semantic colors
- Feature categories:
  - Performance → `text-primary`
  - Security → `text-chart-2`
  - Reliability → `text-chart-3`

---

## Implementation Approach

### Step 1: Read Each File
Read each component file to identify hardcoded colors.

### Step 2: Apply Pattern Matching
Use the common patterns table above to replace colors systematically.

### Step 3: Preserve Technical Semantics
Ensure semantic colors are preserved:
- **Blue:** Information, data flow, in-progress
- **Green:** Success, ready, active
- **Amber:** Primary features, highlights
- **Red:** Errors, failures, warnings
- **Purple:** Keywords, special states

### Step 4: Technical Clarity
Features pages should prioritize:
- **Clear status indicators:** Green=ready, Blue=loading, Red=error
- **Readable code blocks:** Good syntax highlighting
- **Visible diagrams:** Clear node/connection colors
- **Scannable cards:** Good contrast, clear hierarchy

### Step 5: Test Each Component
Verify each component in both light and dark modes.

---

## Verification Checklist

For each component:
- [ ] Renders correctly in light mode
- [ ] Renders correctly in dark mode
- [ ] Status indicators are clear (green/blue/red)
- [ ] Code blocks have good syntax highlighting
- [ ] Diagrams are readable
- [ ] Progress bars are visible
- [ ] Error states are prominent
- [ ] Feature cards are scannable
- [ ] Icons are visible
- [ ] No hardcoded `slate-*`, `amber-*`, `blue-*`, `green-*`, `red-*`, `purple-*` classes remain

---

## Estimated Complexity

**Medium-High** - 9 components with technical diagrams, status indicators, and complex visualizations. Requires careful attention to semantic colors.

---

## Notes

### Status Indicator Colors

Use consistent colors for status across all components:
- **Ready/Success:** Green (`chart-3`)
- **Loading/In Progress:** Blue (`chart-2`)
- **Error/Failed:** Red (`destructive`)
- **Pending/Idle:** Muted (`muted-foreground`)

### Code Block Syntax

Use consistent syntax highlighting:
- **Keywords:** Purple (`chart-4`)
- **Strings:** Green (`chart-3`)
- **Functions:** Blue (`chart-2`)
- **Comments:** Muted (`muted-foreground`)
- **Text:** Foreground (`foreground`)

### Diagram Colors

For network/architecture diagrams:
- **Active nodes:** Primary (`primary`)
- **Inactive nodes:** Muted (`muted`)
- **Connections:** Border (`border`)
- **Data flow:** Blue (`chart-2`)
- **Errors:** Red (`destructive`)

### Progress Indicators

For progress bars and loading states:
- **Fill:** Primary (`primary`)
- **Background:** Muted (`muted`)
- **Complete:** Green (`chart-3`)
- **Failed:** Red (`destructive`)
