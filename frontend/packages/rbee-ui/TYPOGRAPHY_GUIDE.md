# rbee Typography Guide: Sans-Serif vs Serif

## 🎯 Quick Reference

**Default:** Serif (IBM Plex Serif)  
**Override with:** `font-sans` (Geist Sans)  
**Monospace:** `font-mono` (Geist Mono)

---

## 📐 The Rule

### **Use SERIF (default) for:**
- ✅ **All body text** - paragraphs, descriptions, long-form content
- ✅ **Headings** (h1-h6) - titles, section headers, hero headlines
- ✅ **Feature titles** - the bold part of feature lists
- ✅ **Card content** - main text in cards
- ✅ **Testimonial quotes** - the actual quote text
- ✅ **Author names** - citation text
- ✅ **Values in key-value pairs** - the emphasized part

### **Use SANS-SERIF (`font-sans`) for:**
- ✅ **UI controls** - buttons, badges, form inputs
- ✅ **Labels** - the key in key-value pairs, form labels, meta text
- ✅ **Technical/system text** - architecture notes, technical specs
- ✅ **Uppercase labels** - "CHOOSE YOUR PATH", "OPEN-SOURCE"
- ✅ **Keyboard shortcuts** - `<Kbd>` component
- ✅ **Small meta text** - timestamps, secondary info, captions
- ✅ **Descriptions in feature lists** - the non-bold explanatory text
- ✅ **Sublabels** - secondary text under main labels

---

## 🔍 Component-by-Component Breakdown

### **Atoms (UI Controls)**

| Component | Font | Rationale |
|-----------|------|-----------|
| `Button` | **Sans** | UI control - needs clean, functional appearance |
| `Badge` | **Sans** | UI label - compact, scannable |
| `Kbd` | **Sans** | Technical/system element |
| `KeyValuePair` label | **Sans** | Label/metadata |
| `KeyValuePair` value | **Serif** (default) | Emphasized content |

### **Molecules (Composite Elements)**

| Component | Element | Font | Example |
|-----------|---------|------|---------|
| `FeatureListItem` | Title | **Serif** (default) | "Independence" |
| `FeatureListItem` | Description | **Serif** (default) | "Build on your hardware..." |
| `PulseBadge` | Text | **Sans** | "Live", "Active" |
| `BulletListItem` | Title | **Serif** (default) | "GPU Scheduling" |
| `BulletListItem` | Description | **Sans** | "Automatically routes..." |
| `BulletListItem` | Meta | **Sans** | "2 min read" |
| `CTAOptionCard` | Body | **Sans** | Instructional text |
| `CTAOptionCard` | Note | **Sans** | Fine print |
| `BeeArchitecture` | Labels | **Serif** (default) | "Queen", "Worker" |
| `BeeArchitecture` | Sublabels | **Sans** | "Orchestrator", "GPU Node" |

### **Organisms (Sections)**

| Section | Element | Font | Why |
|---------|---------|------|-----|
| `WhatIsRbee` | Heading (h2) | **Serif** (default) | Main content heading |
| `WhatIsRbee` | Subhead paragraph | **Serif** (default) | Body text |
| `WhatIsRbee` | Feature titles | **Serif** (default) | Content emphasis |
| `WhatIsRbee` | Feature descriptions | **Serif** (default) | Body text |
| `WhatIsRbee` | Technical accent | **Sans** | Technical/system note |
| `WhatIsRbee` | Badge | **Sans** | UI element |
| `SocialProofSection` | Heading (h2) | **Serif** (default) | Main heading |
| `SocialProofSection` | Quote text | **Serif** (default) | Body content |
| `SocialProofSection` | Opening quote mark | **Serif** (explicit) | Decorative typography |
| `SocialProofSection` | Author name | **Serif** (default) | Content emphasis |
| `AudienceSelector` | Uppercase label | **Sans** | UI label/eyebrow |
| `AudienceSelector` | Heading (h2) | **Sans** | Exception - UI-focused section |
| `AudienceSelector` | Description | **Sans** | Matches heading style |

---

## 🎨 Design Philosophy

### **Serif = Content & Emotion**
IBM Plex Serif brings:
- **Warmth** - approachable, human
- **Authority** - trustworthy, established
- **Readability** - optimized for long-form reading
- **Elegance** - sophisticated brand presence

Use serif when you want the user to **read, understand, and feel**.

### **Sans-Serif = Function & Clarity**
Geist Sans brings:
- **Clarity** - clean, unambiguous
- **Efficiency** - quick scanning
- **Modernity** - technical, precise
- **Neutrality** - doesn't compete with content

Use sans-serif when you want the user to **scan, click, and act**.

---

## ❌ Common Mistakes (What You Were Doing Wrong)

### **Mistake 1: Sans-serif on feature descriptions**
```tsx
// ❌ WRONG
<div className="text-sm text-muted-foreground font-sans">
  {description}
</div>

// ✅ CORRECT (remove font-sans)
<div className="text-sm text-muted-foreground">
  {description}
</div>
```
**Why:** Descriptions are body content, not UI labels. Let them inherit serif.

### **Mistake 2: Sans-serif on headings (usually)**
```tsx
// ❌ WRONG (in most cases)
<h2 className="font-sans text-3xl">
  What is rbee?
</h2>

// ✅ CORRECT (remove font-sans)
<h2 className="text-3xl">
  What is rbee?
</h2>
```
**Why:** Headings are content hierarchy, not UI elements. Exception: `AudienceSelector` uses sans for a more UI-focused, action-oriented feel.

### **Mistake 3: Serif on technical/system notes**
```tsx
// ❌ WRONG (missing font-sans)
<p className="text-xs text-muted-foreground">
  Architecture at a glance: Smart/Dumb separation
</p>

// ✅ CORRECT
<p className="text-xs text-muted-foreground font-sans">
  Architecture at a glance: Smart/Dumb separation
</p>
```
**Why:** Technical specs are system information, not narrative content.

### **Mistake 4: Serif on uppercase labels**
```tsx
// ❌ WRONG (missing font-sans)
<p className="text-sm uppercase tracking-wider text-primary">
  Choose your path
</p>

// ✅ CORRECT
<p className="text-sm font-sans uppercase tracking-wider text-primary">
  Choose your path
</p>
```
**Why:** Uppercase labels are UI signifiers, not content.

---

## 🧪 Decision Tree

When adding text, ask yourself:

```
Is this text...

├─ A UI control? (button, badge, input)
│  └─ ✅ Use font-sans
│
├─ A label or metadata? (key in key-value, timestamp, caption)
│  └─ ✅ Use font-sans
│
├─ Technical/system information? (specs, architecture notes)
│  └─ ✅ Use font-sans
│
├─ Uppercase/eyebrow text?
│  └─ ✅ Use font-sans
│
├─ A heading or body text?
│  └─ ✅ Use serif (default, no class needed)
│
├─ A feature title or description?
│  └─ ✅ Use serif (default, no class needed)
│
├─ A quote or testimonial?
│  └─ ✅ Use serif (default, no class needed)
│
└─ Not sure?
   └─ ✅ Use serif (default) - it's the safer choice
```

---

## 📊 Real Examples from Codebase

### **Correct Usage**

#### `WhatIsRbee.tsx` - Line 136
```tsx
<p className="text-xs text-muted-foreground pt-2 font-sans">
  <strong>Architecture at a glance:</strong> Smart/Dumb separation
</p>
```
✅ **Correct:** Technical note = sans-serif

#### `FeatureListItem.tsx` - Line 45-46
```tsx
<div className="text-base text-foreground">
  <strong className="font-semibold">{title}:</strong> {description}
</div>
```
✅ **Correct:** Both title and description use default serif (no font-sans)

#### `Button.tsx` - Line 7
```tsx
"... text-sm font-sans font-medium ..."
```
✅ **Correct:** UI control = sans-serif

#### `Badge.tsx` - Line 7
```tsx
"... text-xs font-sans font-medium ..."
```
✅ **Correct:** UI label = sans-serif

#### `KeyValuePair.tsx` - Line 42
```tsx
<span className="text-muted-foreground font-sans">{label}</span>
<span className={cn(valueVariants[valueVariant])}>{value}</span>
```
✅ **Correct:** Label = sans, Value = serif (default)

#### `BulletListItem.tsx` - Lines 109, 115
```tsx
<div className="text-xs text-muted-foreground whitespace-nowrap font-sans">
  {meta}
</div>
<div className="text-sm text-muted-foreground font-sans">
  {description}
</div>
```
✅ **Correct:** Meta and descriptions are secondary/explanatory = sans-serif

#### `SocialProofSection.tsx` - Line 115
```tsx
<span className="absolute -left-1 -top-2 text-5xl font-serif leading-none text-primary/40">
  &ldquo;
</span>
```
✅ **Correct:** Explicit `font-serif` for decorative quote mark (though default would work)

---

## 🚀 Quick Fixes for Your Current Code

Based on the grep results, here are components that might need review:

### **Check These:**

1. **`AudienceSelector.tsx`** - Lines 27, 29, 33
   - ✅ **Correct as-is:** This section uses sans throughout for a UI-focused feel

2. **`EnterpriseCTA.tsx`** - Line 22
   - ✅ **Correct:** Uppercase label = sans

3. **`CTAOptionCard.tsx`** - Lines 99, 109
   - ✅ **Correct:** Instructional text and notes = sans

4. **`BeeArchitecture.tsx`** - Lines 48, 97, 124, 167
   - ✅ **Correct:** Sublabels and host labels = sans

5. **`PulseBadge.tsx`** - Line 52
   - ✅ **Correct:** Badge = sans

---

## 🎓 Summary

**The Golden Rule:**
> If it's **content** (headings, body text, features, quotes), use **serif** (default).  
> If it's **interface** (buttons, labels, meta, technical notes), use **sans** (explicit).

**When in doubt:**
- Omit `font-sans` → defaults to serif
- Serif is the "safe" choice for most text
- Only add `font-sans` when you're certain it's UI/system/meta text

---

## 📝 Checklist for New Components

When creating a new component:

- [ ] Headings: serif (default) ✅
- [ ] Body text: serif (default) ✅
- [ ] Feature titles: serif (default) ✅
- [ ] Feature descriptions: serif (default) ✅
- [ ] Buttons: sans (explicit) ✅
- [ ] Badges: sans (explicit) ✅
- [ ] Labels: sans (explicit) ✅
- [ ] Meta text: sans (explicit) ✅
- [ ] Technical notes: sans (explicit) ✅
- [ ] Uppercase eyebrows: sans (explicit) ✅

---

**Last Updated:** 2025-01-15  
**Applies to:** rbee Design System v1.x
