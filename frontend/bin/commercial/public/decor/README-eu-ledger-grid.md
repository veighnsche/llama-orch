# EU Ledger Grid Decorative Image

## Required Asset: `eu-ledger-grid.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Solution "How It Works" section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **EU-blue ledger grid** with structured, geometric lines suggesting data organization
- **Softly glowing checkpoints** at grid intersections implying validation and immutability
- **Immutable audit trails** visual metaphor through connected nodes and paths
- **Data sovereignty** theme with EU-centric color palette
- **Premium dark UI aesthetic** matching the enterprise theme
- **Subtle amber accents** for warmth and trust

### Visual Elements

- Grid structure with clean, geometric lines (not too dense)
- Glowing nodes at strategic intersections (subtle, not overpowering)
- EU color palette: blues (#3b82f6, #2563eb) with amber highlights (#f59e0b)
- Abstract representation of data flow and audit checkpoints
- Minimalist, modern design language
- Should work well at low opacity (15%) with slight blur

### Technical Requirements

- Dimensions: 1200×640px
- Format: WebP
- File size: < 200KB (optimized)
- Color space: sRGB
- Transparent or dark background (#0f172a or similar)

### Placement

The image is positioned:
- `left: 50%`
- `top: 2rem` (8px)
- `transform: translateX(-50%)` (centered)
- `width: 52rem`
- `opacity: 0.15`
- `blur: 0.5px`
- `z-index: -10`
- Hidden on mobile, visible on `md:` breakpoint and up

### Visual Inspiration

Think of:
- Blockchain node networks (but more structured/grid-like)
- Audit trail visualizations with checkpoints
- Data sovereignty maps with EU boundaries
- Immutable ledger concepts
- Enterprise compliance dashboards

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).

---

**Related Asset:** `/illustrations/audit-ledger.webp` (for EnterpriseHero)  
**Difference:** This grid version is more structured and checkpoint-focused, while the audit-ledger is more flow-oriented.
