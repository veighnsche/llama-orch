# Compliance Ledger Decorative Image

## Required Asset: `compliance-ledger.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Compliance section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **EU-blue ledger lines** with structured, interconnected paths
- **Checkpoint nodes** at key intersections suggesting validation and verification
- **Immutable audit trails** visual metaphor through connected, tamper-evident chains
- **Multi-standard compliance** theme (GDPR, SOC2, ISO 27001)
- **Premium dark UI aesthetic** matching the enterprise theme
- **Subtle amber accents** for warmth and trust

### Visual Elements

- Interconnected ledger lines forming a network (not too dense)
- Glowing checkpoint nodes at strategic points (subtle, professional)
- EU color palette: blues (#3b82f6, #2563eb) with amber highlights (#f59e0b)
- Abstract representation of compliance frameworks converging
- Hash chain or blockchain-inspired visual elements (subtle)
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
- `top: 1.5rem` (6px)
- `transform: translateX(-50%)` (centered)
- `width: 50rem`
- `opacity: 0.15`
- `blur: 0.5px`
- `z-index: -10`
- Hidden on mobile, visible on `md:` breakpoint and up

### Visual Inspiration

Think of:
- Compliance frameworks as interconnected systems
- Audit checkpoints and verification nodes
- Tamper-evident hash chains (blockchain-inspired)
- Multi-standard alignment (GDPR + SOC2 + ISO 27001)
- Immutable ledger concepts with checkpoint validation
- Enterprise security dashboards

### Key Differences from Other Assets

- **audit-ledger.webp** (EnterpriseHero): Flow-oriented, single audit trail
- **eu-ledger-grid.webp** (EnterpriseSolution): Structured grid with glowing checkpoints
- **compliance-ledger.webp** (THIS): Multi-path convergence, checkpoint nodes, compliance framework alignment

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).

---

**Related Assets:**
- `/illustrations/audit-ledger.webp` (EnterpriseHero)
- `/decor/eu-ledger-grid.webp` (EnterpriseSolution)
