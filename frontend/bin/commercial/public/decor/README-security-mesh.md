# Security Mesh Decorative Image

## Required Asset: `security-mesh.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Security section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **Dark security mesh** with interconnected nodes suggesting defense-in-depth
- **Linked nodes** representing the five security crates working together
- **Amber highlights** for warmth and trust (hash-chain validation points)
- **Zero-trust architecture** visual metaphor through isolated, verified connections
- **Time-bounded execution** suggested by flowing, time-aware paths
- **Premium dark UI aesthetic** matching the enterprise theme

### Visual Elements

- Mesh network with interconnected security nodes (not too dense)
- Five distinct node clusters representing the security crates
- Amber/gold highlights at validation/verification points
- Abstract representation of:
  - Constant-time operations (uniform node spacing)
  - Hash chains (linked, tamper-evident connections)
  - Zero-trust (isolated, verified paths)
  - Time budgets (flowing, time-aware indicators)
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
- Zero-trust network architectures
- Defense-in-depth security layers
- Hash-chain validation networks
- Time-bounded execution flows
- Constant-time operation visualizations
- Enterprise security dashboards

### Key Differences from Other Assets

- **audit-ledger.webp** (EnterpriseHero): Single audit trail flow
- **eu-ledger-grid.webp** (EnterpriseSolution): Structured grid with checkpoints
- **compliance-ledger.webp** (EnterpriseCompliance): Multi-path compliance convergence
- **security-mesh.webp** (THIS): Interconnected mesh with five node clusters, defense-in-depth

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).

---

**Related Assets:**
- `/illustrations/audit-ledger.webp` (EnterpriseHero)
- `/decor/eu-ledger-grid.webp` (EnterpriseSolution)
- `/decor/compliance-ledger.webp` (EnterpriseCompliance)
