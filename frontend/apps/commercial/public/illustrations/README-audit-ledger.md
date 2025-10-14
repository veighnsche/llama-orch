# Audit Ledger Decorative Image

## Required Asset: `audit-ledger.webp`

**Dimensions:** 1200×640px  
**Format:** WebP (optimized for web)  
**Purpose:** Decorative background illustration for Enterprise Hero section

### Design Brief

Create an abstract, premium-quality illustration that conveys:

- **EU-blue ledger lines** forming an immutable audit trail pattern
- **Soft amber highlights** suggesting security and trust
- **Premium dark UI aesthetic** matching the enterprise theme
- **Subtle, non-distracting** — should sit at 15% opacity with slight blur

### Visual Elements

- Abstract geometric lines suggesting data flow and audit chains
- EU color palette (blues) with warm amber accents
- Minimalist, modern design language
- Should work well at low opacity and with blur effect

### Technical Requirements

- Dimensions: 1200×640px
- Format: WebP
- File size: < 200KB (optimized)
- Color space: sRGB
- Transparent or dark background

### Placement

The image is positioned:
- `left: -10%`
- `top: -15%`
- `width: 52rem`
- `opacity: 0.15`
- `blur: 0.5px`
- Hidden on mobile, visible on `md:` breakpoint and up

### Alternative

Until the asset is created, the component will gracefully handle the missing image (Next.js Image component will show a placeholder or fail silently with the `hidden md:block` classes preventing layout shift).
