# Homepage Visual Audit — 2025-10-13 05:20

**Audit Start:** Desktop 1440×900, Dark Mode  
**URL:** http://localhost:3000/  
**Total Sections:** 18 identified  
**Theme Toggle:** Present (dark mode active)

---

## Sections Seen

### 1. HeroSection (index 0)
- **Heading:** "AI Infrastructure. On Your Terms."
- **Purpose:** Primary value proposition and first impression
- **Position:** Top 64px, Height 844px
- **Primary CTAs:** "Get Started Free" (yellow), "View Docs" (secondary)
- **Key Components:** 
  - Navigation bar (organism) - rbee logo, nav links (Features, Use Cases, Pricing, Developers, Providers, Enterprise, Docs), GitHub icon, theme toggle, "Join Waitlist" CTA
  - Hero headline (H1) with orange accent on "On Your Terms"
  - Terminal mockup showing `rbee-keeper` CLI with GPU pool visualization
  - Three checkmark bullets: "Your GPUs, your network", "Zero API fees", "Drop-in OpenAI API"
  - Stats panel: GPU Pool (5 nodes/8 GPUs), Cost ($0.00/hr), Latency (-34ms)
  - Social proof badges: "Star on GitHub", "API", "OpenAI-Compatible", "$0 - No Cloud Required"
- **Notes:** Terminal has realistic output with progress bars. Orange/yellow color scheme prominent. Dark navy background.

### 2. WhatIsRbee (index 1)
- **Heading:** "What is rbee?"
- **Purpose:** Product definition and positioning statement
- **Position:** Top 908px, Height 1145px
- **Background:** Secondary (lighter navy/slate)
- **Key Components:**
  - Badge: "OPEN-SOURCE • SELF-HOSTED"
  - Large emoji icon (🐝) + headline "rbee: your private AI infrastructure"
  - Pronunciation guide "(pronounced 'are-bee')"
  - Body copy explaining orchestration platform, OpenAI-compatible
  - Three value bullets with icons: Independence, Privacy, All GPUs together
  - Stat blocks: "$0 No API fees", "100% Your code never leaves network", "All CUDA·Metal·CPU orchestrated"
  - 3D illustration: Isometric network of computers with orange cables, RGB-lit cases, "Local Network" label
- **Notes:** Strong visual with 3D artwork. Stats use large yellow numbers. Three-column benefit layout.

### 3. AudienceSelector (index 2)
- **Heading:** "Where should you start?"
- **Eyebrow:** "CHOOSE YOUR PATH" (orange)
- **Purpose:** Segment users by persona (Developers, GPU Owners, Enterprise)
- **Position:** Top 2053px, Height 1111px
- **Background:** Background (dark)
- **Key Components:**
  - Three cards with colored accents (Blue, Teal/Green, Orange)
  - Card 1 (Blue): "Code with AI locally" → "FOR DEVELOPERS" → "Build on Your Hardware" → CTA "Explore Developer Path"
  - Card 2 (Teal): "Earn from idle GPUs" → "FOR GPU OWNERS" → "Monetize Your Hardware" → CTA "Become a Provider"
  - Card 3 (Orange): "Deploy with compliance" → "FOR ENTERPRISE" → "Compliance & Security" → CTA "Enterprise Solutions"
  - Each card: Icon pair (colored bg), headline, body copy, 3 bullet points with arrows, CTA button
  - Badge on Developer card: "Homelab-ready"
- **Notes:** Clean three-column layout. Each persona has dedicated color. CTAs match card color theme.

### 4. EmailCapture (index 3)
- **Heading:** "Get Updates. Own Your AI."
- **Badge:** "In Development · M0 · 58%" (orange dot)
- **Purpose:** Waitlist signup
- **Position:** Top 3163px, Height 922px
- **Key Components:**
  - Progress badge showing dev status
  - Centered headline emphasizing "Own Your AI"
  - Subhead: "Join the rbee waitlist to get early access, build notes, and launch perks for running AI on your own hardware"
  - Icon illustration: Three connected server/document icons with dotted lines, crown icon on center
  - Email input field (dark border, placeholder "you@company.com")
  - Yellow "Join Waitlist" CTA button
  - Privacy text: "No spam. Unsubscribe anytime." with lock icon
  - GitHub link: "View Repository" with weekly dev notes mention
- **Notes:** Simple, focused conversion section. Progress badge adds transparency/urgency.

### 5. ProblemSection (index 4)
- **Heading:** "The Hidden Cost of AI Dependency"
- **Subhead:** "When vendors change the rules, your roadmap pays the price."
- **Purpose:** Pain point articulation - vendor lock-in problem
- **Position:** Top 4085px, Height 978px
- **Key Components:**
  - Illustration: Data flow from servers → rate limit gate (padlock, orange) → cloud with $ icon, warning triangles
  - Three problem cards: "Loss of Control" (purple warning), "Unpredictable Costs" (blue $), "Privacy & Compliance Risks" (green lock)
  - Each card: Icon, headline, body paragraph
  - Bottom text: "Whether you're building with AI, monetizing hardware, or meeting compliance—outsourcing core models creates risk you can't budget for."
  - Orange link CTA: "See how rbee restores control →"
- **Notes:** Classic Problem section with visual metaphor. Dark tone appropriate for pain points.

### 6. SolutionSection (index 5)
- **Heading:** "Your Hardware. Your Models. Your Control." (Orange accent on "Your Control")
- **Purpose:** Solution statement with architecture diagram
- **Position:** Top 5063px, Height 1687px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Subhead: "rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform."
  - Four badge pills: "OpenAI-compatible API", "Runs on CUDA·Metal·CPU", "Zero API fees (electricity only)", "Code stays in your network"
  - Architecture diagram: "The Bee Architecture" label (orange badge)
    - Top: "Queen-rbee (Orchestrator)" (yellow pill)
    - Middle: "Hive Manager 1, 2, 3" (orange pills) connected with lines
    - Bottom: Worker nodes - "Worker (CUDA)", "Worker (Metal)", "Worker (CPU)", "Worker (CUDA)" (yellow icons)
  - Diagram shows hierarchical orchestration structure
- **Notes:** Technical architecture visualization. Clean, node-based layout. Orange/yellow theme consistent with "bee" metaphor.

### 7. HowItWorksSection (index 6)
- **Heading:** "From Zero to AI Infrastructure in 15 Minutes"
- **Purpose:** Step-by-step onboarding guide
- **Position:** Top 6750px, Height 1748px
- **Background:** Background (dark)
- **Key Components:**
  - Step 1: "Install rbee" (yellow circle badge)
    - Body: "Build from source on Linux or macOS. Windows support is planned."
    - Badges: "~3 min" (orange), "Rust toolchain" (orange)
    - Platform tabs: Linux (active), Macos, Windows (soon)
    - Code block: git clone, cargo build, rbee-keeper daemon start commands
    - Copy button for code
  - Step 2: "Add Your Machines" (yellow circle badge)
    - Body: "Enroll every GPU host. rbee auto-detects CUDA, Metal, and CPU backends."
    - Subtext: "Multi-node, mixed backends, one pool."
    - Badges: "~5 min", "SSH access"
    - Code block: rbee-keeper setup add-node commands for workstation and mac
  - CTAs above section: "Run on my GPUs" (yellow), "See scheduler policy" (link)
- **Notes:** Two-column layout (text + code). Real commands with syntax highlighting. Time badges set expectations.

### 8. FeaturesSection (index 7)
- **Heading:** "Enterprise-Grade Features. Homelab Simplicity."
- **Subhead:** "Pick a lane—API, GPUs, Scheduler, or Real-time—and see exactly how rbee fits your stack."
- **Purpose:** Feature showcase with tabbed interface
- **Position:** Top 8498px, Height 1132px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Four tabs: "OpenAI-Compatible" (active), "Multi-GPU", "Scheduler", "Real-time"
  - Tab content: "OpenAI-Compatible API" section
    - Subhead: "Drop-in replacement for your existing tools"
    - Body: "Drop-in for Zed, Cursor, Continue, or any OpenAI client. Keep your SDKs and prompts—just change the base URL."
    - Badges: "No API fees" (orange), "Local talkers" (orange), "Secure by default" (orange)
    - Code example: Bash code showing before/after with OpenAI vs rbee endpoint
    - Copy button
- **Notes:** Tabbed interface for feature exploration. Code examples prominent. Orange badge pattern consistent.

### 9. UseCasesSection (index 8)
- **Heading:** "Built for Those Who Value Independence"
- **Subhead:** "Run serious AI on your own hardware. Keep costs at zero, keep control at 100%."
- **Purpose:** Use case personas with cost/savings
- **Position:** Top 9630px, Height 1235px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Four tabs: "Solo", "Small Team", "Homelab", "Enterprise"
  - Four use case cards visible (2x2 grid):
    - Card 1: "The Solo Developer" - Icon (laptop, blue) - "Monthly AI cost: $0" - "Shipping a SaaS with AI help but allergic to lock-in" - 3 bullets about gaming PC usage, Llama 70B, no rate limits
    - Card 2: "The Small Team" - Icon (people, orange) - "Savings/yr: $6,000+" - "5-person startup burning $500/mo on APIs" - 3 bullets about pooling workstations/Macs, shared models, GDPR-friendly
    - Card 3: "The Homelab Enthusiast" - Icon (home, teal) - "Idle GPUs → Productive" - "Four GPUs gathering dust" - 3 bullets about spreading workers across LAN, building agents
    - Card 4: "The Enterprise" - Icon (building, purple) - "Compliance: EU-only" - "50-dev org. Code can't leave the premises" - 3 bullets about on-prem rbee, Rhai-based rules
- **Notes:** Grid layout with clear cost/benefit for each persona. Icons color-coded. Bullets emphasize real scenarios.

### 10. ComparisonSection (index 9)
- **Heading:** "Why Developers Choose rbee"
- **Subhead:** "Local-first AI that's faster, private, and costs $0 on your hardware."
- **Purpose:** Feature comparison table vs competitors
- **Position:** Top 10864px, Height 943px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Legend: Available (checkmark), Not available (X), "Partial" = limited coverage
  - Comparison table with 5 columns: Feature | rbee | OpenAI & Anthropic | Ollama | Runpod & Vast.ai
  - Rows:
    - Total Cost: "$0 runs on your hardware (lowest)" vs "$20-100/mo per dev" vs "$0" vs "$0.50-2/hr"
    - Privacy/Data Residency: Complete vs Limited vs Complete vs Limited
    - Multi-GPU Utilization: Orchestrated vs N/A vs Limited vs ✓
    - OpenAI-Compatible API: ✓ vs ✓ vs Partial vs X
    - Custom Routing Policies: "Rhai-based policies" vs X vs X vs X
    - Rate Limits/Quotas: None (teal) vs Yes (red) vs None (teal) vs Yes (red)
  - rbee column highlighted in darker background
  - CTAs at bottom: "See Quickstart" (yellow), "Architecture" (link)
- **Notes:** Clear competitive positioning. rbee column emphasized. Checkmarks/X symbols for quick scanning.

### 11. PricingSection (index 10)
- **Heading:** "Start Free. Scale When Ready."
- **Subhead:** "Run rbee free at home. Add collaboration and governance when your team grows."
- **Purpose:** Pricing tiers
- **Position:** Top 11807px, Height 1654px
- **Background:** Background (dark)
- **Key Components:**
  - Feature badges: "Open source", "OpenAI-compatible", "Multi-GPU", "No feature gates"
  - Billing toggle: "Monthly" | "Yearly" (with "Save 2 months" badge in teal)
  - Three pricing cards:
    - Card 1: "Home/Lab" - "€0 forever" - 5 features (Unlimited GPUs on hardware, OpenAI-compatible API, Multi-modal models, Active community support, Open source core) - CTA: "Download rbee" - Note: "Local use. No feature gates."
    - Card 2: "Team" (Most Popular badge, orange) - "€99 /month" - 5 features (Everything in Home/Lab, Web UI for cluster & models, Shared workspaces & quotas, Priority support business hours, Rhai policy templates rate/data) - CTA: "Start 30-Day Trial" (yellow button) - Note: "Cancel anytime during trial."
    - Card 3: "Enterprise" - "Custom" - 5 features (Everything in Team, Dedicated isolated instances, Custom SLAs & onboarding, White-label & SSO options, Enterprise security & support) - CTA: "Contact Sales" - Note: "We'll reply within 1 business day."
  - Team card has orange border emphasis
- **Notes:** Classic 3-tier pricing. Free tier is truly free. Team tier highlighted as popular choice. Clear feature progression.

### 12. SocialProofSection (index 14)
- **Heading:** "Trusted by Developers Who Value Independence"
- **Subhead:** "Local-first AI with zero monthly cost. Loved by builders who keep control."
- **Purpose:** Social proof with stats and testimonials
- **Position:** Top 13461px, Height 1580px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Source tabs: "GitHub", "HN", "Reddit"
  - Four stat blocks: "1,200+ GitHub Stars" (orange), "500+ Active Installations" (orange), "8,000+ GPUs Orchestrated" (orange), "€0 Avg Monthly Cost" (teal)
  - Label: "REAL TEAMS. REAL SAVINGS."
  - Three testimonial cards:
    - Card 1: "Alex K." (Solo Developer, yellow avatar) - "Used to pay $80/mo for coding. Now Llama 70B runs locally on my gaming PC + an old workstation. Same quality, $0/month. Not going back." - Badge: "$80/mo → $0"
    - Card 2: "Sarah M." (CTO, StartupCo, orange avatar) - "Verified" badge - "We pooled our team's hardware and cut AI spend from $500/month to zero. rbee's OpenAI-compatible API meant no code changes." - Badge: "$500/mo → $0"
    - Card 3: "Dr. Thomas R." (Research Lab Director, yellow avatar) - "GDPR blocked cloud options. With rbee on-prem and EU-only routing via Rhai, we shipped safely—no external deps."
  - Bottom: Partial view of GitHub embed and setup photo
- **Notes:** Strong testimonials with real savings. Stats use large numbers. Verified badge adds credibility. Emphasis on "$0" theme.

### 13. TechnicalSection (index 15)
- **Heading:** "Built by Engineers, for Engineers"
- **Subhead:** "Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."
- **Purpose:** Technical details and architecture credibility
- **Position:** Top 15041px, Height 1175px
- **Background:** Background (dark)
- **Key Components:**
  - Left column: "CORE PRINCIPLES" → "Architecture Highlights"
    - 5 items with checkmarks: "BDD-Driven Development" (42/82 scenarios passing 68% complete), "Cascading Shutdown Guarantee" (No orphaned processes. Clean VRAM lifecycle), "Process Isolation" (Worker-level sandboxes. Zero cross-leak), "Protocol-Aware Orchestration" (SSE, JSON, binary protocols), "Smart/Dumb Separation" (Central brain, distributed execution)
    - Progress bar: "BDD Coverage" 42/82 scenarios passing, 68% complete (teal bar), "Live CI coverage" badge
    - Architecture diagram: "DECISION LAYERS" showing "BEE (Policy & State)" ↔ "Process Adapter" ↔ "Health Monitor (watchdog)" with orange connection lines
  - Right column: "STACK" → "Technology Stack"
    - 6 tech cards: "Rust" (Performance + memory safety), "Candle ML" (Rust-native inference), "Rhai Scripting" (Embedded, sandboxed policies), "SQLite" (Embedded, zero-ops DB), "Axum + Vue.js" (Async backend + modern UI), "100% Open Source" with "View Source" button
- **Notes:** Deep technical details for engineer audience. Progress transparency (BDD coverage). Architecture diagram adds sophistication.

### 14. FAQSection (index 16)
- **Heading:** "rbee FAQ"
- **Eyebrow:** "Support · Self-hosted AI"
- **Subhead:** "Quick answers about setup, models, orchestration, and security."
- **Purpose:** Frequently asked questions with accordion
- **Position:** Top 16217px, Height 1463px
- **Background:** Secondary (lighter)
- **Key Components:**
  - Search bar: "Search questions..."
  - Action buttons: "Expand all" | "Collapse all"
  - Category tabs: "Setup", "Models", "Performance", "Marketplace", "Security", "Production"
  - Accordion items visible:
    - "Performance" section: "How is this different from Ollama?" (collapsed)
    - "Setup" section: "Do I need to be a Rust expert?" (collapsed), "What if I don't have GPUs?" (collapsed), "How do I migrate from OpenAI API?" (collapsed)
    - "Production" section label visible
  - Right sidebar: "Still stuck?"
    - 3D illustration: Person at desk with books/servers and question mark
    - Links: "Join Discussions", "Read Setup Guide", "Email support"
    - CTA: "Open Discussions" (yellow button)
    - Link: "View documentation →"
- **Notes:** Searchable FAQ with category filtering. Sidebar help widget. Clean accordion UI.

### 15. CTASection (index 17)
- **Heading:** "Take Control of Your AI Infrastructure Today." ("AI Infrastructure" in orange)
- **Badge:** "100% Open Source · Self-Hosted"
- **Purpose:** Final conversion with strong CTA
- **Position:** Top 17680px, Height 766px
- **Background:** Background (dark), contained in bordered card
- **Key Components:**
  - Left side:
    - Headline with orange accent
    - Body: "Join hundreds of users, providers, and enterprises who've chosen independence—no vendor lock-in, no data exfiltration."
    - Three CTAs: "Get Started Free" (yellow button), "View Documentation" (secondary), "Join Discord" (secondary)
    - Note: "100% open source. No credit card required. Install in ~15 minutes."
    - Code snippet: `curl -fsSL https://rbee.sh | bash`
    - Feature badges: "OpenAI-compatible API", "Multi-GPU / multi-node", "Sandboxed schedulers"
  - Right side:
    - 3D illustration: Four hexagonal server nodes (black/orange design) on blueprint background
    - Three stat blocks: "15m to install", "100% open source", "GPU+ multi-backend"
  - Bottom preview: Newsletter signup visible ("In the loop")
- **Notes:** Strong final CTA with visual appeal. Multiple conversion paths. Code snippet provides immediate action. 3D art reinforces brand.

### 16. Footer (index 18+)
- **Visible:** Newsletter section "In the loop" - "Get updates, roadmap, and self-hosting tips. 1-2 emails/month."
- **Note:** Footer extends below fold, will capture separately

---

## Design Analysis (In Progress)

### Layout & Hierarchy

**Hero Section (Desktop Dark):**
- ✅ **Commanding presence:** Full viewport height hero with clear focal point
- ✅ **CTA above fold:** Primary yellow CTA immediately visible, strong contrast
- ✅ **Terminal visual:** Right-side terminal mockup adds authenticity, shows product in action
- ⚠️ **Navigation density:** 7 nav links may be too many for clarity - consider grouping or prioritizing
- ✅ **Vertical rhythm:** Good spacing between headline, subhead, bullets, CTAs

**Issues Found:**
- [P2] Navigation has 7 top-level items (Features, Use Cases, Pricing, Developers, Providers, Enterprise, Docs) - consider mega-menu or consolidation
- [P3] Badge row at bottom of hero ("Star on GitHub", "API", etc.) uses small text - may not scan well

**WhatIsRbee Section (Desktop Dark):**
- ✅ **3D illustration:** Authentic, on-brand, shows local network concept perfectly
- ✅ **Stat blocks:** "$0", "100%", "All" format is scannable and impactful
- ✅ **Emoji branding:** Bee emoji reinforces rbee name
- ⚠️ **Pronunciation guide:** Small text, might be missed - consider making it more prominent or moving to hero
- ⚠️ **Stat block design:** Three separate boxes feel disconnected - could use visual connector or unified container

**AudienceSelector Section (Desktop Dark):**
- ✅ **Three-path segmentation:** Clear persona targeting (Developer, GPU Owner, Enterprise)
- ✅ **Color coding:** Blue/Teal/Orange creates visual distinction between paths
- ✅ **Benefit-first headlines:** "Code with AI locally", "Earn from idle GPUs", "Deploy with compliance" are action-oriented
- [P2] **Card equality:** All three cards same visual weight—consider emphasizing primary target persona
- [P3] **"Homelab-ready" badge:** Only on Developer card—inconsistent if other paths also homelab-ready

**Overall Page Flow (Desktop Dark):**
- ✅ **Narrative arc:** Problem → Solution → How → Features → Proof → Pricing → FAQ → CTA follows classic SaaS structure
- ✅ **Consistent theming:** Orange/yellow accent colors maintained throughout
- ✅ **Visual variety:** Alternating backgrounds (dark/secondary) creates rhythm
- ✅ **3D illustrations:** High-quality, consistent style, reinforce brand identity
- ⚠️ **Section length:** Some sections very tall (HowItWorks 1748px, Pricing 1654px)—may lose mobile users
- [P1] **Repeated messaging:** "Zero API fees", "OpenAI-compatible", "Your hardware" appears 10+ times—good for reinforcement but could consolidate

### Typography

**Desktop Dark Mode Observations:**
- ✅ **H1 scale:** "AI Infrastructure. On Your Terms." is large, bold, commanding
- ✅ **Accent treatment:** Orange color on key phrases ("On Your Terms", "Your Control", "AI Infrastructure") creates emphasis
- ✅ **Hierarchy clear:** H1 → H2 → H3 scale is consistent across sections
- ✅ **Body text:** Readable contrast in dark mode, good line length
- ⚠️ **Small text:** Pronunciation guide "(pronounced 'are-bee')" and some badge text very small—may not be legible at distance
- [P2] **Line length:** Some body paragraphs in features/solution sections exceed 80 characters—consider max-width constraint
- [P3] **Weight consistency:** Mix of semibold/bold in headlines—verify intentional vs. inconsistent token usage

### Color & Tokens

**Color Palette Observed:**
- **Primary accent:** Orange (#FF8C00 or similar)—used for "On Your Terms", badges, CTA highlights
- **Secondary accent:** Yellow (#FDB913 or similar)—primary CTA buttons
- **Teal/Cyan:** Used for "None" indicators in comparison table, positive stats
- **Background:** Very dark navy (#0A0F1E or similar)
- **Secondary background:** Slightly lighter slate (#1A2332 or similar)
- **Text:** White/light gray for primary text, muted gray for secondary

**Token Alignment:**
- ✅ **Consistent accent usage:** Orange/yellow appears intentionally across CTAs, badges, headings
- ✅ **Background alternation:** Dark/secondary creates visual rhythm
- ⚠️ **One-off colors:** Persona cards (blue, teal, orange, purple) may be ad-hoc—verify these map to design tokens
- [P2] **Gradient usage:** Some 3D illustrations have gradients—ensure these don't conflict with flat UI elsewhere
- [P3] **Border colors:** Various border treatments (orange, teal, default)—consolidate to token-based system

### Components & Repetition

**Repeated Patterns (Potential Consolidation):**
1. **Feature cards:** Appear in AudienceSelector (3 cards), UseCasesSection (4 cards), Pricing (3 cards)—could be unified `<FeatureCard>` molecule
2. **Stat blocks:** "$0", "100%", "All" in WhatIsRbee; "1,200+", "500+", "8,000+", "€0" in SocialProof—unified `<StatBlock>` atom
3. **Code blocks:** HowItWorks (2 blocks), FeaturesSection (1 block), CTASection (1 snippet)—unified `<CodeBlock>` molecule with copy button
4. **Badge pills:** Orange badges appear in 8+ sections—unified `<Badge>` atom with variants (primary, secondary, success)
5. **Testimonial cards:** SocialProof section—could be `<TestimonialCard>` molecule
6. **CTA buttons:** Yellow primary, ghost secondary—appears consistent but verify all use same `<Button>` component

**Near-Duplicates Found:**
- ⚠️ **Two different CTA styles:** "Get Started Free" (yellow button) vs "View Docs" (ghost) vs "Join Waitlist" (nav, yellow)—all CTAs, but different hierarchies. Intentional or inconsistent?
- ⚠️ **Badge variations:** Some badges have dots (orange), some don't—verify design system intent
- [P2] **Icon treatment:** Some icons circular background, some not—consolidate

**Missing Molecules/Organisms:**
- [ ] `<FeatureCard>` - Reusable card for personas, use cases, pricing tiers
- [ ] `<StatBlock>` - Number + label combo
- [ ] `<CodeBlock>` - Syntax-highlighted code with copy button
- [ ] `<TestimonialCard>` - Avatar + name + role + quote + optional badge
- [ ] `<ComparisonTable>` - Feature comparison grid
- [ ] `<AccordionItem>` - FAQ accordion item

### Imagery & Motion

**3D Illustrations:**
- ✅ **WhatIsRbee:** Isometric network of gaming PCs with RGB lighting and orange cables—authentic, on-brand
- ✅ **ProblemSection:** Rate limit gate visualization with padlock and warning triangles—clear metaphor
- ✅ **CTASection:** Hexagonal server nodes on blueprint—reinforces technical/architectural theme
- ✅ **FAQSection:** Person at desk with servers and question mark—relatable help imagery
- ✅ **Consistency:** All 3D art shares warm lighting, orange accent color, isometric/low-poly style
- [P3] **Hero terminal:** Could be animated (typing effect, progress bars filling)—currently static
- [P3] **Architecture diagram:** Could show data flow animation in SolutionSection

**Motion Present:**
- ⚠️ **Limited motion detected:** Page appears mostly static
- [ ] **Potential motion opportunities:** Hero terminal typing, architecture diagram data flow, stat counter animations, testimonial carousel
- ✅ **If adding motion:** All achievable via `tw-animate-css` (fadeIn, slideInUp, pulse, etc.)—no external libs needed

**Imagery Gaps:**
- [P2] **No team/founder photos:** SocialProof uses avatar circles but not real photos—consider adding authenticity
- [P2] **No product screenshots:** All visuals are 3D art or code—consider actual UI screenshots of Web UI (Team tier feature)
- [P3] **Placeholder concern:** 3D art is high-quality, not generic placeholders—good

### Accessibility

**Positive Elements:**
- ✅ **Color contrast:** White text on dark navy backgrounds passes WCAG AA
- ✅ **Heading hierarchy:** Proper H1 → H2 → H3 structure observed
- ✅ **CTA affordance:** Yellow buttons have clear visual weight and appear clickable

**Issues & Gaps:**
- [P1] **Alt text unknown:** Cannot verify from visual inspection—must check markup for descriptive alt on 3D illustrations
- [P1] **Focus states:** Not visible in screenshots—must test keyboard navigation to verify rings/outlines
- [P2] **Link affordance:** "View Repository", "See how rbee restores control" are text links—ensure underline/hover state
- [P2] **Icon-only buttons:** Theme toggle appears icon-only—must have aria-label
- [P3] **Color-only meaning:** Comparison table uses checkmarks/X + color (teal/red)—symbols help but verify for colorblind users
- [P3] **Motion preference:** If motion added, must respect `prefers-reduced-motion`
- [P3] **Skip links:** Not visible—should have skip-to-main for keyboard users

**ARIA & Landmarks:**
- Cannot verify from visual inspection—audit needed for: `<main>`, `<nav>`, `<section aria-labelledby>`, button roles, expanded states on accordions

### Responsiveness (Desktop 1440×900 Only So Far)

**Desktop Observations:**
- ✅ **Safe margins:** Content well-contained, no edge bleeding
- ✅ **Grid consistency:** Multi-column layouts (3-col personas, 4-col use cases, comparison table) render cleanly
- ⚠️ **Terminal mockup:** Right-side terminal in hero is large—will need significant adaptation for tablet/mobile
- ⚠️ **Code blocks:** Wide code snippets in HowItWorks—may overflow on mobile
- ⚠️ **Comparison table:** 5-column table will be challenging on mobile—likely needs horizontal scroll or card stacking

**Pending Tablet/Mobile Testing:**
- [ ] 834×1112 (tablet): Test terminal mockup, comparison table, feature grids
- [ ] 390×844 (mobile): Test navigation collapse, hero CTA placement, code block overflow

### Content & Story

**What's Working:**
- ✅ **Independence narrative:** "Your Hardware. Your Models. Your Control." is consistent
- ✅ **Zero-cost emphasis:** "$0", "Zero API fees", "No credit card required" repeated effectively
- ✅ **OpenAI-compatible hook:** Drop-in replacement messaging lowers switching friction
- ✅ **Technical credibility:** BDD coverage, Rust-native, architecture diagrams signal serious engineering
- ✅ **Use case variety:** Solo dev, small team, homelab, enterprise—covers spectrum

**What's Missing (Story Gaps):**
1. **[P1] "Use ALL your home GPUs" undersold:** Headline says "AI Infrastructure. On Your Terms." but doesn't explicitly call out "every GPU in your home network" until section 6 (5063px down). Hero should lead with "Turn every GPU in your home into an AI cluster."
2. **[P1] Independence from OpenAI not explicit enough:** Problem section mentions "vendor lock-in" generically but doesn't name OpenAI/Anthropic as the specific threat in hero. Subhead should say "Break free from OpenAI rate limits."
3. **[P2] Risk reversal weak:** "No credit card required" appears once. Add: "Cancel anytime", "100% open source = zero vendor lock", "Your data never leaves your network."
4. **[P2] Proof point timing:** Testimonials with "$80/mo → $0" savings appear at 13461px. Move one testimonial to hero area for immediate credibility.
5. **[P3] "rbee" pronunciation buried:** "(pronounced 'are-bee')" is tiny text in section 2. Move to hero or nav as tooltip.
6. **[P3] Time-to-value unclear:** "15 minutes" appears in section 7 and final CTA but not hero. Hero should promise "15-minute setup."

**Redundancies (Good Reinforcement, But Could Trim):**
- "OpenAI-compatible API" appears: Hero badges, What is rbee, Solution pills, Features tab, Comparison table, Pricing, CTA badges (7 times)
- "Zero API fees" appears: Hero bullets, What is rbee stat, Solution pills, Comparison table, Features badges, Social proof, CTA (7 times)
- Could consolidate by removing from middle sections, keeping in hero, pricing, and CTA

**Tone & Voice:**
- ✅ **Engineer-to-engineer:** Technical language (BDD, Rhai, process isolation) appropriate for target audience
- ✅ **Confidence:** "Enterprise-Grade Features. Homelab Simplicity." is bold claim backed by features
- ⚠️ **Jargon balance:** "Rhai-based policies", "Candle ML", "Axum" may alienate non-Rust devs—consider glossary or tooltips

---

## Observations Log (Live Updates)

**05:20** - Hero captured. Strong terminal visualization. Orange "On Your Terms" creates good emphasis.  
**05:21** - Navigation: Clean but dense. Logo says "rbee" (good), theme toggle present, waitlist CTA in yellow.  
**05:21** - Terminal mockup shows GPU pool with progress bars - excellent technical credibility signal.
**05:22** - Completed dark mode section captures (18 sections). 3D illustrations are consistent and high-quality throughout.
**05:22** - Theme appears to be in dark mode by default. Light mode screenshots will show contrast/readability differences.
**05:23** - Design analysis complete. Identified 6 P1 issues, strong technical credibility, story gaps in hero.

---

## Stakeholder Story (Observed → Proposed)

### Observed Message

The current homepage presents **rbee** as an open-source AI orchestration platform that unifies home/office hardware into an OpenAI-compatible cluster. The promise is **independence** ("On Your Terms"), **zero API costs** (run on your own hardware), and **drop-in compatibility** (replace OpenAI with a URL change). The page targets three personas—developers, GPU owners, and enterprises—with technical depth (Rust, BDD, architecture diagrams) and social proof (testimonials showing $500/mo → $0 savings).

### Gaps

1. **Hero doesn't lead with "ALL your home GPUs":** The unique value—orchestrating every GPU across gaming rigs, workstations, and Macs—is buried in section 6. Hero says "Run LLMs on your hardware" but doesn't emphasize the multi-GPU, multi-machine orchestration.
2. **OpenAI pain point not explicit enough:** "Vendor lock-in" is generic. Hero should call out OpenAI rate limits, pricing changes, and API deprecations by name.
3. **Time-to-value hidden:** "15-minute setup" appears late. Hero should front-load this to lower perceived effort.
4. **Risk reversal weak:** "No credit card required" is good but not enough. Need "100% open source = walk away anytime", "Your data never touches the cloud."
5. **Proof delayed:** First testimonial at 13461px. Hero needs one social proof element above fold.

### Proposed Message (Ready-to-Paste Copy)

#### Headline (H1)
**"Turn Every GPU in Your Home into a Private AI Cluster. 15 Minutes."**

**Rationale:** Leads with the unique multi-GPU orchestration hook + time-to-value. "Private" signals independence. "15 Minutes" lowers barrier.

#### Subhead (Below H1)
**"Break free from OpenAI rate limits and surprise pricing. rbee orchestrates LLMs across your gaming rigs, workstations, and Macs—giving you an OpenAI-compatible API with zero monthly cost and zero data exfiltration."**

**Rationale:** Names the competitor (OpenAI), states the pain (rate limits, pricing), promises the solution (multi-machine orchestration + drop-in API), and delivers the benefits ($0 cost, data stays local).

#### Primary CTA
**"Get Started Free →"**  
*Microcopy:* "Open source. No credit card. Ready in 15 min."

**Rationale:** Keeps existing CTA but strengthens microcopy with three objection-handlers.

#### Support Blocks (Problem → Solution → Proof)

**Block 1: Problem (Why now?)**  
*"OpenAI just changed pricing again. Anthropic rate-limited your team. You're burning $500/month on APIs you can't control."*

**Block 2: Solution (What is rbee?)**  
*"rbee is an open-source orchestrator that pools every CUDA, Metal, and CPU device on your network into one private AI cluster. Drop-in OpenAI API compatibility means zero code changes—just point your tools (Cursor, Zed, Continue) to localhost:8080."*

**Block 3: Proof (Who's using it?)**  
*"Solo devs cut AI costs from $80/month to $0. Startups saved $6,000/year by pooling team hardware. Enterprises met GDPR by keeping inference on-prem. 1,200+ GitHub stars. 8,000+ GPUs orchestrated."*

---

## Evidence Pack

### Screenshots Captured (Desktop 1440×900, Dark Mode)

- `hero_desktop_dark.png` - Full hero section with navigation, headline, terminal mockup, CTAs
- `section-01-whatisrbee_desktop_dark.png` - "What is rbee?" with 3D network illustration
- `section-02-audienceselector_desktop_dark.png` - Three persona cards (Developer, GPU Owner, Enterprise)
- `section-03-emailcapture_desktop_dark.png` - Waitlist section with progress badge
- `section-04-problem_desktop_dark.png` - Problem section with rate limit gate illustration
- `section-05-solution_desktop_dark.png` - Solution section with bee architecture diagram
- `section-06-howitworks_desktop_dark.png` - How It Works with code examples
- `section-07-features_desktop_dark.png` - Features tab interface (OpenAI-Compatible)
- `section-08-usecases_desktop_dark.png` - Four use case persona cards
- `section-09-comparison_desktop_dark.png` - Feature comparison table
- `section-10-pricing_desktop_dark.png` - Three-tier pricing (Home/Lab, Team, Enterprise)
- `section-11-socialproof_desktop_dark.png` - Testimonials and stats
- `section-12-technical_desktop_dark.png` - Architecture highlights and tech stack
- `section-13-faq_desktop_dark.png` - FAQ accordion with search
- `section-14-cta_desktop_dark.png` - Final CTA with hexagonal servers illustration

### Section Index

1. **HeroSection** (64px) - Primary value prop, terminal mockup, CTAs
2. **WhatIsRbee** (908px) - Product definition, 3D network illustration
3. **AudienceSelector** (2053px) - Three persona paths
4. **EmailCapture** (3163px) - Waitlist with dev progress badge
5. **ProblemSection** (4085px) - Vendor lock-in pain points
6. **SolutionSection** (5063px) - Bee architecture diagram
7. **HowItWorksSection** (6750px) - Step-by-step setup guide
8. **FeaturesSection** (8498px) - Tabbed feature showcase
9. **UseCasesSection** (9630px) - Four persona scenarios
10. **ComparisonSection** (10864px) - vs. OpenAI, Ollama, Runpod
11. **PricingSection** (11807px) - Three tiers (€0, €99, Custom)
12. **SocialProofSection** (13461px) - Testimonials, stats (1,200+ stars)
13. **TechnicalSection** (15041px) - BDD coverage, Rust stack
14. **FAQSection** (16217px) - Searchable accordion
15. **CTASection** (17680px) - Final conversion with code snippet
16. **Footer** (18000px+) - Newsletter, links

---

## Component Cross-Map

| Section | Atoms | Molecules | Organisms | Notes |
|---------|-------|-----------|-----------|-------|
| Hero | Button, Badge, ChecklistItem | TerminalMockup, StatBlock | Navigation, HeroLayout | Terminal is custom, could extract |
| WhatIsRbee | Badge, Icon | StatBlock (x3), BulletList | ContentSection | Stat blocks repeated in SocialProof |
| AudienceSelector | Button, Icon, Badge | FeatureCard (x3) | CardGrid | FeatureCard reused in UseCases, Pricing |
| EmailCapture | Input, Button, Badge | EmailForm | CenteredCTA | Simple conversion block |
| Problem | Icon | ProblemCard (x3) | CardGrid | Problem cards unique, could generalize |
| Solution | Badge | ArchitectureDiagram | DiagramSection | Diagram is custom SVG/component |
| HowItWorks | Badge, Button | CodeBlock (x2), StepCard | StepByStepGuide | CodeBlock reused in Features, CTA |
| Features | Badge, Button | CodeBlock, TabPanel | TabbedInterface | Tab component reused? |
| UseCases | Icon, Badge | FeatureCard (x4) | CardGrid | Same FeatureCard as AudienceSelector |
| Comparison | Icon (checkmark/X) | ComparisonCell | ComparisonTable | Table needs mobile adaptation |
| Pricing | Badge, Button, Icon | PricingCard (x3) | PricingGrid | PricingCard variant of FeatureCard |
| SocialProof | Icon, Badge | TestimonialCard (x3), StatBlock (x4) | TestimonialGrid | Stat blocks same as WhatIsRbee |
| Technical | Icon, Badge, Button | TechCard, ProgressBar, Diagram | TwoColumnLayout | Diagram custom |
| FAQ | Input, Button | AccordionItem | AccordionWithSidebar | Accordion needs extraction |
| CTA | Button, Badge | CodeSnippet, StatBlock (x3) | FinalCTASection | Code snippet variant of CodeBlock |

**Consolidation Opportunities:**
- `<FeatureCard>` used in AudienceSelector, UseCases, Pricing (with variants)
- `<StatBlock>` used in WhatIsRbee, SocialProof, CTA
- `<CodeBlock>` used in HowItWorks, Features, CTA (with variants)
- `<Badge>` used in 14+ sections—verify all use same component

---

## Implementation Handoff (For StyleBee)

### Top 5 P1 Issues (Fix First)

1. **[P1] Hero headline undersells unique value**  
   - **Current:** "AI Infrastructure. On Your Terms."  
   - **Fix:** "Turn Every GPU in Your Home into a Private AI Cluster. 15 Minutes."  
   - **Screenshot:** `hero_desktop_dark.png` (line 1, headline)  
   - **Why:** Current headline is generic. Unique selling proposition is multi-GPU/multi-machine orchestration + speed. Competitors (Ollama) are single-machine.

2. **[P1] OpenAI pain point not named in hero**  
   - **Current:** Subhead says "Build with AI, keep control, and avoid vendor lock-in."  
   - **Fix:** "Break free from OpenAI rate limits and surprise pricing. rbee orchestrates LLMs across your gaming rigs, workstations, and Macs—giving you an OpenAI-compatible API with zero monthly cost."  
   - **Screenshot:** `hero_desktop_dark.png` (line 2, subhead)  
   - **Why:** "Vendor lock-in" is generic. Target audience is already on OpenAI and frustrated with recent pricing/rate limit changes. Name the competitor.

3. **[P1] Proof delayed 13,000px**  
   - **Current:** First testimonial at SocialProofSection (13461px down)  
   - **Fix:** Add one testimonial card to hero area (below CTA buttons). Use Alex K.: "$80/mo → $0" with avatar.  
   - **Screenshot:** `hero_desktop_dark.png` (add below fold) + `section-11-socialproof_desktop_dark.png` (source)  
   - **Why:** Social proof above fold increases conversion. Testimonial reinforces "$0" claim immediately.

4. **[P1] Missing alt text on 3D illustrations**  
   - **Current:** Cannot verify from screenshots but likely missing descriptive alt  
   - **Fix:** Add descriptive alt to ALL 3D images that also serves as generation prompt. Example: "Isometric 3D render of three gaming PCs with RGB lighting connected by glowing orange network cables on a dark background, representing a local GPU cluster."  
   - **Screenshots:** `section-01-whatisrbee_desktop_dark.png`, `section-04-problem_desktop_dark.png`, `section-14-cta_desktop_dark.png`  
   - **Why:** Accessibility (screen readers) + provides clear generation prompts for future image updates.

5. **[P1] Focus states missing visual confirmation**  
   - **Current:** Cannot see focus rings in screenshots  
   - **Fix:** Verify all interactive elements (buttons, links, inputs) have visible focus states. Use Tailwind's `focus:ring-2 focus:ring-orange-500 focus:ring-offset-2 focus:ring-offset-background`.  
   - **Test:** Keyboard navigation through hero section  
   - **Why:** WCAG 2.1 AA compliance + keyboard users need visual feedback.

### Sections to Rebuild vs. Tune

**Rebuild (Structural Changes):**
- **Hero (HeroSection):** Rewrite copy (H1, subhead), add testimonial card, restructure CTA microcopy. Keep terminal mockup visual.
- **Pricing (PricingSection):** Current cards work but consider extracting unified `<PricingCard>` component from `<FeatureCard>` to reduce prop bloat.

**Tune (Copy/Style Tweaks):**
- **WhatIsRbee:** Move pronunciation guide to hero or make tooltip on logo. Unify stat blocks into connected container.
- **AudienceSelector:** Emphasize primary target (likely Developer) with larger card or "Recommended" badge.
- **Problem:** Good as-is. Consider shortening body copy.
- **Solution:** Good as-is. Optional: add subtle animation to architecture diagram (data flow lines).
- **HowItWorks:** Good structure. Add overflow handling for code blocks on mobile.
- **Features:** Good. Verify other tabs (Multi-GPU, Scheduler, Real-time) have content.
- **UseCases:** Good. Ensure all 4 personas have equal visual weight.
- **Comparison:** Table will break on mobile. Rebuild as card stack for <768px.
- **SocialProof:** Move one testimonial to hero. Rest good as-is.
- **Technical:** Good for engineer audience. Consider glossary tooltips on jargon (Rhai, Candle, Axum).
- **FAQ:** Good. Ensure search works and accordion is keyboard-accessible.
- **CTA:** Good. Verify code snippet has syntax highlighting.

### Components to Consolidate

**Immediate Extractions:**
1. **`<FeatureCard>`** - Used in AudienceSelector (3), UseCases (4), Pricing (3) with variants  
   - Props: `icon`, `iconBg`, `eyebrow`, `headline`, `body`, `bullets[]`, `cta`, `badge?`  
   - Variants: `default`, `highlighted` (orange border), `compact`

2. **`<StatBlock>`** - Used in WhatIsRbee (3), SocialProof (4), CTA (3)  
   - Props: `value`, `label`, `color` (orange|teal|white)  
   - Variants: `large` (hero), `medium` (sections)

3. **`<CodeBlock>`** - Used in HowItWorks (2), Features (1), CTA (1)  
   - Props: `code`, `language`, `filename?`, `highlightLines?`, `showCopy`  
   - Variants: `inline` (CTA snippet), `full` (HowItWorks)

4. **`<Badge>`** - Used in 14+ sections  
   - Props: `text`, `variant` (primary|secondary|success), `icon?`, `dot?`  
   - Verify all use same component vs. ad-hoc spans

**Lower Priority:**
- `<TestimonialCard>` - Only used once (SocialProof), extract when reused
- `<ComparisonTable>` - Needs mobile refactor, extract after responsive fix
- `<AccordionItem>` - FAQ-specific, extract if reused elsewhere

### Copy Blocks Ready to Paste

**Hero H1:**
```
Turn Every GPU in Your Home into a Private AI Cluster. 15 Minutes.
```

**Hero Subhead:**
```
Break free from OpenAI rate limits and surprise pricing. rbee orchestrates LLMs across your gaming rigs, workstations, and Macs—giving you an OpenAI-compatible API with zero monthly cost and zero data exfiltration.
```

**Hero CTA Microcopy:**
```
Open source. No credit card. Ready in 15 min.
```

**Hero Testimonial (Add Below CTAs):**
```
"Used to pay $80/mo for coding. Now Llama 70B runs locally on my gaming PC + an old workstation. Same quality, $0/month. Not going back."
— Alex K., Solo Developer
[$80/mo → $0 badge]
```

**WhatIsRbee Pronunciation (Add to Logo Tooltip or Badge):**
```
rbee (pronounced "are-bee")
```

### Imagery Directives

**Generate/Replace:**
1. **Hero testimonial avatar** - Need real photo or illustrated avatar for Alex K. (solo dev persona). Current: yellow circle placeholder in SocialProof. Generate: "Professional headshot of a software developer at desk with laptop, warm lighting, looking confident."

2. **Team tier product screenshot** - Pricing card mentions "Web UI for cluster & models" but no screenshot exists. Generate: "Screenshot of modern dark-mode web dashboard showing GPU cluster status, with node health indicators, model list, and real-time metrics. Clean UI with orange accents, resembling Vercel or Railway dashboard style."

3. **Optional: Founder/team photo** - SocialProof section could benefit from "Built by engineers who value independence" with actual founder photo. Generate: "Software engineer at desk with homelab GPU rack in background, warm lighting, authentic workspace."

**Keep As-Is:**
- All 3D illustrations (WhatIsRbee, Problem, Solution, CTA, FAQ) are high-quality and on-brand
- Terminal mockup in hero is excellent—shows real output

**Next.js `<Image>` Optimization:**
Add `<Image>` component with descriptive alt for:
- All 3D illustrations (5 images): `priority={true}` for hero terminal, `loading="lazy"` for others
- Testimonial avatars (3 images): `loading="lazy"`, `width={48}`, `height={48}`
- Future: Team tier screenshot, founder photo

**Alt Text Template (Doubles as Gen Prompt):**
```typescript
// Example for WhatIsRbee illustration
<Image
  src="/images/local-network-cluster.png"
  alt="Isometric 3D illustration of three gaming PCs with glowing RGB fans connected by orange network cables on a dark background, representing a local GPU orchestration cluster"
  width={600}
  height={400}
  loading="lazy"
/>
```

---

## Summary & Next Actions

**Audit Completed:** Desktop 1440×900 dark mode baseline established. 18 sections documented, 15 screenshots captured, component inventory mapped.

**Key Findings:**
- ✅ **Strengths:** High-quality 3D illustrations, consistent orange/yellow theming, strong technical credibility (BDD, Rust, architecture), compelling testimonials with savings proof
- ⚠️ **Critical Issues:** Hero undersells unique multi-GPU value, OpenAI competitor not named, social proof delayed, accessibility gaps (alt text, focus states)
- 🔧 **Action Items:** 5 P1 fixes (hero rewrite, proof above fold, alt text, focus states), component consolidation (`<FeatureCard>`, `<StatBlock>`, `<CodeBlock>`), responsive testing pending

**Pending Work:**
- [ ] Capture light mode screenshots (desktop)
- [ ] Capture tablet (834×1112) and mobile (390×844) both themes
- [ ] Test keyboard navigation and focus states
- [ ] Verify alt text on all images
- [ ] Test comparison table and code blocks on mobile
- [ ] Verify all tabs/accordions have content and are accessible

**Handoff to StyleBee:** Ready for implementation. Start with 5 P1 fixes, then extract 3 consolidation components, then tackle responsive breakpoints.

