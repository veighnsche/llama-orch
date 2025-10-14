# Hero Section Images

## Required Image

### homelab-network.png

**Dimensions:** 1280×720 (16:9 aspect ratio)  
**Location:** `/public/images/homelab-network.png`  
**Status:** ⚠️ Placeholder active — replace with generated image

---

## Detailed Image Generation Prompt

Use this prompt with AI image generation tools (Midjourney, DALL-E 3, Stable Diffusion XL, etc.):

```
Isometric 3D illustration of a distributed homelab AI orchestration network in a cozy, dimly-lit room with warm ambient lighting.

NETWORK TOPOLOGY (Hub-and-Spoke):
CENTRAL HUB (most important, visually centered):
- Small mini PC (Intel NUC or similar compact form factor) in the CENTER of the composition
- Acting as orchestrator/coordinator node
- Glowing amber status LED on front
- Compact and unassuming but clearly the network hub
- Slightly elevated on a small platform or stand
- All cables radiate outward FROM this hub

WORKER NODES (arranged in spoke pattern around the hub):
1. Gaming PC #1 (front-left, LARGE and prominent) - full tower case with tempered glass side panel, large illuminated GPU (RTX 4090 style) clearly visible, RGB fans, aggressive gaming aesthetic, glowing components
2. Gaming PC #2 (front-right, LARGE and prominent) - mid-tower with visible dual-GPU setup through glass panel, amber LED strips, modern gaming build
3. Gaming PC #3 (back-left, medium size) - compact gaming build with single high-end GPU visible through mesh panel, colorful RGB accents
4. Workstation tower (back-right, medium-large) - professional black case with subtle RGB, multiple GPU configuration visible, server-like appearance
5. Mac Studio (back-center, SMALLEST and least prominent) - tiny silver aluminum cube, minimal presence, tucked in background, barely noticeable

SIZE HIERARCHY (visual importance):
- Mini PC hub: MEDIUM (central, elevated, hub of activity)
- Gaming PCs: LARGEST (3 gaming towers, most prominent, GPUs clearly visible)
- Workstation: LARGE (professional but less flashy than gaming PCs)
- Mac Studio: SMALLEST (background element, minimal presence)

NETWORK CONNECTIONS:
- Thick amber/orange (#f59e0b) ethernet cables connecting mini PC hub to EACH worker node
- Clear star/hub-spoke pattern (NOT mesh - all cables go through the hub)
- Cables have slight glow effect
- Small animated data packets (glowing dots) flowing FROM mini PC hub TO worker nodes (showing orchestration)
- Visual emphasis on the centralized coordination

LABELS & UI OVERLAY:
- Mini PC labeled: "Orchestrator" (prominent label)
- Gaming PCs labeled: "Gaming PC 1", "Gaming PC 2", "Gaming PC 3"
- Workstation labeled: "Workstation"
- Mac Studio labeled: "Mac Studio" (small, subtle label)
- Semi-transparent UI overlay panel (frosted glass effect) floating above showing:
  * Network topology diagram (hub-spoke visualization)
  * GPU pool metrics: "5 nodes / 8 GPUs"
  * Real-time orchestration status
  * Prominent badge: "Cost: $0.00/hr" in bright emerald green (#10b981) with subtle glow
  * Small icons: orchestration icon, GPU pool icon, network mesh icon

BACKGROUND & LIGHTING:
- Deep navy blue background (#0f172a) with subtle gradient to darker edges (#0a0f1a)
- Cinematic volumetric god rays (light beams) streaming from top-left corner at 45-degree angle
- Soft ambient occlusion shadows beneath each computer
- Subtle desk/surface beneath the computers (dark wood or matte black)
- Atmospheric depth with slight fog/haze effect

STYLE & MOOD:
- Isometric perspective: 30-degree angle, slight tilt for dynamic composition
- Minimal but detailed: focus on hardware components, clean lines
- Technical but approachable: not too sci-fi, grounded in reality
- High detail on GPU fans, cable connectors, computer ports
- Soft shadows and realistic materials (metal, glass, plastic)
- Modern tech aesthetic: 2024-2025 hardware design language

COLOR PALETTE:
- Background: Navy blue (#0f172a) to dark navy (#0a0f1a)
- Primary accents: Amber/orange (#f59e0b) for cables, GPU glow, highlights
- Success indicator: Emerald green (#10b981) for cost badge
- Labels & text: Off-white (#f1f5f9)
- Secondary highlights: Subtle blue (#3b82f6) for network activity
- Shadows: Deep black with 40% opacity

MOOD & NARRATIVE:
Empowering, professional, "turn idle gaming PCs into AI infrastructure" vibe. Should communicate: distributed GPU orchestration, central coordination by mini PC, gaming hardware repurposed for AI, zero cloud costs, homelab pride. Visual story: small orchestrator coordinates powerful gaming GPUs across the house. Warm and inviting but technically credible.

TECHNICAL SPECS:
- Resolution: 1280×720 pixels (16:9)
- Format: PNG with transparency support (or solid navy background)
- Style: 3D isometric illustration, not photorealistic
- Detail level: High, but optimized for web (not overly complex)
- File size target: <500KB after optimization
```

---

## Alternative Prompts for Different Tools

### For Midjourney
```
isometric 3D illustration, distributed AI homelab orchestration network, small mini PC hub in center connected to multiple gaming PCs in spoke pattern, three large gaming towers with glowing GPUs (RTX 4090 visible through glass), one workstation, tiny Mac Studio in background, amber ethernet cables radiating from central hub, floating UI overlay showing "5 nodes 8 GPUs" and "Cost: $0.00/hr" in green, deep navy background #0f172a, cinematic volumetric lighting highlighting the hub, modern tech aesthetic, clear visual hierarchy, gaming PCs most prominent, 1280x720, --ar 16:9 --style raw --v 6
```

### For DALL-E 3
```
Create an isometric 3D illustration of a homelab AI orchestration network. CENTER: small mini PC (Intel NUC) acting as hub/orchestrator with glowing amber LED. AROUND IT: 5 worker computers in spoke pattern - three large gaming PC towers (most prominent, with visible GPUs through glass panels), one professional workstation, and one tiny Mac Studio (smallest, in background). Thick orange ethernet cables connect hub to each worker in star pattern. Gaming PCs should be largest with clearly visible graphics cards. Add floating UI panel showing "5 nodes / 8 GPUs" and "Cost: $0.00/hr" in green. Dark navy background (#0f172a), cinematic lighting from top. Modern, minimal style. 1280x720 pixels.
```

### For Stable Diffusion XL
```
isometric 3d illustration, distributed homelab AI network, mini PC hub center, multiple gaming PC towers with visible GPUs, hub-spoke topology, amber orange ethernet cables radiating from center, gaming PCs largest and most prominent, small mac studio in background, floating ui overlay, gpu pool metrics, cost $0 badge in green, navy blue background, volumetric lighting, god rays, modern tech aesthetic, clear visual hierarchy, high detail on gaming hardware, soft shadows, professional render, 16:9 aspect ratio
Negative prompt: photorealistic, cluttered, messy, low quality, blurry, text artifacts, watermark, mac prominent, equal sizing
```

---

## Manual Design Guidelines

If creating manually in Figma, Blender, or similar:

1. **Canvas:** 1280×720px, navy background (#0f172a)
2. **Grid:** Set up isometric grid at 30° angle
3. **Layout:** Place mini PC hub in center, arrange 5 workers in spoke pattern around it
4. **Size hierarchy:** Gaming PCs = largest (3x), Workstation = large, Mac Studio = smallest
5. **Computers:** Use simple geometric shapes (boxes, cylinders), gaming PCs with visible GPU details
6. **Cables:** Bezier curves radiating from center hub, 8-12px stroke, amber color, add outer glow
7. **Labels:** Inter or Geist Sans font, 18-24px for gaming PCs, 14px for Mac Studio, white with 80% opacity
8. **UI Overlay:** Rounded rectangle with backdrop-blur effect, show "5 nodes / 8 GPUs", 20% white fill
9. **Lighting:** Add gradient overlay from top highlighting the central hub (white 5% opacity)
10. **Export:** PNG, optimize with TinyPNG or similar

---

## Placeholder Status

A temporary placeholder image has been created at:
```
/public/images/homelab-network.png
```

This shows a gray background with text indicating where to place your generated image. The Next.js Image component will display this until you replace it with the actual artwork.

**To replace:**
1. Generate image using one of the prompts above
2. Save as `homelab-network.png` (1280×720)
3. Place in `/public/images/` (overwrite placeholder)
4. Refresh browser to see new image

---

## Usage in Component

The image is used in `HeroSection.tsx`:
```tsx
<Image
  src="/images/homelab-network.png"
  width={1280}
  height={720}
  priority
  alt="[Detailed alt text for accessibility]"
/>
```

**Visibility:** Only shown on large screens (lg: 1024px+)  
**Purpose:** Visual storytelling to reinforce "your hardware, your rules" narrative
