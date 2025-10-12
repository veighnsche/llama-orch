# Quick Image Generation Guide

## 🎯 What You Need

Generate an **isometric 3D illustration** of a homelab AI setup for the hero section.

**File:** `homelab-network.png` (1280×720)  
**Location:** Already placed at `/public/images/homelab-network.png` (placeholder active)

---

## 🚀 Quick Start (Copy & Paste)

### Option 1: Midjourney (Recommended)

```
/imagine isometric 3D illustration, distributed AI homelab orchestration network, small mini PC hub in center connected to multiple gaming PCs in spoke pattern, three large gaming towers with glowing GPUs (RTX 4090 visible through glass), one workstation, tiny Mac Studio in background, amber ethernet cables radiating from central hub, floating UI overlay showing "5 nodes 8 GPUs" and "Cost: $0.00/hr" in green, deep navy background #0f172a, cinematic volumetric lighting highlighting the hub, modern tech aesthetic, clear visual hierarchy, gaming PCs most prominent, 1280x720, --ar 16:9 --style raw --v 6
```

### Option 2: DALL-E 3 (ChatGPT Plus)

```
Create an isometric 3D illustration of a homelab AI orchestration network. CENTER: small mini PC (Intel NUC) acting as hub/orchestrator with glowing amber LED. AROUND IT: 5 worker computers in spoke pattern - three large gaming PC towers (most prominent, with visible GPUs through glass panels), one professional workstation, and one tiny Mac Studio (smallest, in background). Thick orange ethernet cables connect hub to each worker in star pattern. Gaming PCs should be largest with clearly visible graphics cards. Add floating UI panel showing "5 nodes / 8 GPUs" and "Cost: $0.00/hr" in green. Dark navy background (#0f172a), cinematic lighting from top. Modern, minimal style. 1280x720 pixels.
```

### Option 3: Stable Diffusion (ComfyUI/Automatic1111)

**Positive Prompt:**
```
isometric 3d illustration, distributed homelab AI network, mini PC hub center, multiple gaming PC towers with visible GPUs, hub-spoke topology, amber orange ethernet cables radiating from center, gaming PCs largest and most prominent, small mac studio in background, floating ui overlay, gpu pool metrics, cost $0 badge in green, navy blue background, volumetric lighting, god rays, modern tech aesthetic, clear visual hierarchy, high detail on gaming hardware, soft shadows, professional render, 16:9 aspect ratio
```

**Negative Prompt:**
```
photorealistic, cluttered, messy, low quality, blurry, text artifacts, watermark, mac prominent, equal sizing
```

---

## 📋 Key Visual Elements

### Must Have:
1. ✅ **Mini PC hub** in CENTER (Intel NUC style)
   - Small, compact orchestrator
   - Glowing amber LED
   - All cables radiate FROM this hub

2. ✅ **5 worker nodes** in spoke pattern around hub:
   - **Gaming PC #1** (LARGE, front-left) - visible GPU through glass
   - **Gaming PC #2** (LARGE, front-right) - dual-GPU setup visible
   - **Gaming PC #3** (medium, back-left) - single GPU visible
   - **Workstation** (large, back-right) - professional, multi-GPU
   - **Mac Studio** (SMALLEST, back-center) - tiny, minimal, background

3. ✅ **Hub-spoke network** topology
   - Amber/orange cables (#f59e0b) radiating from center
   - NOT mesh - star pattern only
   - Data packets flowing FROM hub TO workers

4. ✅ **Clear size hierarchy**:
   - Gaming PCs = LARGEST (most prominent)
   - Workstation = Large
   - Mini PC hub = Medium (but central)
   - Mac Studio = SMALLEST (background element)

5. ✅ **Labels**: "Orchestrator", "Gaming PC 1/2/3", "Workstation", "Mac Studio"

6. ✅ **UI overlay** showing:
   - "5 nodes / 8 GPUs"
   - Network topology diagram
   - "Cost: $0.00/hr" in green (#10b981)

7. ✅ **Navy background** (#0f172a)

8. ✅ **Cinematic lighting** highlighting the central hub

### Style:
- Isometric 3D (30° angle)
- Clear visual hierarchy (gaming PCs dominant)
- Modern tech aesthetic
- Gaming hardware prominently featured
- Not photorealistic

---

## 🎨 Exact Color Codes

| Element | Color | Hex |
|---------|-------|-----|
| Background | Navy Blue | `#0f172a` |
| Cables & GPU glow | Amber/Orange | `#f59e0b` |
| Cost badge | Emerald Green | `#10b981` |
| Labels | Off-white | `#f1f5f9` |
| Network activity | Blue | `#3b82f6` |

---

## 📥 After Generation

1. **Download** your generated image
2. **Resize** to exactly 1280×720 if needed
3. **Save as** `homelab-network.png`
4. **Replace** the file at `/public/images/homelab-network.png`
5. **Refresh** browser at http://localhost:3000

The image will appear in the hero section on large screens (desktop/laptop).

---

## 🔍 Current Status

✅ Placeholder created (gray with text)  
✅ Component ready and waiting  
✅ Alt text configured (detailed accessibility description)  
✅ Dev server running on http://localhost:3000  

**Next:** Generate image → Replace placeholder → Done!

---

## 💡 Tips

- **File size:** Keep under 500KB (use TinyPNG.com to compress)
- **Format:** PNG preferred (supports transparency)
- **Quality:** High detail but web-optimized
- **Visual hierarchy:** Gaming PCs should be LARGEST and most eye-catching
- **Mac Studio:** Should be smallest, tucked in background, barely noticeable
- **Network:** Clear hub-spoke pattern (NOT mesh), cables radiate from center
- **Mood:** "Turn idle gaming PCs into AI infrastructure" - gaming hardware repurposed for AI

---

## 📖 Full Details

See `README.md` in this directory for:
- Complete detailed prompt (all specifications)
- Manual design guidelines (Figma/Blender)
- Technical requirements
- Usage information

---

**Quick check:** Visit http://localhost:3000 to see the placeholder in action!
