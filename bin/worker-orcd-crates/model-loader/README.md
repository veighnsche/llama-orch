# model-loader

**Loads model weights from RAM cache into VRAM**

Coordinates with pool-managerd's model-cache to fetch model weights and load them into GPU memory. Works with vram-residency to ensure proper pinning.

**Key responsibilities:**
- Fetch model weights from pool-managerd's RAM cache
- Load weights into VRAM via CUDA
- Verify model integrity (checksums)
- Report load progress and completion