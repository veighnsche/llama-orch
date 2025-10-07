#!/usr/bin/env python3
"""
TEAM PRINTER - Convert binary checkpoint files to numpy .npz format
"""

import numpy as np
import json
import sys
from pathlib import Path

def convert_manifest_to_npz(manifest_path: Path, output_npz: Path):
    """Convert checkpoint manifest + binary files to single .npz file."""
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    checkpoints = manifest['checkpoints']
    print(f"[TEAM PRINTER] Loading {len(checkpoints)} checkpoints...")
    
    arrays = {}
    for cp in checkpoints:
        name = cp['name']
        token_idx = cp['token_idx']
        count = cp['count']
        data_file = Path(cp['file'])
        
        # Read binary data
        data = np.fromfile(data_file, dtype=np.float32, count=count)
        
        # Store with key: name_tokN
        key = f"{name}_tok{token_idx}"
        arrays[key] = data
        
        print(f"  {key}: shape={data.shape}, min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
    
    # Save to npz
    np.savez(output_npz, **arrays)
    print(f"[TEAM PRINTER] Saved {len(arrays)} arrays to {output_npz}")
    
    # Clean up binary files
    for cp in checkpoints:
        data_file = Path(cp['file'])
        if data_file.exists():
            data_file.unlink()
    
    # Clean up manifest
    manifest_path.unlink()
    print(f"[TEAM PRINTER] Cleaned up temporary files")

def main():
    if len(sys.argv) < 2:
        print("Usage: convert_to_npz.py <manifest.json>")
        print("Example: convert_to_npz.py ours.checkpoints.npz.manifest.json")
        sys.exit(1)
    
    manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        print(f"❌ ERROR: {manifest_path} not found")
        sys.exit(1)
    
    # Output npz path: remove .manifest.json suffix
    output_npz = Path(str(manifest_path).replace('.manifest.json', ''))
    
    convert_manifest_to_npz(manifest_path, output_npz)
    print("[TEAM PRINTER] ✅ Conversion complete")

if __name__ == "__main__":
    main()
