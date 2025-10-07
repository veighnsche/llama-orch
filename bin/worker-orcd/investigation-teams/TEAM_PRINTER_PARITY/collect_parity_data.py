#!/usr/bin/env python3
"""
TEAM PRINTER - Parity Data Collection and Comparison Script

Collects checkpoint tensors from both our engine and llama.cpp,
then generates a detailed diff report.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_checkpoints(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load checkpoint tensors from npz file."""
    return dict(np.load(npz_path))

def compute_stats(arr: np.ndarray) -> Dict:
    """Compute min/max/mean statistics for an array."""
    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'shape': list(arr.shape)
    }

def compute_diff_stats(ours: np.ndarray, theirs: np.ndarray) -> Dict:
    """Compute difference statistics between two arrays."""
    if ours.shape != theirs.shape:
        return {
            'error': f'Shape mismatch: ours={ours.shape} vs theirs={theirs.shape}'
        }
    
    diff = ours - theirs
    abs_diff = np.abs(diff)
    
    # Find first significant mismatch (threshold: 1e-5)
    threshold = 1e-5
    mismatches = np.where(abs_diff > threshold)
    first_mismatch_idx = None
    if len(mismatches[0]) > 0:
        first_idx = mismatches[0][0]
        if len(ours.shape) > 1:
            first_mismatch_idx = tuple(m[0] for m in mismatches)
        else:
            first_mismatch_idx = int(first_idx)
    
    return {
        'l2_norm': float(np.linalg.norm(diff)),
        'linf_norm': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'max_abs_diff': float(np.max(abs_diff)),
        'first_mismatch_idx': first_mismatch_idx,
        'first_mismatch_ours': float(ours.flat[mismatches[0][0]]) if first_mismatch_idx is not None else None,
        'first_mismatch_theirs': float(theirs.flat[mismatches[0][0]]) if first_mismatch_idx is not None else None,
        'num_mismatches': int(np.sum(abs_diff > threshold)),
        'mismatch_percentage': float(100.0 * np.sum(abs_diff > threshold) / abs_diff.size)
    }

def generate_diff_report(ours_path: Path, theirs_path: Path, output_path: Path):
    """Generate comprehensive diff report."""
    
    print(f"[TEAM PRINTER] Loading checkpoints...")
    ours = load_checkpoints(ours_path)
    theirs = load_checkpoints(theirs_path)
    
    print(f"[TEAM PRINTER] Our checkpoints: {len(ours)} tensors")
    print(f"[TEAM PRINTER] Their checkpoints: {len(theirs)} tensors")
    
    report_lines = []
    report_lines.append("# TEAM PRINTER - Parity Diff Report")
    report_lines.append("")
    report_lines.append("**Generated:** 2025-10-07T01:24:35Z")
    report_lines.append("**Mission:** Side-by-side comparison of CUDA/Rust engine vs llama.cpp")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Find common checkpoints
    common_keys = sorted(set(ours.keys()) & set(theirs.keys()))
    only_ours = sorted(set(ours.keys()) - set(theirs.keys()))
    only_theirs = sorted(set(theirs.keys()) - set(ours.keys()))
    
    if only_ours:
        report_lines.append(f"⚠️ **Checkpoints only in ours:** {', '.join(only_ours)}")
        report_lines.append("")
    if only_theirs:
        report_lines.append(f"⚠️ **Checkpoints only in theirs:** {', '.join(only_theirs)}")
        report_lines.append("")
    
    report_lines.append(f"## Checkpoint Comparison ({len(common_keys)} common)")
    report_lines.append("")
    
    first_divergence = None
    
    for key in common_keys:
        our_arr = ours[key]
        their_arr = theirs[key]
        
        report_lines.append(f"### {key}")
        report_lines.append("")
        
        # Our stats
        our_stats = compute_stats(our_arr)
        report_lines.append(f"**Ours:** shape={our_stats['shape']}, min={our_stats['min']:.6f}, max={our_stats['max']:.6f}, mean={our_stats['mean']:.6f}")
        
        # Their stats
        their_stats = compute_stats(their_arr)
        report_lines.append(f"**Theirs:** shape={their_stats['shape']}, min={their_stats['min']:.6f}, max={their_stats['max']:.6f}, mean={their_stats['mean']:.6f}")
        
        # Diff stats
        diff_stats = compute_diff_stats(our_arr, their_arr)
        if 'error' in diff_stats:
            report_lines.append(f"**❌ ERROR:** {diff_stats['error']}")
        else:
            report_lines.append(f"**Diff:** L2={diff_stats['l2_norm']:.6e}, L∞={diff_stats['linf_norm']:.6e}, mean_abs={diff_stats['mean_abs_diff']:.6e}")
            
            if diff_stats['first_mismatch_idx'] is not None:
                report_lines.append(f"**First mismatch (>1e-5):** index={diff_stats['first_mismatch_idx']}, ours={diff_stats['first_mismatch_ours']:.6f}, theirs={diff_stats['first_mismatch_theirs']:.6f}")
                report_lines.append(f"**Mismatches:** {diff_stats['num_mismatches']} / {our_arr.size} ({diff_stats['mismatch_percentage']:.2f}%)")
                
                if first_divergence is None and diff_stats['linf_norm'] > 1e-3:
                    first_divergence = key
                    report_lines.append(f"**🔥 FIRST SIGNIFICANT DIVERGENCE DETECTED (L∞ > 1e-3)**")
            else:
                report_lines.append(f"**✅ MATCH:** All values within tolerance (1e-5)")
        
        report_lines.append("")
    
    # Summary
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    if first_divergence:
        report_lines.append(f"**🎯 First significant divergence:** `{first_divergence}`")
        report_lines.append("")
        report_lines.append("This is the earliest checkpoint where our values diverge from llama.cpp.")
        report_lines.append("Investigation should focus on the computation leading to this checkpoint.")
    else:
        report_lines.append("**✅ No significant divergences detected**")
        report_lines.append("")
        report_lines.append("All checkpoints match within tolerance. If output is still incorrect,")
        report_lines.append("the bug may be in sampling, tokenizer decode, or a checkpoint we didn't capture.")
    
    # Write report
    output_path.write_text('\n'.join(report_lines))
    print(f"[TEAM PRINTER] Diff report written to: {output_path}")

def main():
    base_dir = Path(__file__).parent
    ours_path = base_dir / "ours.checkpoints.npz"
    theirs_path = base_dir / "llamacpp.checkpoints.npz"
    output_path = base_dir / "diff_report.md"
    
    if not ours_path.exists():
        print(f"❌ ERROR: {ours_path} not found")
        print("Run our engine with checkpoint logging first.")
        sys.exit(1)
    
    if not theirs_path.exists():
        print(f"❌ ERROR: {theirs_path} not found")
        print("Run llama.cpp with checkpoint logging first.")
        sys.exit(1)
    
    generate_diff_report(ours_path, theirs_path, output_path)
    print("[TEAM PRINTER] ✅ Parity data collection complete")

if __name__ == "__main__":
    main()
