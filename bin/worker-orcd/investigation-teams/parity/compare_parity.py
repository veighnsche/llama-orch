#!/usr/bin/env python3
"""
TEAM PICASSO - Parity Comparison Script
Compares logits between llama.cpp and worker-orcd
"""

import json
import sys
from statistics import mean
from pathlib import Path

def load_jsonl(path):
    """Load JSONL file and return list of records"""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def compute_diff(values_a, values_b):
    """Compute max and mean absolute difference"""
    n = min(len(values_a), len(values_b))
    diffs = [abs(values_a[i] - values_b[i]) for i in range(n)]
    return max(diffs), mean(diffs)

def main():
    # Load both JSONL files
    try:
        llama_records = load_jsonl("llama_hidden_states.jsonl")
        our_records = load_jsonl("our_hidden_states.jsonl")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nExpected files:")
        print("  - llama_hidden_states.jsonl (llama.cpp ground truth)")
        print("  - our_hidden_states.jsonl (worker-orcd output)")
        sys.exit(1)
    
    # Filter for logits only and index by token_idx
    llama_logits = {
        r["token_idx"]: r 
        for r in llama_records 
        if r.get("checkpoint") == "logits"
    }
    our_logits = {
        r["token_idx"]: r 
        for r in our_records 
        if r.get("checkpoint") == "logits"
    }
    
    # Find common token indices
    common_tokens = sorted(set(llama_logits.keys()) & set(our_logits.keys()))
    
    if not common_tokens:
        print("ERROR: No common (token_idx, 'logits') records found")
        print(f"llama.cpp has {len(llama_logits)} logit entries")
        print(f"worker-orcd has {len(our_logits)} logit entries")
        sys.exit(2)
    
    # Print header
    print("# TEAM PICASSO Parity Report")
    print(f"# llama.cpp entries: {len(llama_logits)}")
    print(f"# worker-orcd entries: {len(our_logits)}")
    print(f"# Common tokens: {len(common_tokens)}")
    print("#")
    print("token_idx,max_abs_diff,mean_abs_diff,llama_team,our_team")
    
    # Compare each common token
    for token_idx in common_tokens:
        llama_vals = llama_logits[token_idx]["values"]
        our_vals = our_logits[token_idx]["values"]
        
        max_diff, mean_diff = compute_diff(llama_vals, our_vals)
        
        llama_team = llama_logits[token_idx].get("team", "unknown")
        our_team = our_logits[token_idx].get("team", "unknown")
        
        print(f"{token_idx},{max_diff:.6e},{mean_diff:.6e},{llama_team},{our_team}")

if __name__ == "__main__":
    main()
