#!/usr/bin/env python3
"""
Download GPT-2 from HuggingFace and extract weights for llorch-cpud validation.

This script:
1. Downloads GPT-2 base (124M) from HuggingFace
2. Extracts first transformer block weights to numpy
3. Generates reference outputs for checkpoint validation

Usage:
    python3 extract_gpt2_weights.py [output_dir]
    
Default output_dir: ../../.test-models/gpt2/extracted_weights
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("GPT-2 Weight Extraction from HuggingFace")
    print("=" * 60)
    print()
    
    # Output directory
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "../../.test-models/gpt2/extracted_weights"
    
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Use GPT-2 base (124M) to match GGUF
    model_name = "gpt2"  # base model, 124M params
    
    print(f"Loading {model_name} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    
    print(f"✅ Model loaded: {model_name}")
    print(f"   Parameters: ~124M")
    print(f"   Hidden size: 768")
    print(f"   Layers: 12")
    print(f"   Heads: 12")
    print()
    
    # Extract first transformer block weights
    block_0 = model.transformer.h[0]
    
    # LayerNorm 1
    ln_1_weight = block_0.ln_1.weight.detach().numpy()
    ln_1_bias = block_0.ln_1.bias.detach().numpy()
    
    # Attention (c_attn combines Q, K, V)
    c_attn_weight = block_0.attn.c_attn.weight.detach().numpy()  # [768, 2304]
    c_attn_bias = block_0.attn.c_attn.bias.detach().numpy()      # [2304]
    
    # Token embeddings
    token_emb = model.transformer.wte.weight.detach().numpy()
    pos_emb = model.transformer.wpe.weight.detach().numpy()
    
    # Save all weights
    np.save(output_dir / "h0_ln_1_weight.npy", ln_1_weight)
    np.save(output_dir / "h0_ln_1_bias.npy", ln_1_bias)
    np.save(output_dir / "h0_c_attn_weight.npy", c_attn_weight)
    np.save(output_dir / "h0_c_attn_bias.npy", c_attn_bias)
    np.save(output_dir / "token_embeddings.npy", token_emb)
    np.save(output_dir / "position_embeddings.npy", pos_emb)
    
    print("✅ Weights extracted:")
    print(f"  ln_1.weight: {ln_1_weight.shape}")
    print(f"  ln_1.bias: {ln_1_bias.shape}")
    print(f"  c_attn.weight: {c_attn_weight.shape}")
    print(f"  c_attn.bias: {c_attn_bias.shape}")
    print(f"  token_emb: {token_emb.shape}")
    print(f"  pos_emb: {pos_emb.shape}")
    print()
    
    # Generate reference outputs for "Hello."
    prompt = "Hello."
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"📝 Test prompt: '{prompt}'")
    print(f"   Tokens: {tokens.tolist()[0]}")
    print()
    
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.shape[1]))
        
        # First LayerNorm
        ln_1_output = block_0.ln_1(inputs_embeds)
        
        # QKV projection
        qkv = torch.nn.functional.linear(ln_1_output, c_attn_weight.T, torch.from_numpy(c_attn_bias))
        
        # Reshape and split
        batch, seq, _ = qkv.shape
        qkv_reshaped = qkv.view(batch, seq, 3, 12, 64)  # GPT-2 base: 12 heads, 64 dim
        
        q = qkv_reshaped[:, :, 0, :, :]
        k = qkv_reshaped[:, :, 1, :, :]
        v = qkv_reshaped[:, :, 2, :, :]
    
    # Save reference outputs
    np.save(output_dir / "embeddings.npy", inputs_embeds.numpy())
    np.save(output_dir / "checkpoint_01_ln1_output.npy", ln_1_output.numpy())
    np.save(output_dir / "checkpoint_02_q.npy", q.numpy())
    np.save(output_dir / "checkpoint_02_k.npy", k.numpy())
    np.save(output_dir / "checkpoint_02_v.npy", v.numpy())
    
    print("✅ Reference outputs generated:")
    print(f"  embeddings: {inputs_embeds.shape}")
    print(f"  checkpoint_01_ln1_output: {ln_1_output.shape}")
    print(f"  checkpoint_02_q: {q.shape}")
    print(f"  checkpoint_02_k: {k.shape}")
    print(f"  checkpoint_02_v: {v.shape}")
    print()
    
    # Metadata
    metadata = {
        'model': 'gpt2-124m',
        'source': 'HuggingFace transformers',
        'prompt': prompt,
        'tokens': tokens.tolist()[0],
        'hidden_size': 768,
        'n_heads': 12,
        'head_dim': 64,
        'n_layers': 12,
        'vocab_size': 50257,
        'files': {
            'weights': [
                'h0_ln_1_weight.npy',
                'h0_ln_1_bias.npy',
                'h0_c_attn_weight.npy',
                'h0_c_attn_bias.npy',
                'token_embeddings.npy',
                'position_embeddings.npy',
            ],
            'reference_outputs': [
                'embeddings.npy',
                'checkpoint_01_ln1_output.npy',
                'checkpoint_02_q.npy',
                'checkpoint_02_k.npy',
                'checkpoint_02_v.npy',
            ]
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to metadata.json")
    print()
    print("=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Files created:")
    for f in sorted(output_dir.glob("*.npy")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:<35} ({size_mb:.2f} MB)")
    print(f"  - metadata.json")
    print()
    print("Next steps:")
    print("  1. cd bin/llorch-cpud")
    print("  2. cargo test --test real_gpt2_checkpoint_01 -- --nocapture")
    print("  3. cargo test --test real_gpt2_checkpoint_02 -- --nocapture")

if __name__ == "__main__":
    main()
