#!/usr/bin/env python3
"""List all tensors in GGUF file"""
import sys
sys.path.insert(0, '/home/vince/Projects/llama-orch/bin/worker-orcd')

# Use the worker-gguf crate's Python bindings if available, otherwise manual parse
import struct

def read_gguf_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def skip_metadata_value(f):
    value_type = struct.unpack('<I', f.read(4))[0]
    
    if value_type in [0, 1, 7]:
        f.read(1)
    elif value_type in [2, 3]:
        f.read(2)
    elif value_type in [4, 5, 6]:
        f.read(4)
    elif value_type == 8:
        read_gguf_string(f)
    elif value_type == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            if elem_type == 8:
                read_gguf_string(f)
            elif elem_type <= 7:
                elem_size = 1 if elem_type in [0, 1, 7] else (2 if elem_type in [2, 3] else 4)
                f.read(elem_size)
            else:
                f.read(8)
    elif value_type in [10, 11, 12]:
        f.read(8)

gguf_path = '/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf'

with open(gguf_path, 'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    metadata_count = struct.unpack('<Q', f.read(8))[0]
    
    print(f"GGUF version {version}, {tensor_count} tensors, {metadata_count} metadata entries")
    
    # Skip metadata
    for _ in range(metadata_count):
        read_gguf_string(f)
        skip_metadata_value(f)
    
    # Read tensor info
    print("\nTensors containing 'norm':")
    for _ in range(tensor_count):
        name = read_gguf_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}Q', f.read(n_dims * 8))
        tensor_type = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        
        if 'norm' in name:
            print(f"  {name}: type={tensor_type}, dims={dims}, offset={offset}")
