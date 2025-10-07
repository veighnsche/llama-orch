#!/usr/bin/env python3
"""
TEAM VAN GOGH - Extract output_norm.weight from GGUF file
"""
import struct
import sys
import numpy as np

def read_gguf_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def skip_metadata_value(f):
    value_type = struct.unpack('<I', f.read(4))[0]
    
    if value_type in [0, 1, 7]:  # uint8, int8, bool
        f.read(1)
    elif value_type in [2, 3]:  # uint16, int16
        f.read(2)
    elif value_type in [4, 5, 6]:  # uint32, int32, float32
        f.read(4)
    elif value_type == 8:  # string
        read_gguf_string(f)
    elif value_type == 9:  # array
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
    elif value_type in [10, 11, 12]:  # uint64, int64, float64
        f.read(8)

def extract_tensor(gguf_path, tensor_name):
    with open(gguf_path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:  # "GGUF"
            raise ValueError("Invalid GGUF magic")
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        # Skip metadata
        for _ in range(metadata_count):
            read_gguf_string(f)  # key
            skip_metadata_value(f)  # value
        
        # Read tensor info
        for _ in range(tensor_count):
            name = read_gguf_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(n_dims * 8))
            tensor_type = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            if name == tensor_name:
                # Calculate size
                size = 1
                for d in dims:
                    size *= d
                
                # Read tensor data
                f.seek(offset)
                
                # Type 1 = FP16
                if tensor_type == 1:
                    data = np.frombuffer(f.read(size * 2), dtype=np.float16)
                elif tensor_type == 0:  # FP32
                    data = np.frombuffer(f.read(size * 4), dtype=np.float32)
                else:
                    print(f"Warning: Unsupported type {tensor_type}, reading as bytes")
                    data = np.frombuffer(f.read(size * 2), dtype=np.float16)
                
                return {
                    'name': name,
                    'dims': dims,
                    'type': tensor_type,
                    'data': data
                }
    
    raise ValueError(f"Tensor {tensor_name} not found")

if __name__ == '__main__':
    gguf_path = '/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf'
    tensor_name = 'output_norm.weight'
    
    print(f"[TEAM VAN GOGH] Extracting {tensor_name} from GGUF file...")
    
    tensor = extract_tensor(gguf_path, tensor_name)
    
    print(f"\nTensor: {tensor['name']}")
    print(f"Dimensions: {tensor['dims']}")
    print(f"Type: {tensor['type']} (1=FP16, 0=FP32)")
    print(f"Total elements: {len(tensor['data'])}")
    
    data = tensor['data'].astype(np.float32)
    
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std:  {np.std(data):.6f}")
    print(f"  Min:  {np.min(data):.6f}")
    print(f"  Max:  {np.max(data):.6f}")
    
    print(f"\nFirst 20 values:")
    for i in range(min(20, len(data))):
        print(f"  [{i:2d}] {data[i]:.6f}")
    
    # Save to file
    output_file = '/home/vince/Projects/llama-orch/bin/worker-orcd/investigation-teams/artifacts/van_gogh/output_norm_from_gguf.txt'
    with open(output_file, 'w') as f:
        f.write(f"TEAM VAN GOGH - output_norm.weight from GGUF\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Source: {gguf_path}\n")
        f.write(f"Tensor: {tensor['name']}\n")
        f.write(f"Dimensions: {tensor['dims']}\n")
        f.write(f"Type: {tensor['type']} (1=FP16, 0=FP32)\n")
        f.write(f"Total elements: {len(tensor['data'])}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Mean: {np.mean(data):.6f}\n")
        f.write(f"  Std:  {np.std(data):.6f}\n")
        f.write(f"  Min:  {np.min(data):.6f}\n")
        f.write(f"  Max:  {np.max(data):.6f}\n\n")
        f.write(f"First 20 values:\n")
        for i in range(min(20, len(data))):
            f.write(f"  [{i:2d}] {data[i]:.6f}\n")
        f.write(f"\nAll {len(data)} values:\n")
        for i in range(len(data)):
            f.write(f"  [{i:3d}] {data[i]:.6f}\n")
    
    print(f"\n✅ Saved to: {output_file}")
