#!/usr/bin/env python3
"""
Inspect weight values directly from GGUF file to verify they're correct.
"""
import struct
import sys

def read_gguf_metadata(path):
    """Read GGUF file and extract first few values of attn_q.weight for layer 0"""
    with open(path, 'rb') as f:
        # Read GGUF header
        magic = f.read(4)
        if magic != b'GGUF':
            print(f"Not a GGUF file: {magic}")
            return
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF version: {version}")
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Tensors: {tensor_count}, Metadata KV: {metadata_kv_count}")
        
        # Skip metadata KV pairs (we don't need them for this check)
        for _ in range(metadata_kv_count):
            # Read key length and key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            # Read value type
            val_type = struct.unpack('<I', f.read(4))[0]
            
            # Skip value based on type
            if val_type == 4:  # String
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif val_type == 5:  # Array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                if arr_type == 4:  # String array
                    for _ in range(arr_len):
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        f.read(s_len)
                else:
                    # Numeric array
                    type_sizes = {0: 1, 1: 1, 2: 2, 3: 4, 6: 4, 7: 8, 8: 1}
                    f.read(arr_len * type_sizes.get(arr_type, 4))
            elif val_type in [6, 7]:  # uint32, int32
                f.read(4)
            elif val_type in [8, 9]:  # float32, bool
                f.read(4)
            elif val_type in [10, 11]:  # uint64, int64
                f.read(8)
            elif val_type == 12:  # float64
                f.read(8)
            else:
                print(f"Unknown type {val_type} for key {key}")
        
        # Read tensor info
        tensors = {}
        for _ in range(tensor_count):
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Read dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            
            # Read type and offset
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            tensors[name] = {
                'dims': dims,
                'dtype': dtype,
                'offset': offset
            }
        
        # Find alignment and data start
        alignment = 32  # Default GGUF alignment
        current_pos = f.tell()
        data_start = (current_pos + alignment - 1) // alignment * alignment
        
        # Read blk.0.attn_q.weight
        target = 'blk.0.attn_q.weight'
        if target not in tensors:
            print(f"Tensor {target} not found!")
            return
        
        info = tensors[target]
        print(f"\n{target}:")
        print(f"  Dimensions: {info['dims']}")
        print(f"  Type: {info['dtype']}")
        print(f"  Offset: {info['offset']}")
        
        # Seek to tensor data
        f.seek(data_start + info['offset'])
        
        # Read first 10 FP16 values (2 bytes each)
        if info['dtype'] == 1:  # FP16
            data = f.read(20)  # 10 * 2 bytes
            values = struct.unpack('<10e', data)  # 'e' is half-precision float
            print(f"  First 10 values: {[f'{v:.4f}' for v in values]}")
        else:
            print(f"  Unexpected dtype: {info['dtype']}")

if __name__ == '__main__':
    path = '/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf'
    read_gguf_metadata(path)
