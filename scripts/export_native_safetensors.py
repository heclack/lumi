#!/usr/bin/env python3
"""
Export Lumi native CUDA checkpoint to safetensors format for Candle inference.

Reads the raw binary tensor files + shape metadata from a native checkpoint
directory and writes a single .safetensors file.

Usage:
    python3 scripts/export_native_safetensors.py checkpoints/step-010000 \
        --output model.safetensors --config training_config.json
"""

import argparse
import json
import os
import sys
import numpy as np

try:
    from safetensors.numpy import save_file
except ImportError:
    os.system(f"{sys.executable} -m pip install safetensors numpy -q")
    from safetensors.numpy import save_file


def load_raw_tensor(bin_path, shape_path):
    """Load a raw f32 binary tensor with shape metadata."""
    data = np.fromfile(bin_path, dtype=np.float32)
    with open(shape_path, 'r') as f:
        shape_str = f.read().strip()
    shape = [int(x.strip()) for x in shape_str.strip('[]').split(',') if x.strip()]
    if shape:
        data = data.reshape(shape)
    return data


def main():
    parser = argparse.ArgumentParser(description='Export native checkpoint to safetensors')
    parser.add_argument('checkpoint_dir', help='Native checkpoint directory')
    parser.add_argument('--output', '-o', default='model.safetensors')
    parser.add_argument('--config', '-c', help='Training config JSON (for architecture info)')
    parser.add_argument('--list', '-l', action='store_true', help='List tensors without saving')
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir

    # Load metadata
    meta_path = os.path.join(ckpt_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Checkpoint: step {meta.get('step', '?')}, loss {meta.get('loss', '?'):.4f}")

    # Find all .bin files and their .shape counterparts
    bin_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.bin'))
    print(f"Found {len(bin_files)} tensor files")

    tensors = {}
    total_params = 0

    for bin_file in bin_files:
        bin_path = os.path.join(ckpt_dir, bin_file)
        shape_path = os.path.join(ckpt_dir, bin_file.replace('.bin', '.shape'))

        if not os.path.exists(shape_path):
            print(f"  WARNING: no shape file for {bin_file}, skipping")
            continue

        # Reconstruct tensor name from filename (reverse the dots→underscores mapping)
        # item_blocks_0_in_proj_weight → item.blocks.0.in_proj.weight
        name = bin_file.replace('.bin', '')
        # Reconstruct dotted name: item_blocks_N_... → item.blocks.N....
        # Strategy: known prefixes + known suffixes
        name = reconstruct_tensor_name(name)

        arr = load_raw_tensor(bin_path, shape_path)

        # Linear weight matrices are stored as [in_features, out_features] in native trainer
        # Candle linear expects [out_features, in_features], so transpose 2D weights
        # Exception: embedding is [vocab, d_model] in both — don't transpose
        if arr.ndim == 2 and '.weight' in name and 'embedding' not in name:
            arr = arr.T.copy()

        tensors[name] = arr
        total_params += arr.size

        if args.list:
            print(f"  {name:60s} {str(arr.shape):>20s}  {arr.size:>10,}")

    print(f"\nTotal tensors: {len(tensors)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Size: {total_params * 4 / 1024 / 1024:.1f} MB (fp32)")

    if not args.list:
        print(f"Saving to: {args.output}")
        save_file(tensors, args.output)
        size = os.path.getsize(args.output)
        print(f"Saved: {size / 1024 / 1024:.1f} MB")

    print("Done!")


def reconstruct_tensor_name(safe_name):
    """Convert filesystem-safe name back to dotted tensor path.

    item_embedding_weight → item.embedding.weight
    item_blocks_7_attn_norm_gamma → item.blocks.7.attn_norm.gamma
    """
    # Known leaf patterns (order matters — match longer patterns first)
    leaf_patterns = [
        ('_attn_out_proj_weight', '.attn_out_proj.weight'),
        ('_in_proj_weight', '.in_proj.weight'),
        ('_out_proj_weight', '.out_proj.weight'),
        ('_q_proj_weight', '.q_proj.weight'),
        ('_k_proj_weight', '.k_proj.weight'),
        ('_v_proj_weight', '.v_proj.weight'),
        ('_mlp_gate_weight', '.mlp_gate.weight'),
        ('_mlp_up_weight', '.mlp_up.weight'),
        ('_mlp_down_weight', '.mlp_down.weight'),
        ('_attn_norm_gamma', '.attn_norm.gamma'),
        ('_mlp_norm_gamma', '.mlp_norm.gamma'),
        ('_b_norm_gamma', '.b_norm.gamma'),
        ('_c_norm_gamma', '.c_norm.gamma'),
        ('_final_norm_gamma', '.final_norm.gamma'),
        ('_norm_gamma', '.norm.gamma'),
        ('_ssm_dt_bias', '.ssm.dt_bias'),
        ('_ssm_h_init', '.ssm.h_init'),
        ('_ssm_d', '.ssm.d'),
        ('_b_bias', '.b_bias'),
        ('_c_bias', '.c_bias'),
        ('_embedding_weight', '.embedding.weight'),
    ]

    for pattern, replacement in leaf_patterns:
        if safe_name.endswith(pattern):
            prefix = safe_name[:len(safe_name) - len(pattern)]
            # Convert prefix: item_blocks_7 → item.blocks.7
            dotted = prefix.replace('item_blocks_', 'item.blocks.')
            dotted = dotted.replace('item_', 'item.')
            return dotted + replacement

    # Fallback: just replace underscores with dots (may not be perfect)
    return safe_name.replace('_', '.')


if __name__ == '__main__':
    main()
