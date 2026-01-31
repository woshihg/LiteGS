#!/usr/bin/env python
import sys
import os
import numpy as np
import argparse
from tqdm import tqdm

# Add the project root to sys.path to allow importing litegs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litegs.io_manager.ply import load_ply, save_ply

def main():
    parser = argparse.ArgumentParser(description="Split a PLY file into multiple PLY files based on feature classes.")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the source PLY file")
    parser.add_argument("--output_dir", type=str, default="output/split_classes", help="Directory to save the split PLY files")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used in the PLY")
    args = parser.parse_args()

    if not os.path.exists(args.ply_path):
        print(f"Error: File {args.ply_path} not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load PLY using litegs utility
    print(f"Loading PLY from {args.ply_path}...")
    # load_ply returns: xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, features
    xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, features = load_ply(args.ply_path, args.sh_degree)

    if features is None:
        print("Error: The PLY file does not contain features to split by.")
        return

    # Determine classes using argmax (assuming multi-channel features)
    # features shape is (C, N)
    print(f"Processing features of shape {features.shape}...")
    if features.shape[0] > 1:
        classes = np.argmax(features, axis=0) # (N,)
    else:
        # If single channel, treat rounding values as classes or prompt error
        classes = np.round(features[0]).astype(int)

    unique_classes = np.unique(classes)
    print(f"Found {len(unique_classes)} unique classes: {unique_classes}")

    # Split and save
    for cls in tqdm(unique_classes, desc="Saving split files"):
        mask = (classes == cls)
        
        # Filter all attributes
        # Note: xyz (3, N), scale (3, N), rot (4, N), sh_0 (1, 3, N), sh_rest (C_sh, 3, N), opacity (1, N), features (C_feat, N)
        cls_xyz = xyz[:, mask]
        cls_scale = scale[:, mask]
        cls_rot = rot[:, mask]
        cls_sh_0 = sh_0[:, :, mask]
        cls_sh_rest = sh_rest[:, :, mask]
        cls_opacity = opacity[:, mask]
        cls_features = features[:, mask]

        output_path = os.path.join(args.output_dir, f"class_{cls}.ply")
        
        # save_ply expectation:
        # xyz (3, N), scale (3, N), rot (4, N), sh_0 (1/SH, 3, N), sh_rest (SH_rest, 3, N), opacity (1, N), features (C, N)
        save_ply(
            output_path, 
            cls_xyz, 
            cls_scale, 
            cls_rot, 
            cls_sh_0, 
            cls_sh_rest, 
            cls_opacity, 
            cls_features
        )

    print(f"\nDone! Split files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
