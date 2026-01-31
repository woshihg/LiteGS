#!/usr/bin/env python
import torch
import numpy as np
import os
import cv2
from argparse import ArgumentParser
import sys
from tqdm import tqdm

# Add the project root to sys.path to allow importing litegs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litegs.io_manager.ply import load_ply
from litegs.io_manager.colmap import load_frames
from litegs.render import render, render_preprocess
from litegs.arguments import PipelineParams, OptimizationParams
from litegs.utils import viewproj_to_frustumplane
from litegs.scene.cluster import cluster_points

def get_color_palette(num_classes=256):
    """Create a color palette for visualization."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        if i == 0:
            continue # Background is usually black
        # Generate distinct colors using bit manipulation
        j = i
        for k in range(8):
            palette[i, 0] |= (((j >> 0) & 1) << (7 - k))
            palette[i, 1] |= (((j >> 1) & 1) << (7 - k))
            palette[i, 2] |= (((j >> 2) & 1) << (7 - k))
            j >>= 3
    # Convert RGB to BGR for OpenCV
    return palette[:, ::-1]

@torch.no_grad()
def main():
    parser = ArgumentParser(description="Render category masks from a PLY file using COLMAP cameras.")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the PLY file")
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to the COLMAP data directory (containing sparse/0/)")
    parser.add_argument("--output_dir", type=str, default="output/rendered_masks", help="Directory to save rendered masks")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used in the PLY")
    parser.add_argument("--images_dir", type=str, default="images", help="Images directory in COLMAP path (used to resolve frame names)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA is not available. This script may be very slow or fail if submodules require CUDA.")

    # Load PLY
    print(f"Loading PLY from {args.ply_path}...")
    # load_ply returns: xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, features
    # features shape is (C, N)
    xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, features = load_ply(args.ply_path, args.sh_degree)
    
    if features is None:
        print("Error: The PLY file does not contain feature attributes.")
        return

    print(f"Found features with dimension {features.shape[0]}")

    # Move to GPU
    xyz = torch.from_numpy(xyz).float().to(device)
    scale = torch.from_numpy(scale).float().to(device)
    rot = torch.from_numpy(rot).float().to(device)
    sh_0 = torch.from_numpy(sh_0).float().to(device)
    sh_rest = torch.from_numpy(sh_rest).float().to(device)
    opacity = torch.from_numpy(opacity).float().to(device)
    features = torch.from_numpy(features).float().to(device)

    # Pipeline parameters
    pp = PipelineParams.get_class_default_obj()
    pp.use_classifier = True # Ensure we use the classification path
    pp.cluster_size = 128 # Default cluster size
    op = OptimizationParams.get_class_default_obj()

    # Clustering
    if pp.cluster_size > 0:
        print(f"Clustering points with size {pp.cluster_size}...")
        xyz, scale, rot, sh_0, sh_rest, opacity, features = cluster_points(
            pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity, features
        )

    # Load COLMAP
    print(f"Loading COLMAP data from {args.colmap_path}...")
    cam_info_dict, frames = load_frames(args.colmap_path, args.images_dir)
    
    # Create output subdirectories
    raw_dir = os.path.join(args.output_dir, "raw")
    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    palette = get_color_palette(256)

    print(f"Rendering {len(frames)} views...")
    for frame in tqdm(frames):
        if frame.camera_id not in cam_info_dict:
            continue
            
        cam = cam_info_dict[frame.camera_id]
        
        view_matrix = torch.from_numpy(frame.view_matrix).float().to(device).unsqueeze(0)
        proj_matrix = torch.from_numpy(cam.get_project_matrix()).float().to(device).unsqueeze(0)
        
        # Frustum culling
        frustumplane = viewproj_to_frustumplane(view_matrix @ proj_matrix)
        
        # Preprocess (Culling, SH to RGB, etc.)
        visible_chunkid, culled_xyz, culled_scale, culled_rot, color, culled_opacity, culled_features = render_preprocess(
            None, None, frustumplane, view_matrix, xyz, scale, rot, sh_0, sh_rest, opacity,
            op, pp, sh_degree, features
        )
        
        # Render
        # returns: img, transmitance, depth, normal, primitive_visible, class_feature
        # class_feature shape is (V, C, H, W)
        _, _, _, _, _, class_feature = render(
            view_matrix, proj_matrix, culled_xyz, culled_scale, culled_rot, color, culled_opacity,
            sh_degree, (cam.height, cam.width), pp, culled_features
        )
        
        if class_feature is not None:
            # Transfer to CPU
            mask_data = class_feature[0].cpu().numpy() # (C, H, W)
            
            # If multi-channel, take argmax to get the class ID
            if mask_data.shape[0] > 1:
                mask = np.argmax(mask_data, axis=0).astype(np.uint8)
            else:
                # If single channel, normalize or clamp
                mask = (mask_data[0]).astype(np.uint8)
                
            # Save raw mask
            safe_name = frame.name.replace("/", "_")
            file_base = os.path.splitext(safe_name)[0]
            cv2.imwrite(os.path.join(raw_dir, f"{file_base}.png"), mask)

            # Save visualizable mask
            vis_mask = palette[mask]
            cv2.imwrite(os.path.join(vis_dir, f"{file_base}_vis.png"), vis_mask)
        else:
            print(f"Warning: No class_feature rendered for {frame.name}")

    print(f"\nFinished rendering. Raw masks: {raw_dir}, Visualization: {vis_dir}")

if __name__ == "__main__":
    main()
