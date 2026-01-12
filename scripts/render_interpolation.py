#!/usr/bin/env python
import torch
import numpy as np
import os
import cv2
from argparse import ArgumentParser
try:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
except ImportError:
    print("Error: scipy is required for this script. Please install it with 'pip install scipy'.")
    exit(1)
from tqdm import tqdm
# 上级目录导入litegs模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from litegs.io_manager.ply import load_ply
from litegs.io_manager.colmap import load_frames
from litegs.render import render, render_preprocess
from litegs.arguments import PipelineParams, ModelParams
from litegs.utils import get_view_matrix, qvec2rotmat, viewproj_to_frustumplane

def interpolate_poses(frames, num_output_frames):
    # Extract quaternions and translations
    qvecs = []
    tvecs = []
    for frame in frames:
        # frame.extr_params is [qvec, tvec]
        qvecs.append(frame.extr_params[:4])
        tvecs.append(frame.extr_params[4:])
    
    qvecs = np.array(qvecs)
    tvecs = np.array(tvecs)
    
    num_input_frames = len(frames)
    # We want to interpolate along the path of all cameras
    # Total segments = num_input_frames - 1
    
    input_times = np.linspace(0, 1, num_input_frames)
    output_times = np.linspace(0, 1, num_output_frames)
    
    # Slerp for rotations
    # scipy uses x,y,z,w for quaternions, COLMAP uses w,x,y,z
    rotations = R.from_quat(qvecs[:, [1, 2, 3, 0]]) 
    slerp = Slerp(input_times, rotations)
    interp_rots = slerp(output_times)
    interp_qvecs_xyzw = interp_rots.as_quat()
    interp_qvecs = interp_qvecs_xyzw[:, [3, 0, 1, 2]] # back to w,x,y,z
    
    # Linear interpolation for translations
    interp_tvecs = np.zeros((num_output_frames, 3))
    for i in range(3):
        interp_tvecs[:, i] = np.interp(output_times, input_times, tvecs[:, i])
        
    return interp_qvecs, interp_tvecs

@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the PLY file")
    parser.add_argument("--colmap_path", type=str, required=True, help="Path to the COLMAP data directory (containing sparse/0/)")
    parser.add_argument("--output_dir", type=str, default="output/render_interp", help="Directory to save rendered images")
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames to render")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree used in the PLY")
    parser.add_argument("--images_dir", type=str, default="images", help="Images directory in COLMAP path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")

    # Load PLY
    print(f"Loading PLY from {args.ply_path}...")
    xyz, scale, rot, sh_0, sh_rest, opacity, inferred_sh_degree, features = load_ply(args.ply_path, args.sh_degree)
    
    # Move to GPU
    xyz = torch.from_numpy(xyz).float().to(device)
    scale = torch.from_numpy(scale).float().to(device)
    rot = torch.from_numpy(rot).float().to(device)
    sh_0 = torch.from_numpy(sh_0).float().to(device)
    sh_rest = torch.from_numpy(sh_rest).float().to(device)
    opacity = torch.from_numpy(opacity).float().to(device)
    if features is not None:
        features = torch.from_numpy(features).float().to(device)

    # Pipeline parameters
    pp = PipelineParams.get_class_default_obj()

    # Cluster points if needed
    from litegs.scene.cluster import cluster_points, get_cluster_AABB
    cluster_origin = None
    cluster_extend = None
    if pp.cluster_size > 0:
        print(f"Clustering points with size {pp.cluster_size}...")
        if features is not None:
            xyz, scale, rot, sh_0, sh_rest, opacity, features = cluster_points(
                pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity, features
            )
        else:
            xyz, scale, rot, sh_0, sh_rest, opacity = cluster_points(
                pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity
            )
        cluster_origin, cluster_extend = get_cluster_AABB(xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0))

    # Load COLMAP
    print(f"Loading COLMAP data from {args.colmap_path}...")
    cam_intrinsics, frames = load_frames(args.colmap_path, args.images_dir)
    
    if not frames:
        print("Error: No frames found in COLMAP data.")
        return

    # Interpolate poses
    print(f"Interpolating {args.num_frames} frames between {len(frames)} cameras...")
    interp_qvecs, interp_tvecs = interpolate_poses(frames, args.num_frames)
    
    # Use the first camera's intrinsics for all rendered frames
    cam_info = list(cam_intrinsics.values())[0]
    proj_matrix = torch.from_numpy(cam_info.get_project_matrix()).float().to(device).unsqueeze(0)
    output_shape = (cam_info.height, cam_info.width)
    
    print("Rendering...")
    for i in tqdm(range(args.num_frames)):
        qvec = interp_qvecs[i]
        tvec = interp_tvecs[i]
        
        R_mat = qvec2rotmat(qvec)
        view_matrix = torch.from_numpy(get_view_matrix(R_mat, tvec).transpose()).float().to(device).unsqueeze(0)
        
        # Calculate frustum planes
        viewproj = view_matrix @ proj_matrix
        frustumplane = viewproj_to_frustumplane(viewproj)
        
        # Preprocess (culling, SH to RGB, etc.)
        visible_chunkid, culled_xyz, culled_scale, culled_rot, color, culled_opacity, culled_features = render_preprocess(
            cluster_origin, cluster_extend, frustumplane, view_matrix,
            xyz, scale, rot, sh_0, sh_rest, opacity,
            None, pp, args.sh_degree, features
        )
        
        # Rasterization
        img, _, _, _, _, _ = render(
            view_matrix, proj_matrix,
            culled_xyz, culled_scale, culled_rot, color, culled_opacity,
            args.sh_degree, output_shape, pp, culled_features
        )
        
        # Save image
        # img is (1, 3, H, W)
        img_np = (img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, f"{i:04d}.png"), img_np)

    print(f"Done! Rendered images saved to {args.output_dir}")

if __name__ == "__main__":
    main()
