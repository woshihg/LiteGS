import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

# Add the project root to sys.path to allow importing litegs
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import litegs
import litegs.config
import litegs.utils

def render_set(model_path, name, loader, xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, pp, op, features, cluster_origin, cluster_extend):
    render_path = os.path.join(model_path, name, "renders")
    os.makedirs(render_path, exist_ok=True)

    with torch.no_grad():
        for index, (view_matrix, proj_matrix, frustumplane, gt_image, idx, gt_mask, gt_depth) in enumerate(tqdm(loader, desc=f"Rendering {name}")):
            view_matrix = view_matrix.cuda()
            proj_matrix = proj_matrix.cuda()
            frustumplane = frustumplane.cuda()
            
            # Cluster culling
            visible_chunkid, culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity, culled_features = litegs.render.render_preprocess(
                cluster_origin, cluster_extend, frustumplane, view_matrix, xyz, scale, rot, sh_0, sh_rest, opacity, op, pp, sh_degree, features
            )
            
            # Render
            img, _, _, _, _, _ = litegs.render.render(
                view_matrix, proj_matrix, culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity,
                sh_degree, gt_image.shape[2:], pp, culled_features
            )
            
            # Save
            img_id = idx[0].item() if torch.is_tensor(idx) else idx[0]
            img_name = loader.dataset.frames[img_id].name
            
            img_np = (img[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(render_path, img_name))

if __name__ == "__main__":
    parser = ArgumentParser(description="Render images script")
    lp_cdo, op_cdo, pp_cdo, dp_cdo = litegs.config.get_default_arg()
    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo, parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo, parser)
    
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    lp = litegs.arguments.ModelParams.extract(args)
    pp = litegs.arguments.PipelineParams.extract(args)
    op = litegs.arguments.OptimizationParams.get_class_default_obj()

    # Load data
    print(f"Loading dataset from {lp.source_path}")
    cameras_info, camera_frames, _, _ = litegs.io_manager.load_colmap_result(lp.source_path, lp.images)
    
    if lp.eval:
        import json
        if os.path.exists(os.path.join(lp.source_path, "train_test_split.json")):
            with open(os.path.join(lp.source_path, "train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames = [c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames = [c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames = [c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames = camera_frames
        test_frames = []

    trainingset = litegs.data.CameraFrameDataset(cameras_info, training_frames, lp.resolution, pp.device_preload)
    testset = litegs.data.CameraFrameDataset(cameras_info, test_frames, lp.resolution, pp.device_preload) if test_frames else None

    from torch.utils.data import DataLoader
    def custom_collate_fn(batch):
        filtered_batch = [item for item in batch if item is not None]
        if len(filtered_batch) == 0: return None
        collated = []
        for i in range(len(filtered_batch[0])):
            field = [item[i] for item in filtered_batch]
            if isinstance(field[0], (torch.Tensor, np.ndarray)):
                if isinstance(field[0], np.ndarray): field = [torch.from_numpy(f) for f in field]
                collated.append(torch.stack(field))
            else: collated.append(field)
        return tuple(collated)

    train_loader = DataLoader(trainingset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn) if testset else None

    # Load Model
    ply_path = lp.model_path
    if not os.path.exists(ply_path):
        # try find any chkpnt file
        ckpt_files = [f for f in os.listdir(lp.model_path) if f.startswith('chkpnt') and f.endswith('.pth')]
        if ckpt_files:
            # sort by epoch number
            ckpt_files.sort(key=lambda x: int(x.split('chkpnt')[-1].split('.pth')[0]))
            latest_ckpt = ckpt_files[-1]
            ckpt_path = os.path.join(lp.model_path, latest_ckpt)
            print(f"Loading checkpoint {ckpt_path}")
            xyz, scale, rot, sh_0, sh_rest, opacity, features, _, _, _, _, _ = litegs.io_manager.load_checkpoint(ckpt_path)
            sh_degree = lp.sh_degree
        else:
            print(f"No PLY or checkpoint found in {lp.model_path}")
            exit(1)
    else:
        print(f"Loading PLY {ply_path}")
        xyz, scale, rot, sh_0, sh_rest, opacity, inferred_sh_degree, features = litegs.io_manager.load_ply(ply_path, lp.sh_degree, pp.reset_load_opacity)
        sh_degree = inferred_sh_degree
        xyz = torch.tensor(xyz, device='cuda', dtype=torch.float32)
        scale = torch.tensor(scale, device='cuda', dtype=torch.float32)
        rot = torch.tensor(rot, device='cuda', dtype=torch.float32)
        sh_0 = torch.tensor(sh_0, device='cuda', dtype=torch.float32)
        sh_rest = torch.tensor(sh_rest, device='cuda', dtype=torch.float32)
        opacity = torch.tensor(opacity, device='cuda', dtype=torch.float32)
        if features is not None: features = torch.tensor(features, device='cuda', dtype=torch.float32)

    # Pre-cluster if needed
    cluster_origin = None
    cluster_extend = None
    if pp.cluster_size > 0:
        print("Clistering points...")
        if features is not None:
            xyz, scale, rot, sh_0, sh_rest, opacity, features = litegs.scene.point.spatial_refine(False, None, xyz, scale, rot, sh_0, sh_rest, opacity, features)
            xyz, scale, rot, sh_0, sh_rest, opacity, features = litegs.scene.cluster.cluster_points(pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity, features)
        else:
            xyz, scale, rot, sh_0, sh_rest, opacity = litegs.scene.point.spatial_refine(False, None, xyz, scale, rot, sh_0, sh_rest, opacity)
            xyz, scale, rot, sh_0, sh_rest, opacity = litegs.scene.cluster.cluster_points(pp.cluster_size, xyz, scale, rot, sh_0, sh_rest, opacity)
        cluster_origin, cluster_extend = litegs.scene.cluster.get_cluster_AABB(xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0))

    if not args.skip_train:
        render_set("./output", "trainingset", train_loader, xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, pp, op, features, cluster_origin, cluster_extend)
    if test_loader:
        render_set("./output", "testset", test_loader, xyz, scale, rot, sh_0, sh_rest, opacity, sh_degree, pp, op, features, cluster_origin, cluster_extend)
    
    print(f"Rendering finished. Results saved in {lp.model_path}")
