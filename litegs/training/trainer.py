import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import math
import os
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import json

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify
from .. import utils
from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint:str=None):
    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.ImageFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)
        if lp.feature_dim > 0:
            mask_path = os.path.join(lp.source_path, "masks", camera_frame.name)
            if not os.path.exists(mask_path):
                base_name = os.path.splitext(camera_frame.name)[0]
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.bmp']:
                    potential_path = os.path.join(lp.source_path, "masks", base_name + ext)
                    if os.path.exists(potential_path):
                        mask_path = potential_path
                        break
            
            res = camera_frame.load_mask(mask_path, lp.resolution)
            if res is None:
                print(f"[ WARNING ] Mask not found for {camera_frame.name} at {mask_path}")

    #Dataset
    if lp.eval:
        if os.path.exists(os.path.join(lp.source_path,"train_test_split.json")):
            with open(os.path.join(lp.source_path,"train_test_split.json"), "r") as file:
                train_test_split = json.load(file)
                training_frames=[c for c in camera_frames if c.name in train_test_split["train"]]
                test_frames=[c for c in camera_frames if c.name in train_test_split["test"]]
        else:
            training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
            test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload) if lp.eval else None

    def custom_collate_fn(batch):
        """
        Custom collate function to handle None values in the batch.
        Filters out None values and stacks the remaining items.
        """
        filtered_batch = [item for item in batch if item is not None]
        if len(filtered_batch) == 0:
            raise ValueError("All elements in the batch are None.")

        # Assuming the batch contains tuples, we need to handle each field separately
        collated = []
        for i in range(len(filtered_batch[0])):
            field = [item[i] for item in filtered_batch]
            if isinstance(field[0], torch.Tensor):
                collated.append(torch.stack(field))
            else:
                collated.append(field)  # Keep as list for non-tensor fields

        return tuple(collated)

    train_loader = DataLoader(trainingset, batch_size=1, shuffle=True, pin_memory=not pp.device_preload, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=not pp.device_preload, collate_fn=custom_collate_fn) if lp.eval else None
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    features=None
    init_points_num=init_xyz.shape[0]
    if start_checkpoint is None:
        ply_path = os.path.join(lp.source_path, "0000.ply")
        if os.path.exists(ply_path):
            print(f"Loading initial Gaussians from {ply_path}")
            xyz,scale,rot,sh_0,sh_rest,opacity,inferred_sh_degree,loaded_features=io_manager.load_ply(ply_path,lp.sh_degree)
            init_points_num=xyz.shape[-1]
            xyz=torch.tensor(xyz,dtype=torch.float32,device='cuda')
            scale=torch.tensor(scale,dtype=torch.float32,device='cuda')
            rot=torch.tensor(rot,dtype=torch.float32,device='cuda')
            sh_0=torch.tensor(sh_0,dtype=torch.float32,device='cuda')
            sh_rest=torch.tensor(sh_rest,dtype=torch.float32,device='cuda')
            opacity=torch.tensor(opacity,dtype=torch.float32,device='cuda')
            if loaded_features is not None:
                features = torch.tensor(loaded_features, dtype=torch.float32, device='cuda')
            else:
                features = None
            
            # Pad sh_rest if inferred degree is less than target degree
            if inferred_sh_degree < lp.sh_degree:
                print(f"Padding SH coefficients from degree {inferred_sh_degree} to {lp.sh_degree}")
                num_points = sh_rest.shape[2]
                target_sh_count = (lp.sh_degree + 1) ** 2 - 1
                current_sh_count = sh_rest.shape[0]
                if target_sh_count > current_sh_count:
                    extra_sh = torch.zeros((target_sh_count - current_sh_count, 3, num_points), device='cuda')
                    sh_rest = torch.cat([sh_rest, extra_sh], dim=0)
        else:
            init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
            init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        
        if lp.feature_dim == 0:
            features = None
        elif features is None:
            features = torch.zeros((lp.feature_dim, xyz.shape[-1]), device='cuda')

        if pp.cluster_size:
            if features is not None:
                xyz,scale,rot,sh_0,sh_rest,opacity,features=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity,features)
            else:
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        if features is not None:
            features=torch.nn.Parameter(features)
            
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp,features)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,features,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
    if pp.cluster_size:
        cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #learnable view matrix
    if op.learnable_viewproj:
        noise_extr=torch.cat([frame.extr_params[None,:] for frame in trainingset.frames])
        denoised_training_extr=torch.nn.Embedding(noise_extr.shape[0],noise_extr.shape[1],_weight=noise_extr.clone(),sparse=True)
        noise_intr=torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0)
        denoised_training_intr=torch.nn.Parameter(torch.tensor(list(trainingset.cameras.values())[0].intr_params,dtype=torch.float32,device='cuda').unsqueeze(0))#todo fix multi cameras
        view_opt=torch.optim.SparseAdam(denoised_training_extr.parameters(),lr=1e-4)
        proj_opt=torch.optim.Adam([denoised_training_intr,],lr=1e-5)

    #init
    total_epoch=int(op.iterations/len(trainingset))
    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    
    writer = SummaryWriter(log_dir=lp.model_path)
    
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)

    for epoch in range(start_epoch,total_epoch):

        with torch.no_grad():
            if pp.cluster_size>0 and (epoch-1)%dp.densification_interval==0:
                xyz,scale,rot,sh_0,sh_rest,opacity=scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for i, (view_matrix,proj_matrix,frustumplane,gt_image,idx,gt_mask) in enumerate(train_loader):
                global_step = epoch * len(train_loader) + i
                nvtx.range_push("Iter Init")
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0
                idx=idx.cuda()
                if gt_mask is not None:
                    if isinstance(gt_mask, (list, tuple)):
                        if gt_mask[0] is not None:
                            gt_mask = gt_mask.cuda().long()
                    else:
                        gt_mask = gt_mask.cuda().long()
                        
                if op.learnable_viewproj:
                    #fix view matrix
                    extr=denoised_training_extr(idx)
                    intr=denoised_training_intr
                    view_matrix,proj_matrix,viewproj_matrix,frustumplane=utils.wrapper.CreateViewProj.apply(extr,intr,gt_image.shape[2],gt_image.shape[3],0.01,5000)
                nvtx.range_pop()
                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,culled_features=render.render_preprocess(
                    cluster_origin,cluster_extend,frustumplane,view_matrix,xyz,scale,rot,sh_0,sh_rest,opacity,op,pp,actived_sh_degree,features)
                
                # Conditionally render classification features
                do_classification = (culled_features is not None) and (global_step % op.classification_iter == 0)
                render_features = culled_features if do_classification else None

                img,transmitance,depth,normal,primitive_visible,class_feature=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_color,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp,render_features)
                
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
                if pp.enable_transmitance:
                    loss+=(1-transmitance).abs().mean()
                
                class_loss = 0
                if do_classification and class_feature is not None and gt_mask is not None:
                    # gt_mask stores category indices [B, 1, H, W], class_feature stores one-hot [B, 16, H, W]
                    gt_one_hot = torch.nn.functional.one_hot(gt_mask.squeeze(1).long(), num_classes=class_feature.shape[1])
                    gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()
                    class_loss = torch.nn.functional.mse_loss(class_feature, gt_one_hot)
                    loss += class_loss
 
                loss.backward()
                
                # TensorBoard logging
                if global_step % 10 == 0:
                    writer.add_scalar('train/total_loss', loss.item(), global_step)
                    writer.add_scalar('train/l1_loss', l1_loss.item(), global_step)
                    writer.add_scalar('train/ssim_loss', ssim_loss.item(), global_step)
                    if class_feature is not None:
                        writer.add_scalar('train/class_loss', class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss, global_step)
                    writer.add_scalar('train/num_points', xyz.shape[-1] * (xyz.shape[-2] if pp.cluster_size else 1), global_step)

                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()
                opt.zero_grad(set_to_none = True)
                if op.learnable_viewproj:
                    view_opt.step()
                    view_opt.zero_grad()
                    # proj_opt.step()
                    # proj_opt.zero_grad()
                schedular.step()

        if epoch in test_epochs:
            with torch.no_grad():
                _cluster_origin = None
                _cluster_extend = None
                if pp.cluster_size:
                    _cluster_origin, _cluster_extend = scene.cluster.get_cluster_AABB(
                        xyz, scale.exp(), torch.nn.functional.normalize(rot, dim=0)
                    )
                psnr_metrics = psnr.PeakSignalNoiseRatio(data_range=(0.0, 1.0)).cuda()
                loaders = {"Trainingset": train_loader}
                if lp.eval:
                    loaders["Testset"] = test_loader
                for name, loader in loaders.items():
                    psnr_list = []
                    for view_matrix, proj_matrix, frustumplane, gt_image, idx, gt_mask in loader:
                        view_matrix = view_matrix.cuda()
                        proj_matrix = proj_matrix.cuda()
                        frustumplane = frustumplane.cuda()
                        gt_image = gt_image.cuda() / 255.0
                        idx = idx.cuda()

                        if gt_mask is not None:
                            if isinstance(gt_mask, (list, tuple)):
                                gt_mask = gt_mask[0]
                            if gt_mask is not None:
                                gt_mask = gt_mask.cuda().long()
                        else:
                            # Handle missing gt_mask by creating a default mask
                            gt_mask = torch.ones_like(gt_image[:, 0:1, :, :], device='cuda')

                        if op.learnable_viewproj:
                            if name == "Trainingset":
                                # Fix view matrix
                                extr = denoised_training_extr(idx)
                                intr = denoised_training_intr
                            else:
                                nearest_idx = (
                                    extr - denoised_training_extr._parameters['weight']
                                ).abs().sum(dim=1).argmin()
                                delta = denoised_training_extr(nearest_idx) - noise_extr[nearest_idx]
                                extr = extr + delta
                            view_matrix, proj_matrix, viewproj_matrix, frustumplane = utils.wrapper.CreateViewProj.apply(
                                extr, intr, gt_image.shape[2], gt_image.shape[3], 0.01, 5000
                            )

                        # Cluster culling
                        visible_chunkid, culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity, culled_features = render.render_preprocess(
                            cluster_origin, cluster_extend, frustumplane, view_matrix, xyz, scale, rot, sh_0, sh_rest, opacity, op, pp, actived_sh_degree, features
                        )
                        img, transmitance, depth, normal, primitive_visible, feature_map = render.render(
                            view_matrix, proj_matrix, culled_xyz, culled_scale, culled_rot, culled_color, culled_opacity,
                            actived_sh_degree, gt_image.shape[2:], pp, culled_features
                        )
                        psnr_list.append(psnr_metrics(img, gt_image).unsqueeze(0))
                    tqdm.write(
                        f"\n[EPOCH {epoch}] {name} Evaluating: PSNR {torch.concat(psnr_list, dim=0).mean()}"
                    )

        params=density_controller.step(opt,epoch)
        xyz,scale,rot,sh_0,sh_rest,opacity=params["xyz"],params["scale"],params["rot"],params["sh_0"],params["sh_rest"],params["opacity"]
        if "features" in params:
            features = params["features"]
        progress_bar.update()  

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                print("{} takes: {}".format(lp.model_path,progress_bar.format_dict['elapsed']))
                save_path=os.path.join(lp.model_path,"point_cloud","finish")
            else:
                save_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch))    

            if pp.cluster_size:
                if features is not None:
                    tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity,features)
                else:
                    tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                if features is not None:
                    tensors=xyz,scale,rot,sh_0,sh_rest,opacity,features
                else:
                    tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(os.path.join(save_path,"point_cloud.ply"),*param_nyp)

            # Save separate PLY files for each category if features exist
            if features is not None and features.shape[0] > 0:
                # features is the last element in tensors
                feat_tensor = tensors[-1]
                categories = torch.argmax(feat_tensor, dim=0)
                unique_cats = torch.unique(categories)
                print(f"Saving {len(unique_cats)} categories to separate PLY files...")
                for cat in unique_cats:
                    mask = (categories == cat)
                    cat_tensors = [t[..., mask] for t in tensors]
                    param_nyp_cat = [t.detach().cpu().numpy() for t in cat_tensors]
                    cat_save_path = os.path.join(save_path, f"point_cloud_cat_{cat.item()}.ply")
                    io_manager.save_ply(cat_save_path, *param_nyp_cat)

            if op.learnable_viewproj:
                torch.save(list(denoised_training_extr.parameters())+[denoised_training_intr],os.path.join(save_path,"viewproj.pth"))

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)
    
    writer.close()
    return