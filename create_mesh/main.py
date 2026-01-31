import os
import sys
import torch
import numpy as np
import nvdiffrast.torch as dr
import cv2
import trimesh
import xatlas
import argparse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from litegs.io_manager.colmap import load_frames

# =================================================================
# 1. 坐标系与矩阵变换工具
# =================================================================

def projection_matrix(fx, fy, cx, cy, w, h, n=0.1, f=1000.0):
    """
    构建投影矩阵，从 COLMAP 相机内参生成。
    
    COLMAP 相机坐标系: X 右, Y 下, Z 前（相机看向 +Z）
    COLMAP 投影: u = fx * x/z + cx, v = fy * y/z + cy
    
    nvdiffrast 使用 OpenGL 风格的 clip space:
    - x_clip/w_clip -> x_ndc in [-1, 1]，左到右
    - y_clip/w_clip -> y_ndc in [-1, 1]，下到上
    - z_clip/w_clip -> z_ndc in [0, 1]
    
    由于 COLMAP 的 v 向下增加，而 OpenGL 的 y_ndc 向上增加，
    渲染结果会上下颠倒。我们可以在投影矩阵中翻转 Y，或者后处理翻转图像。
    这里选择在投影矩阵中处理。
    """
    proj = torch.zeros(4, 4)
    
    # 从像素坐标 (u, v) 到 NDC (x_ndc, y_ndc):
    # x_ndc = (u - cx) / fx * (2*fx/w) = 2*(u - cx)/w = 2*u/w - 2*cx/w
    # 要得到 x_ndc = 2*u/w - 1，需要 cx = w/2（图像中心）
    # 一般情况: x_ndc = 2*fx/w * x/z + (2*cx/w - 1)
    
    # X 方向
    proj[0, 0] = 2.0 * fx / w
    proj[0, 2] = 2.0 * cx / w - 1.0
    
    # Y 方向：不翻转，因为 litegs 的 view_matrix 已经处理了坐标系
    proj[1, 1] = 2.0 * fy / h
    proj[1, 2] = 2.0 * cy / h - 1.0
    
    # Z 方向：映射 [n, f] 到 [0, 1]
    # z_ndc = (f * z - f*n) / (z * (f-n)) = f/(f-n) - f*n/((f-n)*z)
    # 在齐次坐标: z_clip = f/(f-n) * z - f*n/(f-n), w_clip = z
    proj[2, 2] = f / (f - n)
    proj[2, 3] = -f * n / (f - n)
    proj[3, 2] = 1.0
    
    return proj


def load_mesh_and_generate_uvs(mesh_path, device):
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    verts = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    
    print(f"正在为具有 {len(verts)} 个顶点和 {len(faces)} 个面片的网格生成 UV...")
    atlas = xatlas.Atlas()
    atlas.add_mesh(verts, faces)
    atlas.generate()
    
    vmapping, indices, uvs = atlas[0]
    
    new_verts = torch.tensor(verts[vmapping], device=device)
    new_faces = torch.tensor(indices.astype(np.int32), device=device).reshape(-1, 3)
    new_uvs = torch.tensor(uvs.astype(np.float32), device=device)
    
    return new_verts, new_faces, new_uvs, new_faces

def load_colmap_data(colmap_path, image_dir, device):
    cameras_dict, frames = load_frames(colmap_path, image_dir)
    processed_frames = []
    for frame in frames:
        cam = cameras_dict[frame.camera_id]
        # 从磁盘加载实际图片
        img = Image.open(frame.img_source).convert('RGB')
        w_img, h_img = img.size  # 实际图片分辨率
        
        # 获取 COLMAP 记录的原始分辨率
        w_orig, h_orig = cam.width, cam.height
        
        # 计算缩放比例
        sw = w_img / w_orig
        sh = h_img / h_orig
        
        # 缩放内参：fx, fy, cx, cy
        fx, fy, cx, cy = cam.params
        fx_scaled, fy_scaled = fx * sw, fy * sh
        cx_scaled, cy_scaled = cx * sw, cy * sh
        
        # 使用缩放后的内参生成投影矩阵（已在其中处理 Y 轴翻转）
        proj = projection_matrix(fx_scaled, fy_scaled, cx_scaled, cy_scaled, w_img, h_img).to(device)
        
        # 获取转置后的 View 矩阵 (来自 litegs，COLMAP 坐标系)
        # 视图矩阵直接使用，不需要转换，因为投影矩阵已处理坐标系差异
        view_T = torch.tensor(frame.view_matrix, device=device)
        
        # 核心：合成转置后的 MVP 矩阵 -> V^T @ P^T
        mvp_T = view_T @ proj.t()
        
        # 将图片转换为 tensor，避免在循环中重复操作
        img_np = np.array(img).astype(np.float32) / 255.0
        gt_img = torch.tensor(img_np, device=device).unsqueeze(0)
        
        processed_frames.append({
            'mvp_T': mvp_T, 
            'gt_img': gt_img,
            'res': [h_img, w_img]
        })
    return processed_frames

# =================================================================
# 2. 新增：导出与可视化工具函数
# =================================================================

def save_textured_obj(filename, verts, faces, uvs, uv_idx, texture_raw):
    """导出标准的 OBJ + MTL + PNG"""
    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    u = uvs.detach().cpu().numpy()
    u_idx = uv_idx.detach().cpu().numpy()
    # 获取最终纹理，并进行垂直翻转以匹配标准图像布局 (Top-Left origin)
    tex_img = (torch.sigmoid(texture_raw)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    tex_img = np.flipud(tex_img)
    
    base_name = os.path.splitext(filename)[0]
    obj_path = f"{base_name}.obj"
    mtl_path = f"{base_name}.mtl"
    png_path = f"{base_name}.png"

    # 保存纹理图
    cv2.imwrite(png_path, cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))

    # 保存 OBJ
    with open(obj_path, 'w') as handler:
        handler.write(f"mtllib {os.path.basename(mtl_path)}\n")
        for vv in v:
            handler.write(f"v {vv[0]} {vv[1]} {vv[2]}\n")
        for uu in u:
            # 由于贴图已经翻转，这里直接写入原始 UV 即可
            handler.write(f"vt {uu[0]} {uu[1]}\n")
        handler.write("usemtl default\n")
        for i in range(len(f)):
            # f 顶点索引 / UV索引 (从1开始)
            v_idx = f[i] + 1
            vt_idx = u_idx[i] + 1
            handler.write(f"f {v_idx[0]}/{vt_idx[0]} {v_idx[1]}/{vt_idx[1]} {v_idx[2]}/{vt_idx[2]}\n")

    # 保存 MTL
    with open(mtl_path, 'w') as handler:
        handler.write("newmtl default\n")
        handler.write(f"map_Kd {os.path.basename(png_path)}\n")
    
    print(f"Mesh 已保存至: {obj_path}")

def visualize_mesh(verts, faces, uvs, uv_idx, texture_raw):
    """使用 Trimesh 进行弹窗交互可视化"""
    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    u = uvs.detach().cpu().numpy()
    # 纹理处理：同样翻转以匹配 Trimesh 的 UV 采样规律
    tex_img = (torch.sigmoid(texture_raw)[0].detach().cpu().numpy() * 255).astype(np.uint8)
    tex_img = np.flipud(tex_img)
    material = trimesh.visual.texture.SimpleMaterial(image=Image.fromarray(tex_img))
    
    # 构造 visual 对象
    # 注意：Trimesh 默认 vertices 和 UVs 数量必须一致，如果你的 uv_idx 和 pos_idx 不同，
    # 需要先进行展开（Unmerge），这里演示简单情况
    visuals = trimesh.visual.TextureVisuals(uv=u, material=material)
    mesh = trimesh.Trimesh(vertices=v, faces=f, visual=visuals, validate=True)
    print("正在打开预览窗口...")
    mesh.show()

# =================================================================
# 3. 核心模型 (保持不变)
# =================================================================
class TextureOptimizer(torch.nn.Module):
    def __init__(self, verts, faces, uvs, face_uvs_idx, res=1024):
        super().__init__()
        # 将 verts 设为可优化参数
        self.verts = torch.nn.Parameter(verts)
        self.faces = faces
        self.uvs = uvs
        self.face_uvs_idx = face_uvs_idx
        
        # 初始化为中性灰色 (0.5)
        # 因为后续使用了 torch.sigmoid，所以初始化为 0 即可得到 0.5
        self.texture_raw = torch.nn.Parameter(torch.zeros(1, res, res, 3, device=verts.device))
        
    def forward(self, glctx, mvp, res):
        verts_homo = torch.cat([self.verts, torch.ones_like(self.verts[:, :1])], dim=-1)
        # vertex transformation: v_clip = v_world @ mvp_transpose
        verts_clip = verts_homo @ mvp
        rast, _ = dr.rasterize(glctx, verts_clip.unsqueeze(0), self.faces, resolution=res)
        texc, _ = dr.interpolate(self.uvs.unsqueeze(0), rast, self.face_uvs_idx)
        color = dr.texture(torch.sigmoid(self.texture_raw), texc, filter_mode='linear')
        mask = torch.clamp(rast[..., 3:4], 0, 1)
        color = color * mask
        return color, mask

# =================================================================
# 4. 主流程
# =================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True, help='Path to the PLY mesh')
    parser.add_argument('--colmap', type=str, required=True, help='Path to the COLMAP directory (containing sparse/0)')
    parser.add_argument('--images', type=str, default='images', help='Subdirectory for images')
    parser.add_argument('--iters', type=int, default=1000, help='Optimization iterations')
    parser.add_argument('--res', type=int, default=1024, help='Texture resolution')
    args = parser.parse_args()

    glctx = dr.RasterizeCudaContext()
    device = 'cuda'

    # 1. 加载网格并生成 UV
    v_pos, pos_idx, v_tex, uv_idx = load_mesh_and_generate_uvs(args.mesh, device)

    # 2. 加载 COLMAP 相机和图片
    processed_frames = load_colmap_data(args.colmap, args.images, device)
    if not processed_frames:
        print("未加载到任何有效相机帧，请检查路径。")
        return
    print(f"成功加载 {len(processed_frames)} 帧数据。")

    # 3. 初始化模型
    model = TextureOptimizer(v_pos, pos_idx, v_tex, uv_idx, res=args.res).to(device)
    
    # 设置不同的学习率：纹理可以用较大的 LR，顶点位置通常需要较小的 LR 以保持几何稳定
    optimizer = torch.optim.Adam([
        {'params': [model.texture_raw], 'lr': 1e-2},
        {'params': [model.verts], 'lr': 1e-4}
    ])

    print("开始优化纹理与几何...")
    output_dir = "output_mesh"
    os.makedirs(output_dir, exist_ok=True)
    render_save_dir = os.path.join(output_dir, "renders")
    os.makedirs(render_save_dir, exist_ok=True)

    for step in range(args.iters + 1):
        # 随机选择一个相机视角
        frame_idx = np.random.randint(len(processed_frames))
        frame = processed_frames[frame_idx]
        mvp_T = frame['mvp_T']
        gt_img = frame['gt_img']
        res = frame['res']

        optimizer.zero_grad()
        out, mask = model(glctx, mvp_T, res=res)

        # 每 100 步保存一次当前视角渲染图
        if step % 1000 == 0:
            debug_out = (out[0].detach().cpu().numpy() * 255).astype(np.uint8)
            debug_gt = (gt_img[0].detach().cpu().numpy() * 255).astype(np.uint8)
            combined = np.hstack([debug_out, debug_gt])
            cv2.imwrite(os.path.join(render_save_dir, f"step_{step:04d}_frame_{frame_idx:03d}.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"已保存第 {step} 步渲染对比图至: {render_save_dir}")

        # 只在有物体的地方计算 Loss，防止背景干扰
        loss = torch.sum(((out - gt_img * mask) ** 2)) / (torch.sum(mask) * 3 + 1e-8)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    # --- 可视化与保存 ---
    print("\n优化完成，执行保存与预览...")
    
    output_dir = "output_mesh"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_model.obj")
    
    # 导出文件
    save_textured_obj(output_path, model.verts, model.faces, model.uvs, model.face_uvs_idx, model.texture_raw)
    
    # 实时预览
    visualize_mesh(model.verts, model.faces, model.uvs, model.face_uvs_idx, model.texture_raw)

if __name__ == "__main__":
    main()