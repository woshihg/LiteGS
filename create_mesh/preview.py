import trimesh
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Mesh Texture Previewer")
    parser.add_argument('--model', type=str, required=True, help='Path to the .obj or .ply model file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 找不到文件 {args.model}")
        return

    print(f"正在加载模型: {args.model} ...")
    
    # trimesh 会根据 .obj 文件自动寻找目录下的 .mtl 和 .png 贴图
    try:
        mesh = trimesh.load(args.model)
        
        # 处理场景对象 (如果 obj 包含多个 mesh)
        if isinstance(mesh, trimesh.Scene):
            print("检测到场景对象，正在合并网格...")
            # 保持纹理信息的合并方式
            if len(mesh.geometry) == 0:
                print("错误: 场景中没有几何体")
                return
            # 如果只有一个几何体，直接提取
            if len(mesh.geometry) == 1:
                mesh = list(mesh.geometry.values())[0]
        
        print("--- 模型信息 ---")
        if hasattr(mesh, 'vertices'):
            print(f"顶点数: {len(mesh.vertices)}")
            print(f"面片数: {len(mesh.faces)}")
        
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            print("检测到 UV 坐标和贴图")
        else:
            print("警告: 该模型似乎没有 UV 或贴图信息")

        print("\n正在打开预览窗口 (使用鼠标旋转，滚动缩放) ...")
        mesh.show()

    except Exception as e:
        print(f"加载模型失败: {e}")

if __name__ == "__main__":
    main()
