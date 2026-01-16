"""
3DGS 不透明度降采样脚本

通过保留不透明度（Opacity）最高的高斯球来减少点数。
通常不透明度高的高斯球对渲染贡献最大，这种方法比随机采样更能保留场景主体。
"""

import numpy as np
import argparse
import os
import sys
from plyfile import PlyData, PlyElement

def load_ply(path: str):
    """加载.ply文件"""
    print(f"正在加载 {path}...")
    plydata = PlyData.read(path)
    
    vertex_element = plydata.elements[0]
    properties = {}
    for prop in vertex_element.properties:
        properties[prop.name] = np.asarray(vertex_element[prop.name])
    
    print(f"加载完成，共 {len(vertex_element.data)} 个点")
    return plydata, properties

def save_ply(path: str, plydata_original, properties, mask):
    """保存降采样后的.ply文件"""
    print(f"正在保存到 {path}...")
    
    element_name = plydata_original.elements[0].name
    filtered_properties = {key: value[mask] for key, value in properties.items()}
    
    dtype_list = []
    for prop in plydata_original.elements[0].properties:
        dtype_list.append((prop.name, filtered_properties[prop.name].dtype))
    
    num_points = np.sum(mask)
    elements = np.empty(num_points, dtype=dtype_list)
    
    for prop in plydata_original.elements[0].properties:
        elements[prop.name] = filtered_properties[prop.name]
    
    el = PlyElement.describe(elements, element_name)
    PlyData([el]).write(path)
    print(f"保存完成，保留 {num_points} 个点")

def main():
    parser = argparse.ArgumentParser(description='根据不透明度对 3DGS .ply 文件进行降采样')
    parser.add_argument('input', type=str, help='输入 .ply 文件路径')
    parser.add_argument('--ratio', '-r', type=float, default=0.2, 
                        help='保留比例 (默认: 0.2，即保留前 20%%)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
        
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_top_{int(args.ratio*100)}{ext}"
    
    plydata, properties = load_ply(args.input)
    
    # 获取不透明度属性 (3DGS 官方格式通常叫 'opacity')
    if 'opacity' not in properties:
        print("错误: 在 PLY 文件中未找到 'opacity' 属性")
        print(f"现有属性: {list(properties.keys())}")
        sys.exit(1)
    
    opacities = properties['opacity']
    num_total = len(opacities)
    num_keep = int(num_total * args.ratio)
    
    print(f"正在计算不透明度阈值，准备保留前 {args.ratio*100:.1f}% ({num_keep} 个点)...")
    
    # 找到第 (1-ratio) 分位数作为阈值
    # 例如保留前 20%，则寻找 80% 分位数
    threshold = np.percentile(opacities, (1 - args.ratio) * 100)
    
    # 生成掩码
    # 注意：如果有很多点具有完全相同的不透明度，np.percentile 后的数量可能略多于 num_keep
    mask = opacities >= threshold
    
    # 如果因为重复值导致点的数量超过预期太多，可以强制截断
    if np.sum(mask) > num_keep + 100:
        print(f"由于存在大量相同不透明度的点，实际保留比例略高 ({(np.sum(mask)/num_total)*100:.2f}%)")

    save_ply(args.output, plydata, properties, mask)
    
    print(f"\n降采样任务完成！")
    print(f"原始点数: {num_total}")
    print(f"保留点数: {np.sum(mask)}")
    print(f"不透明度阈值 (sigmoid前): {threshold:.6f}")

if __name__ == '__main__':
    main()
