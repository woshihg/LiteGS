"""
3DGS 点云降采样脚本

通过随机采样（Random Sampling）减少高斯球的数量，
适用于减小模型体积、加速渲染预览或调试。
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
    
    # 获取第一个元素（通常是 'vertex'）
    vertex_element = plydata.elements[0]
    
    # 提取所有属性
    properties = {}
    for prop in vertex_element.properties:
        properties[prop.name] = np.asarray(vertex_element[prop.name])
    
    print(f"加载完成，共 {len(vertex_element.data)} 个点")
    return plydata, properties

def save_ply(path: str, plydata_original, properties, mask):
    """保存降采样后的.ply文件"""
    print(f"正在保存到 {path}...")
    
    # 获取原始元素名称和属性
    element_name = plydata_original.elements[0].name
    
    # 根据mask过滤所有属性
    filtered_properties = {}
    for key, value in properties.items():
        filtered_properties[key] = value[mask]
    
    # 构建dtype (必须与原始数据类型一致)
    dtype_list = []
    for prop in plydata_original.elements[0].properties:
        # 使用过滤后数组本身的 dtype，避免直接访问 PlyProperty.dtype 可能导致的问题
        current_dtype = filtered_properties[prop.name].dtype
        dtype_list.append((prop.name, current_dtype))
    
    num_points = np.sum(mask)
    # 创建结构化数组
    elements = np.empty(num_points, dtype=dtype_list)
    
    # 填充数据
    for prop in plydata_original.elements[0].properties:
        elements[prop.name] = filtered_properties[prop.name]
    
    # 创建并保存
    el = PlyElement.describe(elements, element_name)
    PlyData([el]).write(path)
    
    print(f"保存完成，保留 {num_points} 个点")

def main():
    parser = argparse.ArgumentParser(description='对 3DGS .ply 文件进行比例降采样')
    parser.add_argument('input', type=str, help='输入 .ply 文件路径')
    parser.add_argument('--ratio', '-r', type=float, default=0.5, 
                        help='保留比例 (0.0 到 1.0, 默认: 0.5)')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='输出 .ply 文件路径 (默认: input_downsampled.ply)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 参数预检查
    if not (0 <= args.ratio <= 1):
        print("错误: ratio 必须在 0 到 1 之间")
        sys.exit(1)
        
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
        
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_down_{int(args.ratio*100)}{ext}"
    
    # 设置随机种子
    np.random.seed(args.seed)
    
    # 执行逻辑
    plydata, properties = load_ply(args.input)
    num_total = len(plydata.elements[0].data)
    
    print(f"执行随机采样 (保留比例: {args.ratio*100:.1f}%)...")
    
    # 生成随机掩码
    mask = np.random.rand(num_total) < args.ratio
    
    # 保存结果
    save_ply(args.output, plydata, properties, mask)
    
    print(f"\n降采样任务完成！")
    print(f"原始点数: {num_total}")
    print(f"保留点数: {np.sum(mask)}")
    print(f"缩减比例: {(1 - np.sum(mask)/num_total)*100:.2f}%")

if __name__ == '__main__':
    main()
