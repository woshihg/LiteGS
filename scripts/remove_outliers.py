"""
去除.ply文件中的离群点脚本

使用统计方法检测和移除高斯点云中的离群点：
1. 基于K近邻距离的统计离群点检测
2. 基于密度的DBSCAN聚类方法
"""

import numpy as np
import argparse
import os
import sys
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def load_ply(path: str):
    """加载.ply文件"""
    print(f"正在加载 {path}...")
    plydata = PlyData.read(path)
    
    # 提取所有属性
    properties = {}
    for prop in plydata.elements[0].properties:
        properties[prop.name] = np.asarray(plydata.elements[0][prop.name])
    
    # 提取xyz坐标
    xyz = np.stack([properties['x'], properties['y'], properties['z']], axis=1)
    
    print(f"加载完成，共 {xyz.shape[0]} 个点")
    return plydata, properties, xyz

def save_ply(path: str, plydata_original, properties, mask):
    """保存过滤后的.ply文件"""
    print(f"正在保存到 {path}...")
    
    # 根据mask过滤所有属性
    filtered_properties = {}
    for key, value in properties.items():
        filtered_properties[key] = value[mask]
    
    # 构建dtype
    dtype_full = [(prop.name, 'f4') for prop in plydata_original.elements[0].properties]
    
    num_points = np.sum(mask)
    elements = np.empty(num_points, dtype=dtype_full)
    
    # 按照原始顺序组合属性
    attributes = []
    for prop in plydata_original.elements[0].properties:
        attributes.append(filtered_properties[prop.name])
    
    attributes = np.stack(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # 创建并保存
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    
    print(f"保存完成，共 {num_points} 个点")

def remove_outliers_statistical(xyz, k=500, std_ratio=2.0):
    """
    使用统计方法去除离群点
    
    参数:
        xyz: 点云坐标 (N, 3)
        k: 近邻点数量
        std_ratio: 标准差倍数，超过mean + std_ratio*std的点被认为是离群点
    
    返回:
        mask: 布尔数组，True表示保留的点
    """
    print(f"\n使用统计方法检测离群点 (k={k}, std_ratio={std_ratio})...")
    
    # 计算每个点到其k个最近邻的平均距离
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    
    # 排除自身（第一个邻居），计算到其他邻居的平均距离
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # 计算统计量
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # 定义离群点阈值
    threshold = global_mean + std_ratio * global_std
    
    # 创建mask
    mask = mean_distances < threshold
    
    num_outliers = np.sum(~mask)
    print(f"检测到 {num_outliers} 个离群点 ({num_outliers/len(xyz)*100:.2f}%)")
    print(f"平均距离: {global_mean:.6f}, 标准差: {global_std:.6f}, 阈值: {threshold:.6f}")
    
    return mask

def remove_outliers_dbscan(xyz, eps=0.05, min_samples=10):
    """
    使用DBSCAN聚类方法去除离群点
    
    参数:
        xyz: 点云坐标 (N, 3)
        eps: DBSCAN的邻域半径
        min_samples: 核心点的最小邻居数
    
    返回:
        mask: 布尔数组，True表示保留的点（属于最大簇）
    """
    print(f"\n使用DBSCAN方法检测离群点 (eps={eps}, min_samples={min_samples})...")
    
    # 执行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_
    
    # 统计每个簇的大小
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(unique_labels) == 0:
        print("警告: 没有找到任何簇，保留所有点")
        return np.ones(len(xyz), dtype=bool)
    
    # 找到最大的簇
    largest_cluster = unique_labels[np.argmax(counts)]
    
    # 创建mask - 保留最大簇的点
    mask = labels == largest_cluster
    
    num_noise = np.sum(labels == -1)
    num_outliers = np.sum(~mask)
    print(f"找到 {len(unique_labels)} 个簇")
    print(f"最大簇包含 {np.sum(mask)} 个点")
    print(f"噪声点: {num_noise} 个")
    print(f"移除 {num_outliers} 个离群点 ({num_outliers/len(xyz)*100:.2f}%)")
    
    return mask

def remove_outliers_radius(xyz, radius=0.1, min_neighbors=5):
    """
    基于半径的离群点检测
    
    参数:
        xyz: 点云坐标 (N, 3)
        radius: 搜索半径
        min_neighbors: 最小邻居数量
    
    返回:
        mask: 布尔数组，True表示保留的点
    """
    print(f"\n使用半径方法检测离群点 (radius={radius}, min_neighbors={min_neighbors})...")
    
    # 使用半径搜索
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(xyz)
    distances, indices = nbrs.radius_neighbors(xyz)
    
    # 计算每个点的邻居数量（排除自身）
    neighbor_counts = np.array([len(idx) - 1 for idx in indices])
    
    # 创建mask
    mask = neighbor_counts >= min_neighbors
    
    num_outliers = np.sum(~mask)
    print(f"检测到 {num_outliers} 个离群点 ({num_outliers/len(xyz)*100:.2f}%)")
    
    return mask

def main():
    parser = argparse.ArgumentParser(description='去除.ply文件中的离群点')
    parser.add_argument('input', type=str, help='输入.ply文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='输出.ply文件路径（默认: input_cleaned.ply）')
    parser.add_argument('--method', '-m', type=str, default='statistical', 
                        choices=['statistical', 'dbscan', 'radius'],
                        help='离群点检测方法 (默认: statistical)')
    
    # 统计方法参数
    parser.add_argument('--k', type=int, default=50, 
                        help='统计方法: K近邻数量 (默认: 50)')
    parser.add_argument('--std_ratio', type=float, default=2.0, 
                        help='统计方法: 标准差倍数 (默认: 2.0)')
    
    # DBSCAN方法参数
    parser.add_argument('--eps', type=float, default=0.05, 
                        help='DBSCAN方法: 邻域半径 (默认: 0.05)')
    parser.add_argument('--min_samples', type=int, default=10, 
                        help='DBSCAN方法: 最小样本数 (默认: 10)')
    
    # 半径方法参数
    parser.add_argument('--radius', type=float, default=0.1, 
                        help='半径方法: 搜索半径 (默认: 0.1)')
    parser.add_argument('--min_neighbors', type=int, default=5, 
                        help='半径方法: 最小邻居数 (默认: 5)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 设置输出文件路径
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_cleaned{ext}"
    
    # 加载.ply文件
    plydata, properties, xyz = load_ply(args.input)
    
    # 根据选择的方法去除离群点
    if args.method == 'statistical':
        mask = remove_outliers_statistical(xyz, k=args.k, std_ratio=args.std_ratio)
    elif args.method == 'dbscan':
        mask = remove_outliers_dbscan(xyz, eps=args.eps, min_samples=args.min_samples)
    elif args.method == 'radius':
        mask = remove_outliers_radius(xyz, radius=args.radius, min_neighbors=args.min_neighbors)
    
    # 保存过滤后的.ply文件
    save_ply(args.output, plydata, properties, mask)
    
    print(f"\n完成！")
    print(f"原始点数: {len(xyz)}")
    print(f"保留点数: {np.sum(mask)}")
    print(f"移除点数: {np.sum(~mask)}")
    print(f"移除比例: {np.sum(~mask)/len(xyz)*100:.2f}%")

if __name__ == '__main__':
    main()
