import os
import json

def generate_splits(full_data_path, split_data_path):
    # 场景列表
    scenes = [
        "bicycle", "bonsai", "counter", "flowers", 
        "garden", "kitchen", "room", "stump", "treehill"
    ]
    
    for scene in scenes:
        # 1. 从完整路径获取所有图像并排序
        full_scene_dir = os.path.join(full_data_path, scene)
        full_image_dir = os.path.join(full_scene_dir, "images")
        
        if not os.path.exists(full_image_dir):
            print(f"[跳过] 场景 {scene} 未找到完整图像目录: {full_image_dir}")
            continue
            
        all_images = sorted([f for f in os.listdir(full_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 2. 从拆分路径获取训练集图像（这是真正的训练集源）
        split_scene_dir = os.path.join(split_data_path, scene)
        split_image_dir = os.path.join(split_scene_dir, "images")
        
        if not os.path.exists(split_image_dir):
            print(f"[跳过] 场景 {scene} 未找到训练集图像目录: {split_image_dir}")
            continue
            
        # 训练集直接读取 split 文件夹下的内容
        train_list = sorted([f for f in os.listdir(split_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 3. 逻辑划分验证集：从全集中取 index % 8 == 0
        test_list = []
        for i, img_name in enumerate(all_images):
            if i % 8 == 0:
                test_list.append(img_name)
        
        # 检查训练集和验证集是否有交集（安全检查）
        overlap = set(train_list).intersection(set(test_list))
        if overlap:
            print(f"[警告] {scene}: 训练集和验证集存在交集点，数量: {len(overlap)}")
        
        split_data = {
            "train": train_list,
            "test": test_list
        }
        
        # 写入 JSON 到完整数据路径（LiteGS 会从这里读取）
        output_file = os.path.join(full_scene_dir, "train_test_split.json")
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=4)
            
        print(f"[完成] {scene}: 训练集 {len(train_list)} 张, 验证集 {len(test_list)} 张 -> {output_file}")

if __name__ == "__main__":
    # 完整图像集合
    full_path = os.path.expanduser("~/360_v2")
    # 官方拆分出的训练集（用于参考/确认）
    split_path = os.path.expanduser("~/360_v2_split")
    
    generate_splits(full_path, split_path)
