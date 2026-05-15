import os
import random
import shutil
import time

def main():
    # 定义路径 [1,3,9](@ref)
    source_dir = "./datasets/generated/product"
    train_dir = "./datasets/generated/product_train"
    val_dir = "./datasets/generated/product_val"
    test_dir = "./datasets/generated/product_test"

    # 确保目标目录存在 [3,9](@ref)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"源目录: {source_dir}")
    print(f"训练目录: {train_dir}")
    print(f"验证目录: {val_dir}")
    print(f"验证目录: {test_dir}")

    # 获取所有文件夹并过滤非目录项 [6,7](@ref)
    all_items = os.listdir(source_dir)
    folders = [item for item in all_items
               if os.path.isdir(os.path.join(source_dir, item))]

    total_folders = len(folders)
    print(f"找到 {total_folders} 个文件夹")

    if total_folders == 0:
        print("错误: 源目录中没有找到任何文件夹")
        return

    # 随机打乱文件夹顺序 [1](@ref)
    random.seed(time.time())  # 使用时间作为随机种子
    random.shuffle(folders)

    # 计算分割点 [1](@ref)
    split_index = int(total_folders * 0.8)
    train_folders = folders[:split_index]
    val_folders = folders[split_index: int(split_index+(total_folders-split_index)/2)]
    test_folders = folders[int(split_index+(total_folders-split_index)/2):]

    print(f"训练集文件夹数: {len(train_folders)}")
    print(f"验证集文件夹数: {len(val_folders)}")
    print(f"测试集文件夹数: {len(test_folders)}")

    # 复制训练集文件夹 [3,9,10](@ref)
    print("\n开始复制训练集文件夹...")
    for i, folder in enumerate(train_folders, 1):
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(train_dir, folder)

        # 如果目标存在则先删除 [3](@ref)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        shutil.copytree(src_path, dest_path)
        print(f"({i}/{len(train_folders)}) 已复制: {folder} -> {dest_path}")

    # 复制验证集文件夹
    print("\n开始复制验证集文件夹...")
    for i, folder in enumerate(val_folders, 1):
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(val_dir, folder)

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        shutil.copytree(src_path, dest_path)
        print(f"({i}/{len(val_folders)}) 已复制: {folder} -> {dest_path}")

    # 复制测试集文件夹
    print("\n开始复制测试集文件夹...")
    for i, folder in enumerate(test_folders, 1):
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(test_dir, folder)

        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)

        shutil.copytree(src_path, dest_path)
        print(f"({i}/{len(test_folders)}) 已复制: {folder} -> {dest_path}")
    
    print("\n操作完成!")

if __name__ == "__main__":
    main()