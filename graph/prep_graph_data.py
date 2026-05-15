import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import cv2
from skimage.transform import estimate_transform, AffineTransform
from skimage.feature import match_template
import matplotlib.pyplot as plt
import shutil
import matplotlib
# 设置matplotlib使用Agg后端，避免需要GUI
matplotlib.use('Agg')

class PrepRelationMatrix():
    def __init__(self, distance_thresh=512, overlap_thresh=0.00):
        """
        初始化关系矩阵构建器
        :param distance_thresh: 中心点距离阈值，用于判断接触关系
        :param overlap_thresh: 重叠面积阈值，用于区分接触和重叠
        """
        self.distance_thresh = distance_thresh
        self.overlap_thresh = overlap_thresh
        # 方向关系编码
        self.direction_codes = {
            "右": 1, "右下": 2, "下": 3, "左下": 4,
            "左": 5, "左上": 6, "上": 7, "右上": 8
        }
        # 接触关系编码
        self.contact_codes = {
            "分离": 0, "接触": 1, "重叠": 2
        }
        # 创建关系矩阵保存目录
        self.relations_dir = Path("relations_matrices")
        self.relations_dir.mkdir(exist_ok=True)
        # 创建可视化结果保存目录
        self.vis_dir = Path("relations_visualization")
        self.vis_dir.mkdir(exist_ok=True)

    def encode_parts(self, part_imgs):
        """为部件图像按序号编码"""
        part_codes = {}
        for idx, part_img in enumerate(part_imgs):
            part_codes[idx] = f"P{idx:02d}"  # P00, P01, P02...
        return part_codes

    def get_direction_relation(self, angle):
        """根据角度返回方向关系"""
        # 将角度标准化到 [-180, 180] 范围
        angle = angle % 360
        if angle > 180:
            angle -= 360

        if -22.5 <= angle < 22.5:
            return "右"
        elif 22.5 <= angle < 67.5:
            return "右下"
        elif 67.5 <= angle < 112.5:
            return "下"
        elif 112.5 <= angle < 157.5:
            return "左下"
        elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
            return "左"
        elif -157.5 <= angle < -112.5:
            return "左上"
        elif -112.5 <= angle < -67.5:
            return "上"
        elif -67.5 <= angle < -22.5:
            return "右上"
        else:
            return "右"  # 默认值

    def check_bbox_overlap(self, bbox1, bbox2):
        """计算两包围盒交并比（IoU）"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算交集区域
        inter_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        inter_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter_area = inter_x * inter_y

        # 计算并集区域
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def analyze_spatial_relation(self, pos_i, pos_j):
        """分析两个部件的空间关系"""
        # 计算中心点距离
        center_dist = np.sqrt((pos_i["center"][0] - pos_j["center"][0])**2 +
                              (pos_i["center"][1] - pos_j["center"][1])**2)

        # 计算包围盒重叠
        overlap_ratio = self.check_bbox_overlap(pos_i["bbox"], pos_j["bbox"])

        # 判断接触类型
        if overlap_ratio > self.overlap_thresh:
            return "重叠"
        elif overlap_ratio > 0 or center_dist < self.distance_thresh:
            return "接触"
        else:
            return "分离"

    def inverse_direction(self, direction):
        """反转方向语义"""
        direction_map = {
            "右": "左", "左": "右",
            "上": "下", "下": "上",
            "右下": "左上", "左上": "右下",
            "左下": "右上", "右上": "左下"
        }
        return direction_map.get(direction, direction)

    def extract_mask(self, img):
        """从图像中提取二值掩码"""
        if len(img.shape) == 3:
            # 彩色图像
            if img.shape[2] == 4:  # 带alpha通道
                mask = img[:, :, 3] > 0
            else:  # RGB图像
                # 转为灰度图
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 二值化
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                mask = mask > 0
        else:
            # 灰度图像
            _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
            mask = mask > 0

        return mask

    def estimate_affine_transform(self, part_img, target_img):
        """
        估计部件图像到目标图像的仿射变换
        使用特征匹配和RANSAC算法
        """
        # 转换为灰度图
        if len(part_img.shape) == 3:
            part_gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
        else:
            part_gray = part_img.copy()

        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_img.copy()

        # 确保数据类型正确
        part_gray = part_gray.astype(np.uint8)
        target_gray = target_gray.astype(np.uint8)

        # 提取特征点和描述符
        try:
            # 使用SIFT特征检测器
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(part_gray, None)
            kp2, des2 = sift.detectAndCompute(target_gray, None)

            if len(kp1) < 4 or len(kp2) < 4:
                # 特征点太少，回退到模板匹配
                return self.find_part_in_target(part_img, target_img)

            # 特征匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 应用比率测试筛选好的匹配
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                # 匹配点太少，回退到模板匹配
                return self.find_part_in_target(part_img, target_img)

            # 获取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC估计仿射变换
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is None:
                # 仿射变换估计失败，回退到模板匹配
                return self.find_part_in_target(part_img, target_img)

            # 计算变换后的四个角点
            h, w = part_gray.shape
            corners = np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_corners = cv2.transform(corners, M)

            # 计算变换后的包围盒
            x_min = np.min(transformed_corners[:, 0, 0])
            y_min = np.min(transformed_corners[:, 0, 1])
            x_max = np.max(transformed_corners[:, 0, 0])
            y_max = np.max(transformed_corners[:, 0, 1])

            # 计算中心点
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # 计算宽高
            width = x_max - x_min
            height = y_max - y_min

            # 返回位置信息和置信度
            return (int(x_min), int(y_min)), 0.9, (int(width), int(height))

        except Exception as e:
            print(f"仿射变换估计失败: {str(e)}")
            # 回退到模板匹配
            return self.find_part_in_target(part_img, target_img)

    def find_part_in_target(self, part_img, target_img, method=cv2.TM_CCOEFF_NORMED):
        """
        在目标图像中找到部件的最佳匹配位置
        使用模板匹配
        """
        try:
            # 验证输入
            if part_img is None or target_img is None:
                raise ValueError("输入图像为空")

            if not isinstance(part_img, np.ndarray) or not isinstance(target_img, np.ndarray):
                raise ValueError("输入必须是numpy数组")

            # 检查图像尺寸
            if part_img.shape[0] > target_img.shape[0] or part_img.shape[1] > target_img.shape[1]:
                print(f"警告: 部件图像 {part_img.shape} 大于目标图像 {target_img.shape}")
                # 缩放部件图像
                scale = min(target_img.shape[0] / part_img.shape[0], target_img.shape[1] / part_img.shape[1]) * 0.9
                new_height = int(part_img.shape[0] * scale)
                new_width = int(part_img.shape[1] * scale)
                part_img = cv2.resize(part_img, (new_width, new_height))

            # 转换为灰度图进行匹配
            if len(part_img.shape) == 3:
                part_gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
            else:
                part_gray = part_img.copy()

            if len(target_img.shape) == 3:
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = target_img.copy()

            # 确保数据类型正确
            part_gray = part_gray.astype(np.uint8)
            target_gray = target_gray.astype(np.uint8)

            # 模板匹配
            result = cv2.matchTemplate(target_gray, part_gray, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            return max_loc, max_val, (part_img.shape[1], part_img.shape[0])

        except Exception as e:
            print(f"模板匹配出错: {str(e)}")
            return (0, 0), 0.0, (part_img.shape[1], part_img.shape[0])

    def process_relations(self, source_paths, target_path):
        """
        主要处理函数：分析部件图像在目标图像中的空间关系
        返回二进制链接矩阵 (0/1)
        """
        try:
            # 处理输入参数 - 支持路径和图像数组两种输入
            if isinstance(target_path, (str, Path)):
                target_img = cv2.imread(str(target_path))
                if target_img is None:
                    raise ValueError(f"无法加载目标图像: {target_path}")
                target_filename = Path(target_path).stem
            else:
                target_img = target_path
                target_filename = "unknown_target"

            # 处理部件图像
            part_imgs = []
            if len(source_paths) > 0 and isinstance(source_paths[0], (str, Path)):
                # 按数字排序后加载所有图像
                sorted_paths = sorted(source_paths, key=lambda x: int(Path(x).stem))
                for path in sorted_paths:
                    img = cv2.imread(str(path))
                    if img is None:
                        print(f"警告: 无法加载部件图像 {path}")
                        continue
                    part_imgs.append(img)
            else:
                part_imgs = source_paths

            N = len(part_imgs)
            if N == 0:
                print("警告: 没有有效的部件图像")
                return np.zeros((0, 0), dtype=np.int32)

            # 初始化数据结构
            part_positions = {}

            # print(f"开始处理 {N} 个部件的空间关系分析...")

            # Step 1: 部件精确定位 - 使用仿射变换或模板匹配
            for idx, part_img in enumerate(part_imgs):
                try:
                    if not isinstance(part_img, np.ndarray) or not isinstance(target_img, np.ndarray):
                        print(f"错误: 图像数据类型不正确")
                        continue

                    # 尝试使用仿射变换定位
                    max_loc, confidence, size = self.estimate_affine_transform(part_img, target_img)
                    x, y = max_loc
                    w, h = size

                    # 计算部件中心点及包围盒
                    center = (x + w//2, y + h//2)
                    bbox = (x, y, w, h)

                    # 存储部件元数据
                    part_positions[idx] = {
                        "center": center,
                        "bbox": bbox,
                        "confidence": confidence
                    }

                    # print(f"部件 {idx}: 位置({x},{y}), 中心({center[0]},{center[1]}), 置信度{confidence:.3f}")

                except Exception as e:
                    print(f"处理部件 {idx} 时出错: {str(e)}")
                    # 设置默认位置
                    part_positions[idx] = {
                        "center": (0, 0),
                        "bbox": (0, 0, 1, 1),
                        "confidence": 0.0
                    }

            # Step 2: 构建关系链接矩阵
            # print("开始分析空间关系...")

            # 初始化关系矩阵 - 使用整数编码表示关系
            # 0: 无关系, 1-8: 方向关系, 10-12: 接触关系(10:分离, 11:接触, 12:重叠)
            relation_matrix = np.zeros((N, N), dtype=np.int32)
            # 初始化二进制链接矩阵 (0/1)
            adjacency_matrix = np.zeros((N, N), dtype=np.int32)

            for i in range(N):
                for j in range(i+1, N):  # 只计算上三角，然后对称填充
                    pos_i = part_positions[i]
                    pos_j = part_positions[j]

                    # 计算方向关系
                    dx = pos_j["center"][0] - pos_i["center"][0]
                    dy = pos_j["center"][1] - pos_i["center"][1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    direction = self.get_direction_relation(angle)
                    direction_code = self.direction_codes[direction]

                    # 计算接触关系
                    contact = self.analyze_spatial_relation(pos_i, pos_j)
                    contact_code = self.contact_codes[contact]

                    # 组合编码: 方向码 + 接触码*10
                    combined_code = direction_code + contact_code * 10

                    # 对称填充邻接矩阵
                    relation_matrix[i][j] = combined_code

                    # 反向关系
                    inverse_direction = self.inverse_direction(direction)
                    inverse_direction_code = self.direction_codes[inverse_direction]
                    inverse_combined_code = inverse_direction_code + contact_code * 10
                    relation_matrix[j][i] = inverse_combined_code

                    # 判断是否连接 (根据接触或重叠关系)
                    is_connected = 0
                    if contact_code > 0:  # 接触(1)或重叠(2)
                        is_connected = 1

                    # 对称填充二进制链接矩阵
                    adjacency_matrix[i][j] = is_connected
                    adjacency_matrix[j][i] = is_connected

            # print(f"\n关系矩阵构建完成! 矩阵大小: {N}x{N}")

            # 保存关系矩阵到文件
            # self.save_relation_matrix(relation_matrix, adjacency_matrix, target_filename)

            # 可视化关系矩阵
            # self.visualize_relation_matrix(relation_matrix, adjacency_matrix, part_positions, target_img, target_filename)

            # 返回二进制链接矩阵 (0/1)
            return adjacency_matrix

        except Exception as e:
            print(f"处理关系时发生错误: {str(e)}")
            return np.zeros((0, 0), dtype=np.int32)

    def save_relation_matrix(self, relation_matrix, adjacency_matrix, target_filename):
        """保存关系矩阵到文件，保持与原文件相同的名称"""
        # 创建保存路径
        save_path = self.relations_dir / f"{target_filename}_relations.npy"
        adj_save_path = self.relations_dir / f"{target_filename}_adjacency.npy"

        # 保存关系矩阵
        np.save(save_path, relation_matrix)
        np.save(adj_save_path, adjacency_matrix)
        # print(f"关系矩阵已保存到: {save_path}")
        # print(f"链接矩阵已保存到: {adj_save_path}")

        # 同时保存可读的文本版本
        txt_path = self.relations_dir / f"{target_filename}_relations.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"部件空间关系矩阵 - {target_filename}\n")
            f.write("="*50 + "\n\n")

            # 写入关系矩阵
            f.write("关系矩阵 (编码):\n")
            N = relation_matrix.shape[0]

            # 表头
            f.write("     ")
            for j in range(N):
                f.write(f"P{j:02d}".rjust(8))
            f.write("\n")

            # 矩阵内容
            for i in range(N):
                f.write(f"P{i:02d} ")
                for j in range(N):
                    code = relation_matrix[i][j]
                    contact_code = code // 10
                    direction_code = code % 10

                    # 解码方向和接触关系
                    direction = "无"
                    for d_name, d_code in self.direction_codes.items():
                        if d_code == direction_code:
                            direction = d_name
                            break

                    contact = "无"
                    for c_name, c_code in self.contact_codes.items():
                        if c_code == contact_code:
                            contact = c_name
                            break

                    if i == j:
                        f.write("自身".rjust(8))
                    else:
                        f.write(f"{direction}{contact}".rjust(8))
                f.write("\n")

            # 写入链接矩阵
            f.write("\n\n链接矩阵 (0/1):\n")

            # 表头
            f.write("     ")
            for j in range(N):
                f.write(f"P{j:02d}".rjust(4))
            f.write("\n")

            # 矩阵内容
            for i in range(N):
                f.write(f"P{i:02d} ")
                for j in range(N):
                    f.write(f"{adjacency_matrix[i][j]}".rjust(4))
                f.write("\n")

        # print(f"关系矩阵文本版本已保存到: {txt_path}")

    def visualize_relation_matrix(self, relation_matrix, adjacency_matrix, part_positions, target_img, target_filename):
        """可视化关系矩阵和部件位置"""
        try:
            # 创建可视化图像
            vis_img = target_img.copy()
            N = relation_matrix.shape[0]

            # 定义颜色
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红色
                (0, 255, 255),  # 黄色
                (128, 0, 0),    # 深蓝色
                (0, 128, 0)     # 深绿色
            ]

            # 绘制部件包围盒和编号
            for idx, pos_info in part_positions.items():
                bbox = pos_info["bbox"]
                center = pos_info["center"]
                color = colors[idx % len(colors)]

                # 绘制包围盒
                x, y, w, h = bbox
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)

                # 绘制中心点
                cv2.circle(vis_img, center, 5, color, -1)

                # 绘制编号
                cv2.putText(vis_img, f"P{idx}", (center[0]-10, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 绘制连接线 - 使用链接矩阵
            for i in range(N):
                for j in range(i+1, N):
                    if adjacency_matrix[i][j] > 0:
                        # 提取关系编码
                        code = relation_matrix[i][j]
                        contact_code = code // 10

                        # 根据接触类型选择线型
                        if contact_code == 2:  # 重叠
                            thickness = 2
                            line_type = cv2.LINE_AA
                        elif contact_code == 1:  # 接触
                            thickness = 1
                            line_type = cv2.LINE_AA
                        else:  # 分离
                            continue  # 不绘制分离关系

                        # 绘制连接线
                        cv2.line(vis_img,
                                 part_positions[i]["center"],
                                 part_positions[j]["center"],
                                 colors[(i+j) % len(colors)],
                                 thickness,
                                 line_type)

            # 保存可视化结果
            vis_path = self.vis_dir / f"{target_filename}_visualization.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            # print(f"关系可视化已保存到: {vis_path}")

            # 可视化关系矩阵
            plt.figure(figsize=(10, 8))
            plt.imshow(relation_matrix, cmap='viridis')
            plt.colorbar(label='Relation Code')
            plt.title(f'Parts Relation Matrix - {target_filename}')
            plt.xlabel('Part Index')
            plt.ylabel('Part Index')

            # 添加数值标签
            for i in range(N):
                for j in range(N):
                    plt.text(j, i, relation_matrix[i, j],
                             ha="center", va="center", color="w")

            # 保存矩阵可视化
            matrix_vis_path = self.vis_dir / f"{target_filename}_matrix.jpg"
            plt.savefig(str(matrix_vis_path))
            plt.close()
            # print(f"关系矩阵可视化已保存到: {matrix_vis_path}")

            # 可视化链接矩阵
            plt.figure(figsize=(8, 6))
            plt.imshow(adjacency_matrix, cmap='binary')
            plt.colorbar(label='Connection (0/1)')
            plt.title(f'Parts Adjacency Matrix - {target_filename}')
            plt.xlabel('Part Index')
            plt.ylabel('Part Index')

            # 添加数值标签
            for i in range(N):
                for j in range(N):
                    plt.text(j, i, adjacency_matrix[i, j],
                             ha="center", va="center",
                             color="white" if adjacency_matrix[i, j] == 0 else "black")

            # 保存链接矩阵可视化
            adj_vis_path = self.vis_dir / f"{target_filename}_adjacency.jpg"
            plt.savefig(str(adj_vis_path))
            plt.close()
            # print(f"链接矩阵可视化已保存到: {adj_vis_path}")

        except Exception as e:
            print(f"可视化关系矩阵时出错: {str(e)}")

    def batch_process_directory(self, data_dir):
        """批量处理目录下的所有样本"""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            print(f"错误: 目录 {data_dir} 不存在")
            return

        # 查找所有目标图像
        target_images = list(data_dir.glob("**/full.*")) + list(data_dir.glob("**/target.*"))

        for target_path in target_images:
            try:
                # 获取目标图像所在目录
                sample_dir = target_path.parent

                # 查找同目录下的部件图像
                part_paths = []
                for i in range(100):  # 假设最多100个部件
                    part_path = sample_dir / f"{i}.png"
                    if part_path.exists():
                        part_paths.append(part_path)
                    else:
                        # 尝试其他可能的扩展名
                        for ext in ['.jpg', '.jpeg', '.bmp']:
                            part_path = sample_dir / f"{i}{ext}"
                            if part_path.exists():
                                part_paths.append(part_path)
                                break

                if not part_paths:
                    print(f"警告: 在 {sample_dir} 中未找到部件图像")
                    continue

                # print(f"\n处理样本: {target_path.name}")
                # print(f"找到 {len(part_paths)} 个部件图像")

                # 处理关系
                self.process_relations(part_paths, target_path)

            except Exception as e:
                print(f"处理样本 {target_path} 时出错: {str(e)}")