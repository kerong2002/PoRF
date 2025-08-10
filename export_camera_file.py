import numpy as np
import os
import trimesh
import argparse
import logging
import colorsys

def create_camera_pyramid(pose, color, scale=0.1):
    """
    為單一相機位姿建立一個 3D 金字塔模型。

    Args:
        pose (np.array): 4x4 的相機 C2W (camera-to-world) 位姿矩陣。
        color (list): 金字塔的顏色 [R, G, B]。
        scale (float): 金字塔的大小。

    Returns:
        tuple: (vertices, faces, vertex_colors)
    """
    # 金字塔的局部座標系頂點
    # 頂點 0 是相機中心 (金字塔尖)
    # 頂點 1-4 是相機的影像平面四角
    vertices_local = np.array([
        [0, 0, 0],
        [-scale, -scale, scale * 2],
        [scale, -scale, scale * 2],
        [scale, scale, scale * 2],
        [-scale, scale, scale * 2]
    ])

    # 金字塔的面 (連接頂點)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # 四個側面
        [1, 3, 2], [1, 4, 3]  # 兩個底面三角形
    ])

    # 將局部座標頂點轉換到世界座標系
    vertices_world = (pose[:3, :3] @ vertices_local.T + pose[:3, 3, np.newaxis]).T

    # 為每個頂點設定顏色
    vertex_colors = np.array([color] * len(vertices_world))

    return vertices_world, faces, vertex_colors

def visualize_poses(work_dir):
    """
    讀取 poses.npy 和 sparse_points.ply，合併並產生一個包含可視化相機的 pose.ply。

    Args:
        work_dir (str): 工作目錄，應包含 poses.npy 和 sparse_points.ply。
    """
    poses_file = os.path.join(work_dir, 'poses.npy')
    points_file = os.path.join(work_dir, 'sparse_points.ply')
    output_file = os.path.join(work_dir, 'pose.ply')

    # --- 檢查檔案是否存在 ---
    if not os.path.exists(poses_file):
        logging.error(f"錯誤: 找不到相機位姿檔案 '{poses_file}'")
        return
    if not os.path.exists(points_file):
        logging.warning(f"警告: 找不到稀疏點雲檔案 '{points_file}'。將只產生相機位姿。")
        base_points = np.zeros((0, 3))
        base_colors = np.zeros((0, 3))
    else:
        # --- 載入稀疏點雲 ---
        try:
            point_cloud = trimesh.load(points_file)
            base_points = np.array(point_cloud.vertices)
            # 如果點雲有顏色就使用，沒有就設為灰色
            if hasattr(point_cloud.visual, 'vertex_colors'):
                base_colors = np.array(point_cloud.visual.vertex_colors)[:, :3]
            else:
                base_colors = np.full_like(base_points, 128, dtype=np.uint8)
            logging.info(f"成功載入 {len(base_points)} 個稀疏點雲。")
        except Exception as e:
            logging.error(f"讀取點雲檔案 '{points_file}' 失敗: {e}")
            base_points = np.zeros((0, 3))
            base_colors = np.zeros((0, 3))

    # --- 載入相機位姿 ---
    poses = np.load(poses_file)  # 格式應為 (N, 3, 5) 或 (N, 4, 4)
    
    # NeRF-style poses (3x5) to standard 4x4 C2W matrices
    if poses.shape[1] == 3 and poses.shape[2] == 5:
        # [R|t|H,W,f] -> 4x4 C2W
        c2w_mats = []
        for p in poses:
            mat = np.eye(4)
            mat[:3, :3] = p[:3, :3]
            mat[:3, 3] = p[:3, 3]
            c2w_mats.append(mat)
        poses = np.array(c2w_mats)
    
    logging.info(f"成功載入 {len(poses)} 個相機位姿。")

    all_vertices = [base_points]
    all_faces = []
    all_colors = [base_colors]

    # --- 為每個相機建立金字塔 ---
    face_offset = len(base_points)
    for i, pose in enumerate(poses):
        # 為相機建立一個獨特的顏色 (使用 HSV 色彩空間)
        hue = i / len(poses)
        # 使用標準函式庫 colorsys 進行轉換，避免 trimesh 版本問題
        color = np.array(colorsys.hsv_to_rgb(hue, 0.8, 1.0)) * 255
        
        verts, faces, colors = create_camera_pyramid(pose, color, scale=0.05)
        
        all_vertices.append(verts)
        all_faces.append(faces + face_offset)
        all_colors.append(colors)
        
        face_offset += len(verts)

    # --- 合併並儲存 ---
    final_vertices = np.concatenate(all_vertices, axis=0)
    final_faces = np.concatenate(all_faces, axis=0) if all_faces else np.zeros((0, 3))
    final_colors = np.concatenate(all_colors, axis=0)

    # 建立 Trimesh 物件並匯出
    final_mesh = trimesh.Trimesh(vertices=final_vertices, faces=final_faces, vertex_colors=final_colors.astype(np.uint8))
    final_mesh.export(output_file)
    
    logging.info(f"成功匯出視覺化位姿檔案至 '{output_file}'")
    logging.info("您現在可以用 MeshLab 或 CloudCompare 等軟體開啟此檔案進行分析。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="視覺化 COLMAP 的相機位姿和稀疏點雲。")
    parser.add_argument('--work_dir', type=str, required=True,
                        help="工作目錄的路徑，例如 './porf_data/dtu/scan24'")
    
    args = parser.parse_args()

    # 設定日誌
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])

    if not os.path.isdir(args.work_dir):
        logging.error(f"錯誤: 找不到指定的目錄 '{args.work_dir}'")
    else:
        visualize_poses(args.work_dir)
