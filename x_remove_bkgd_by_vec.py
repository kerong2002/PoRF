import open3d as o3d
import numpy as np

# 讀取 PLY 檔案
pcd = o3d.io.read_point_cloud("C:/Users/hsu/Desktop/project/exp_dtu/scan37_1/dtu_sift_porf/meshes/00050000.ply")
print(len(pcd.points))
# 計算法向量
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))

# 計算法向量平均方向
normals = np.asarray(pcd.normals)
average_normal = np.mean(normals, axis=0)
average_normal = average_normal / np.linalg.norm(average_normal)  # 標準化

# 計算每個點法向量與平均法向量的夾角
angle_threshold = np.pi / 3  # 設定夾角閾值，例：30度
filtered_indices = [
    i for i, normal in enumerate(normals)
    if np.arccos(np.dot(normal, average_normal)) < angle_threshold
]
# 過濾並保留目標物體點
pcd_filtered = pcd.select_by_index(filtered_indices)
print(len(pcd_filtered.points))

# 可視化結果
o3d.visualization.draw_geometries([pcd_filtered])
print()