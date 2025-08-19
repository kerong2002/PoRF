import open3d as o3d
import numpy as np

# def remove_bkgd(pts, pcd):
# 載入兩個 PLY 檔案
# mesh = o3d.io.read_triangle_mesh("D:/Desktop/project/exp_dtu/images_c3/dtu_sift_porf/meshes/000500004.ply")
# pcd2 = o3d.io.read_point_cloud("D:/Desktop/project/porf_data/dtu/images_c3/sparse_points_interest.ply")

# mesh = o3d.io.read_triangle_mesh("D:/Desktop/project/exp_dtu/images_g2/dtu_sift_porf/meshes/000500004.ply")
# pcd2 = o3d.io.read_point_cloud("D:/Desktop/project/porf_data/dtu/images_g2/sparse_points_interest.ply")

# mesh = o3d.io.read_triangle_mesh("D:/Desktop/project/exp_dtu/images_m5/dtu_sift_porf/meshes/000500004.ply")
# pcd2 = o3d.io.read_point_cloud("D:/Desktop/project/porf_data/dtu/images_m5/sparse_points_interest.ply")

mesh = o3d.io.read_triangle_mesh("D:/Desktop/project/exp_dtu/pot5/dtu_sift_porf/meshes/000500004.ply")
pcd2 = o3d.io.read_point_cloud("D:/Desktop/project/porf_data/dtu/pot5/sparse_points_interest.ply")

print(len(mesh.vertices))


# 將 TriangleMesh 轉換為 PointCloud
pcd1 = mesh.sample_points_uniformly(number_of_points=np.array(mesh.vertices).shape[0])  # 使用均勻抽樣來獲取相同數量的點

# 對第二個點雲進行下採樣和法線估計
pcd2_down = pcd2.voxel_down_sample(voxel_size=0.01)
pcd2_down.estimate_normals()

# 設定配準參數
threshold = 0.02  # 設定配準閾值
trans_init = np.eye(4)  # 初始化轉換矩陣
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd2_down, pcd1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# 進行變換
pcd2.transform(reg_p2p.transformation)

# 計算 AABB 並擴大範圍
aabb = pcd2.get_axis_aligned_bounding_box()
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()
center = (min_bound + max_bound) / 2
size = max_bound - min_bound
expansion_factor = 0.1
new_size = size * (1 + expansion_factor)
new_min_bound = center - new_size / 2
new_max_bound = center + new_size / 2

# 獲取篩選後的點和顏色數據
points = np.asarray(mesh.vertices)
colors = np.asarray(mesh.vertex_colors)

# 確保不會因為沒有顏色而出錯
if colors.shape[0] == 0:
    colors = np.zeros_like(points)  # 如果沒有顏色，設置顏色為黑色

# 篩選點、顏色和面
mask = np.all(np.logical_and(points >= new_min_bound, points <= new_max_bound), axis=1)
filtered_points = points[mask]
filtered_colors = colors[mask]
print(len(filtered_points))

# 更新面索引
triangles = np.asarray(mesh.triangles)
filtered_triangles = []
filtered_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(mask)[0])}

for triangle in triangles:
    if all(vtx in filtered_indices_map for vtx in triangle):
        # 更新面的頂點索引
        filtered_triangles.append([filtered_indices_map[vtx] for vtx in triangle])

filtered_triangles = np.array(filtered_triangles)

# 創建新的三角形網格
filtered_mesh = o3d.geometry.TriangleMesh()
filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_points)
filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(filtered_colors)
filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

# 可視化篩選後的網格
o3d.visualization.draw_geometries([filtered_mesh])

# 可選：保存篩選後的三角形網格
o3d.io.write_triangle_mesh("filtered_object.ply", filtered_mesh)
