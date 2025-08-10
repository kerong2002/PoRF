import os
import open3d as o3d
import numpy as np

def remove_noise_from_ply(ply_file, output_file, nb_neighbors=10, std_ratio=100000):
    # 读取 PLY 文件
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 使用统计滤波法去除噪声
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    
    # 只保留非噪声点
    inlier_cloud = pcd.select_by_index(ind)
    
    # 保存处理后的点云
    o3d.io.write_point_cloud(output_file, inlier_cloud)
    
    print(f"Filtered point cloud saved to {output_file}")

def noise_cancel(work_dir):
    input_ply = os.path.join(work_dir, 'sparse_points.ply')
    output_ply = os.path.join(work_dir, 'sparse_points_interest.ply')
    
    remove_noise_from_ply(input_ply, output_ply, nb_neighbors=70, std_ratio=4) # scan37
    # remove_noise_from_ply(input_ply, output_ply, nb_neighbors=30, std_ratio=15) # cup2
