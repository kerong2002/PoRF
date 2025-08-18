import numpy as np
import trimesh

# 讀取 PLY 檔
mesh = trimesh.load('D:/Desktop/project/exp_dtu/images17/dtu_sift_porf/meshes/000030003.ply')

# 獲取頂點座標與顏色資訊
vertices = mesh.vertices
colors = mesh.visual.vertex_colors
faces = mesh.faces

c_list = []
v_list = []
f_list = []
mesh_list = []
# 列出每個頂點的座標與顏色
count = 0
for i in range(len(vertices)):
    if colors[i][2] < colors[i][1] or colors[i][2] < colors[i][0]:
        count += 1
        print(f"Point {i}: Position = {vertices[i]}, Color = {colors[i]}, faces = {faces[i]}")
        c_list.append(colors[i])
        v_list.append(vertices[i])
        f_list.append(faces[i])
        # mesh_list.append(mesh[i])
print(len(vertices), count)

triangles = np.random.randint(0, 100, size=(100, 3))
mesh = trimesh.Trimesh(v_list, faces=triangles, vertex_colors=c_list)
mesh.export('TEST.ply')
