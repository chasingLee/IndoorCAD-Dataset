import os
import json
import math
import numpy as np
import trimesh
import pyrender
from PIL import Image

def get_fibonacci_views(n_views=32, radius=2.5):
    """利用斐波那契球面算法在球面上均匀生成相机位姿"""
    poses = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_views):
        y = 1.0 - (i / float(n_views - 1)) * 2.0
        radius_at_y = math.sqrt(1.0 - y * y)
        theta = phi * i

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y

        eye = np.array([x, y, z]) * radius
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = eye
        poses.append(pose)
    return poses

def render_mesh_dataset(base_dir, n_views=32):
    # 相机内参设置 (1024x1024)
    W, H = 1024, 1024
    yfov = np.pi / 3.0
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    
    f = (H / 2.0) / np.tan(yfov / 2.0)
    intrinsics = {
        "fl_x": f, "fl_y": f,
        "cx": W / 2.0, "cy": H / 2.0,
        "w": W, "h": H
    }

    poses = get_fibonacci_views(n_views, radius=2.2)
    
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],
        metallicFactor=0.3,
        roughnessFactor=0.5
    )

    r = pyrender.OffscreenRenderer(W, H)

    # 统计信息
    skipped_count = 0
    processed_count = 0

    items = os.listdir(base_dir)
    print(f"找到 {len(items)} 个潜在项目，开始检查进度...")

    for item in items:
        subfolder_path = os.path.join(base_dir, item)
        if not os.path.isdir(subfolder_path):
            continue
        
        # 定义输出路径和标志性文件
        output_path = os.path.join(subfolder_path, "multiview_picture")
        marker_file = os.path.join(output_path, "transforms.json")

        # --- 断点续传逻辑 ---
        # 如果标志文件 transforms.json 已经存在，说明渲染已完成
        if os.path.exists(marker_file):
            skipped_count += 1
            # 使用 \r 实现动态刷新显示，不刷屏
            print(f"[跳过] 已完成: {item} (累计跳过: {skipped_count})", end='\r')
            continue
        
        # 寻找 STL 文件
        stl_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith('.stl')]
        if not stl_files:
            continue
        
        print(f"\n[渲染] 正在处理: {item} -> {stl_files[0]}")
        
        stl_path = os.path.join(subfolder_path, stl_files[0])
        os.makedirs(output_path, exist_ok=True)

        try:
            # 加载并处理网格
            t_mesh = trimesh.load(stl_path)
            if isinstance(t_mesh, trimesh.Scene):
                if len(t_mesh.geometry) == 0: continue
                t_mesh = trimesh.util.concatenate([g for g in t_mesh.geometry.values()])

            center = t_mesh.bounding_box.centroid
            t_mesh.vertices -= center
            scale = np.max(np.linalg.norm(t_mesh.vertices, axis=1))
            if scale > 0:
                t_mesh.vertices /= scale
                
            mesh = pyrender.Mesh.from_trimesh(t_mesh, material=material)
            camera_info = {"intrinsics": intrinsics, "frames": []}

            for i, pose in enumerate(poses):
                scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0])
                scene.add(mesh)
                scene.add(camera, pose=pose)
                
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                scene.add(light, pose=pose)
                
                color, _ = r.render(scene)
                
                img_name = f"view_{i:03d}.png"
                Image.fromarray(color).save(os.path.join(output_path, img_name))
                
                camera_info["frames"].append({
                    "file_path": img_name,
                    "transform_matrix": pose.tolist()
                })

            # 只有当 32 张图都渲染完，才保存 json 标志文件
            with open(marker_file, 'w') as f_json:
                json.dump(camera_info, f_json, indent=4)
            
            processed_count += 1

        except Exception as e:
            print(f"\n[错误] 处理 {item} 时发生异常: {e}")
            continue

    r.delete()
    print(f"\n\n任务结束。")
    print(f"本次新增处理: {processed_count} 个项目")
    print(f"自动跳过: {skipped_count} 个项目")

if __name__ == '__main__':
    INPUT_DIRECTORY = r'D:\Lzm_Temp_Data\Li_temp_project\Dataset_1\9001_10000' 
    render_mesh_dataset(INPUT_DIRECTORY, n_views=32)