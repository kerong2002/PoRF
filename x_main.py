import os
import shutil
from path import Path
from PIL import Image
from imgs2poses import imgs2poses
from gen_cameras import gen_cameras
from export_colmap_matches import export_colmap_matches
from train import train


if __name__ == '__main__':
    data_dirs = Path('porf_data/dtu')
    case_name = 'cup2'
    image_dir = 'D:/Desktop/porf/idr/NeuS_main/preprocess_custom_data/colmap_preprocess/cup2/images'
    target_height = 1440
    
    work_dir = os.path.join(data_dirs, case_name)
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'sparse'), exist_ok=True)
        
    new_images_dir = os.path.join(data_dirs, case_name, 'images')
    os.makedirs(new_images_dir, exist_ok=True)
    # new_image_dir = os.path.join(data_dirs, case_name, 'image')
    # os.makedirs(new_image_dir, exist_ok=True)
    
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            
            output_path = os.path.join(new_images_dir, filename)
            img.save(output_path)
            
            # width, height = img.size
            # aspect_ratio = width / height
            # new_width = int(target_height * aspect_ratio)
            # img = img.resize((new_width, target_height), Image.LANCZOS)
            
            # output_path = os.path.join(new_image_dir, filename)
            # img.save(output_path)
            
    
    imgs2poses(work_dir)
    
    a = input('處理sparse_interest.ply')
    # 處理sparse_interest.ply
    gen_cameras(work_dir)
    export_colmap_matches(work_dir)
    train(case_name)
            
           
