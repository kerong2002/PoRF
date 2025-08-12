import platform
import re
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
from path import Path
from PIL import Image
from imgs2poses import imgs2poses
from gen_cameras import gen_cameras
from export_colmap_matches import export_colmap_matches
from train import train
from noise_cancel import noise_cancel


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class FolderSelectorApp:
    def __init__(self, root):
        # 設定視窗大小和位置（這裡設定為 400x300 像素）
        root.geometry("600x350")

        self.root = root
        self.root.title("Folder Selector")
        
        # Label to display folder name
        self.folder_label = tk.Label(root, text="No folder selected")
        self.folder_label.pack(pady=10)
        
        # Button to select folder
        self.select_button = tk.Button(root, text="Select Image Folder", command=self.select_folder)
        self.select_button.pack(pady=10)
        
        # Label for input
        self.input_label = tk.Label(root, text="Enter project name:")
        self.input_label.pack(pady=5)
        
        # Entry widget for user input
        self.text_entry = tk.Entry(root, width=50)
        self.text_entry.pack(pady=5)
        self.text_entry.config(state=tk.NORMAL)
        
        # Button to Auto Generate sparse_points_interest.ply
        self.generate_auto_button = tk.Button(root, text="Auto Generate sparse_points_interest.ply", command=self.start_auto)
        self.generate_auto_button.pack(pady=10)
        self.generate_auto_button.config(state=tk.DISABLED)
        
        # Button to Waiting Manually Create sparse_points_interest.ply
        self.generate_manual_button = tk.Button(root, text="Waiting Manually Create sparse_points_interest.ply", command=self.start_manual)
        self.generate_manual_button.pack(pady=10)
        self.generate_manual_button.config(state=tk.DISABLED)
        
        # self.image_status_label = tk.Label(root, text="image use: ")
        # self.image_status_label.pack(pady=10)
        
        # self.status_label = tk.Label(root, text="training status: ")
        # self.status_label.pack(pady=10)
        
    def select_folder(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            self.folder_label.config(text=f"Selected folder: {self.image_dir}")
            self.generate_auto_button.config(state=tk.NORMAL)
            self.generate_manual_button.config(state=tk.NORMAL)
            self.text_entry.delete(0, tk.END)
            self.text_entry.insert(0, self.image_dir.split('/')[-1])
    
    def start_auto(self):
        self.select_button.config(state=tk.DISABLED)
        self.generate_auto_button.config(state=tk.DISABLED)
        self.generate_manual_button.config(state=tk.DISABLED)
        
        self.case_name = self.text_entry.get()
        if not re.match(r'^[a-zA-Z0-9_]+$', self.case_name):
            self.select_button.config(state=tk.NORMAL)
            self.generate_auto_button.config(state=tk.NORMAL)
            self.generate_manual_button.config(state=tk.NORMAL)
            messagebox.showinfo('Error', 'project name have only have english or digit or underline and cannot be null')
            return
        self.text_entry.config(state=tk.DISABLED)
        
        self.create_dir()
        
        noise_cancel(self.work_dir)
        
        gen_cameras(self.work_dir)
        export_colmap_matches(self.work_dir)
        # self.status_label.config(text="training status: training")
        # train(self.status_label, self.case_name)
        train(self.case_name)
        
        self.select_button.config(state=tk.NORMAL)
        self.generate_auto_button.config(state=tk.NORMAL)
        self.generate_manual_button.config(state=tk.NORMAL)
        self.text_entry.config(state=tk.NORMAL)
        
    def start_manual(self):
        self.select_button.config(state=tk.DISABLED)
        self.generate_auto_button.config(state=tk.DISABLED)
        self.generate_manual_button.config(state=tk.DISABLED)
        
        self.case_name = self.text_entry.get()
        if not re.match(r'^[a-zA-Z0-9_]+$', self.case_name):
            self.select_button.config(state=tk.NORMAL)
            self.generate_auto_button.config(state=tk.NORMAL)
            self.generate_manual_button.config(state=tk.NORMAL)
            messagebox.showinfo('Error', 'project name have only have english or digit or underline and cannot be null')
            return
        self.text_entry.config(state=tk.DISABLED)
        
        self.create_dir()
        
        if not os.path.exists(os.path.join(self.work_dir, 'sparse_points_interest.ply')):
            meshlab_path = r"MeshLab/meshlab.exe"

            if os.path.exists(meshlab_path):
                subprocess.Popen([meshlab_path])
                
            if platform.system() == "Windows":
                os.startfile(self.work_dir)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open {self.work_dir}")
            else:  # Linux
                os.system(f"xdg-open {self.work_dir}")
            
        while not os.path.exists(os.path.join(self.work_dir, 'sparse_points_interest.ply')):
            messagebox.showinfo('Notice', 'when the notice is closed, the process will continue\nplease create sparse_points_interest.ply before close this notice')
            
        gen_cameras(self.work_dir)
        export_colmap_matches(self.work_dir)
        train(self.case_name)
        
        self.select_button.config(state=tk.NORMAL)
        self.generate_auto_button.config(state=tk.NORMAL)
        self.generate_manual_button.config(state=tk.NORMAL)
        self.text_entry.config(state=tk.NORMAL)
        
    def create_dir(self):
        self.data_dirs = Path('porf_data/dtu')
        
        self.work_dir = os.path.join(self.data_dirs, self.case_name)
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, 'sparse'), exist_ok=True)
            
        new_images_dir = os.path.join(self.data_dirs, self.case_name, 'images')
        
        # new_image_dir = os.path.join(data_dirs, case_name, 'image')
        # os.makedirs(new_image_dir, exist_ok=True)
        
        target_height = 1200
        
        if os.listdir(self.image_dir)[0].endswith((".png")) and Image.open(os.path.join(self.image_dir, os.listdir(self.image_dir)[0])).size[1] <= target_height:
            shutil.copytree(os.path.join(self.image_dir), os.path.join(new_images_dir))
        else:
            os.makedirs(new_images_dir, exist_ok=True)
            for filename in os.listdir(self.image_dir):
                if filename.endswith((".jpg", ".jpeg", '.png')):
                    img_path = os.path.join(self.image_dir, filename)
                    img = Image.open(img_path)
                    
                    width, height = img.size
                    aspect_ratio = width / height
                    new_width = int(target_height * aspect_ratio)
                    img = img.resize((new_width, target_height), Image.LANCZOS)
                    
                    output_path = os.path.join(new_images_dir, filename.split('.')[0] + '.png')
                    print(output_path)
                    img.save(output_path, 'PNG')

        # self.status_label.config(text="training status: generating poses")
        
        num_used_image = imgs2poses(self.work_dir)
        
        # self.image_status_label.config(text=f"image use: {num_used_image}/{len(os.listdir(self.image_dir))}")
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FolderSelectorApp(root)
    root.mainloop()


# png
# 拍新照片
# 測試exe