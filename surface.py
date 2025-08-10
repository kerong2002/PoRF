import platform
import re
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import logging
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
        # 設定視窗大小和位置
        root.geometry("600x350")

        self.root = root
        self.root.title("PoRF 處理工具")
        
        # 顯示資料夾名稱的標籤
        self.folder_label = tk.Label(root, text="尚未選擇資料夾")
        self.folder_label.pack(pady=10)
        
        # 選擇資料夾的按鈕
        self.select_button = tk.Button(root, text="選擇影像資料夾", command=self.select_folder)
        self.select_button.pack(pady=10)
        
        # 輸入專案名稱的標籤
        self.input_label = tk.Label(root, text="輸入專案名稱:")
        self.input_label.pack(pady=5)
        
        # 使用者輸入的文字框
        self.text_entry = tk.Entry(root, width=50)
        self.text_entry.pack(pady=5)
        self.text_entry.config(state=tk.NORMAL)
        
        # 自動產生 sparse_points_interest.ply 的按鈕
        self.generate_auto_button = tk.Button(root, text="自動產生 sparse_points_interest.ply", command=self.start_auto)
        self.generate_auto_button.pack(pady=10)
        self.generate_auto_button.config(state=tk.DISABLED)
        
        # 等待手動建立 sparse_points_interest.ply 的按鈕
        self.generate_manual_button = tk.Button(root, text="等待手動建立 sparse_points_interest.ply", command=self.start_manual)
        self.generate_manual_button.pack(pady=10)
        self.generate_manual_button.config(state=tk.DISABLED)
        
    def select_folder(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            self.folder_label.config(text=f"已選擇資料夾: {self.image_dir}")
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
            messagebox.showinfo('錯誤', '專案名稱只能包含英文、數字或底線，且不能為空')
            return
        self.text_entry.config(state=tk.DISABLED)
        
        self.create_dir()
        
        noise_cancel(self.work_dir)
        
        gen_cameras(self.work_dir)
        export_colmap_matches(self.work_dir)
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
            messagebox.showinfo('錯誤', '專案名稱只能包含英文、數字或底線，且不能為空')
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
            messagebox.showinfo('提示', '當此提示關閉後，流程將繼續。\n請在關閉此提示前，手動建立 sparse_points_interest.ply 檔案。')
            
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
        
        target_height = 1200
        
        # 檢查是否需要縮放圖片
        first_image_path = os.path.join(self.image_dir, os.listdir(self.image_dir)[0])
        if first_image_path.endswith((".png")) and Image.open(first_image_path).size[1] <= target_height:
            shutil.copytree(os.path.join(self.image_dir), os.path.join(new_images_dir))
            logging.info(f"直接複製影像至: {new_images_dir}")
        else:
            os.makedirs(new_images_dir, exist_ok=True)
            logging.info(f"開始縮放影像至高度 {target_height}px...")
            for filename in os.listdir(self.image_dir):
                if filename.endswith((".jpg", ".jpeg", '.png')):
                    img_path = os.path.join(self.image_dir, filename)
                    img = Image.open(img_path)
                    
                    width, height = img.size
                    aspect_ratio = width / height
                    new_width = int(target_height * aspect_ratio)
                    img = img.resize((new_width, target_height), Image.LANCZOS)
                    
                    output_path = os.path.join(new_images_dir, filename.split('.')[0] + '.png')
                    logging.info(f"儲存縮放後的影像: {output_path}")
                    img.save(output_path, 'PNG')

        imgs2poses(self.work_dir)
        
        
if __name__ == "__main__":
    root = tk.Tk()
    app = FolderSelectorApp(root)
    root.mainloop()