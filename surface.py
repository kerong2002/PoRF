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
        # 設定視窗大小和位置（這裡設定為 600x350 像素）
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

        # Reconstruction technology selection
        self.recon_tech_label = tk.Label(root, text="Select reconstruction technology:")
        self.recon_tech_label.pack(pady=5)
        self.recon_tech = tk.StringVar(value="colmap")
        self.colmap_radio = tk.Radiobutton(root, text="COLMAP", variable=self.recon_tech, value="colmap")
        self.colmap_radio.pack(pady=2)
        self.glomap_radio = tk.Radiobutton(root, text="GloMAP", variable=self.recon_tech, value="glomap")
        self.glomap_radio.pack(pady=2)

        # Button to Auto Generate sparse_points_interest.ply
        self.generate_auto_button = tk.Button(root, text="Auto Generate sparse_points_interest.ply",
                                              command=self.start_auto)
        self.generate_auto_button.pack(pady=10)
        self.generate_auto_button.config(state=tk.DISABLED)

        # Button to Waiting Manually Create sparse_points_interest.ply
        self.generate_manual_button = tk.Button(root, text="Waiting Manually Create sparse_points_interest.ply",
                                                command=self.start_manual)
        self.generate_manual_button.pack(pady=10)
        self.generate_manual_button.config(state=tk.DISABLED)

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
            messagebox.showinfo('Notice',
                                'when the notice is closed, the process will continue\nplease create sparse_points_interest.ply before close this notice')

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

        # 檢查圖片是否需要縮放
        first_image_path = os.path.join(self.image_dir, os.listdir(self.image_dir)[0])
        with Image.open(first_image_path) as img:
            height = img.height

        if first_image_path.endswith(".png") and height <= target_height:
            shutil.copytree(self.image_dir, new_images_dir)
        else:
            os.makedirs(new_images_dir, exist_ok=True)
            for filename in os.listdir(self.image_dir):
                if filename.lower().endswith((".jpg", ".jpeg", '.png')):
                    img_path = os.path.join(self.image_dir, filename)
                    img = Image.open(img_path)

                    width, height = img.size
                    aspect_ratio = width / height
                    new_width = int(target_height * aspect_ratio)
                    img = img.resize((new_width, target_height), Image.LANCZOS)

                    output_path = os.path.join(new_images_dir, os.path.splitext(filename)[0] + '.png')
                    print(f"Resizing and saving to: {output_path}")
                    img.save(output_path, 'PNG')

        # --- ★ 修改部分 START ---
        tech = self.recon_tech.get()
        total_time = -1

        if tech == "colmap":
            try:
                from colmap_wrapper import run_colmap
                total_time = run_colmap(self.work_dir, 'exhaustive_matcher')
            except ImportError:
                messagebox.showerror("Error", "colmap_wrapper.py not found!")
                return
        else:
            try:
                from glomap_wrapper import run_glomap
                total_time = run_glomap(self.work_dir, 'exhaustive_matcher')
            except ImportError:
                messagebox.showerror("Error", "glomap_wrapper.py not found!")
                return

        # 在終端機印出重建所花費的時間
        if total_time != -1:
            time_message = f"--- {tech.upper()} 重建流程總耗時: {total_time:.2f} 秒 ---"
            print(time_message)
            # 也可以彈出一個訊息視窗顯示時間
            messagebox.showinfo('Reconstruction Time', time_message)
        # --- ★ 修改部分 END ---

        num_used_image = imgs2poses(self.work_dir)


if __name__ == "__main__":
    root = tk.Tk()
    app = FolderSelectorApp(root)
    root.mainloop()
