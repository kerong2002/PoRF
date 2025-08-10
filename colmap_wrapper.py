import os
import subprocess
import sys

def run_colmap(basedir, match_type, colmap_executable='colmap'):
    """
    使用 COLMAP 處理位於 basedir 的資料集。

    Args:
        basedir (str): 資料集的根目錄，應包含 'images' 子目錄。
        match_type (str): COLMAP 使用的匹配器類型，例如 'exhaustive_matcher'。
        colmap_executable (str): COLMAP 可執行檔的路徑。預設為 'colmap'，
                                 假設它在系統 PATH 中。如果不在，請提供完整路徑。
    """

    # --- 路徑和參數檢查 ---
    if not os.path.isdir(basedir):
        print(f"錯誤：找不到資料集目錄 '{basedir}'")
        return

    image_path = os.path.join(basedir, 'images')
    if not os.path.isdir(image_path):
        print(f"錯誤：在 '{basedir}' 中找不到 'images' 子目錄。")
        return

    database_path = os.path.join(basedir, 'database.db')
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    sparse_path = os.path.join(basedir, 'sparse')

    # --- 使用 with open 確保日誌檔案會被妥善關閉 ---
    with open(logfile_name, 'w') as logfile:
        
        # --- 步驟 1: 特徵提取 ---
        print("步驟 1: 正在提取特徵...")
        logfile.write("--- 步驟 1: 特徵提取 ---\n")
        feature_extractor_args = [
            colmap_executable, 'feature_extractor',
            '--database_path', database_path,
            '--image_path', image_path,
            '--ImageReader.single_camera', '1',
        ]
        
        try:
            # 使用 subprocess.run 可以更好地控制輸出和錯誤
            result = subprocess.run(feature_extractor_args, capture_output=True, text=True, check=True)
            logfile.write(result.stdout)
            logfile.write(result.stderr)
            print("特徵提取完成。")
        except FileNotFoundError:
            print(f"錯誤：找不到 COLMAP 可執行檔 '{colmap_executable}'。")
            print("請確認 COLMAP 已安裝，且其路徑已加入系統 PATH 環境變數，或透過 'colmap_executable' 參數指定完整路徑。")
            logfile.write(f"錯誤：找不到 COLMAP 可執行檔 '{colmap_executable}'。\n")
            return
        except subprocess.CalledProcessError as e:
            print(f"特徵提取失敗！COLMAP 錯誤訊息請見日誌檔。")
            logfile.write(e.stdout)
            logfile.write(e.stderr)
            return

        # --- 步驟 2: 特徵匹配 ---
        print(f"\n步驟 2: 使用 '{match_type}' 進行特徵匹配...")
        logfile.write(f"\n--- 步驟 2: 特徵匹配 ({match_type}) ---\n")
        exhaustive_matcher_args = [
            colmap_executable, match_type,
            '--database_path', database_path,
        ]

        try:
            result = subprocess.run(exhaustive_matcher_args, capture_output=True, text=True, check=True)
            logfile.write(result.stdout)
            logfile.write(result.stderr)
            print("特徵匹配完成。")
        except subprocess.CalledProcessError as e:
            print(f"特徵匹配失敗！COLMAP 錯誤訊息請見日誌檔。")
            logfile.write(e.stdout)
            logfile.write(e.stderr)
            return

        # --- 步驟 3: 稀疏重建 (Mapper) ---
        print("\n步驟 3: 正在建立稀疏地圖...")
        logfile.write("\n--- 步驟 3: 稀疏重建 ---\n")
        if not os.path.exists(sparse_path):
            os.makedirs(sparse_path)

        mapper_args = [
            colmap_executable, 'mapper',
            '--database_path', database_path,
            '--image_path', image_path,
            '--output_path', sparse_path,
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
        ]

        try:
            result = subprocess.run(mapper_args, capture_output=True, text=True, check=True)
            logfile.write(result.stdout)
            logfile.write(result.stderr)
            print("稀疏地圖建立完成。")
        except subprocess.CalledProcessError as e:
            print(f"稀疏地圖建立失敗！COLMAP 錯誤訊息請見日誌檔。")
            logfile.write(e.stdout)
            logfile.write(e.stderr)
            return

    print(f'\nCOLMAP 流程執行完畢，詳細日誌請見: {logfile_name}')


# ===================================================================
# 主要執行區塊：讓此腳本可以獨立執行以進行測試
# ===================================================================
if __name__ == "__main__":
    # ！！！！【請根據您的環境修改以下路徑】！！！！

    # --- 設定 1: COLMAP 可執行檔路徑 ---
    # 如果您的 'colmap.exe' 不在系統 PATH 中，請提供它的完整路徑。
    # Windows 範例: "C:/Program Files/COLMAP/bin/colmap.exe"
    # Linux/macOS 範例: "/usr/local/bin/colmap"
    # 如果 colmap 已經在 PATH 中，保留預設值 'colmap' 即可。
    colmap_path = 'colmap'

    # --- 設定 2: 資料集根目錄 ---
    # 請將此路徑替換為您存放資料集的真實路徑。
    # 這個目錄下應該要有一個名為 "images" 的資料夾。
    # 範例: "C:/Users/krameri120/Desktop/porf/porf_data/dtu/scan24"
    dataset_base_dir = "C:/Users/krameri120/Desktop/porf/porf_data/dtu/scan24"

    # --- 設定 3: 匹配器類型 ---
    # 'exhaustive_matcher' 適用於影像較少的場景，比較徹底。
    # 'sequential_matcher' 適用於影片序列。
    # 'spatial_matcher' 適用於有 GPS 資訊的影像。
    matcher = "exhaustive_matcher"

    # --- 執行 COLMAP 流程 ---
    print(f"準備執行 COLMAP...")
    print(f"  - 資料集路徑: {dataset_base_dir}")
    print(f"  - COLMAP 執行檔: {colmap_path}")
    print(f"  - 匹配器: {matcher}")
    print("-" * 30)

    # 檢查設定的路徑是否存在
    if not os.path.isdir(dataset_base_dir):
        print(f"錯誤：設定的資料集路徑不存在: '{dataset_base_dir}'")
        print("請修改 __main__ 區塊中的 'dataset_base_dir' 變數。")
        sys.exit(1) # 終止程式

    run_colmap(dataset_base_dir, matcher, colmap_executable=colmap_path)