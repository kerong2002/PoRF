import os
import subprocess


def run_glomap(basedir, match_type):
    """
    使用 GloMAP 執行稀疏重建流程。
    這個流程會先使用 COLMAP 進行特徵提取和匹配，
    然後用 GloMAP 取代 COLMAP mapper 來進行高速的稀疏重建。
    """

    # 將日誌檔名更改以反映使用的是 GloMAP
    logfile_name = os.path.join(basedir, 'glomap_output.txt')
    logfile = open(logfile_name, 'w')

    print("--- STAGE 1: Feature Extraction ---")
    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
        # 如果您有支援 CUDA 的 GPU，建議取消下面這行的註解以加速
        # '--SiftExtraction.use_gpu', '1',
    ]
    print(f"Executing: {' '.join(feature_extractor_args)}")
    try:
        feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(feat_output)
        print('Features extracted successfully.\n')
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction:\n{e.output}")
        logfile.write(e.output)
        logfile.close()
        return

    print("--- STAGE 2: Feature Matching ---")
    # 根據傳入的 match_type 決定匹配器
    matcher_args = [
        'colmap', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
    ]
    print(f"Executing: {' '.join(matcher_args)}")
    try:
        match_output = subprocess.check_output(matcher_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(match_output)
        print('Features matched successfully.\n')
    except subprocess.CalledProcessError as e:
        print(f"Error during feature matching:\n{e.output}")
        logfile.write(e.output)
        logfile.close()
        return

    # 確保 sparse 資料夾存在
    sparse_path = os.path.join(basedir, 'sparse')
    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)

    print("--- STAGE 3: Sparse Reconstruction with GloMAP ---")
    # ==========================================================
    #  核心修改：將 colmap mapper 替換成 glomap mapper
    # ==========================================================
    glomap_mapper_args = [
        'glomap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', sparse_path,
        # 您可以根據需要添加或修改 GloMAP 的特定參數
        # 例如: '--TrackEstablishment.max_num_tracks', '5000'
    ]
    print(f"Executing: {' '.join(glomap_mapper_args)}")
    try:
        map_output = subprocess.check_output(glomap_mapper_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(map_output)
        print('Sparse map created successfully with GloMAP.')
    except subprocess.CalledProcessError as e:
        print(f"Error during GloMAP mapping:\n{e.output}")
        logfile.write(e.output)
    finally:
        logfile.close()

    print(f'\nFinished running GloMAP, see {logfile_name} for logs.')


if __name__ == "__main__":
    # 使用範例：
    # 您需要提供一個基礎目錄 (basedir)，其中應包含一個名為 'images' 的子資料夾
    # 例如： python your_script_name.py --basedir /path/to/your/dataset --matcher exhaustive_matcher

    import argparse

    parser = argparse.ArgumentParser(description="Run the GloMAP sparse reconstruction pipeline.")
    parser.add_argument("--basedir", default="Dataset/scan24",  help="Path to the base directory containing the 'images' folder.")
    parser.add_argument("--matcher", default="exhaustive_matcher", choices=["exhaustive_matcher", "sequential_matcher"],
                        help="Feature matcher to use.")

    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.basedir, 'images')):
        print(f"Error: 'images' folder not found in {args.basedir}")
    else:
        run_glomap(args.basedir, args.matcher)