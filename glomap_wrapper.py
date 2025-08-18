import os
import subprocess
import time


def run_glomap(basedir, match_type):
    """
    使用 GloMAP 執行稀疏重建流程，並記錄每個步驟的時間。
    這個流程會先使用 COLMAP 進行特徵提取和匹配，然後用 GloMAP 取代 COLMAP mapper。
    """
    logfile_name = os.path.join(basedir, 'glomap_output.txt')
    logfile = open(logfile_name, 'w')

    total_start_time = time.time()

    # --- STAGE 1: 特徵提取 (Feature Extraction) ---
    print("--- GloMAP Pipeline STAGE 1: 特徵提取 (使用 COLMAP) ---")
    feature_start_time = time.time()
    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.use_gpu', '1',  # 啟用 GPU 加速
    ]
    print(f"執行中: {' '.join(feature_extractor_args)}")
    try:
        feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(feat_output)
        feature_time = time.time() - feature_start_time
        print(f'特徵提取成功。耗時: {feature_time:.2f} 秒。\n')
    except subprocess.CalledProcessError as e:
        print(f"特徵提取出錯:\n{e.output}")
        logfile.write(e.output)
        return -1

    # --- STAGE 2: 特徵匹配 (Feature Matching) ---
    print("--- GloMAP Pipeline STAGE 2: 特徵匹配 (使用 COLMAP) ---")
    match_start_time = time.time()
    matcher_args = [
        'colmap', match_type,
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.use_gpu', '1',  # 啟用 GPU 加速
    ]
    print(f"執行中: {' '.join(matcher_args)}")
    try:
        match_output = subprocess.check_output(matcher_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(match_output)
        match_time = time.time() - match_start_time
        print(f'特徵匹配成功。耗時: {match_time:.2f} 秒。\n')
    except subprocess.CalledProcessError as e:
        print(f"特徵匹配出錯:\n{e.output}")
        logfile.write(e.output)
        return -1

    # 確保 sparse 資料夾存在
    sparse_path = os.path.join(basedir, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)

    # --- STAGE 3: 稀疏重建 (Sparse Reconstruction with GloMAP) ---
    print("--- GloMAP Pipeline STAGE 3: 稀疏重建 (使用 GloMAP) ---")
    map_start_time = time.time()
    glomap_mapper_args = [
        'glomap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', sparse_path,
    ]
    print(f"執行中: {' '.join(glomap_mapper_args)}")
    try:
        map_output = subprocess.check_output(glomap_mapper_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(map_output)
        map_time = time.time() - map_start_time
        print(f'GloMAP 稀疏重建成功。耗時: {map_time:.2f} 秒。\n')
    except subprocess.CalledProcessError as e:
        print(f"GloMAP Mapper 出錯:\n{e.output}")
        logfile.write(e.output)
    finally:
        logfile.close()

    total_time = time.time() - total_start_time
    print(f'GloMAP 流程總耗時: {total_time:.2f} 秒。詳情請見 {logfile_name}')
    return total_time
