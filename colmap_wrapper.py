import os
import subprocess
import time


def run_colmap(basedir, match_type):
    """
    使用 COLMAP 執行完整的稀疏重建流程，並記錄每個步驟的時間。
    """
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    total_start_time = time.time()

    # --- STAGE 1: 特徵提取 (Feature Extraction) ---
    print("--- COLMAP STAGE 1: 特徵提取 ---")
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
    print("--- COLMAP STAGE 2: 特徵匹配 ---")
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

    # --- STAGE 3: 稀疏重建 (Sparse Reconstruction) ---
    print("--- COLMAP STAGE 3: 稀疏重建 (Mapper) ---")
    map_start_time = time.time()
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', sparse_path,
    ]
    print(f"執行中: {' '.join(mapper_args)}")
    try:
        map_output = subprocess.check_output(mapper_args, universal_newlines=True, stderr=subprocess.STDOUT)
        logfile.write(map_output)
        map_time = time.time() - map_start_time
        print(f'COLMAP 稀疏重建成功。耗時: {map_time:.2f} 秒。\n')
    except subprocess.CalledProcessError as e:
        print(f"COLMAP Mapper 出錯:\n{e.output}")
        logfile.write(e.output)
    finally:
        logfile.close()

    total_time = time.time() - total_start_time
    print(f'COLMAP 流程總耗時: {total_time:.2f} 秒。詳情請見 {logfile_name}')
    return total_time
