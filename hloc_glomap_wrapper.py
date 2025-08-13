import os
import sys
from pathlib import Path
import argparse
import subprocess
import pycolmap
import shutil

# 將 hloc 的路徑加入到系統路徑中
hloc_path = Path(__file__).parent / 'Hierarchical-Localization-1.4'
sys.path.append(str(hloc_path))

from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils.database import COLMAPDatabase
from hloc.triangulation import import_features, import_matches
from export_colmap_matches import export_colmap_matches

def run_hloc_glomap(basedir):
    """
    使用 hloc 進行特徵提取和匹配，然後使用 glomap 進行稀疏重建。
    """
    
    image_src_dir = Path(basedir) / 'images'
    image_dir = Path(basedir) / 'image'
    image_dir.mkdir(exist_ok=True)
    
    # 複製影像到 image 資料夾
    for f in image_src_dir.iterdir():
        if f.is_file():
            shutil.copy(f, image_dir / f.name)

    outputs = Path(basedir) / 'hloc_glomap_outputs'
    outputs.mkdir(exist_ok=True)
    
    db_path = outputs / 'database.db'
    if db_path.exists():
        db_path.unlink()

    # hloc 設定
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # --- STAGE 1: hloc 特徵提取和匹配 ---
    print("--- STAGE 1: hloc feature extraction and matching ---")
    
    # 1. 產生影像對
    print(f"Generating pairs from images in {image_dir}")
    image_list = [p.name for p in image_dir.iterdir() if p.is_file()]
    if not image_list:
        raise ValueError(f"No images found in {image_dir}")
    
    pair_path = outputs / 'pairs-exhaustive.txt'
    pairs_from_exhaustive.main(output=pair_path, image_list=image_list)

    # 2. 提取特徵
    feature_path = extract_features.main(feature_conf, image_dir, outputs)

    # 3. 匹配特徵
    match_path = match_features.main(matcher_conf, pair_path, feature_conf['output'], outputs)

    # --- STAGE 2: 將 hloc 結果匯入 COLMAP 資料庫 ---
    print("--- STAGE 2: Import hloc results into COLMAP database ---")
    
    # 使用 colmap 建立初始資料庫
    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', str(db_path),
        '--image_path', str(image_dir),
        '--ImageReader.single_camera', '1',
    ], check=True)
    
    # 【=========== 新增的修正程式碼區塊 ===========】
    print("Clearing SIFT features from database before importing SuperPoint features.")
    db = COLMAPDatabase.connect(db_path)
    db.execute("DELETE FROM keypoints;")  # 刪除舊的 keypoints
    db.execute("DELETE FROM descriptors;") # 刪除舊的 descriptors
    db.commit()                           # 確認變更
    db.close()
    # 【===========================================】

    # 讀取 colmap 產生的 image_ids
    db = COLMAPDatabase.connect(db_path)
    image_ids = dict((c, i) for i, c in db.execute("SELECT image_id, name FROM images"))
    db.close()

    import_features(image_ids, db_path, feature_path)
    import_matches(image_ids, db_path, pair_path, match_path, skip_geometric_verification=True)


    # --- STAGE 3: GloMAP 稀疏重建 ---
    print("--- STAGE 3: Sparse Reconstruction with GloMAP ---")
    
    sparse_path = Path(basedir) / 'sparse'
    sparse_path.mkdir(exist_ok=True)
    
    glomap_mapper_args = [
        'glomap', 'mapper',
        '--database_path', str(db_path),
        '--image_path', str(image_dir),
        '--output_path', str(sparse_path),
    ]
    print(f"Executing: {' '.join(glomap_mapper_args)}")
    subprocess.run(glomap_mapper_args, check=True)

    # --- STAGE 4: 匯出 porf 需要的匹配檔案 ---
    print("--- STAGE 4: Exporting matches for porf ---")
    export_colmap_matches(outputs)

    print(f'\nFinished running hloc+GloMAP pipeline.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the hloc+GloMAP pipeline.")
    parser.add_argument("--basedir", default="Dataset/scan24", help="Path to the base directory.")
    args = parser.parse_args()
    run_hloc_glomap(args.basedir)