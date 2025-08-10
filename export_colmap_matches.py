from path import Path
import numpy as np
import sqlite3
import os
import logging

def pair_id_to_image_ids(pair_id):
    """
    將 COLMAP 資料庫中的 pair_id 解碼為兩個影像的 ID。
    這是 COLMAP 用來唯一標識一對影像的編碼方式。
    """
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1 - 1, image_id2 - 1

def get_keypoints(cursor, image_id):
    """從資料庫中獲取指定影像的所有特徵點 (keypoints)。"""
    image_id += 1
    cursor.execute("SELECT data FROM keypoints WHERE image_id = ?;", (image_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    raw_data = row[0]
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 6)
    return kypnts[:, :2]  # 只取 x, y 座標

def process_one_scene(scene_dir):
    """
    處理單一場景的主函式。
    從 database.db 讀取特徵匹配，並將其匯出為後續流程所需的 .txt 和 .npz 格式。
    """
    filename_db = Path(scene_dir) / 'database.db'
    outdir = scene_dir / 'colmap_matches'
    image_dir = scene_dir / 'image'
    logging.info(f"正在開啟資料庫: {filename_db}")

    if not os.path.exists(filename_db):
        logging.error('錯誤: database.db 不存在!')
        return
    if not os.path.exists(image_dir):
        logging.error('錯誤: image 資料夾不存在!')
        return

    os.makedirs(outdir, exist_ok=True)

    logging.info(f'正在清理舊的匹配檔案於: {outdir}')
    for f in Path(outdir).glob('*'):
        try:
            os.remove(f)
        except OSError as e:
            logging.error(f"移除檔案 {f} 失敗: {e}")

    connection = sqlite3.connect(filename_db)
    cursor = connection.cursor()

    # 從資料庫讀取影像列表
    img_ids_to_names_dict = {}
    cursor.execute('SELECT image_id, name FROM images')
    for row in cursor:
        image_idx, name = row
        img_ids_to_names_dict[image_idx - 1] = name
    
    num_image_ids = len(img_ids_to_names_dict)
    
    # 檢查哪些影像是實際存在的 (處理資料庫與檔案系統不同步的情況)
    exist = [0] * num_image_ids
    for i in range(num_image_ids):
        if i in img_ids_to_names_dict and os.path.exists(os.path.join(image_dir, img_ids_to_names_dict[i])):
            exist[i] = 1

    # 讀取所有雙視圖幾何匹配
    cursor.execute('SELECT pair_id, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id, raw_data = row
        if raw_data is None:
            continue
        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(-1, 2)
        if len(matches) < 5:
            continue
        all_matches[pair_id] = matches

    # 計算索引修正量，用於處理被移除的影像
    minus = [0] * num_image_ids
    for j, x in enumerate(exist):
        if not x:
            for i in range(j, num_image_ids):
                minus[i] += 1
    logging.info(f"影像存在陣列: {exist}")
    logging.info(f"索引修正陣列: {minus}")

    # 在資料庫中建立一個名為 new_images 的新表，只包含存在的影像，並重新編號
    cursor.execute("DROP TABLE IF EXISTS new_images")
    cursor.execute('CREATE TABLE new_images AS SELECT * FROM images WHERE image_id IN ({})'.format(','.join('?' for _ in [i + 1 for i, e in enumerate(exist) if e == 1])))
    connection.commit()
    
    # 更新 new_images 中的 image_id
    for image_id, value_to_subtract in enumerate(minus):
        if exist[image_id]:
            cursor.execute('UPDATE new_images SET image_id = image_id - ? WHERE image_id = ?', (value_to_subtract, image_id + 1))
            connection.commit()

    # 處理並儲存匹配
    for pair_id, matches in all_matches.items():
        id1, id2 = pair_id_to_image_ids(pair_id)

        # 確保兩個影像都存在
        if not (exist[int(id1)] and exist[int(id2)]):
            continue

        keys1 = get_keypoints(cursor, id1)
        keys2 = get_keypoints(cursor, id2)
        
        if keys1 is None or keys2 is None:
            logging.warning(f"無法獲取影像 {id1} 或 {id2} 的特徵點，跳過此對。")
            continue

        match_positions = np.empty([matches.shape[0], 4])
        for i in range(matches.shape[0]):
            idx1, idx2 = matches[i]
            match_positions[i, :] = np.array([keys1[idx1, 0], keys1[idx1, 1], keys2[idx2, 0], keys2[idx2, 1]])

        # 使用修正後的索引儲存
        id1_new = id1 - minus[int(id1)]
        id2_new = id2 - minus[int(id2)]
        outfile = os.path.join(outdir, f'{id1_new:06d}_{id2_new:06d}.txt')
        np.savetxt(outfile, match_positions, delimiter=' ')

        # 儲存反向匹配
        match_positions_reverse = np.concatenate([match_positions[:, 2:4], match_positions[:, 0:2]], axis=1)
        outfile_reverse = os.path.join(outdir, f'{id2_new:06d}_{id1_new:06d}.txt')
        np.savetxt(outfile_reverse, match_positions_reverse, delimiter=' ')

    cursor.close()
    connection.close()

    # 將每個影像的匹配對打包成 .npz 檔案
    num_new_images = sum(exist)
    for idx2 in range(num_new_images):
        two_view = {"src_idx": [], "match": []}
        
        files = sorted(Path(outdir).glob(f'{idx2:06d}_*.txt'))
        for f in files:
            j = int(os.path.basename(f)[7:13])
            one_pair = np.loadtxt(f)
            two_view["src_idx"].append(j)
            two_view["match"].append(one_pair)

        np.savez(outdir / f'{idx2:06d}.npz', **two_view)
    
    logging.info("匯出 COLMAP 匹配完成。")

def export_colmap_matches(scene_dir):
    """
    匯出 COLMAP 匹配的進入點函式。
    """
    process_one_scene(scene_dir)

if __name__ == '__main__':
    # 作為獨立腳本執行時的範例
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 請將此路徑替換為您的場景路徑
    # example_scene_dir = './porf_data/dtu/scan24'
    # if os.path.exists(example_scene_dir):
    #     export_colmap_matches(example_scene_dir)
    # else:
    #     logging.error(f"範例路徑 '{example_scene_dir}' 不存在，請修改腳本。")
    pass