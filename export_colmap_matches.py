from pathlib import Path
import numpy as np
import sqlite3
import os


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1 - 1, image_id2 - 1


def image_ids_to_pair(image_id1, image_id2):
    image_id1 += 1
    image_id2 += 1
    pair_id = image_id2 + 2147483647 * image_id1
    return pair_id


def get_keypoints(cursor, image_id):
    image_id += 1
    cursor.execute("SELECT * FROM keypoints WHERE image_id = ?;", (image_id,))
    image_idx, n_rows, n_columns, raw_data = cursor.fetchone()
    kypnts = np.frombuffer(raw_data, dtype=np.float32).reshape(n_rows, n_columns).copy()
    kypnts = kypnts[:, 0:2]
    return kypnts


def process_one_scene(scene_dir):

    # --- 智慧路徑判斷 START ---
    # 優先嘗試 "前面流程" 的路徑 (資料庫在 parent 目錄)
    filename_db = Path(scene_dir).parent / 'database.db'

    # 如果找不到，則嘗試 "後面流程" 的路徑 (資料庫在當前目錄)
    if not filename_db.exists():
        filename_db = Path(scene_dir) / 'database.db'
    # --- 智慧路徑判斷 END ---
    outdir = scene_dir/'colmap_matches'
    image = Path(scene_dir) / 'images' # 優先嘗試 'images'
    if not image.exists():
        image = Path(scene_dir) / 'image' # 若 'images' 不存在，則嘗試 'image'
    print("Opening database: " + str(filename_db))

    if not os.path.exists(filename_db):
        print('Error db does not exist!')
        exit()
        
    if not os.path.exists(image):
        print('Error image does not exist!')
        exit()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print(f'Clean old matches in {outdir}')
    for f in outdir.glob('*'):
        if f.is_file():
            os.remove(f)

    connection = sqlite3.connect(filename_db)
    cursor = connection.cursor()

    list_image_ids = []
    img_ids_to_names_dict = {}
    cursor.execute('SELECT image_id, name, cameras.width, cameras.height FROM images LEFT JOIN cameras ON images.camera_id == cameras.camera_id;')
    for row in cursor:
        image_idx, name, width, height = row
        list_image_ids.append(image_idx - 1)
        img_ids_to_names_dict[image_idx - 1] = name

    num_image_ids = len(list_image_ids)
    # print(Path(image).files('*'))
    exist = [1] * (num_image_ids)
    for i in img_ids_to_names_dict:
        if not (image / img_ids_to_names_dict[i]).exists():
            # print(os.path.join(image, img_ids_to_names_dict[i]))
            exist[i] = 0
    

    # Iterate over entries in the two-view geometry table
    cursor.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries;')
    all_matches = {}
    for row in cursor:
        pair_id = row[0]
        rows = row[1]
        cols = row[2]
        raw_data = row[3]
        if (rows < 5):
            continue

        matches = np.frombuffer(raw_data, dtype=np.uint32).reshape(rows, cols)

        if matches.shape[0] < 5:
            continue

        all_matches[pair_id] = matches

    # print(len(all_matches))
    minus = [0] * (num_image_ids)    
    for j, x in enumerate(exist):
        if not x:
            for i in range(j, num_image_ids):
                minus[i] += 1 
    print(exist, minus)    

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='new_images'")
    table_exists = cursor.fetchone()
    if not table_exists:
        cursor.execute('CREATE TABLE new_images AS SELECT * FROM images')
        connection.commit()
        for image_id, value_to_subtract in enumerate(minus):
            if not exist[image_id]:
                cursor.execute('''
                    DELETE FROM new_images 
                    WHERE image_id = ?
                ''', (image_id + 1,))
                connection.commit()
            else:
                cursor.execute('''
                    UPDATE new_images 
                    SET image_id = image_id - 1 - ? 
                    WHERE image_id = ?
                ''', (value_to_subtract, image_id + 1))
                connection.commit()
    
        
    for key in all_matches:
        pair_id = key
        matches = all_matches[key]

        # # skip if too few matches are given
        # if matches.shape[0] < 300:
        #     continue

        id1, id2 = pair_id_to_image_ids(pair_id)
        image_name1 = img_ids_to_names_dict[id1]
        image_name2 = img_ids_to_names_dict[id2]

        keys1 = get_keypoints(cursor, id1)
        keys2 = get_keypoints(cursor, id2)

        match_positions = np.empty([matches.shape[0], 4])
        for i in range(0, matches.shape[0]):
            match_positions[i, :] = np.array([keys1[matches[i, 0]][0], keys1[matches[i, 0]][1], keys2[matches[i, 1]][0], keys2[matches[i, 1]][1]])

        # print(match_positions.shape)
        # outfile = os.path.join(outdir, image_name1.split("/")[0].split(".jpg")[0] + "_" + image_name2.split("/")[0].split(".jpg")[0] + ".txt")

        if exist[int(id1)] and exist[int(id2)]:
            id1 = id1 - minus[int(id1)]
            id2 = id2 - minus[int(id2)]
            outfile = os.path.join(outdir, '{:06d}_{:06d}.txt'.format(int(id1), int(id2)))

            np.savetxt(outfile, match_positions, delimiter=' ')

            # reverse and save
            match_positions_reverse = np.concatenate([match_positions[:, 2:4], match_positions[:, 0:2]], axis=1)
            # outfile = os.path.join(outdir, image_name2.split("/")[0].split(".jpg")[0] + "_" + image_name1.split("/")[0].split(".jpg")[0] + ".txt")
            outfile = os.path.join(outdir, '{:06d}_{:06d}.txt'.format(int(id2), int(id1)))
            np.savetxt(outfile, match_positions_reverse, delimiter=' ')

    cursor.close()
    connection.close()

    for idx in range(num_image_ids):
        if not exist[idx]:
            continue
        idx2 = idx - minus[idx]

        two_view = {}
        two_view["src_idx"] = []
        two_view["match"] = []

        files = sorted(outdir.glob('{:06d}_*.txt'.format(idx2)))
        for f in files:
            j = int(os.path.basename(f)[7:13])

            one_pair = np.loadtxt(f)
            # one_list = []
            # for i in one_pair:
            #     # print(list(i))
            #     one_list.append(list(i))   
            # print("\n") 
            # print(len(one_list))
            two_view["src_idx"].append(j)
            # two_view["match"].append(one_list)
            two_view["match"].append(one_pair)

        # print(two_view)
        np.savez(outdir/'{:06d}.npz'.format(idx2), **two_view)


def export_colmap_matches(scene_dir):
    process_one_scene(scene_dir)