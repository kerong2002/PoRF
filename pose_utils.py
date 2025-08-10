import sqlite3
import numpy as np
import os
import sys
from PIL import Image
import logging

from path import Path
import imageio
import skimage.transform
import trimesh

from colmap_wrapper import run_colmap
import colmap_read_model as read_model


def load_colmap_data(realdir):
    """
    從 COLMAP 的輸出目錄 (sparse/0) 載入重建資料。

    Args:
        realdir (str): 資料集的根目錄，包含 sparse/0 子目錄。

    Returns:
        num_used_image (int): 已註冊的影像數量。
        poses (np.array): 相機位姿矩陣。
        pts3d (dict): 3D 點雲資料。
        perm (np.array): 根據檔名排序的影像索引排列。
    """
    
    # 讀取相機參數 (cameras.bin)
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # 假設所有影像都使用同一個相機模型
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    logging.info(f'相機數量: {len(cam)}')

    # 取得影像高、寬、焦距
    h, w, f = cam.height, cam.width, cam.params[0]
    hwf = np.array([h,w,f]).reshape([3,1])
    
    # 讀取影像資訊 (images.bin)
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    # 取得所有已註冊影像的檔名
    names = [imdata[k].name for k in imdata]
    
    # 為了方便後續處理，將 images/ 中的原始圖片複製一份到 image/
    image_dir = os.path.join(realdir, 'images')
    new_image_dir = os.path.join(realdir, 'image')
    os.makedirs(new_image_dir, exist_ok=True)
    
    logging.info("正在複製圖片至 'image' 資料夾...")
    for filename in names:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        output_path = os.path.join(new_image_dir, filename)
        img.save(output_path)
        
    logging.info("已排序的圖片檔名:")
    # 根據檔名排序，並記錄其順序
    perm = np.argsort(names)
    for i in perm:
        logging.info(names[i])
        
    logging.info(f'已註冊的影像數量: {len(names)}')
    num_used_image = len(names)
    logging.info(f"影像排序索引: {perm}")

    logging.info("影像資料 (image.bin) 的 Key:")
    # 從 imdata 中提取旋轉 (qvec) 和平移 (tvec) 向量，建構 world-to-camera (w2c) 矩陣
    for k in imdata:
        logging.info(k)
        im = imdata[k]
        R = im.qvec2rotmat() # 四元數轉旋轉矩陣
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0)
    # 透過求逆矩陣得到 camera-to-world (c2w) 矩陣
    c2w_mats = np.linalg.inv(w2c_mats)
    
    # 整理成 NeRF 所需的格式 [3, 4, N] -> [R|t]
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # 將影像高、寬、焦距附加到 poses 矩陣中
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    logging.info(f"Poses 矩陣形狀: {poses.shape}")
    
    # 讀取 3D 點雲資料 (points3D.bin)
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    
    # 座標系轉換：從 COLMAP 的 [r, u, -t] 轉為 NeRF 常用的 [-u, r, -t]
    # (r, u, t 分別代表 right, up, translation)
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    return num_used_image, poses, pts3d, perm


def save_poses(basedir, poses, pts3d, perm):
    """
    儲存處理後的相機位姿和點雲資料。

    Args:
        basedir (str): 資料集的根目錄。
        poses (np.array): 相機位姿矩陣。
        pts3d (dict): 3D 點雲資料。
        perm (np.array): 影像排序索引。
    """
    filename_db = basedir + '/database.db'
    image_dir_path = basedir + '/image'
    if not os.path.exists(filename_db):
        logging.error('錯誤: database.db 不存在!')
        return
    if not os.path.exists(image_dir_path):
        logging.error('錯誤: image 資料夾不存在!')
        return
    
    # --- 這段程式碼似乎是用來處理資料庫中存在但影像檔案遺失的情況 ---
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
    
    exist = [1] * (num_image_ids)
    for i in img_ids_to_names_dict:
        if os.path.join(image_dir_path, img_ids_to_names_dict[i]) not in Path(image_dir_path).files('*'):
            exist[i] = 0
            
    minus = [0] * (num_image_ids)    
    for j, x in enumerate(exist):
        if not x:
            for i in range(j, num_image_ids):
                minus[i] += 1 
    logging.info(f"影像存在陣列: {exist}")
    logging.info(f"索引修正陣列: {minus}")
    # --- ---
    
    
    # 計算每個 3D 點被哪些相機看到 (可見性)
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * (poses.shape[-1])
        for ind in pts3d[k].image_ids:
            # 修正相機索引
            corrected_ind = ind - minus[ind - 1] - 1
            if len(cams) < corrected_ind:
                logging.error(f"相機索引超出範圍。Cams 長度: {len(cams)}, 需要的索引: {corrected_ind}")
                logging.error('錯誤：無法獲取當前點的正確相機位姿')
                return
            cams[corrected_ind] = 1
        vis_arr.append(cams)

    # 將點雲儲存為 .ply 檔案，方便視覺化檢查
    pts = np.stack(pts_arr, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    logging.info(f'點雲形狀: {pts_arr.shape}, 可見性陣列形狀: {vis_arr.shape}')

    # 儲存最終的 poses 矩陣為 .npy 檔案
    poses = np.moveaxis(poses, -1, 0)
    logging.info(f"用於位姿的排序索引: {perm}")
    # poses = poses[perm] # 這行程式碼被註解掉了，保持原樣
    np.save(os.path.join(basedir, 'poses.npy'), poses)


def minify_v0(basedir, factors=[], resolutions=[]):
    # (此函式為舊版降採樣，暫不加註解)
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)
        
        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                order=1, mode='constant', cval=0, clip=True, preserve_range=False, 
                                                 anti_aliasing=True, anti_aliasing_sigma=None)
        
        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(i)), (255*imgs_down[i]).astype(np.uint8))
            


def minify(basedir, factors=[], resolutions=[]):
    # (此函式為降採樣，暫不加註解)
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    # (此函式為 LLFF 資料集格式的讀取器，暫不加註解)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs
            
    
def gen_poses(basedir, match_type, factors=None):
    """
    從影像生成相機位姿的主函式。

    Args:
        basedir (str): 資料集的根目錄。
        match_type (str): COLMAP 使用的匹配器類型。
        factors (list, optional): 用於降採樣的因子。 Defaults to None.
    """
    
    # --- 設定日誌 ---
    # 建立一個日誌檔案，路徑為 basedir/pose_utils.log
    log_file = os.path.join(basedir, 'pose_utils.log')
    # 為避免重複寫入，如果已有 handler 則先移除
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # 設定日誌的基本組態
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='w'), # 寫入到檔案
                            # logging.StreamHandler() # 如果也想在終端機看到 INFO，可以取消此行註解
                        ])

    # 檢查 COLMAP 的稀疏重建結果是否存在
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    
    # 如果缺少任何一個 .bin 檔案，就執行 COLMAP
    if not all([f in files_had for f in files_needed]):
        print( '需要執行 COLMAP' )
        run_colmap(basedir, match_type)
    else:
        print('不需執行 COLMAP (已存在重建結果)')
        
    print('正在進行 COLMAP 後處理...')
    
    # 載入 COLMAP 資料
    num_used_image, poses, pts3d, perm = load_colmap_data(basedir)

    # 儲存位姿和點雲
    save_poses(basedir, poses, pts3d, perm)
    
    # 如果提供了降採樣因子，則執行影像降採樣
    if factors is not None:
        print( '降採樣因子:', factors)
        minify(basedir, factors)
    
    print( 'imgs2poses 流程執行完畢' )
    
    return num_used_image
