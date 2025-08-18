import os
import logging
import argparse
import random
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork, RenderingNetwork
from models.renderer import NeuSRenderer
from models.networks import LearnPose, PoRF
import utils

# 嘗試導入 wandb，如果沒有安裝則設定一個標記
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

print(f"torch version: {torch.__version__}")
print(f"wandb available: {WANDB_AVAILABLE}")


class PoseRunner:
    # def __init__(self, status_label, conf_path, mode='train', case='CASE_NAME'):
    def __init__(self, conf_path, mode='train', case='CASE_NAME'):
        # self.status_label = status_label
        self.device = torch.device('cuda')  # 設定使用的計算設備為 CUDA (GPU)

        # --- 1. 讀取與設定配置 ---
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        # 將設定檔中的 'CASE_NAME' 佔位符替換為實際的案例名稱
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        print("--- 讀取的設定檔內容 ---")
        print(conf_text)
        print("--------------------------")
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']  # 實驗結果的儲存根目錄
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])  # 初始化資料集
        self.iter_step = 0  # 初始化迭代計數器

        # --- 2. 讀取訓練參數 ---
        self.end_iter = self.conf['train.pose_end_iter']  # 總訓練迭代次數
        self.val_freq = self.conf.get_int('train.pose_val_freq')  # 驗證頻率
        self.report_freq = self.conf.get_int('train.report_freq')  # 報告 (log) 頻率
        self.batch_size = self.conf.get_int('train.batch_size')  # 每批次的光線數量
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')  # 驗證影像的解析度等級
        self.learning_rate = self.conf.get_float('train.learning_rate')  # 場景網路的學習率
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')  # 學習率衰減的最終比例
        self.pose_learning_rate = self.conf.get_float('train.pose_learning_rate')  # 相機姿態的學習率
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')  # 是否使用白色背景
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)  # 學習率預熱階段的結束點
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)  # NeuS 中 s_val 退火的結束點

        # PoRF (姿態優化) 相關參數
        self.use_porf = self.conf.get_bool('train.use_porf')
        self.inlier_threshold = self.conf.get_float('train.inlier_threshold')  # 對極幾何中內點的判斷閾值
        self.num_pairs = self.conf.get_int('train.num_pairs')  # 計算對極損失時，採樣的影像對數量

        # --- 3. 讀取損失函數的權重 ---
        self.color_loss_weight = self.conf.get_float('train.color_loss_weight')  # 顏色損失權重
        self.igr_weight = self.conf.get_float('train.igr_weight')  # Eikonal 損失權重 (幾何正規化)
        self.epipolar_loss_weight = self.conf.get_float('train.epipolar_loss_weight')  # 對極損失權重 (姿態優化)
        self.mode = mode

        # --- 4. 初始化日誌記錄工具 ---
        # TensorBoard: 用於本地查看訓練曲線
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'pose_logs'))

        # Weights & Biases: 用於雲端實驗追蹤與比較
        if WANDB_AVAILABLE:
            # --- ★ 錯誤修正 START ---
            # 將 pyhocon 物件轉換為標準的 Python 字典，以避免 wandb 初始化錯誤
            config_dict = self.conf.as_plain_ordered_dict()
            wandb.init(
                project="porf-neus-project",  # 您可以更改此專案名稱
                name=case,  # 實驗名稱，直接使用案例名稱
                config=config_dict,  # ★ 使用轉換後的字典
                dir=self.base_exp_dir  # 將 wandb 的本地檔案也存在實驗資料夾中
            )
            # --- ★ 錯誤修正 END ---
            print("Weights & Biases 日誌記錄功能已啟用。")
        else:
            print("未安裝 wandb，將跳過 wandb 日誌記錄。若要使用，請執行: pip install wandb")

        # --- 5. 初始化神經網路模型 ---
        params_to_train = []
        # SDF 網路: 學習場景的符號距離函數 (幾何形狀)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        # Variance 網路: 學習 NeuS 中的 s_val，控制 SDF 到密度的轉換
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # Rendering 網路: 學習從空間點、視角等資訊到最終顏色的映射
        self.render_network = RenderingNetwork(**self.conf['model.render_network']).to(self.device)

        # 將所有場景相關網路的參數加入到待訓練清單中
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.render_network.parameters())

        # 設定場景網路的優化器
        optim_params = [{'params': params_to_train, 'lr': self.learning_rate}]
        self.optimizer = torch.optim.Adam(optim_params)

        # 初始化 NeuS 渲染器，它會整合上述三個網路來執行體渲染
        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.render_network,
                                     **self.conf['model.neus_renderer'])

        # 初始化相機姿態優化網路
        if self.use_porf:
            # PoRF 是一個更複雜的姿態優化模型
            self.pose_param_net = PoRF(
                self.dataset.n_images,
                init_c2w=self.dataset.pose_all,
                scale=self.conf.get_float('train.scale')
            ).to(self.device)
        else:
            # LearnPose 是一個較簡單的姿態優化模型
            self.pose_param_net = LearnPose(
                self.dataset.n_images,
                init_c2w=self.dataset.pose_all
            ).to(self.device)

        # 設定姿態網路的優化器
        self.optimizer_pose = torch.optim.Adam(self.pose_param_net.parameters(),
                                               lr=self.pose_learning_rate)

        # 在訓練開始前，先驗證一次初始姿態的誤差
        if self.iter_step == 0:
            self.validate_pose(initial_pose=True)

    def train(self):
        # --- 主要訓練迴圈 ---
        self.update_learning_rate()  # 更新學習率
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step)):

            self.update_image_index()  # 隨機選擇一張圖片進行訓練

            # --- A. 姿態優化部分 ---
            # 從資料集中取樣匹配點，用於計算對極損失
            intrinsic, pose, intrinsic_src_list, pose_src_list, match_list = self.dataset.sample_matches(self.img_idx,
                                                                                                         self.pose_param_net)
            # 計算投影矩陣
            P_src_list = []
            for cam, p in zip(intrinsic_src_list, pose_src_list):
                P_src_list.append(utils.compute_P_from_KT(cam, p))

            # 計算對極損失 (Epipolar Loss) 和內點率
            avg_inlier_rate, epipolar_loss = utils.evaluate_pose(intrinsic,
                                                                 pose,
                                                                 P_src_list,
                                                                 match_list,
                                                                 self.num_pairs,
                                                                 self.inlier_threshold)

            # --- B. NeuS 渲染部分 ---
            # 在當前選擇的圖片中，隨機採樣一批光線
            data = self.dataset.gen_random_rays_at(self.img_idx,
                                                   self.batch_size,
                                                   pose)

            rays_o, rays_d = data[:, :3], data[:, 3: 6]  # 光線起點與方向
            true_rgb = data[:, 6: 9]  # 真實的像素顏色 (Ground Truth)
            # 計算光線的最近和最遠渲染邊界
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            # 執行 NeuS 的體渲染過程
            render_out = self.renderer.render(rays_o,
                                              rays_d,
                                              near,
                                              far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            # 從渲染結果中提取各項數值
            color = render_out['color']  # 渲染出的顏色
            s_val = render_out['s_val']  # NeuS 的退火參數
            cdf = render_out['cdf']  # 沿光線的權重累積分布
            gradient_error = render_out['gradient_error']  # Eikonal 損失
            weight_max = render_out['weight_max']  # 權重最大值
            dist_loss = render_out['dist_loss']  # 失真損失

            # --- C. 計算總損失函數 ---
            mask = torch.ones_like(color[:, :1])
            mask_sum = mask.sum() + 1e-5  # 加上一個極小值避免除以零

            # 顏色損失 (L1 Loss)，也是 PSNR 的基礎
            color_loss = F.l1_loss((color - true_rgb) * mask, torch.zeros_like(color), reduction='sum') / mask_sum
            # 計算 PSNR (峰值信噪比)
            psnr = 20.0 * torch.log10(1.0 / torch.sqrt(((color - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)))

            # Eikonal 損失，用於約束 SDF 梯度為 1，使幾何更平滑
            eikonal_loss = gradient_error

            # 加權總損失
            loss = color_loss * self.color_loss_weight + \
                   eikonal_loss * self.igr_weight + \
                   dist_loss * 0.001 + \
                   epipolar_loss * self.epipolar_loss_weight

            # --- D. 反向傳播與優化 ---
            self.optimizer.zero_grad()  # 清空場景網路的梯度
            self.optimizer_pose.zero_grad()  # 清空姿態網路的梯度
            loss.backward()  # 計算梯度
            self.optimizer.step()  # 更新場景網路的權重
            self.optimizer_pose.step()  # 更新姿態網路的權重

            self.iter_step += 1

            # --- E. 記錄日誌 ---
            # 記錄到 TensorBoard
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/dist_loss', dist_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', cdf[:, :1].mean(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', weight_max.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            self.writer.add_scalar('Statistics/inlier_rate', avg_inlier_rate, self.iter_step)
            self.writer.add_scalar('Loss/epipolar_loss', epipolar_loss, self.iter_step)

            if not self.use_porf:
                r_grad = torch.linalg.norm(
                    self.pose_param_net.r.grad).mean() if self.pose_param_net.r.grad is not None else 0
                t_grad = torch.linalg.norm(
                    self.pose_param_net.t.grad).mean() if self.pose_param_net.t.grad is not None else 0
                self.writer.add_scalar('Statistics/r_grad', r_grad, self.iter_step)
                self.writer.add_scalar('Statistics/t_grad', t_grad, self.iter_step)

            # 記錄到 Weights & Biases
            if WANDB_AVAILABLE:
                log_dict = {
                    'Loss/loss': loss.item(),
                    'Loss/color_loss': color_loss.item(),
                    'Loss/eikonal_loss': eikonal_loss.item(),
                    'Loss/dist_loss': dist_loss.item(),
                    'Statistics/s_val': s_val.mean().item(),
                    'Statistics/cdf': cdf[:, :1].mean().item(),
                    'Statistics/weight_max': weight_max.mean().item(),
                    'Statistics/psnr': psnr.item(),
                    'Statistics/inlier_rate': avg_inlier_rate.item(),
                    'Loss/epipolar_loss': epipolar_loss.item(),
                    'LearningRate/lr': self.optimizer.param_groups[0]['lr']
                }
                if not self.use_porf:
                    log_dict['Statistics/r_grad'] = r_grad.item()
                    log_dict['Statistics/t_grad'] = t_grad.item()

                wandb.log(log_dict, step=self.iter_step)

            # --- F. 定期報告與驗證 ---
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.val_freq == 0:
                self.validate_pose()
                self.validate_image()  # 在驗證姿態的同時，也產生一張驗證影像

            self.update_learning_rate()

        # --- 訓練結束後，儲存最終結果 ---
        self.save_checkpoint()
        self.validate_mesh()
        self.validate_image()

        if WANDB_AVAILABLE:
            wandb.finish()

    def update_image_index(self):
        self.img_idx = np.random.randint(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network': self.sdf_network.state_dict(),
            'variance_network': self.deviation_network.state_dict(),
            'render_network': self.render_network.state_dict(),
            'pose_param_net': self.pose_param_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        out_dir = os.path.join(self.base_exp_dir, 'pose_checkpoints')
        os.makedirs(out_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.dataset.gen_rays_at(idx, self.pose_param_net, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color'].detach().cpu().numpy())
            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)

        # 將渲染結果與真實影像上下拼接，便於比較
        val_dir = os.path.join(self.base_exp_dir, 'validations')
        os.makedirs(val_dir, exist_ok=True)

        gt_img = self.dataset.image_at(idx, resolution_level=resolution_level)
        composite_img = np.concatenate([img_fine, gt_img], axis=0)

        val_img_path = os.path.join(val_dir, '{:0>8d}_{}.png'.format(self.iter_step, idx))
        cv.imwrite(val_img_path, composite_img)

        # 將拼接後的驗證影像上傳到 wandb
        if WANDB_AVAILABLE:
            wandb.log({f"Validation/{idx}": wandb.Image(composite_img, caption=f"Iter: {self.iter_step}")},
                      step=self.iter_step)

    def validate_mesh(self, world_space=True, resolution=256, threshold=0.0):
        bound_min = self.dataset.object_bbox_min
        bound_max = self.dataset.object_bbox_max

        with torch.no_grad():
            vertices, triangles, normals, vertices2, triangles2 = \
                self.renderer.extract_geometry(
                    bound_min, bound_max, resolution=resolution, threshold=threshold)
            os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices_w = vertices * \
                         self.dataset.scale_mats_np[0][0, 0] + \
                         self.dataset.scale_mats_np[0][:3, 3][None]

            vertices_w2 = vertices2 * \
                          self.dataset.scale_mats_np[0][0, 0] + \
                          self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices_w, triangles)
        mesh_path = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step))
        mesh.export(mesh_path)

        mesh2 = trimesh.Trimesh(vertices_w2, triangles2)
        mesh2_path = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}2.ply'.format(self.iter_step))
        mesh2.export(mesh2_path)

        # 將產生的 3D 模型 (.ply) 上傳到 wandb
        if WANDB_AVAILABLE:
            wandb.log({"Mesh/geometry": wandb.Object3D(mesh_path)}, step=self.iter_step)
            wandb.log({"Mesh/geometry2": wandb.Object3D(mesh2_path)}, step=self.iter_step)

        # ... (其餘的 mesh 著色與匯出邏輯保持不變) ...

        logging.info('End validate_mesh')

    def validate_pose(self, initial_pose=False):
        pose_dir = os.path.join(
            self.base_exp_dir, 'poses_{:06d}'.format(self.iter_step))
        os.makedirs(pose_dir, exist_ok=True)

        scale_mat = self.dataset.object_scale_mat

        pred_poses = []
        for idx in range(self.dataset.n_images):
            if initial_pose:
                p = self.pose_param_net.get_init_pose(idx)
            else:
                p = self.pose_param_net(idx)
            p = p.detach().cpu().numpy()
            # scale and transform
            t = scale_mat @ p[:, 3].T
            p = np.concatenate([p[:, :3], t[:, None]], axis=1)
            pred_poses.append(p)
        pred_poses = np.stack(pred_poses)

        np.savetxt(os.path.join(pose_dir, 'refined_pose.txt'),
                   pred_poses.reshape(-1, 16),
                   fmt='%.8f', delimiter=' ')

        gt_poses = self.dataset.get_gt_pose()  # np, [n44]

        # 如果資料集中沒有提供真實姿態 (Ground Truth)，則跳過姿態驗證
        if gt_poses is None:
            return

        pred_poses = utils.pose_alignment(pred_poses, gt_poses)

        # ate
        ate_rots, ate_trans = utils.compute_ATE(gt_poses, pred_poses)
        ate_errs = np.stack([ate_rots, ate_trans], axis=-1)
        ate_errs = np.concatenate([ate_errs, np.mean(ate_errs, axis=0).reshape(-1, 2)], axis=0)

        ate_rot_deg = np.mean(ate_errs[:-1, 0]) / np.pi * 180
        ate_trans_mean = np.mean(ate_errs[:-1, 1])

        self.writer.add_scalar('Val/ate_rot_deg', ate_rot_deg, self.iter_step)
        self.writer.add_scalar('Val/ate_trans', ate_trans_mean, self.iter_step)

        # 將姿態驗證結果 (ATE) 上傳到 wandb
        if WANDB_AVAILABLE:
            wandb.log({
                'Val/ate_rot_deg': ate_rot_deg,
                'Val/ate_trans': ate_trans_mean
            }, step=self.iter_step)


def train(case_name):
    print(torch.__config__)
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/dtu_sift_porf.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default=case_name)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = PoseRunner(args.conf, args.mode, args.case)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_pose':
        runner.validate_pose()


if __name__ == "__main__":
    # 這只是一個 trimesh 的使用範例，在實際訓練時不會被執行
    try:
        vertices = np.random.rand(100, 3)
        triangles = np.random.randint(0, 100, size=(100, 3))
        vertex_colors = np.random.randint(0, 255, size=(100, 3))
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=vertex_colors)
        os.makedirs("./meshes_test", exist_ok=True)
        mesh.export("./meshes_test/test.ply")
    except Exception as e:
        print(f"Trimesh 測試失敗 (這對訓練不是關鍵問題): {e}")

