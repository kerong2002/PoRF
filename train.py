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

# torch.autograd.set_detect_anomaly(True)

class PoseRunner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME'):
        self.device = torch.device('cuda')

        # --- 組態設定 (Configuration) ---
        self.conf_path = conf_path
        with open(self.conf_path, 'r') as f:
            conf_text = f.read()
        
        # 將設定檔中的 CASE_NAME 替換為實際的案例名稱
        conf_text = conf_text.replace('CASE_NAME', case)
        
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        # --- 設定日誌 (Logging) ---
        log_file = os.path.join(self.base_exp_dir, 'train.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[
                                logging.FileHandler(log_file, mode='w'),
                                logging.StreamHandler() # 同時輸出到終端機
                            ])
        
        # 將載入的設定檔內容寫入日誌，而不是直接 print
        logging.info("--- 載入的 HOCON 設定檔 ---")
        logging.info(conf_text)
        logging.info("--------------------------")

        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # --- 訓練參數 (Training parameters) ---
        self.end_iter = self.conf['train.pose_end_iter']
        self.val_freq = self.conf.get_int('train.pose_val_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.pose_learning_rate = self.conf.get_float('train.pose_learning_rate')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # --- PoRF 參數 ---
        self.use_porf = self.conf.get_bool('train.use_porf')
        self.inlier_threshold = self.conf.get_float('train.inlier_threshold')
        self.num_pairs = self.conf.get_int('train.num_pairs')

        # --- 損失權重 (Loss Weights) ---
        self.color_loss_weight = self.conf.get_float('train.color_loss_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.epipolar_loss_weight = self.conf.get_float('train.epipolar_loss_weight')
        self.mode = mode

        # --- 初始化網路和優化器 (Networks and Optimizers) ---
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'pose_logs'))

        # 初始化 SDF, Variance, Rendering 網路
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.render_network = RenderingNetwork(**self.conf['model.render_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.render_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.render_network,
                                     **self.conf['model.neus_renderer'])

        # 初始化相機位姿優化網路 (PoRF 或 LearnPose)
        if self.use_porf:
            self.pose_param_net = PoRF(
                self.dataset.n_images,
                init_c2w=self.dataset.pose_all,
                scale=self.conf.get_float('train.scale')
            ).to(self.device)
        else:
            self.pose_param_net = LearnPose(
                self.dataset.n_images,
                init_c2w=self.dataset.pose_all
            ).to(self.device)

        self.optimizer_pose = torch.optim.Adam(self.pose_param_net.parameters(),
                                               lr=self.pose_learning_rate)

        # 驗證初始位姿
        if self.iter_step == 0:
            self.validate_pose(initial_pose=True)

    def train(self):
        """主訓練循環"""
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        for iter_i in tqdm(range(res_step), desc="Training"):

            self.update_image_index()

            # --- PoRF ---
            # 從資料集中取樣匹配點對，用於計算對極幾何損失
            intrinsic, pose, intrinsic_src_list, pose_src_list, match_list = self.dataset.sample_matches(self.img_idx,
                                                                                                         self.pose_param_net)

            P_src_list = []
            for cam, p in zip(intrinsic_src_list, pose_src_list):
                P_src_list.append(utils.compute_P_from_KT(cam, p))

            # 評估當前位姿的品質，計算對極損失
            avg_inlier_rate, epipolar_loss = utils.evaluate_pose(intrinsic,
                                                              pose,
                                                              P_src_list,
                                                              match_list,
                                                              self.num_pairs,
                                                              self.inlier_threshold)

            # --- NeuS ---
            # 在當前影像中隨機取樣光線
            data = self.dataset.gen_random_rays_at(self.img_idx,
                                                   self.batch_size,
                                                   pose
                                                   )

            rays_o, rays_d = data[:, :3], data[:, 3: 6]
            true_rgb = data[:, 6: 9]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            # 渲染光線，得到顏色、SDF值等
            render_out = self.renderer.render(rays_o,
                                              rays_d,
                                              near,
                                              far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            # --- 計算損失 (Loss Calculation) ---
            color = render_out['color']
            s_val = render_out['s_val']
            cdf = render_out['cdf']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            dist_loss = render_out['dist_loss']

            mask = torch.ones_like(color[:, :1])
            mask_sum = mask.sum()

            # 顏色損失 (L1)
            color_error = (color - true_rgb) * mask
            color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # Eikonal 損失，約束 SDF 梯度為 1
            eikonal_loss = gradient_error

            # 總損失
            loss = color_loss * self.color_loss_weight +\
                eikonal_loss * self.igr_weight +\
                dist_loss * 0.001 +\
                epipolar_loss * self.epipolar_loss_weight

            # --- 反向傳播與優化 ---
            self.optimizer.zero_grad()
            self.optimizer_pose.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.optimizer_pose.step()

            self.iter_step += 1

            # --- 寫入 Tensorboard ---
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

            # 檢查位姿梯度 (用於除錯)
            if not self.use_porf:
                r_grad_norms = torch.linalg.norm(self.pose_param_net.r.grad,
                                                 dim=-1,
                                                 keepdim=True).expand_as(self.pose_param_net.r.grad)

                t_grad_norms = torch.linalg.norm(self.pose_param_net.t.grad,
                                                 dim=-1,
                                                 keepdim=True).expand_as(self.pose_param_net.t.grad)
                r_grad = r_grad_norms[r_grad_norms > 0].mean()
                t_grad = t_grad_norms[t_grad_norms > 0].mean()

                self.writer.add_scalar('Statistics/r_grad', r_grad, self.iter_step)
                self.writer.add_scalar('Statistics/t_grad', t_grad, self.iter_step)

            # --- 定期報告與驗證 ---
            if self.iter_step % self.report_freq == 0:
                logging.info(f"Iter: {self.iter_step:8>d} Loss: {loss.item():.4f} LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.iter_step % self.val_freq == 0:
                self.validate_pose()

            self.update_learning_rate()

        self.save_checkpoint()
        self.validate_mesh()
        self.validate_image()

    def update_image_index(self):
        """隨機選擇一張圖片進行訓練"""
        self.img_idx = np.random.randint(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        """計算 Cosine Annealing 的比例"""
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        """根據 warm-up 和 cosine decay 更新學習率"""
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / \
                (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) +
                               1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def save_checkpoint(self):
        """儲存模型權重"""
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
        logging.info(f"Checkpoint saved to {out_dir}")

    def validate_image(self, idx=-1, resolution_level=-1):
        """渲染一張影像並與真實影像比較，用於驗證"""
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        logging.info(f'Validating image: iter={self.iter_step}, camera_idx={idx}')

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx,
                                                  self.pose_param_net,
                                                  resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb = []
        out_normal = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            with torch.no_grad():
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color'):
                out_rgb.append(render_out['color'].detach().cpu().numpy()[..., :3])
            if feasible('gradients') and feasible('weights'):
                n_samples = render_out['gradients'].shape[1]
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal.append(normals)
            del render_out

        img = None
        if len(out_rgb) > 0:
            img = (np.concatenate(out_rgb, axis=0).reshape(
                [H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal) > 0:
            normal_img = np.concatenate(out_normal, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]
                                    ).reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img.shape[-1]):
            if len(out_rgb) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def validate_mesh(self, world_space=True, resolution=256, threshold=0.0):
        """從 SDF 中提取 Mesh 表面"""
        bound_min = self.dataset.object_bbox_min
        bound_max = self.dataset.object_bbox_max
            
        with torch.no_grad():
            vertices, triangles, normals, vertices2, triangles2 =\
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
        logging.info(f"Mesh saved to {mesh_path}")
    
        mesh = trimesh.Trimesh(vertices_w2, triangles2)
        mesh_path_2 = os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}2.ply'.format(self.iter_step))
        mesh.export(mesh_path_2)
        logging.info(f"Mesh 2 saved to {mesh_path_2}")
        
        # --- 以下為著色部分 ---
        if vertices.shape == normals.shape:
            with torch.no_grad():
                pts = torch.tensor(vertices.copy()).to(torch.float32)
                sdf_nn_output = self.sdf_network(pts)
                sdf_features = sdf_nn_output[:, 1:]
            
            gradients = self.sdf_network.gradient(pts, True).squeeze()
            
            with torch.no_grad():
                normals *= -1
                sampled_color3 = self.render_network(pts,
                                            gradients,
                                            torch.tensor(normals.copy()).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()

                mesh = trimesh.Trimesh(vertices_w, triangles, vertex_colors=(sampled_color3[:, [2, 1, 0]]))
                mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                            '{:0>8d}4.ply'.format(self.iter_step)))
                
                sampled_color = self.render_network(pts,
                                            gradients,
                                            torch.tensor(np.random.rand(pts.shape[0], 3) * 2 - 1).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()
                
                mesh = trimesh.Trimesh(vertices_w, triangles, vertex_colors=(sampled_color[:, [2, 1, 0]]))
                mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                            '{:0>8d}2.ply'.format(self.iter_step)))
        
        else:
            with torch.no_grad():
                pts = torch.tensor(vertices2.copy()).to(torch.float32)
                sdf_nn_output = self.sdf_network(pts)
                sdf_features = sdf_nn_output[:, 1:]

            gradients = self.sdf_network.gradient(pts, True).squeeze()
            
            with torch.no_grad():
                normals *= -1
                sampled_color3 = self.render_network(pts,
                                            gradients,
                                            torch.tensor(normals.copy()).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()

                mesh = trimesh.Trimesh(vertices_w2, triangles2, vertex_colors=(sampled_color3[:, [2, 1, 0]]))
                mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                            '{:0>8d}4.ply'.format(self.iter_step)))
                
                sampled_color = self.render_network(pts,
                                            gradients,
                                            torch.tensor(np.random.rand(pts.shape[0], 3) * 2 - 1).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()
                
                mesh = trimesh.Trimesh(vertices_w2, triangles2, vertex_colors=(sampled_color[:, [2, 1, 0]]))
                mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                            '{:0>8d}2.ply'.format(self.iter_step)))
                
        normals2 = trimesh.Trimesh(vertices, triangles).vertex_normals
        if normals2.shape == vertices.shape:
            with torch.no_grad():
                sampled_color4 = self.render_network(pts,
                                            gradients,
                                            torch.tensor(normals2.copy()).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()
            
            mesh = trimesh.Trimesh(vertices_w, triangles, vertex_colors=(sampled_color4[:, [2, 1, 0]]))
            mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                        '{:0>8d}5.ply'.format(self.iter_step)))
            
        if vertices.shape != normals.shape:    
            with torch.no_grad():
                pts = torch.tensor(vertices.copy()).to(torch.float32)
                sdf_nn_output = self.sdf_network(pts)
                sdf_features = sdf_nn_output[:, 1:]
            
            gradients = self.sdf_network.gradient(pts, True).squeeze()
            
            with torch.no_grad():
                normals *= -1
                sampled_color3 = self.render_network(pts,
                                            gradients,
                                            torch.tensor(normals[:pts.shape[0]].copy()).to(torch.float32),
                                            sdf_features).detach().cpu().numpy()

                mesh = trimesh.Trimesh(vertices_w, triangles, vertex_colors=(sampled_color3[:, [2, 1, 0]]))
                mesh.export(os.path.join(self.base_exp_dir, 'meshes',
                            '{:0>8d}42.ply'.format(self.iter_step)))

        logging.info('Mesh validation finished.')

    def validate_pose(self, initial_pose=False):
        """驗證相機位姿，並計算 ATE (Absolute Trajectory Error)"""
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
            # 縮放與變換
            t = scale_mat @ p[:, 3].T
            p = np.concatenate([p[:, :3], t[:, None]], axis=1)
            pred_poses.append(p)
        pred_poses = np.stack(pred_poses)

        np.savetxt(os.path.join(pose_dir, 'refined_pose.txt'),
                   pred_poses.reshape(-1, 16),
                   fmt='%.8f', delimiter=' ')

        gt_poses = self.dataset.get_gt_pose()  # 真實位姿

        # 將預測位姿與真實位姿對齊
        pred_poses = utils.pose_alignment(pred_poses, gt_poses)

        # 計算 ATE (旋轉與平移誤差)
        ate_rots, ate_trans = utils.compute_ATE(gt_poses, pred_poses)
        ate_errs = np.stack([ate_rots, ate_trans], axis=-1)
        ate_errs = np.concatenate([ate_errs, np.mean(ate_errs, axis=0).reshape(-1, 2)], axis=0)

        self.writer.add_scalar('Val/ate_rot', np.mean(ate_errs, axis=0)[0] / np.pi * 180, self.iter_step)
        self.writer.add_scalar('Val/ate_trans', np.mean(ate_errs, axis=0)[1], self.iter_step)
        logging.info(f"Pose validation finished. ATE_rot: {np.mean(ate_errs, axis=0)[0] / np.pi * 180:.4f}, ATE_trans: {np.mean(ate_errs, axis=0)[1]:.4f}")


def train(case_name):
    """訓練流程的進入點"""
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    # 這個區塊似乎是用於測試 trimesh 功能，與主流程無關
    vertices = np.random.rand(100, 3)
    triangles = np.random.randint(0, 100, size=(100, 3))
    vertex_colors = list(np.array([[0.1 + i * 0.005, 0, 0.6 - i * 0.005] for i in range(100)]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=vertex_colors)
    mesh.export("D:/Desktop/project/exp_dtu/images14/dtu_sift_porf/meshes/test.ply")
