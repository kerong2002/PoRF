import torch
import torch.nn.functional as F
import numpy as np
import mcubes
from models.embedder import get_embedder


def eff_distloss(w, t_vals, t_near, t_far):
    '''
    Efficient O(N) realization of distortion loss.
    There are B rays each with N sampled points.
    w:        Float tensor in shape [B,N]. Volume rendering weights of each point.
    m:        Float tensor in shape [B,N]. Midpoint distance to camera of each point.
    interval: Scalar or float tensor in shape [B,N]. The query interval of each point.
    '''
    s_vals = t_vals.clone().detach()

    t_near = t_near.expand_as(t_vals)
    t_far = t_far.expand_as(t_vals)

    mask = t_vals < t_far
    s_vals[mask] = ((t_vals[mask] - t_near[mask]) /
                    (t_far[mask] - t_near[mask]) * 0.5).clamp(0, 0.5)
    s_vals[~mask] = (1.0 - 0.5 / t_vals[~mask]).clamp(0.5, 1)
    s_vals = s_vals.clamp(0, 1)

    interval = s_vals[:, 1:] - s_vals[:, :-1]
    interval = torch.cat([interval, interval[:, -1, None]], dim=-1)
    m = s_vals + interval * 0.5

    loss_uni = (1 / 3) * (interval * w.pow(2)).sum(dim=-1).mean()
    wm = (w * m)
    w_cumsum = w.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(
                        len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N +
                                                        len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    vertices, triangles = mcubes.marching_cubes(u, threshold)
    from skimage import measure
    # 使用 scikit-image 的 marching_cubes 來同時獲取法向量
    vertices2, triangles2, normals, _ = measure.marching_cubes(u, level=threshold)

    b_max_np = bound_max
    b_min_np = bound_min

    vertices = vertices / (resolution - 1.0) * \
               (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    # scikit-image 的頂點座標也需要轉換到世界座標系
    vertices2 = vertices2 / (resolution - 1.0) * \
                (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

    return vertices, triangles, normals, vertices2, triangles2


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 render_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.render_network = render_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).to(rays_o.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False, progress=1.0):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            # ★ 修改：在重新計算 SDF 時傳入 progress
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), progress=progress).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    near,
                    far,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):

        # ★ 新增：將 cos_anneal_ratio 作為訓練進度 progress
        # 這個值會從 0.0 平滑增長到 1.0
        progress = cos_anneal_ratio

        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist]).to(rays_o.device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 2.0).float().detach()

        # ★ 修改：將 progress 傳遞給 SDF 網路
        sdf_nn_output = self.sdf_network(pts, progress=progress)
        sdf = sdf_nn_output[:, :1]
        sdf_features = sdf_nn_output[:, 1:]

        gradients = self.sdf_network.gradient(pts).squeeze()
        inv_s = self.deviation_network(torch.zeros([1, 3]).to(rays_o.device))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        sampled_color = self.render_network(pts, gradients, dirs, sdf_features).reshape(batch_size, n_samples, 3)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)

        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        dist_loss = eff_distloss(weights, mid_z_vals, near, far)

        return {
            'color': color,
            'pts': pts.reshape(batch_size, n_samples, 3),
            'sdf': sdf,
            'dists': dists,
            'dist_loss': dist_loss,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, -1),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside).to(rays_o.device)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]).to(rays_o.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]]).to(rays_o.device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                # ★ 修改：傳入 progress
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), progress=cos_anneal_ratio).reshape(batch_size,
                                                                                                  self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    # ★ 修改：傳入 progress
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf,
                                                  last=(i + 1 == self.up_sample_steps), progress=cos_anneal_ratio)

            n_samples = self.n_samples + self.n_importance

        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            z_vals = z_vals_feed
            n_samples = n_samples + self.n_outside

        render_core_out = self.render_core(rays_o, rays_d, z_vals, sample_dist, near, far,
                                           background_rgb=background_rgb,
                                           cos_anneal_ratio=cos_anneal_ratio)

        color = render_core_out['color']
        weights = render_core_out['weights']
        pts = render_core_out['pts']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = render_core_out['gradients']
        s_val = render_core_out['s_val'].reshape(batch_size, n_samples)
        surface_pts = torch.sum(weights[:, :, None] * pts, dim=1)

        return {
            'color': color,
            'surface_pts': surface_pts,
            'pts': pts,
            'dist_loss': render_core_out['dist_loss'],
            's_val': s_val.mean(dim=-1, keepdim=True),
            'cdf': render_core_out['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': render_core_out['gradient_error'],
            'inside_sphere': render_core_out['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        # 在提取幾何時，我們使用完整的 positional encoding (progress=1.0)
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts, progress=1.0))
