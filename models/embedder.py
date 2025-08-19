import torch
import torch.nn as nn
import numpy as np


def get_embedder(multires, input_dims=3):
    """
    工廠函數，根據給定的參數創建位置編碼器。
    multires: int, 最高頻率的 log2 值 (例如 10 -> 2^10)
    """
    embed_kwargs = {
        'include_input': True,  # 是否在最終輸出中包含原始輸入向量
        'input_dims': input_dims,  # 輸入向量的維度 (例如 3D 座標為 3)
        'max_freq_log2': multires - 1,  # 最高頻率的 log2 值
        'num_freqs': multires,  # 要使用的頻率總數
        'log_sampling': True,  # 是否在對數尺度上對頻率進行採樣
        'periodic_fns': [torch.sin, torch.cos],  # 用於編碼的週期函數
    }

    embedder_obj = PositionalEncoder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class PositionalEncoder(nn.Module):
    """
    位置編碼器模組，實現了 BARF 中提出的由粗到精的頻率遮罩策略。
    """

    def __init__(self, include_input, input_dims, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        super().__init__()
        self.kwargs = {
            'include_input': include_input,
            'input_dims': input_dims,
            'max_freq_log2': max_freq_log2,
            'num_freqs': num_freqs,
            'log_sampling': log_sampling,
            'periodic_fns': periodic_fns,
        }

        self.out_dim = 0
        self.embed_fns = []
        if include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += input_dims

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, steps=num_freqs)

        for freq in self.freq_bands:
            for p_fn in periodic_fns:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.out_dim += input_dims

    def forward(self, inputs, progress=1.0):
        """
        前向傳播函數，根據訓練進度對輸入進行編碼。
        inputs: torch.Tensor, 輸入的 3D 座標，形狀為 [N, 3]
        progress: float, 訓練進度，值從 0 到 1
        """
        # alpha 控制了頻率遮罩的平滑過渡，隨著 progress 從 0 增長到 num_freqs
        alpha = self.kwargs['num_freqs'] * progress

        outputs = []
        # 第一個 embed_fn 總是 x 本身 (如果 include_input 為 True)，直接加入
        if self.kwargs['include_input']:
            outputs.append(self.embed_fns[0](inputs))
            start_idx = 1
        else:
            start_idx = 0

        # 遍歷 sin 和 cos 函數對
        for i in range(start_idx, len(self.embed_fns), 2):
            # i 是 sin, i+1 是 cos
            # 計算當前頻率的索引
            freq_idx = (i - start_idx) // 2

            # --- ★ 錯誤修正 START ---
            # 計算權重 (BARF 中的 w_k)
            # 確保所有參與 torch.min 運算的都是 Tensor，並且在同一個 device 上
            alpha_minus_freq = torch.tensor(alpha - freq_idx, device=inputs.device)
            one_tensor = torch.tensor(1.0, device=inputs.device)

            weight = torch.clamp(torch.min(alpha_minus_freq, one_tensor), 0.0)
            # --- ★ 錯誤修正 END ---

            # 應用 sin 和 cos 函數
            sin_val = self.embed_fns[i](inputs)
            cos_val = self.embed_fns[i + 1](inputs)

            # 應用權重並加入到輸出列表
            outputs.append(weight * sin_val)
            outputs.append(weight * cos_val)

        return torch.cat(outputs, -1)
