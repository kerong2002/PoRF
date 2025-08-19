# 效 3D 重建與姿態優化專案 README

本專案旨在實現一個從影像到高品質 3D 模型的自動化管線。其核心是結合 **GloMAP** 的高速運動恢復結構 (Structure from Motion, SfM) 能力與 **NeuS** 神經渲染模型的精細重建品質，並整合了先進的實驗追蹤與效能評測工具。

---

## 核心功能與增強部分
相較於基礎版本，我們引入了以下幾項關鍵增強，以提升效率、可比較性與最終重建品質 (PSNR)：

### 1. 雙重建管線與效能評測
為了能客觀評估 GloMAP 帶來的效率提升，我們建立了兩套平行且可比較的 SfM 管線：

- **glomap_wrapper.py (推薦)**: 採用 COLMAP 進行特徵提取與匹配，但使用速度更快的 GloMAP 進行核心的三維重建。  
- **colmap_wrapper.py**: 使用純 COLMAP 流程，作為效能比較的基準。  

✨ **新增功能**:
- **自動計時**: 兩套管線都內建了詳細的計時功能。當您透過 GUI (`surface.py`) 執行重建時，程式會自動在終端機和彈出視窗中報告總耗時，讓您可以直觀地比較兩者的速度差異。

---

### 2. Weights & Biases (WandB) 雲端實驗追蹤
為了更專業、更系統化地管理與分析實驗結果，我們在 `train.py` 中深度整合了 **Weights & Biases**。

✨ **新增功能**:
- **自動日誌**: 訓練過程中的所有關鍵指標 (Loss, PSNR, ATE 等) 都會自動上傳到您的 wandb 雲端儀表板。  
- **視覺化結果**:
  - **驗證影像**: 定期將渲染出的影像與真實影像拼接後上傳，讓您能直觀地看到模型進步的過程。  
  - **3D 模型**: 定期將重建出的 `.ply` 網格模型上傳，您可以在網頁上直接進行 3D 互動預覽。  
- **實驗比較**: wandb 強大的儀表板讓您可以輕鬆比較不同設定（例如，使用 GloMAP vs. COLMAP，或使用不同超參數）下的訓練曲線與最終成果。  

---

### 3. 高 PSNR 增強策略
這是讓 GloMAP 流程最終品質超越 COLMAP 的最關鍵強化。我們透過修改神經網路架構和訓練策略，使其能夠在 GloMAP 提供的良好幾何基礎上，學習到前所未有的細節。

✨ **新增功能**:
- **增強版設定檔 (`confs/dtu_sift_porf_enhanced.conf`)**:
  - 增加模型容量: 加深並加寬了 SDF 和渲染網路，使其能學習更複雜的細節。  
  - 增加光線採樣數: 大幅提升了渲染時沿光線的採樣點數量，直接提升渲染精度。  
  - 延長訓練時間: 給予模型更充分的時間來收斂到一個高品質的解。  

- **由粗到精的位置編碼 (Coarse-to-Fine Positional Encoding)**:  
  - **原理**: 訓練初期，模型只學習場景的低頻輪廓（像一張模糊的草圖），建立穩定的幾何基礎。隨著訓練進行 (`anneal_end` 參數控制)，逐漸解鎖高頻細節，讓模型在穩定的基礎上進行精雕細琢。  
  - **效果**: 雖然 PSNR 在訓練初期成長較慢，但在中期會出現拐點並加速攀升，最終達到遠高於傳統訓練方法的峰值，從而讓 GloMAP 流程在速度和品質上實現雙重超越。  


## 如何執行

### 1. 啟動應用程式

整個流程可以透過一個簡單的圖形化介面 (GUI) 啟動。

```bash
python surface.py
```

### 2. 監控訓練過程

您可以使用兩種工具來視覺化和監控訓練的進度：

#### TensorBoard (本地監控)

用於查看損失函數曲線、PSNR 等指標。

```bash
tensorboard --logdir exp_dtu\scan24\dtu_sift_porf\pose_logs
```
*請將 `exp_dtu\scan24\dtu_sift_porf` 替換為您實際的實驗輸出路徑。*

#### Weights & Biases (雲端儀表板)

提供更強大的實驗追蹤、比較與視覺化功能。您可以在訓練開始後，從終端機的輸出中找到對應的 `wandb` 專案連結，並在瀏覽器中打開。

-   **優點**:
    -   雲端儲存，隨時隨地存取。
    -   可輕鬆比較不同實驗的結果。
    -   除了指標曲線，還可以視覺化 3D 點雲、渲染影像等。
---

## 4. 核心修改部分詳解

### 4.1. `embedder.py`：策略的最終執行者
最關鍵的邏輯發生在 **PositionalEncoder** 類別中，我們修改了其 `forward` 方法，使其能根據訓練進度動態地遮蔽高頻的位置編碼。

- **新增 progress 參數**：`forward` 函式新增了一個 `progress` 參數（值域 0.0 到 1.0），用以接收當前的訓練進度。
- **動態頻率遮罩 (Dynamic Frequency Masking)**：
  - 計算 Alpha 值：`alpha = self.kwargs['num_freqs'] * progress`
  - 計算權重：隨訓練進度逐步開啟高頻細節。
  - 應用權重：將權重乘上 sin/cos 編碼值以控制高頻資訊。

程式碼示例：
```python
class PositionalEncoder(nn.Module):
    def forward(self, inputs, progress=1.0):
        alpha = self.kwargs['num_freqs'] * progress
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(self.embed_fns[0](inputs))
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, len(self.embed_fns), 2):
            freq_idx = (i - start_idx) // 2
            alpha_minus_freq = torch.tensor(alpha - freq_idx, device=inputs.device)
            one_tensor = torch.tensor(1.0, device=inputs.device)
            weight = torch.clamp(torch.min(alpha_minus_freq, one_tensor), 0.0)
            sin_val = self.embed_fns[i](inputs)
            cos_val = self.embed_fns[i + 1](inputs)
            outputs.append(weight * sin_val)
            outputs.append(weight * cos_val)

        return torch.cat(outputs, -1)
```

---

### 4.2. `fields.py`：參數的中間傳遞者
**SDFNetwork** 負責學習場景幾何，我們修改它作為 `progress` 參數的傳遞橋樑。

- 在 `forward` 與 `sdf` 方法中新增 `progress` 參數。
- 呼叫位置編碼器時傳遞進 `progress`，確保整體遵循由粗到精的學習策略。

程式碼示例：
```python
class SDFNetwork(nn.Module):
    def forward(self, inputs, progress=1.0):
        inputs_scaled = inputs * self.scale
        if self.embed_fn_fine is not None:
            embedded_inputs = self.embed_fn_fine(inputs_scaled, progress)
        else:
            embedded_inputs = inputs_scaled
        x = embedded_inputs
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, progress=1.0):
        return self.forward(x, progress)[:, :1]
```

---

### 4.3. `renderer.py`：策略的發起者
**NeuSRenderer** 作為渲染流程的總指揮，新增了 `cos_anneal_ratio` 參數來驅動整個 coarse-to-fine 機制。

- 在 `render` 與 `render_core` 方法中新增 `cos_anneal_ratio`。
- 將其作為 `progress` 傳遞給 `SDFNetwork`。

程式碼示例：
```python
class NeuSRenderer:
    def render_core(self, rays_o, rays_d, z_vals, sample_dist, near, far,
                    background_rgb=None, cos_anneal_ratio=0.0):
        progress = cos_anneal_ratio
        sdf_nn_output = self.sdf_network(pts, progress=progress)
        return { ... }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1,
               background_rgb=None, cos_anneal_ratio=0.0):
        sdf = self.sdf_network.sdf(pts.reshape(-1, 3), progress=cos_anneal_ratio)
        render_core_out = self.render_core(rays_o, rays_d, z_vals, sample_dist, near, far,
                                           background_rgb=background_rgb,
                                           cos_anneal_ratio=cos_anneal_ratio)
        return { ... }
```

---

### 4.4. 由設定檔驅動的增強
對應的 `dtu_sift_porf_high_psnr.conf` 設定檔調整：

- 增加模型容量 (`n_layers`, `d_hidden`)
- 增加光線採樣數 (`n_samples`, `n_importance`)
- 啟用高頻位置編碼 (`multires = 10`)

設定檔示例：
```conf
model {
    sdf_network {
        d_hidden = 256
        n_layers = 8
        multires = 10
    }
    render_network {
        d_hidden = 256
        n_layers = 4
    }
    neus_renderer {
        n_samples = 128
        n_importance = 128
    }
}
```

---

### 4.5. 總結
本次改動實現了一個動態的「由粗到精」學習機制，並透過設定檔提升了模型容量與渲染精度。兩者相輔相成，使模型能在訓練初期建立穩定幾何，後期專注於學習細節，最終在速度與品質上雙重超越。