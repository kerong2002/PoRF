# 3D 重建與姿態優化專案

本專案旨在實現一個高效的 3D 重建與相機姿態優化管線。核心是利用 `GloMAP` 的高速全域最佳化能力來改善傳統 `COLMAP` 的 SfM (Structure from Motion) 流程，並結合一個基於 NeRF (神經輻射場) 的模型 (`train.py`) 來進行場景重建與相機姿態的聯合優化。

---

## 核心流程：GloMAP 增強的 SfM 管線

### 概述

為了提升大規模場景下 SfM 的效率與穩健性，本專案採用 `GloMAP` 作為核心的稀疏重建引擎，取代了 `COLMAP` 中較為耗時的 `mapper` 步驟。整個流程由 `glomap_wrapper.py` 進行封裝與自動化。

### 執行步驟詳解

1.  **特徵提取與匹配 (COLMAP)**:
    -   `colmap feature_extractor`: 首先，利用 COLMAP 強大的特徵提取功能，處理所有輸入影像。
    -   `colmap exhaustive_matcher`: 接著，進行窮舉匹配，找出影像之間的特徵對應關係。所有特徵和匹配資訊都會儲存在 `database.db` 檔案中。

2.  **稀疏重建 (GloMAP)**:
    -   `glomap mapper`: `GloMAP` 讀取由 COLMAP 產生的 `database.db`，並利用其高效的全域最佳化演算法，快速完成相機姿態估計和稀疏點雲重建。這一步顯著加速了整個 SfM 過程。

3.  **為後續步驟產生輸出**:
    -   重建完成後，腳本會自動從資料庫中提取所需的匹配資訊，並產生 `poses.npy` 等檔案，以供姿態優化網路使用。

---

## 關鍵腳本說明

-   `glomap_wrapper.py`: 自動化 SfM 流程的總控制器。它負責依序呼叫 `COLMAP` 和 `GloMAP`，並處理它們之間的資料傳遞。
-   `train.py`: 姿態優化與場景重建的核心。它會載入由 SfM 產生的初始相機姿態，並在一個端到端的框架中，同時優化相機姿態 (Pose) 和場景的幾何表示 (NeRF)。
-   `utils.py`: 包含專案所需的各種輔助函式，例如姿態計算、資料處理等。

---

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
