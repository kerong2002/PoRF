# 3D 重建流程優化專案

本專案旨在實現一個高效且高品質的 3D 重建管線，透過深度整合 `hloc` (SuperPoint+SuperGlue) 與 `GloMAP`，為 `porf` 專案提供最優質的相機姿態估計。

---

## 最新流程 (2025-08-13)：hloc + GloMAP 混合式 SfM 管線

### 核心更新：結合 hloc 的精度與 GloMAP 的速度

為了追求極致的重建品質與效率，我們設計了一套全新的混合式 Structure from Motion (SfM) 管線。此管線取代了原有的 COLMAP 特徵提取與匹配步驟，並保留了 GloMAP 的高速全域最佳化能力。

這個策略旨在結合兩者的最強優勢：

-   **hloc (Hierarchical Localization)**:
    -   **特徵提取 (SuperPoint)**: 利用基於深度學習的 SuperPoint 模型，能夠在各種具挑戰性的場景中（如弱紋理、光照變化、重複圖案）偵測到極其穩健的特徵點。
    -   **特徵匹配 (SuperGlue)**: 透過圖神經網路 (GNN) 進行特徵匹配，其效果遠優於傳統的最近鄰匹配，能大幅提升匹配的準確性與數量。

-   **GloMAP**:
    -   **稀疏重建**: 作為一個專為大規模場景設計的高速全域 SfM 解算器，GloMAP 能快速地根據高品質的匹配資料，完成相機姿態估計和稀疏點雲重建，有效避免了傳統 `colmap mapper` 的效能瓶頸。

### 新的混合管線詳解

新的核心邏輯被封裝在 `hloc_glomap_wrapper.py` 中，其執行流程如下：

1.  **影像準備**: 腳本會自動將 `images` 資料夾中的影像複製到 `image` 資料夾，以符合後續流程的格式要求。

2.  **hloc 特徵提取與匹配**:
    -   `pairs_from_exhaustive`: 首先，產生一個包含所有影像對的列表。
    -   `extract_features`: 呼叫 `hloc` 提取所有影像的 SuperPoint 特徵，並儲存為 `.h5` 檔案。
    -   `match_features`: 根據影像對列表，使用 SuperGlue 進行特徵匹配，並將結果儲存為 `.h5` 檔案。

3.  **資料庫橋接**:
    -   `colmap feature_extractor`: 我們巧妙地利用此指令來快速建立一個包含所有相機和影像資訊的 `database.db` 檔案。
    -   **清除 SIFT 特徵**: 為了避免特徵衝突，腳本會**自動清除**上一步產生的預設 SIFT 特徵。
    -   `import_features` & `import_matches`: 將 `hloc` 產生的 SuperPoint 特徵和 SuperGlue 匹配結果，匯入到這個乾淨的 `database.db` 中。這一步是連接 `hloc` 和 `GloMAP` 的關鍵橋樑。

4.  **GloMAP 稀疏重建**:
    -   `glomap mapper`: `GloMAP` 讀取這個包含了高品質匹配的資料庫，並快速執行全域最佳化，完成稀疏重建。重建結果（`cameras.bin`, `images.bin`, `points3D.bin`）會儲存在 `sparse/0` 資料夾中。

5.  **為 porf 產生輸出**:
    -   `export_colmap_matches`: 在重建完成後，腳本會自動呼叫此函式，從資料庫中提取匹配資訊，並產生 `porf` 所需的 `colmap_matches` 資料夾和 `.npz` 檔案。
    -   `save_poses`: 最後，[`pose_utils.py`](pose_utils.py:) 會讀取重建結果，並產生 `poses.npy` 和 `sparse_points.ply` 點雲檔案。

### 帶來的優勢

-   **極高的重建品質**：SuperPoint+SuperGlue 的組合在準確性和穩健性上均顯著優於傳統方法。
-   **維持高速重建**：繼續利用 GloMAP 的速度優勢，實現快速的全域最佳化。
-   **全自動化與無縫整合**：整個複雜的流程被完整封裝，使用者只需執行一個指令即可完成所有操作，並得到 `porf` 所需的全部檔案。

---

## 如何執行

整個 SfM 流程已經被整合到 `imgs2poses.py` 中。您可以透過執行 `surface.py` 或直接執行 `imgs2poses.py` 來啟動。

```bash
# 建議的執行方式 (透過 GUI)
python surface.py

# 或直接執行 (指定工作目錄)
python imgs2poses.py --work_dir path/to/your/dataset
```

-   `--work_dir`: 指向您的資料集路徑，該路徑下應包含一個名為 `images` 的資料夾。

執行完成後，所有中間檔案會儲存在 `hloc_glomap_outputs` 資料夾中，而最終的稀疏重建結果會像以前一樣，儲存在 `sparse` 資料夾內，以供 `porf` 的後續步驟使用。
