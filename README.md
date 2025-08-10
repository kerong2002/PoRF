# PoRF 專案除錯與優化日誌

本文件記錄了對 PoRF (Pose-Free Radiance Fields) 專案進行的一系列除錯與程式碼優化過程。

## 專案目標

原始專案在執行時遇到設定檔解析錯誤、輸出日誌混亂以及在提取表面網格 (mesh) 時崩潰等問題。本次修改的目標是：
1.  修復所有導致程式中斷的錯誤。
2.  整理與優化日誌輸出，使終端機介面乾淨，並將詳細資訊導入日誌檔案。
3.  為主要腳本加上中文註解，提高程式碼的可讀性與可維護性。

## 修改摘要

### 1. COLMAP 包裝腳本 (`colmap_wrapper.py`)
- **問題**: 原始腳本在 `output.txt` 中沒有輸出，且難以獨立測試。
- **修改**:
    - 加入 `if __name__ == "__main__"` 區塊，使其可以作為獨立腳本執行，方便單獨測試 COLMAP 流程。
    - 增強了錯誤處理，確保無論成功或失敗，COLMAP 的詳細輸出都會被重新導向到 `colmap_output.txt`。
    - 增加了 `colmap_executable` 參數，允許使用者明確指定 COLMAP 執行檔的路徑。
    - 補全了中文註解。

### 2. 位姿處理工具 (`pose_utils.py`)
- **問題**: 執行時會在終端機打印大量陣列和檔案列表，造成資訊混亂。
- **修改**:
    - 引入 Python 的 `logging` 模組。
    - 將所有詳細的 `print` 輸出替換為 `logging.info()`，並將其導向到位於各個 `scan` 資料夾下的 `pose_utils.log` 檔案中。
    - 只保留了關鍵的進度訊息在終端機上顯示。
    - 補全了詳盡的中文註解，解釋了從 COLMAP 讀取資料、計算可見性、儲存位姿等關鍵步驟。

### 3. HOCON 設定檔 (`confs/dtu_sift_porf.conf`)
- **問題**: 使用了不符合 HOCON 語法規範的分號 (`;`) 作為註解，導致 `pyparsing.exceptions.ParseSyntaxException` 錯誤。
- **修改**:
    - 將錯誤的分號註解 ` ; ` 修改為正確的井號註解 ` # `。

### 4. 主訓練腳本 (`train.py`)
- **問題**:
    1.  會直接 `print` 整個設定檔內容到終端機。
    2.  在訓練後期呼叫 `validate_mesh` 時，因 `marching_cubes` 錯誤而崩潰。
- **修改**:
    - **日誌優化**: 移除了 `print(conf_text)`，改為使用 `logging` 模組將設定檔內容寫入實驗目錄下的 `train.log`。
    - **崩潰修復**: 透過修改 `models/renderer.py` 間接解決了此問題（見下一節）。
    - **中文註解**: 為 `PoseRunner` 類別的初始化、訓練迴圈、驗證等主要邏輯區塊加上了中文註解。

### 5. 渲染器 (`models/renderer.py`)
- **問題**: 在 `extract_geometry` 函式中，當 SDF 網路收斂不佳時，呼叫 `skimage.measure.marching_cubes` 會因 `ValueError: Surface level must be within volume data range.` 而使整個程式崩潰。
- **修改**:
    - 在呼叫 `marching_cubes` 之前，增加了保護機制。
    - 新增程式碼以檢查 SDF 體積的最大值和最小值，判斷給定的 `threshold` 是否在此範圍內。
    - 如果不在範圍內，則記錄一條警告日誌並跳過此次的網格提取，回傳空值，從而避免程式崩潰。

### 6. 資料集處理 (`models/dataset.py`) & GUI 介面 (`surface.py`)
- **問題**: 仍有部分 `print` 語句會輸出到終端機。
- **修改**:
    - 將這兩個檔案中殘餘的 `print` 語句全部替換為 `logging.info()`，確保所有日誌都得到統一管理。

## 目前狀態

經過上述修改，專案現在應該能夠：
- **順利執行**：不會再因為設定檔或 Marching Cubes 錯誤而中斷。
- **日誌清晰**：終端機只會顯示簡潔的進度，所有詳細資訊都分門別類地儲存在對應的 `.log` 檔案中。
- **易於理解**：主要程式碼都附有中文註解。

您可以再次執行 `surface.py` 來啟動完整的訓練流程。
