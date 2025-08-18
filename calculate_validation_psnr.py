import os
import cv2
import numpy as np
import glob
import argparse


def calculate_psnr_from_split_image(image_path):
    """
    讀取一張上下拼接的圖片，將其分割並計算 PSNR。

    Args:
        image_path (str): 拼接圖片的檔案路徑。

    Returns:
        float: 計算出的 PSNR 值，如果圖片無法讀取或格式錯誤則返回 None。
    """
    # 讀取圖片
    composite_img = cv2.imread(image_path)

    if composite_img is None:
        print(f"警告：無法讀取圖片 {image_path}")
        return None

    # 獲取圖片高度和寬度
    height, width, _ = composite_img.shape

    # 確保圖片高度為偶數，以方便分割
    if height % 2 != 0:
        # 如果高度是奇數，忽略最下面的一行像素以確保能平均分割
        composite_img = composite_img[:-1, :, :]
        height = composite_img.shape[0]
        print(f"注意：圖片 {os.path.basename(image_path)} 高度為奇數，已忽略最後一行像素進行計算。")

    # 計算垂直中間點
    mid_point = height // 2

    # 分割圖片
    # 上半部：產生的圖片 (Generated)
    generated_img = composite_img[0:mid_point, :]

    # 下半部：原始的真實圖片 (Ground Truth)
    ground_truth_img = composite_img[mid_point:height, :]

    # 確保分割後的兩張圖片大小完全相同
    if generated_img.shape != ground_truth_img.shape:
        print(
            f"錯誤：分割後的圖片 '{os.path.basename(image_path)}' 尺寸不匹配。上半部: {generated_img.shape}, 下半部: {ground_truth_img.shape}")
        return None

    # 使用 OpenCV 的內建函數計算 PSNR
    # cv2.PSNR(原始圖片, 失真/產生圖片)
    psnr = cv2.PSNR(ground_truth_img, generated_img)

    return psnr


def main(args):
    """
    主執行函數
    """
    # --- 設定路徑 ---
    # 因為您是在 porf 資料夾執行，所以我們使用相對路徑
    # 使用者可以透過 --scan 和 --model 參數來自訂
    validations_dir = os.path.join("exp_dtu", args.scan, args.model, "validations")

    print(f"正在搜尋目標資料夾: {os.path.abspath(validations_dir)}")

    if not os.path.isdir(validations_dir):
        print("錯誤：找不到指定的 validations 資料夾。請確認路徑是否正確，以及您是否在 'porf' 資料夾中執行此程式。")
        print(f"預期路徑: {os.path.abspath(validations_dir)}")
        return

    # --- 尋找所有 .png 和 .jpg 圖片 ---
    # 使用 glob 可以輕鬆匹配多種副檔名
    image_paths = glob.glob(os.path.join(validations_dir, '*.png')) + \
                  glob.glob(os.path.join(validations_dir, '*.jpg'))

    if not image_paths:
        print("錯誤：在指定的資料夾中找不到任何 .png 或 .jpg 圖片。")
        return

    psnr_values = []

    print("\n--- 開始計算每張圖片的 PSNR ---")
    for img_path in sorted(image_paths):
        psnr = calculate_psnr_from_split_image(img_path)

        if psnr is not None:
            filename = os.path.basename(img_path)
            print(f"檔案: {filename:<25} | PSNR: {psnr:.4f} dB")
            psnr_values.append(psnr)

    # --- 計算並顯示平均 PSNR ---
    if psnr_values:
        average_psnr = np.mean(psnr_values)
        print("\n-------------------------------------")
        print(f"處理圖片總數: {len(psnr_values)}")
        print(f"平均 PSNR: {average_psnr:.4f} dB")
        print("-------------------------------------")


if __name__ == "__main__":
    # 設定命令列參數，使其更具彈性
    parser = argparse.ArgumentParser(description="計算 porf 專案中 validation 圖片的 PSNR。")
    parser.add_argument('--scan', type=str, default='scan24', help='要處理的 scan 資料夾名稱。')
    parser.add_argument('--model', type=str, default='dtu_sift_porf', help='模型實驗的資料夾名稱。')

    args = parser.parse_args()
    main(args)
