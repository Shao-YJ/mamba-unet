import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

from matplotlib.font_manager import FontProperties

def center_crop_or_pad(image, target_h=224, target_w=224):
    """
    对 2D 图像进行中心裁剪。如果原图比目标尺寸小，则进行补零填充。
    """
    h, w = image.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        h, w = image.shape
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return image[start_y : start_y + target_h, start_x : start_x + target_w]

def generate_paper_grid_3x3(base_dir, cases, save_path="final_segmentation_grid_3x3.png"):
    """
    生成 3行 x 3列 的高清分割结果展示图
    """
    # 确保 SimHei.ttf 已下载到当前目录
    my_font = FontProperties(fname="SimHei.ttf", size=18) 
    
    colors = ['black', 'red', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 修改为 3x3 画布
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), dpi=300)
    
    # 横轴说明文字 (共3列)
    col_titles = ["原始图像", "真实标注", "分割结果"]

    def get_middle_slice_cropped(case_name):
        img_path = os.path.join(base_dir, f"{case_name}_img.nii.gz")
        gt_path = os.path.join(base_dir, f"{case_name}_gt.nii.gz")
        pred_path = os.path.join(base_dir, f"{case_name}_pred.nii.gz")
        
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            z = img.shape[0] // 2 
            
            img_slice = center_crop_or_pad(img[z], 224, 224)
            gt_slice = center_crop_or_pad(gt[z], 224, 224)
            pred_slice = center_crop_or_pad(pred[z], 224, 224)
            return img_slice, gt_slice, pred_slice
        except Exception as e:
            print(f"读取 {case_name} 失败: {e}")
            empty = np.zeros((224, 224))
            return empty, empty, empty

    # 循环 3 行
    for row in range(3):
        case_name = cases[row]
        img, gt, pred = get_middle_slice_cropped(case_name)
        
        # 每一行包含 3 张图的数据
        plot_data = [
            (img, None),  # 原始图像
            (img, gt),    # 真实标注
            (img, pred)   # 分割结果
        ]
        
        for col in range(3):
            ax = axes[row, col]
            bg_img, mask_img = plot_data[col]
            
            ax.imshow(bg_img, cmap='gray')
            
            if mask_img is not None:
                mask_masked = np.ma.masked_where(mask_img == 0, mask_img)
                ax.imshow(mask_masked, cmap=cmap, norm=norm, alpha=0.5)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 在最后一行显示底部标签
            if row == 2:
                ax.set_xlabel(col_titles[col], fontproperties=my_font, fontweight='bold', labelpad=15)
                
            # 在第一列显示患者编号
            # if col == 0:
            #     ax.set_ylabel(f"患者 {row+1}", fontproperties=my_font, fontweight='bold', labelpad=15)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 3x3 对比图已生成：{save_path}")

if __name__ == "__main__":
    # ⚠️ 配置区：请根据你的实际情况修改以下路径和文件名
    
    # 预测结果所在的文件夹（确保里面有 _img.nii.gz, _gt.nii.gz, _pred.nii.gz）
    RESULT_DIR = "/root/autodl-tmp/Mamba-UNet-main/model/ACDC/Semi_Mamba_UNet_L7_7/mambaunet/mambaunet_best_model1_predictions/"
    
    # 选取 6 个患者，每个患者选取 2 个时相 (Frame)
    # ACDC 数据集通常一个病人有两个帧 (如 ED 舒张末期, ES 收缩末期)
    # PATIENT_CASES = [
    #     ("patient001_frame01", "patient001_frame02"), # Patient 1
    #     ("patient008_frame01", "patient008_frame02"), # Patient 2
    #     ("patient052_frame01", "patient052_frame02"), # Patient 3
    #     ("patient059_frame01", "patient059_frame02"), # Patient 4
    #     ("patient080_frame01", "patient080_frame02"), # Patient 5
    #     ("patient093_frame01", "patient093_frame02")  # Patient 6
    # ]
    
    PATIENT_CASES = [
        "patient001_frame01", 
        "patient052_frame01", 
        "patient093_frame01",
    ]
    
    generate_paper_grid_3x3(RESULT_DIR, PATIENT_CASES, save_path="paper_figure_3x3.png")