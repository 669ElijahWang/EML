import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 读取.nii文件
nii_image_gt = nib.load(
    r'D:\Project\changshu_gaijin\Q-Net-main\data\CMR\cmr_MR_normalized\image_16.nii.gz')
nii_image_pred = nib.load(
    r'D:\Project\changshu_gaijin\Q-Net-main\runs\Ours_train_CMR_cv2\1\interm_preds\prediction_16_LV-MYO.nii.gz')

# 获取图像数据
image_data_gt = nii_image_gt.get_fdata()
image_data_pred = nii_image_pred.get_fdata()

# 获取切片数量
num_slices_gt = image_data_gt.shape[2]
num_slices_pred = image_data_pred.shape[2]

# 计算网格布局（行数和列数）
rows_gt = int(np.ceil(np.sqrt(num_slices_gt)))
cols_gt = int(np.ceil(num_slices_gt / rows_gt))

# 创建一个大图，用于包含所有的子图
fig, axes_gt = plt.subplots(rows_gt, cols_gt, figsize=(cols_gt * 4, rows_gt * 4))

# 选择一个彩色colormap
pred_cmap = plt.cm.Oranges_r

# 定义预测切片开始覆盖的起始位置
pred_start_slice = 1  # 第8张切片（0-based索引为7）

# 遍历所有的原始图像切片，并将它们添加到子图中
for i in range(num_slices_gt):
    row = i // cols_gt
    col = i % cols_gt
    ax = axes_gt[row, col]

    # 显示ground truth图像（灰度）
    ax.imshow(image_data_gt[:, :, i], cmap='gray')
    ax.set_title(f'Slice {i + 1}')
    ax.axis('off')

    # 计算对应的预测切片索引
    pred_slice_idx = i - pred_start_slice

    # 如果预测切片索引有效且在范围内，则添加覆盖
    if 0 <= pred_slice_idx < num_slices_pred:
        # 只显示预测的非零区域
        mask = image_data_pred[:, :, pred_slice_idx] > 0
        overlay = np.ma.masked_where(~mask, image_data_pred[:, :, pred_slice_idx])
        im = ax.imshow(overlay, cmap=pred_cmap, alpha=0.7)  # alpha控制透明度

        # 如果是最后一个子图，添加colorbar
        if i == num_slices_gt - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 隐藏多余的子图
for i in range(num_slices_gt, rows_gt * cols_gt):
    row = i // cols_gt
    col = i % cols_gt
    fig.delaxes(axes_gt[row, col])

plt.tight_layout()
plt.show()