# EML：边缘感知多原型学习（Edge-aware Multi-Prototype Learning）用于小样本医学图像分割

本仓库提供论文 “Edge-aware Multi-Prototype Learning for Few-Shot Medical Image Segmentation (EML)” 的参考实现代码（ScienceDirect 链接：https://www.sciencedirect.com/science/article/abs/pii/S1746809425014296）。EML 面向 MRI/CT/超声等医学影像的小样本分割任务，针对边界模糊、噪声干扰和标注昂贵等痛点，提出了边缘感知的多原型学习框架，在三类公开数据集上取得了更优的边界定位和细节恢复能力。

## 方法概览
- Locality-Attentive Fusion Prototyper（LAFP）：采用多尺度特征融合与空间-通道协同注意机制，精确捕获前景特征并构建多原型表示。
- Dual-Stage Prototype Optimization（DPO）：以查询特征为参照对原型进行两阶段优化，提升判别性与鲁棒性，缓解复杂背景下的性能退化。
- Multi-Scale Prototype Matching：在 0.5/1.0/2.0 等尺度进行原型匹配与自适应融合，保留细粒度局部信息。
- Geometry-Aware Edge Optimization Loss（GEOL）：基于几何约束的边缘优化损失，显著增强复杂边界定位能力。

模型总体流程由 `FewShotSeg` 实现，骨干网络为 ResNet-101 编码器，并集成了对齐损失与边缘优化分支。

## 代码结构
- `train.py`：训练入口，使用 Sacred 管理实验与日志。
- `test.py`：评估入口，支持单类 1-shot/多切片推理与指标统计。
- `models/`：
  - `fewshot.py`：EML 主体，包括原型生成、注意力融合、边缘分支与损失。
  - `encoder.py`：ResNet-101 特征编码器，含预训练权重加载。
- `dataloaders/`：
  - `datasets.py`：训练/测试数据集与切片采样。
  - `dataset_specifics.py`：标签与折分定义。
  - `image_transforms.py`：仿射与弹性形变增强。
- `utils.py`：日志、指标与通用工具（Dice/IoU 计算）。
- `data/`：示例数据结构与预处理脚本（SABS、CMR、CHAOST2）。

## 环境依赖
建议使用 Python 3.8+（示例 notebook 为 3.8.12），CUDA 环境下运行。

- `torch`、`torchvision`
- `numpy`、`scipy`
- `opencv-python`
- `SimpleITK`
- `sacred`

示例安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy opencv-python SimpleITK sacred
```

> 注意：如需不同 CUDA 版本，请根据本机环境选择对应的 PyTorch 轮子。

## 数据准备
本仓库包含 SABS 与 CMR 的规范化示例；CHAOS-T2 需按同样结构准备。

- SABS（CT）：`data/SABS/sabs_CT_normalized/`
  - `image_*.nii.gz`、`label_*.nii.gz`
- CMR（MRI）：`data/CMR/cmr_MR_normalized/`
  - `image_*.nii.gz`、`label_*.nii.gz`
- CHAOS-T2（MRI）：`data/CHAOST2/chaos_MR_T2_normalized/`
  - 期望 `image_*.nii.gz`、`label_*.nii.gz`

训练阶段支持使用超体素作为弱标签：需在对应数据目录下生成 `supervoxels_<n_sv>/super_*.nii.gz`。仓库提供了 3D Felzenszwalb 的 Cython 实现以供生成，可结合您的数据流程产出超体素体积。

## 预训练权重
编码器默认从本地路径加载 Deeplabv3-ResNet101 的预训练权重。请将 `deeplabv3_resnet101_coco-586e9e4e.pth` 下载到本机并在 `models/encoder.py` 中更新为实际路径，或改为 `None` 禁用自定义加载并仅使用随机初始化。

## 运行示例
Sacred 作为实验入口，配置项可通过 `with` 语法覆盖。

### 训练
```bash
python train.py with mode=train dataset=SABS eval_fold=0 n_steps=30000 n_sv=5000 use_gt=False gpu_id=0
```
- 快照保存：`runs/Ours_train_<DATASET>_cv<EVAL_FOLD>/<ID>/snapshots`
- 日志保存：`runs/Ours_train_<DATASET>_cv<EVAL_FOLD>/<ID>/logger.log`

### 评估
在 `config.py` 设置/覆盖 `reload_model_path` 为训练得到的权重：
```bash
python test.py with mode=test dataset=SABS eval_fold=0 supp_idx=0 n_part=3 alpha=0.9 gpu_id=0 reload_model_path="c:/path/to/model.pth"
```
- 中间预测：`runs/Ours_test_<DATASET>_cv<EVAL_FOLD>/<ID>/interm_preds`
- 指标与日志：`runs/Ours_test_<DATASET>_cv<EVAL_FOLD>/<ID>/logger.log`

## 关键参数
- `dataset`：`SABS`、`CMR`、`CHAOST2`
- `n_shot`/`n_way`/`n_query`：小样本设定（默认 1/1/1）
- `alpha`：双阶段融合系数（`models/fewshot.py:21`）
- `n_sv`：超体素数量，决定 `supervoxels_<n_sv>` 路径（`config.py:41`-`config.py:44`）
- `supp_idx`/`n_part`：评估支持切片与查询分块（`test.py:98`、`test.py:122`）

## 指标与可视化
- Dice/IoU 统计：`utils.py:78`
- 评估流程：`test.py:86` 起遍历标签，跳过背景 `BG`。

## 许可
本仓库仅用于学术研究目的。请在符合目标数据集许可与隐私规范的前提下使用。