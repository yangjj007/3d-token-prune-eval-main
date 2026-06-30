# ShapeLLM-Omni 项目概述

## 项目简介

**ShapeLLM-Omni** 是一个用于3D生成和理解的原生多模态大型语言模型（LLM）。该项目基于清华大学的研究成果，在NeurIPS 2025上发表，支持从文本和图像输入生成高质量的3D模型，并具备3D理解和编辑能力。

## 项目架构

### 核心组件

1. **多模态LLM骨干网络**: 基于Qwen2.5-VL模型，通过指令微调实现3D生成和理解
2. **TRELLIS生成引擎**: 提供高质量的3D生成能力，支持文本到3D、图像到3D的转换
3. **VQ-VAE压缩模块**: 使用3DVQVAE对3D结构进行高效压缩和重建
4. **数据预处理工具链**: 完整的obj文件预处理流水线

### 技术栈

- **深度学习框架**: PyTorch
- **多模态模型**: Qwen2.5-VL, DINOv2
- **3D处理库**: Open3D, Trimesh, Blender
- **数据处理**: Pandas, NumPy, utils3d
- **UI界面**: Gradio

## OBJ文件预处理流程

### 1. 数据获取
- 支持多个数据集：ObjaverseXL, 3D-FUTURE, ABO, HSSD, Toys4k等
- 从HuggingFace下载原始obj文件和元数据
- 过滤低质量数据（基于美学评分等指标）

### 2. 多视角渲染
```python
# 使用Blender进行多视角渲染
BLENDER_PATH = '/tmp/blender-3.0.1-linux-x64/blender'
views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
         for y, p, r, f in zip(yaws, pitchs, radius, fov)]
```

- 默认生成150个视角的渲染图像
- 相机参数：半球面Hammersley序列采样
- 输出：transforms.json + 多张RGB图像

### 3. 体素化处理
```python
# 将三角网格转换为64x64x64体素
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh, voxel_size=1/64,
    min_bound=(-0.5, -0.5, -0.5),
    max_bound=(0.5, 0.5, 0.5)
)
```

- 坐标归一化到[-0.5, 0.5]范围
- 体素分辨率：64³
- 输出格式：PLY点云文件

### 4. 特征提取
- 使用DINOv2-ViT-L/14提取图像patch特征
- 支持多个特征模型：dinov2_vitl14_reg等
- 输出：稀疏特征表示（patchtokens + indices）

### 5. Latent编码
```python
# 结构化latent编码
feats = sp.SparseTensor(
    feats=torch.from_numpy(feats['patchtokens']).float(),
    coords=torch.cat([
        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
        torch.from_numpy(feats['indices']).int(),
    ], dim=1),
)
latent = encoder(feats, sample_posterior=False)
```

- 使用预训练的SLat-VAE编码器
- 输出：压缩的latent表示（feats + coords）

### 6. 稀疏结构编码
```python
# 体素结构编码
ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
latent = encoder(ss.float(), sample_posterior=False)
```

- 使用Conv3D-VAE编码器
- 输出：结构latent（用于几何重建）

## 模型训练流程

### 1. VAE预训练

#### Sparse Structure VAE训练
```python
# 损失函数：BCE + KL散度
terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float())
terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
terms["loss"] = terms["bce"] + lambda_kl * terms["kl"]
```

- **目标**: 重建3D体素结构
- **损失**: 二元交叉熵 + KL正则化
- **架构**: 编码器-解码器结构

#### Structured Latent VAE训练
- **目标**: 压缩多视角特征为latent空间
- **输入**: 稀疏图像特征
- **输出**: 结构化latent表示

### 2. Flow Matching生成训练

#### 核心算法
```python
# Flow Matching目标函数
x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
pred = model(x_t, t * 1000, cond)
target = (1 - sigma_min) * noise - x_0
loss = F.mse_loss(pred, target)
```

- **扩散过程**: 最小噪声sigma_min=1e-5
- **时间采样**: LogitNormal分布
- **条件生成**: 支持文本和图像条件

#### 条件生成变体
- **Text-conditioned**: 使用文本嵌入作为条件
- **Image-conditioned**: 使用图像特征作为条件
- **CFG**: Classifier-Free Guidance提升生成质量

### 3. LLM指令微调

#### 数据集：3D-Alpaca
- 包含50k+高质量3D编辑样本对
- 指令格式：文本描述 → 3D模型token序列
- 多轮对话支持

#### 微调策略
- 基于LLaMA-Factory框架
- 冻结视觉编码器，微调语言模型
- 支持3D生成和编辑任务

## 推理流程

### 文本到3D生成
1. 用户输入文本提示
2. LLM生成3D token序列（<mesh>标记）
3. VQVAE解码为体素结构
4. 后处理生成最终网格

### 图像到3D生成
1. 图像输入经视觉编码器处理
2. 结合文本条件进行生成
3. 流匹配采样生成latent
4. 解码器重建3D几何

### 3D理解和编辑
- 支持3D模型输入分析
- 基于多模态对话的编辑能力
- Nano3D算法支持无mask编辑

## 技术亮点

1. **原生多模态**: LLM直接处理3D数据，无需额外适配器
2. **高效压缩**: VQ-VAE将3D数据压缩为离散token序列
3. **高质量生成**: 基于TRELLIS的先进3D生成技术
4. **统一接口**: 支持文本、图像、3D多模态输入输出
5. **可编辑性**: 支持3D模型的理解和精确编辑

## 部署和使用

### 环境配置
```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
pip install -r requirements.txt
```

### 推理启动
```bash
cd /data/xujinyi/junjie_llm/3d-token-prune-eval-main
conda activate token-prune-shapellm
python archive/demo/app.py
```

### 训练配置
- 当前评测配置使用 `configs/runs/*.yaml` 与 `configs/eval/*.json`
- 原始生成 demo 的 JSON 配置已归档到 `archive/demo/configs/generation/`
- 支持分布式训练（DDP）
- 弹性内存管理

## 未来计划

- [ ] 发布完整3D-Alpaca数据集
- [ ] 发布训练代码
- [ ] 支持多轮对话和3D编辑的模型权重
- [ ] Nano3D-Edit-100k数据集开源

## 引用

```bibtex
@article{ye2025shapellm,
  title={ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding},
  author={Ye, Junliang and Wang, Zhengyi and Zhao, Ruowen and Xie, Shenghao and Zhu, Jun},
  journal={arXiv preprint arXiv:2506.01853},
  year={2025}
}
```
