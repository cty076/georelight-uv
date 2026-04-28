# GeoRelight-UV

GeoRelight-UV 是一个轻量级 3D 纹理去光照与 PBR 清理研究脚手架。

项目聚焦的问题是：

> 能否用一个小型、感知几何信息的模型，去除生成或扫描 3D 纹理中烘焙进去的阴影和高光，从而得到更干净、更容易重新打光的 albedo 贴图？

当前版本包含可复现的合成数据生成流程，以及一个可以在单张 16 GB 显卡上运行的 Tiny U-Net 基线模型。

## 当前范围

项目目前支持生成成对的合成训练样本，包括：

- 带光照的 RGB 渲染图
- 干净的 albedo 目标图
- normal map 条件输入
- AO 条件输入
- shadow residual 目标

项目也包含一个轻量级去光照基线：

- 输入：带光照 RGB + normal + AO
- 输出：干净 albedo + shadow residual

评估指标包括：

- albedo MAE
- shadow residual MAE
- 多光照条件下的 albedo 一致性
- 保存可视化预测结果

第一版使用程序化材质 patch，因此即使没有 Blender，也可以跑通完整流程。后续数据方向是加入 MatSynth、ambientCG、Poly Haven 等 PBR 资产，并使用 Blender/Cycles 渲染更真实的数据。

## 快速开始

生成一个 smoke 测试数据集：

```powershell
python scripts/generate_synthetic_dataset.py --config configs/synth_smoke.json --out data/synth_smoke
```

从 ambientCG 的 CC0 PBR 资产生成一个小型真实材质数据集：

```powershell
python scripts/generate_ambientcg_dataset.py --config configs/ambientcg_real_small.json --out data/ambientcg_real_small --raw data/raw/ambientcg
```

快速训练 1 个 epoch：

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke --epochs 1 --batch-size 4 --base-channels 16
```

评估模型：

```powershell
python scripts/evaluate_model.py --data data/synth_smoke --checkpoint runs/smoke/checkpoint.pt --out runs/smoke_eval --split val
```

运行启发式基线：

```powershell
python scripts/evaluate_heuristics.py --data data/synth_smoke --out runs/smoke_heuristics --split val
```

训练 RGB-only 消融实验：

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke_rgb --epochs 1 --batch-size 4 --base-channels 16 --input-mode rgb
```

训练更大的模型：

```powershell
python scripts/train_tiny_unet.py --data data/synth_smoke --out runs/smoke_residual --epochs 1 --batch-size 4 --base-channels 16 --model residual_unet
```

可用模型名称：

```text
tiny_unet
residual_unet
attention_unet
convnext_unet
nafnet
restormer_lite
retinex_physics
```

运行测试：

```powershell
pytest -q
```

## 项目结构

```text
configs/                         小型实验配置
docs/                            研究笔记与数据集说明
georelight/dataset/              数据集 schema、生成器、PyTorch 数据集
georelight/models/               轻量级基线模型
georelight/train.py              训练入口
georelight/evaluate.py           评估入口
scripts/                         简薄 CLI 包装脚本
tests/                           smoke tests
```
