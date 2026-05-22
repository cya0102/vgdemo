# 视频时序定位 Web 演示系统

基于 CPL、PPS 、CPL-MoE三个模型的视频时序定位 Web 演示，GitHub 链接：https://github.com/xxxx/vgdemo，本地服务器地址：192.168.1.107:/data/chenyuan/vgdemo

## 项目结构

```
vgdemo/
├── cpl-main/                  # CPL 模型代码（CVPR 2022 / TPAMI 2025）
│   ├── train.py               # 训练/评估入口
│   ├── config/                # JSON 配置文件
│   ├── checkpoints/           # 预训练权重
│   └── data/                  # 数据集标注与词表
├── cplmoe-main/               # CPL-MoE 模型代码
│   ├── train_moe.py           # 训练/评估入口
│   ├── config/                # JSON 配置文件
│   └── checkpoints/           # 预训练权重
├── pps-main/                  # PPS 模型代码（AAAI 2024）
│   ├── train.py               # 训练/评估入口
│   ├── config/                # JSON 配置文件
│   └── checkpoints/           # 预训练权重
├── inference_server.py        # 主推理服务（端口 8100，CPL + CPL-MoE）
├── pps_inference_server.py    # PPS 推理服务（端口 8200）Net I3D 特征提取
├── start_servers.sh           # 一键启动服务
├── cache/                     # 推理缓存目录
└── README.md                  # 说明文档
```

## 项目预览

## 快速开始

### 1. 环境准备

```bash
# CPL / CPL-MoE 环境（Python 3.8）
conda create -n cpl python=3.8
conda activate cpl
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install h5py nltk fairseq tqdm fastapi uvicorn

# PPS 环境（Python 3.10）
conda create -n pps python=3.10
conda activate pps
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install h5py nltk fairseq tqdm fastapi uvicorn
```

下载 NLTK 数据：

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 2. 下载 I3D 权重（可选，仅特征提取需要）

将 `rgb_imagenet.pt` 和 `rgb_charades.pt` 放入 `weights/` 目录。

### 3. 启动服务

**一键启动：**

```bash
bash start_servers.sh
```

### 4. 访问

打开浏览器访问：`http://127.0.0.1:8100`

## 支持的模型

| 模型 | 论文 | 特征维度 | 推理策略 |
|------|------|----------|----------|
| CPL | CVPR 2022 | C3D 500-d / I3D 1024-d | ActivityNet：投票，Charades：损失最小 |
| PPS | AAAI 2024 | C3D 500-d / I3D 1024-d | 统一使用加权投票 |
| CPL-MoE | 未发表 | C3D 500-d / I3D 1024-d | 同 CPL |

| 数据集 | 特征文件 |
|--------|----------|
| ActivityNet | sub_activitynet_v1-3.c3d.hdf5 |
| Charades-STA | i3d_features.hdf5 |

## 配置

模型配置位于各子项目的 `config/` 目录下的 JSON 文件中：

| 配置项 | 说明 |
|--------|------|
| `dataset.feature_path` | HDF5 特征文件路径 |
| `dataset.vocab_path` | GloVe 词表路径 |
| `train.model_saved_path` | 模型保存路径 |
| `model.num_props` | 高斯提议数量 |
| `loss.lambda` | 对比损失权重（敏感参数） |

## 注意事项

1. 确保 HDF5 特征文件和 GloVe 词表存在且路径正确（见各 `config/*.json` 中的 `feature_path` 和 `vocab_path`）
2. PPS 与 CPL/CPL-MoE 使用不同的 Conda 环境，需分别在两个终端启动
3. 模型权重需自行下载并放入对应 `checkpoints/` 目录
4. 首次请求时模型采用懒加载，等待约 10-30 秒
5. CPL 的 `lambda` 参数对结果敏感，如无法复现论文指标请微调（0.125 → 0.135）
6. 如需使用在线 I3D 特征提取，需安装 `ffmpeg` 和 `opencv-python`，并下载 I3D 预训练权重
