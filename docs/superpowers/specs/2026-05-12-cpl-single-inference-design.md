# CPL Single Video Inference — Frontend + Backend

## Scope

为 CPL 模型构建单视频推理前后端。用户上传视频文件 + 输入查询文本，返回视频中匹配文本的时间区间（真实秒数）。支持 ActivityNet Captions 和 Charades-STA 两个数据集。

**不做：** CPL-MoE 模型、PPS 模型、多 proposal 可视化。

## Files

```
vgdemo/
├── inference_server.py    # FastAPI 服务（含内嵌 HTML）
```

复用 `cpl-main/` 下的模型代码、配置、数据。

## Backend: FastAPI

### Startup（`@app.on_event("startup")`）

同时加载两个数据集的两个模型：

1. **ActivityNet 模型：**
   - 配置：`cpl-main/config/activitynet/main.json`
   - 词表：`cpl-main/data/activitynet/glove.pkl`
   - 权重：`cpl-main/checkpoints/activitynet/model-best.pt`
   - 特征：`feature_path`（C3D, 500维）

2. **Charades-STA 模型：**
   - 配置：`cpl-main/config/charades/main.json`
   - 词表：`cpl-main/data/charades/glove.pkl`
   - 权重：`cpl-main/checkpoints/charades/model-best.pt`
   - 特征：`feature_path`（I3D, 1024维）

3. 模型移至 GPU（若可用），设 `eval()` 模式

### 数据集自动识别

根据上传视频的文件名模式识别数据集：

| 视频名示例 | 模式 | 数据集 |
|-----------|------|--------|
| `v_uw9x69DT8_g.mp4` | 以 `v_` 开头 | ActivityNet |
| `3MSZA.mp4` | 不以 `v_` 开头 | Charades-STA |

去掉扩展名后得到 `video_id`，用于查 HDF5 和 duration 表。

### GET `/` — 返回内嵌 HTML 页面

- 视频文件上传控件
- 文本输入框
- 提交按钮
- 结果显示区（时间区间 + 所属数据集）

### POST `/predict` — multipart/form-data

**输入：**
- `video`: 上传视频文件
- `query`: 查询文本

**处理流程：**

```
1. 取文件名，根据命名模式识别数据集
2. 去掉扩展名 → video_id
3. ffprobe 读取视频时长 duration
4. 查对应 HDF5 → 视频特征 (N帧, 500或1024维)
5. 特征采样到 200 帧
6. query → nltk 分词 + POS tagging → GloVe 词向量 (≤20词, 300维)
7. 构造 batch(bsz=1) → 送入对应模型(epoch=0)
8. 从输出取 center/width/words_logit → 计算 8 个 proposal 的 NLL，排序
9. ActivityNet: vote 机制选最佳 proposal（IoU 加权投票）
   Charades: 直接选 NLL 最小的 proposal（loss-based）
10. proposal [center-width/2, center+width/2] × duration → 真实秒数
```

**输出 JSON：**
```json
{
  "success": true,
  "video_name": "v_uw9x69DT8_g.mp4",
  "video_id": "v_uw9x69DT8_g",
  "dataset": "activitynet",
  "interval": [0.92, 56.74],
  "duration": 159.99
}
```

**错误处理：**
- 文件名无法识别数据集 → 400
- video_id 不在 HDF5 中 → 400 `"video not found in feature file"`
- ffprobe 读取时长失败 → 400 `"failed to read video duration"`
- 其他异常 → 500

## Vote vs Loss-based 选择

这是 CPL 模型两种不同的 proposal 选择策略，参考 `runners/main_runner.py` 的 `eval()` 方法：

**Vote（ActivityNet）：**
1. 按 NLL 排序 8 个 proposal
2. 计算每个 proposal 与其他 proposal 的 IoU，累加为投票分
3. 选投票分最高的 proposal

**Loss-based（Charades）：**
1. 按 NLL 排序 8 个 proposal
2. 直接选 NLL 最低的 proposal

推理细节：
- 模型中的 `use_negative` 从配置读取（ActivityNet: `true`），虽然会计算负样本但对 inference 无影响（只用 positive 路径的 `words_logit`、`center`、`width`）
- 有 GPU 则用 GPU，无则 CPU

## Design Decisions

| 决策 | 选择 | 原因 |
|------|------|------|
| Web 框架 | FastAPI | 用户指定 |
| 前端方案 | 内嵌 HTML | 零额外文件 |
| 数据集识别 | 文件名模式匹配 | 无需用户手动选择 |
| Proposal 选择 | ActivityNet: vote / Charades: loss | 与训练时 eval 行为一致 |
| 视频时长 | ffprobe 读取 | 直接从上传文件获取，无需查 JSON |
| 文本处理 | 复用 BaseDataset 的 nltk 分词 | 与训练一致 |
| Device | GPU 优先，CPU 兜底 | 兼容有无 GPU 的服务器 |
