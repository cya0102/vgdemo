# Frontend UI Optimization

## Scope
美化 CPL 推理服务前端页面，增加时间轴可视化。纯 CSS 实现，零外部依赖。

## Changes

### 整体美化
- 渐变标题栏（蓝紫色渐变 + 图标）
- 文件上传区改为虚线拖拽框样式
- 按钮 hover 渐变过渡，加载时禁用态 pulsate
- 卡片阴影加深，统一圆角
- 结果区时间区间大字 + 渐变背景突出

### 时间轴可视化
- 水平条 (0s — duration)，浅灰底色
- 预测区间蓝色高亮块
- 区间两端标注 start/end 时间

### Layout
```
渐变标题栏
  输入卡片
    虚线文件上传框
    文本输入框
    [Find Segment]
  结果卡片
    大字时间区间
    时间轴进度条
    元信息 (video, duration, dataset, query)
```

## Implementation
单文件改动 `inference_server.py`，替换 HTML 字符串。
