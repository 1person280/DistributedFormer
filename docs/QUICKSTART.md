# 快速开始

> 依赖极轻：核心只需要 `numpy`。

## 安装

```bash
pip install -e .
# 可选: 真实行情数据源
pip install -e ".[realtime]"
```

## 端到端演示

3 只股票的持续监控脉冲网络（模拟数据）：

```bash
dformer demo --cycles 10 --interval 0.5
```

## 作为库使用

```python
from distributedformer import CubeGPT

gpt = CubeGPT(depth=2, dim=16)   # 深度2 = 281K 参数

# 四模态面独立编码, 立方体棱环形侧连, 顶层输出模块生成动作脉冲
spikes = gpt.step({
    "numeric": 1.5,              # 标量 / 向量
    "text": "market surges",     # 文本
    "timeseries": [1, 2, 4, 3],  # 时序数组
    "image": gray_image_2d,      # 2D 图像
})
```

## 命令行

```bash
# 长驻监控服务（Docker/生产默认入口）
dformer serve --tickers AAPL TSLA NVDA --interval 300
dformer serve --realtime --interval 300        # 使用真实行情 (yfinance)

# 训练: 默认 rust / Markdown / 时序异常检测
dformer train --depth 1 --epochs 10                 # 真实 Rust 基准
dformer train --dataset md --depth 0 --epochs 10    # Markdown 内容类型
dformer train --dataset ts --depth 1 --epochs 4     # 真实时序异常检测

# 基准: 多任务 / 异常检测 / 时序流持续学习 / 全面统一基准
dformer benchmark-multi
dformer benchmark-anomaly
dformer benchmark-stream
dformer benchmark-all

# Web 图形界面 (本地浏览器控制台) / 终端聊天 / 模块自检
dformer ui --port 8001
dformer chat
dformer test
```

## 远程服务器部署

```bash
cd src/deployment
docker compose up -d          # Redis + 2 个监控节点
docker compose --profile monitoring up -d   # 加 Prometheus
```

Kubernetes 清单（Deployment / Service / HPA / ConfigMap）见
[`src/deployment/k8s/`](../src/deployment/k8s/)。