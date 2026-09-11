# 架构说明

## 分层视图

```
┌─────────────────────────────────────────────────────┐
│ 数据源层   StockDataSimulator │ YFinanceDataSource   │  可插拔接口:
│            (fetch_price / fetch_news / history)      │  fetch_price, fetch_news,
└──────────────────────┬──────────────────────────────┘  get_price_history
                       ▼
┌─────────────────────────────────────────────────────┐
│ 顶层输入模块  InputModule × 4 (每种模态独立)          │
│   numeric 16单元 │ text 16单元                        │
│   timeseries 16单元 │ image 16单元 (4×4池化编码)      │
│   原始数据在各模块内编码, 模块间互不干扰              │
└──────────────────────┬──────────────────────────────┘
                       ▼
         模态融合层 (modality_weights 加权 → tanh)
                       ▼
┌─────────────────────────────────────────────────────┐
│ CubeGPT (depth=2, 4面×~4,400单元 ≈ 281K 参数)         │
│                                                      │
│   ┌─ numeric face ─→棱─→ text face ─┐               │
│   │   端口16+皮层4368      端口16+皮层4368            │
│   └←─棱─ image face ←─棱─ timeseries face ←┘        │
│        (环形侧连: 每步本面脉冲注入邻面)               │
│                    ▼ 融合                            │
│   顶层输出模块 OutputModule (16 单元)                 │
│   节律调制: 120 步周期 (80 思考 + 40 抑制)            │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│ 多智能体工作流  SpikeWorkflowEngine                   │
│   感知 → 推理(×2, 侧向互联) → 动作 → 记忆            │
│   RhythmAgent 广播全局节律; 共享 KVStack              │
│   脉冲每跳强度 ×0.7, hops ≤ 3                        │
└─────────────────────────────────────────────────────┘
```

## 16 参数脉冲单元

每个 `SpikingUnit` 恰好 16 个标量参数:

```
w_in  b_in  w_state  w_out  b_out  w_attn  b_attn  decay
gain  w_global  threshold  refractory_period
fatigue_rate  recovery_rate  spontaneous_rate  w_lateral
```

动力学:

```
state(t+1) = (gain·tanh(w_in·mean(input)+b_in)      # 输入门控
            + w_attn·mean(attn) + b_attn             # KV 检索 (v0.5.0 起为真实检索)
            + w_state·mean(state)                    # 状态反馈
            + w_global·global_modulation) · decay    # 全局节律
            + 0.1·state
output = state·w_out + b_out
发射条件: output ≥ threshold, 或以 spontaneous_rate 自发放电
```

## KV 堆 (KVStack)

- 写入: `push(key, key_vec, value_vec)`
- 检索: 余弦相似度 × 时间衰减, 取 top-k
- 淘汰: LRU + 7 天过期; `save_to_disk / load_from_disk` JSON 持久化
- 分布式后端: `deployment/redis_kv.py` 提供 Redis 实现 + 多租户前缀 (多节点工作流中尚未启用)

## 学习规则

无反向传播。两种机制协同:

1. **STDP** — 脉冲时序依赖可塑性 (LTP τ≈20ms / LTD), 作用于单元间连接
2. **监督引导** — 临时调制输出层/思考层单元的 gain/threshold ("注入→前向→恢复"),
   由发放率 + 目标激活 + 抑制 + 思考层对齐四项损失驱动权重更新

## CLI 与部署

```
dformer demo    # 端到端演示 (模拟)
dformer serve   # 长驻服务, Docker CMD; --realtime 接 yfinance
dformer train   # 合成数据训练
dformer test    # 模块自检
```

容器: `distributedformer/deployment/Dockerfile` + `docker-compose.yml` (Redis + 2 节点 + 可选 Prometheus)。
K8s: `distributedformer/deployment/k8s/`。

## CubeGPT: 立方体连接的多模态脉冲模型 (v0.4.0)

**命名**: Cube 指连接方式——四个模态面构成立方体侧面, 以"棱"环形侧连;
GPT 致敬 ChatGPT (Generative Pulse Transformer)。

**规模**: 4 面 × ~4,400 单元 × 16 参数 ≈ 281K 参数。
精确值: 每面 = 输入端口 16 单元 + 深度 2 分形皮层 4,368 单元 = 4,384 单元;
4 面 + 输出头 16 单元 = 17,552 单元 / 280,832 参数。

**数据流**:
1. 每面端口编码原始数据 → 端口模式 + 邻面棱传入的 inbox → tanh → 面皮层计算
2. 面间棱: 本面皮层脉冲聚合 (hash 散列到 dim 维, tanh 归一化) → 环形邻面下一拍 inbox
3. 各面输出融合 → OutputModule 头部 → 动作解码 (SpikeDecoder, 位于 agent 层)

**与 DistributedFormer 的关系**: DistributedFormer (共享思考层结构) 保留为
训练基底, trainer 的思考层监督实验基于它; 生产/演示路径的智能体内嵌网络
已切换为 CubeGPT。
