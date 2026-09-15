# DistributedFormer Pre0.1 — 发行说明

**发布日期**: 2025-07-16  
**版本号**: Pre0.1 (Preview Release 0.1)  
**发布性质**: 首次对外公开发行版，发出即公开  

---

## 版本概述

DistributedFormer Pre0.1 是本项目首次面向公众发布的完整归档版本。本版本包含了从原型开发到实验验证的全部内容，包括但不限于：

- 完整源代码（核心架构、编解码、智能体、工作流、训练、部署）
- 全部实验记录（消融实验、对照实验、扩展实验）
- 训练模型权重（5组实验检查点 + 最终模型，共约 240MB）
- 实验报告与可视化图表
- 部署配置（Docker、K8s、云原生）

---

## 包含内容清单

### 1. 核心架构 (`core/`)
| 文件 | 说明 |
|------|------|
| `distributedformer.py` | DistributedFormer 脉冲神经网络核心实现，含分形递归、KV堆、异步事件驱动机制 |

### 2. 编解码层 (`codec/`)
| 文件 | 说明 |
|------|------|
| `spike_codec.py` | 脉冲编码/解码（标量/文本/图像→脉冲信号） |
| `multimodal_codec.py` | 多模态融合编解码器 |

### 3. 智能体框架 (`agents/`)
| 文件 | 说明 |
|------|------|
| `base_agent.py` | 5种脉冲智能体基类（感知/推理/动作/记忆/节律） |

### 4. 工作流引擎 (`workflow/`)
| 文件 | 说明 |
|------|------|
| `engine.py` | 脉冲驱动的工作流编排引擎 |

### 5. 训练系统 (`training/`)
| 文件 | 说明 |
|------|------|
| `train.py` | 训练入口脚本 |
| `trainer.py` | 脉冲网络训练器（BPTT兼容） |
| `data_generator.py` | 合成数据集生成器 |
| `checkpoints/` | 模型权重目录（见下方"训练检查点"） |

### 6. 实验记录 (`experiments/`)
| 文件 | 说明 |
|------|------|
| `ablation_study.py` | 消融实验脚本（标准版） |
| `ablation_study_extended.py` | 消融实验脚本（扩展版） |
| `run_single_exp.py` | 单实验运行器 |
| `ablation_results.json` | 标准消融实验原始数据 |
| `ablation_results_extended.json` | 扩展消融实验原始数据 |
| `ablation_report.md` | 标准实验报告 |
| `ablation_report_extended.md` | 扩展实验报告 |
| `ablation_comparison.png` | 标准实验对比图 |
| `ablation_comparison_extended.png` | 扩展实验对比图 |
| `partial_E1_Baseline.json` | E1基线实验部分结果 |

### 7. 实验报告 (`reports/`)
| 文件 | 说明 |
|------|------|
| `training_report.md` | 训练过程详细报告 |
| `training_summary.json` | 训练摘要数据 |
| `elevator_question_experiment_report.md` | 电梯问题实验报告 |
| `elevator_question_computation.md` | 电梯问题计算过程记录 |

### 8. 可视化 (`visualization/`)
| 文件 | 说明 |
|------|------|
| `dashboard.py` | 实时训练监控面板 |
| `training_curves.png` | 训练曲线图 |
| `class_accuracy.png` | 类别准确率图 |

### 9. 部署配置 (`deployment/`)
| 文件 | 说明 |
|------|------|
| `Dockerfile` | 容器镜像构建文件 |
| `docker-compose.yml` | Docker Compose 编排 |
| `deploy.sh` / `deploy.ps1` | 部署脚本（Unix/Windows） |
| `cloud_config.py` | 云端配置管理 |
| `redis_kv.py` | Redis KV堆后端 |
| `monitoring.py` | 监控与可观测性 |
| `k8s/*.yaml` | Kubernetes 完整部署清单（Deployment/Service/HPA/ConfigMap） |

### 10. 演示 (`demos/`)
| 文件 | 说明 |
|------|------|
| `stock_monitor.py` | 智能股票监控端到端演示 |

### 11. 根目录文件
| 文件 | 说明 |
|------|------|
| `main.py` | 主入口点，支持模块级测试与演示 |
| `requirements.txt` | Python 依赖清单 |
| `elevator_question_full_computation.py` | 电梯问题完整计算脚本 |
| `test_import.py` | 模块导入测试 |
| `test_perf.py` | 性能基准测试 |
| `computation_output.txt` | 计算输出记录 |

---

## 训练检查点 (Training Checkpoints)

| 检查点 | 说明 | 文件名 |
|--------|------|--------|
| E1_Baseline | 基线实验（标准配置） | `best_model.npz` / `final_model.npz` |
| E2_MoreData | 更多数据实验 | `best_model.npz` / `final_model.npz` |
| E3_BiggerNet | 更大网络实验 | `best_model.npz` / `final_model.npz` |
| E4_ThinkSuper | 超级思考实验 | `best_model.npz` / `final_model.npz` |
| E5_v52_Full | v5.2 完整版实验 | `best_model.npz` / `final_model.npz` |
| (根目录) | 最终统一模型 | `best_model.npz` / `final_model.npz` / `test_model_v52.npz` |

---

## 实验日志

> **说明**: `logs/` 目录在 Pre0.1 版本中为空（预留目录，用于运行时日志输出）。
> 所有实验记录均已以 JSON/PNG/MD 形式固化于 `experiments/` 和 `reports/` 目录中。

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整测试
python main.py

# 运行股票监控演示
python main.py stock --cycles 20 --interval 0.5

# 运行消融实验
python experiments/ablation_study.py

# 查看帮助
python main.py help
```

---

## 技术参数

| 参数 | 数值 |
|------|------|
| 分形深度 | 2 |
| 基础单元数 | 4,096 |
| 总参数 | ~69,888 |
| 思考层 | 20层分形递归 |
| KV堆容量 | 256 条目/单元 |
| 智能体类型 | 5种（感知/推理/动作/记忆/节律） |

---

## 许可声明

本版本（Pre0.1）作为完整研究归档对外发布。所有代码、实验数据、模型权重均包含于本归档中。

---

*DistributedFormer Pre0.1*  
*发布日期: 2025-07-16*  
*发出即公开*
