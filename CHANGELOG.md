# 更新日志

## v0.2.0 (2026-09-11)

产品化重构版本。核心算法保持 Pre0.1 原貌, 工程链路全面修复:

### 新增
- 可安装的 `distributedformer` Python 包 (`pip install -e .`), `dformer` 命令行入口
- `dformer serve` 长驻监控服务 (容器部署默认入口, 支持 `--realtime` 切换 yfinance 真实行情)
- 可插拔行情数据源接口: 内置模拟器 / `YFinanceDataSource`
- pytest 测试套件 (12 项) 与 GitHub Actions CI (Python 3.9 / 3.11 / 3.12)
- MIT License, CHANGELOG, 本 README

### 修复
- Docker 镜像 CMD 引用不存在的 `python main.py server` 模式 → 改为 `dformer serve`, 镜像可正常启动
- docker-compose 构建上下文路径失效 → 指向包化后的 Dockerfile
- 所有模块导入路径统一为 `distributedformer.*`

### 变更
- 训练权重 (~240MB npz) 不再随仓库分发, 通过 GitHub Release 附件提供
- 历史实验报告 (experiments/, reports/) 作为归档保留

## Pre0.1 (2025-07-16)

首次对外公开发行版, 完整归档实验日志、训练权重、消融实验记录 (详见 RELEASE_NOTES.md)。
