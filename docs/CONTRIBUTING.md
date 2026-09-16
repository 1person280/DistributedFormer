# 参与贡献

欢迎 Issue 与 PR。

1. Fork → 建分支 (`git checkout -b feature/x`)
2. 开发: `pip install -e ".[dev]"`, 运行 `pytest -q` 保证全绿
3. 提交 PR, 描述变更动机与验证方式

约定:
- 核心算法改动请附上对应实验/基准结果 (src/experiments/ 下有现成的评估框架)
- 已知问题 (见 README) 是当前优先的贡献方向
