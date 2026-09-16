"""src/experiments — 实验脚本、结果与报告 (真实数据, 无合成)

    readout_validation.py             R1: 训练方法学验证
                                      (5 种子 × 5 折交叉验证, v0.8.3+)
    rust_benchmark.py                 R2: Rust 编码真实需求基准
                                      (rustc 错误码 5 分类)
    distributed_architecture.py       新阶段: 主模型知识迁移到思考插件
                                      后的端到端评估 (v0.8.6, 76.7%)
    *_results.json / *_report.md      对应结果与报告 (随仓库交付)

约定: 脚本头部将仓库根 (上溯 3 级) 插入 sys.path 后导入 src 包,
输出写在脚本所在目录; 运行例: python src/experiments/rust_benchmark.py
"""
