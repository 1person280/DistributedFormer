"""src/tests — pytest 测试套件 (157 项)

按被测对象分文件, 与 src/ 各子包一一对应:

    test_cutemamen.py        插件标准全链路 (内核/生命周期/.CuteMamen
                             包格式/解码器/热加载/知识迁移, 最大)
    test_face_pkg.py         模态面包格式 (.dfpkg 往返/兼容检查)
    test_cube_gpt.py         CubeGPT 内核形态 (路由/输出头/节律)
    test_core.py             DistributedFormer 神经底座
    test_codec.py            多模态脉冲编解码
    test_multimodal.py       多模态注入端到端
    test_rust_data.py        真实 Rust 语料结构特征
    test_data_sources.py     数据源
    test_fixes.py            历史修复回归
    test_workflow.py         工作流引擎
    test_chat.py             终端聊天
    test_security_*.py       运行时安全监控 (骨架/单元/集成)

约定: 测试文件头部将仓库根 (上溯 3 级) 插入 sys.path 后
`from src... import ...`; 运行: python -m pytest
"""
