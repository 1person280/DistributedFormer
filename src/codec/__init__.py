"""codec — 编解码层 (真实世界 ↔ 脉冲信号的翻译官)

    spike_codec.py        SpikeEncoder: 数值/文本/时序 → 16 维脉冲信号
                          (文本走代码感知 TF-IDF, crc32 确定性哈希);
                          MultiModalCodec + 脉冲 → 动作解码
    multimodal_codec.py   多模态编解码器 (CubeGPT 4 模态面配套)
    class_token.py        64 比特分类式 token: 高 32 位分组 + 低 32 位载荷;
                          0x00000000 组保留为 utf8-mb4 字符 token
                          (低 32 位 = Unicode 码点), embed_tokens 桥接脉冲信号

只做"信号翻译", 不含任何学习或路由逻辑。
"""
