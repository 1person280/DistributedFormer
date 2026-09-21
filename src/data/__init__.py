"""data — 真实数据 (纯真实, 无合成样本, 项目硬约束)

    rust_coding.py        Rust 编码基准语料: 502 段真实代码
                          (rustc 错误索引官方样例 + 真实 crate 编译
                          失败样本; move/borrow/lifetime/type/ok 五类),
                          及 static_metrics (10 维语法) /
                          structure_metrics (6 维结构) 特征、
                          分层划分 / K 折工具
    real_dataset.py       RustCodingTrainingDataset: 语料 → 双模态
                          脉冲训练样本 (numeric=语法特征, text=代码原文)
    md_text.py            第二个真实任务: 仓库真实多字节 Markdown 语料
                          → 文本内容类型分类 (heading/code/list/table/
                          paragraph), 直接消费 64 比特 utf8-mb4 分类式
                          token (embed_tokens 折叠 + token_seq 序列)
    real_dialogues.py     真实对话语料 (互联网抓取 + 清洗: MandarinHero
                          100 中文对话 / NTU Mini-conversations 英文 /
                          DailyDialog), → 4K 参数统计对话模型
                          (ChatTransitionModel, 字符二元转移训练于真实语料)
"""
