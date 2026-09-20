# -*- coding: utf-8 -*-
"""
真实多字节文本训练数据集 (v0.11.0) — 第二个真实任务, 纯真实无合成

语料: 本仓库内真实已撰写的 Markdown 文档 (*.md), 含大量多字节
utf8-mb4 字符 (中文/全角符号), 全部为真实成稿文字, 非任何生成/合成。

任务: 静态识别每一行真实文本的**内容类型** (由行首真实语法判定):
    heading   标题    (setext/pound 标题行)
    code      代码块  (``` fenced code block 内部行)
    list      列表    (- * + 或 有序列表)
    table     表格行  (| col | 管道表格)
    paragraph 段落    (其余非空正文行)
这是 lint / 文档解析 / 结构化阅读的基础能力。

样本编码 (与 Rust 数据通路对齐):
- input_signal: 64 比特 class-token (utf8-mb4 字符组) 序列经
  embed_tokens 折叠为 dim 维脉冲信号 —— 直接消费分类式 token
- token_seq: 完整 64 比特 token 序列 (供端到端可训练嵌入层)
- target_pattern: 5 类监督输出模式 (维度 0-4 对应真实内容类型)

评估基线 (5 分类): 随机 20%; 多数类 ~ 正文段落占比 (如实统计)。

复用 src.data.rust_coding 的 stratified_split / stratified_kfold
做分层划分与 K 折 (保持 5 种子 × 5 折评估约定)。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.codec.class_token import encode as token_encode, embed_tokens
from src.data.rust_coding import stratified_split, stratified_kfold

# 仓库根 (src/data/<file> → 上溯 3 级)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 真实内容类型 (5 类) ──────────────────────────────────────
LABELS = ["heading", "code", "list", "table", "paragraph"]
LABEL_NAMES = {
    "heading": "标题",
    "code": "代码块",
    "list": "列表",
    "table": "表格行",
    "paragraph": "段落",
}


def _line_type(line: str) -> str:
    """单行真实文本 → 真实内容类型 (语法判定)"""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#") and (
            len(stripped.lstrip("#")) > 0 and stripped.lstrip("#")[0] in " \t"):
        return "heading"
    if stripped.startswith("|") and stripped.endswith("|"):
        return "table"
    if (stripped.startswith(("- ", "* ", "+ ")) or
            (len(stripped) > 1 and stripped[0].isdigit() and
             stripped[1] in ". )")):
        return "list"
    return "paragraph"


def _collect_doc_md_files() -> List[str]:
    """递归收集 docs/ 分类子目录下的真实 *.md 文档 (排除许可原文/译文).

    docs/ 按分类子目录组织 (技术/历史/规范/发行/合规), 故需递归而非平铺
    listdir; 许可文本非"内容类型"语料, 不纳入训练材料。
    """
    doc_root = os.path.join(_REPO_ROOT, "docs")
    files = []
    for root, _dirs, names in os.walk(doc_root):
        for name in names:
            if not name.endswith(".md"):
                continue
            if name.upper().startswith("LICENSE"):
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def load_md_text() -> List[Dict]:
    """扫描仓库内真实 *.md 文档, 返回每行真实样本列表

    逐行解析, 依据 fenced code block 状态把 ``` 内部行标为 code,
    其余行交给 _line_type; 空行 (无判别信息) 跳过。
    返回: [{"text": str, "type": "heading|code|list|table|paragraph"}]。
    """
    sources = _collect_doc_md_files()
    for root_f in ("README.md",):
        p = os.path.join(_REPO_ROOT, root_f)
        if os.path.exists(p):
            sources.append(p)

    rows = []
    in_code = False
    for path in sources:
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                # 代码块内的空行无判别信息, 与整体"空行跳过"规则一致
                # (空行 token 序列为空, 折叠出的脉冲信号全零, 不应成样本)
                if not stripped:
                    continue
                # 复用 rust_coding.stratified_* 需要 "label" 键 (类别名字符串)
                rows.append({"text": line, "type": "code",
                             "label": "code"})
                continue
            lt = _line_type(line)
            if lt is not None:
                rows.append({"text": line, "type": lt, "label": lt})
    return rows


@dataclass
class TextSample:
    """单个真实文本行训练样本"""
    sample_id: str
    category: int           # 0=heading 1=code 2=list 3=table 4=paragraph
    category_name: str
    input_signal: np.ndarray     # dim 维脉冲 (class-token 折叠)
    target_pattern: np.ndarray   # 5 类监督输出模式
    metadata: Dict
    token_seq: List[int] = None  # 完整 64 比特 class-token 序列
    static_signal: np.ndarray = None

    def multimodal_input(self) -> Dict:
        """双模态输入: numeric=token 脉冲, text=原始行 (向后兼容)"""
        return {"numeric": self.input_signal, "text": self.metadata["text"]}


class MarkdownTextDataset:
    """真实 Markdown 文本内容类型数据集 (纯真实, 无合成样本)

    类别 (真实内容类型):
      0: heading   标题
      1: code      代码块
      2: list      列表
      3: table     表格行
      4: paragraph 段落
    """

    CATEGORIES = list(LABELS)
    N_CLASSES = len(LABELS)
    RANDOM_BASELINE = 1.0 / len(LABELS)          # 20%

    _TARGET_STRENGTH = {i: 0.7 for i in range(len(LABELS))}

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._corpus = load_md_text()  # 真实文档行

    def _make_target_pattern(self, category: int) -> np.ndarray:
        pattern = np.ones(self.dim) * 0.05
        pattern[category] = self._TARGET_STRENGTH.get(category, 0.6)
        for i in range(self.N_CLASSES):
            if i != category:
                pattern[i] = 0.05
        return np.clip(pattern, 0.0, 1.0)

    def _to_sample(self, row: Dict, idx: int) -> TextSample:
        text = row["text"]
        tokens = token_encode(text)             # 64 比特 utf8-mb4 字符组 token
        category = LABELS.index(row["type"])
        return TextSample(
            sample_id=f"{row['type']}_{idx:04d}",
            category=category,
            category_name=row["type"],
            input_signal=embed_tokens(tokens, self.dim),
            target_pattern=self._make_target_pattern(category),
            metadata={"text": text,
                      "text_len": len(text),
                      "token_len": len(tokens)},
            token_seq=tokens,
        )

    def generate_dataset(self, train_ratio: float = 0.75,
                         seed: int = 0) -> Tuple[List[TextSample],
                                                 List[TextSample]]:
        train_raw, val_raw = stratified_split(
            self._corpus, train_ratio=train_ratio, seed=seed)
        index_of = {id(s): i for i, s in enumerate(self._corpus)}
        train = [self._to_sample(s, index_of[id(s)]) for s in train_raw]
        val = [self._to_sample(s, index_of[id(s)]) for s in val_raw]
        return train, val

    def kfold_datasets(self, n_folds: int = 5,
                       seed: int = 0) -> List[Tuple[List[TextSample],
                                                    List[TextSample]]]:
        index_of = {id(s): i for i, s in enumerate(self._corpus)}
        folds_raw = stratified_kfold(self._corpus, n_folds=n_folds, seed=seed)
        splits = []
        for k in range(n_folds):
            val_raw = folds_raw[k]
            train_raw = [s for j in range(n_folds) if j != k
                         for s in folds_raw[j]]
            splits.append((
                [self._to_sample(s, index_of[id(s)]) for s in train_raw],
                [self._to_sample(s, index_of[id(s)]) for s in val_raw]))
        return splits

    def get_class_distribution(self, samples: List[TextSample]) -> Dict:
        counts = {lab: 0 for lab in LABELS}
        for s in samples:
            counts[s.category_name] += 1
        return counts

    def majority_baseline(self, samples: List[TextSample]) -> float:
        if not samples:
            return 0.0
        dist = self.get_class_distribution(samples)
        return max(dist.values()) / len(samples)


# ═══════════════════════════════════════════════════════════════
# 自测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("真实 Markdown 文本数据集测试 (100% 真实多字节语料)")
    print("=" * 60)

    dataset = MarkdownTextDataset(dim=16)
    train, val = dataset.generate_dataset(train_ratio=0.75)
    print(f"\n[数据规模]")
    print(f"  语料行数: {len(dataset._corpus)}")
    print(f"  训练集: {len(train)} | 验证集: {len(val)}")
    print(f"  训练集分布: {dataset.get_class_distribution(train)}")
    print(f"  随机基线: {dataset.RANDOM_BASELINE:.0%}")
    print(f"  多数类基线: {dataset.majority_baseline(train):.0%}")

    print("\n[各类真实样本示例]")
    for cat in range(5):
        sample = next(s for s in train if s.category == cat)
        print(f"\n  {sample.category_name:10s}: {sample.metadata['text'][:50]!r}")
        print(f"    输入脉冲非零维: {np.count_nonzero(sample.input_signal)} | "
              f"token 数: {len(sample.token_seq)}")