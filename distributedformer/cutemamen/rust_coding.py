"""RustCodingPlugin: Rust coding 思考插件 (填补训练材料空白)

后台携带 v0.6.0 的 Rust coding 真实需求语料 (100 段真实风格 Rust 代码
× 5 类真实 rustc 编译错误) 作为**训练材料** —— 直接消费真实代码文本,
不再是合成随机数据。每个类别从语料里蒸馏出一个带权原型特征向量
(static_metrics 空间的类别质心), 作为可存档 / 可加载的"可学习权重"。
这让内核无需额外数据源即可拿到一份真实训练/评估材料, 见路线图
"实数据集上端到端训练"的数据通路。

route = "rust": on_think 把一段 Rust 代码分类到 5 类编译错误之一,
并把结果发布到事件总线 ("rust.classified") 供其他插件订阅。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .plugin import ExpertPlugin, PluginContext

# 相对导入真实语料 (distributedformer.data.rust_coding)
try:
    from ..data.rust_coding import LABELS, RUST_SNIPPETS, static_metrics
except Exception:  # pragma: no cover - 极罕见时序问题, 见 _ensure_corpus
    LABELS, RUST_SNIPPETS, static_metrics = [], [], None


class RustCodingPlugin(ExpertPlugin):
    """静态识别一段 Rust 代码命中哪一类编译错误的思考插件

    训练材料: 内嵌 100 段真实 Rust 代码 + 真实 rustc 错误类别 (move /
    borrow / lifetime / type / ok)。原型权重 = 每类别 static_metrics
    均值; 分类 = 最近原型 (特征空间欧氏距离), 置信度 = softmax(-距离)。

    事件格式: {"topic": "rust", "data": <代码 str 或 {"code": str}>}
    返回: {"label", "label_name", "rustc", "confidence"},
    并广播 "rust.classified"。
    """

    BASE_MODEL = "rust.coding"
    CAPABILITY = ("静态识别 Rust 代码命中的编译错误类别 "
                  "(move/borrow/lifetime/type/ok), 内嵌 100 段真实语料作训练材料")

    def __init__(self, name: str = "rust-coding", *,
                 route: Optional[str] = None, **kwargs):
        super().__init__(name, route=route or "rust", **kwargs)
        # 原型权重: label → 类别质心 (特征空间), 权重存档用
        self.prototypes: Dict[str, np.ndarray] = {}
        self._sample_counts: Dict[str, int] = {}
        self._corpus_loaded = False

    # ── 真实训练材料 (填补训练数据空白) ─────────────────────
    def _ensure_corpus(self) -> None:
        """惰性加载真实语料并蒸馏每类别原型 (若权重未另行注入)"""
        if self._corpus_loaded:
            return
        if static_metrics is None:
            return
        features: Dict[str, List[np.ndarray]] = {lab: [] for lab in LABELS}
        for s in RUST_SNIPPETS:
            lab = s["label"]
            if lab in features:
                features[lab].append(static_metrics(s["code"]))
        for lab, arrs in features.items():
            if arrs:
                self.prototypes[lab] = np.mean(np.stack(arrs), axis=0)
                self._sample_counts[lab] = len(arrs)
        self._corpus_loaded = True

    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """导出真实训练材料: (特征矩阵, 标签索引), 供端到端训练/评估"""
        self._ensure_corpus()
        xs, ys = [], []
        for s in RUST_SNIPPETS:
            xs.append(static_metrics(s["code"]))
            ys.append(LABELS.index(s["label"]))
        return np.stack(xs), np.array(ys)

    def corpus_size(self) -> int:
        self._ensure_corpus()
        return sum(self._sample_counts.values())

    # ── 生命周期 ────────────────────────────────────────────
    def on_load(self, ctx: PluginContext) -> None:
        self._ensure_corpus()
        self.memory.set("base_model", self.BASE_MODEL)
        self.memory.set("corpus_size", self.corpus_size())
        self.memory.set("labels", LABELS)
        super().on_load(ctx)

    def on_think(self, event: Dict[str, Any],
                 ctx: PluginContext) -> Optional[Dict[str, Any]]:
        """分类一段 Rust 代码 → 5 类编译错误之一"""
        super().on_think(event, ctx)
        code = _extract_code(event.get("data"))
        if code is None:
            return None
        self._ensure_corpus()
        if not self.prototypes:
            return {"label": "ok", "label_name": "合法代码 (可编译)",
                    "rustc": "-", "confidence": 0.0, "needs_corpus": True}
        f = static_metrics(code)
        probs = _classify(f, self.prototypes)
        label = LABELS[int(np.argmax(probs))]
        confidence = float(probs[np.argmax(probs)])
        # 触发一次工作记忆写入 + 事件总线广播 (插件间通信)
        if ctx is not None:
            ctx.working_memory.push(
                f"rust_{ctx.kernel_version}_{len(ctx.working_memory.recent_events)}",
                f.astype(float), np.array([confidence]))
            ctx.emit("rust.classified",
                     {"label": label, "confidence": confidence,
                      "code_len": len(code)})
        return {"label": label, "label_name": _label_name(label),
                "rustc": _rustc_for(label), "confidence": confidence}

    def on_unload(self) -> None:
        self.memory.consolidate()
        super().on_unload()

    # ── 权重序列化 (可学习权重 = 类别原型, 可存档/加载) ──────
    def save_weights(self) -> Dict[str, np.ndarray]:
        self._ensure_corpus()
        out: Dict[str, np.ndarray] = {}
        for lab, proto in self.prototypes.items():
            out[f"proto_{lab}"] = np.asarray(proto, dtype=float)
        for lab, n in self._sample_counts.items():
            out[f"count_{lab}"] = np.array([n], dtype=int)
        return out

    def load_weights(self, weights: Dict[str, np.ndarray],
                     manifest: Dict[str, Any]) -> None:
        self.prototypes = {}
        self._sample_counts = {}
        for lab in LABELS:
            if f"proto_{lab}" in weights:
                self.prototypes[lab] = weights[f"proto_{lab}"].astype(float)
            if f"count_{lab}" in weights:
                self._sample_counts[lab] = int(weights[f"count_{lab}"][0])
        if self.prototypes:
            self._corpus_loaded = True  # 权重优先, 无需再从源码重蒸馏

    def build_manifest(self, **extra: Any) -> Dict[str, Any]:
        self._ensure_corpus()
        extra.setdefault("capability", self.CAPABILITY)
        extra.setdefault("corpus_size", self.corpus_size())
        extra.setdefault("labels", LABELS)
        return super().build_manifest(**extra)

    def stats(self) -> Dict[str, Any]:
        s = super().stats()
        s["labels"] = LABELS
        s["corpus_size"] = self.corpus_size()
        s["prototypes"] = {lab: len(proto)
                           for lab, proto in self.prototypes.items()}
        return s


# ── 辅助 ──────────────────────────────────────────────────

def _extract_code(data: Any) -> Optional[str]:
    """从事件取代码文本: 支持 str 或 {"code": str}"""
    if isinstance(data, str):
        return data
    if isinstance(data, dict) and isinstance(data.get("code"), str):
        return data["code"]
    if isinstance(data, dict) and "code" in data:
        return str(data["code"])
    return None


def _classify(feature: np.ndarray,
              prototypes: Dict[str, np.ndarray]) -> np.ndarray:
    """最近原型 (欧氏距离) + softmax(-距离) 给出类别置信度分布"""
    labels = LABELS
    d = np.array([float(np.linalg.norm(np.asarray(feature, dtype=float)
                                       - prototypes[lab]))
                  for lab in labels if lab in prototypes])
    present = [lab for lab in labels if lab in prototypes]
    if not present:
        return np.ones(len(labels)) / len(labels) * np.nan
    # softmax(-d), 距离越近置信度越高
    e = np.exp(-d - np.max(-d))
    probs = e / e.sum()
    out = np.zeros(len(labels))
    for i, lab in enumerate(present):
        out[LABELS.index(lab)] = probs[i]
    return out


_RUSTC_BY_LABEL = {
    "move": "E0382/E0505/E0507",
    "borrow": "E0502/E0499",
    "lifetime": "E0597/E0106/E0515/E0716",
    "type": "E0308/E0277/E0599/E0300",
    "ok": "-",
}


def _rustc_for(label: str) -> str:
    return _RUSTC_BY_LABEL.get(label, "-")


def _label_name(label: str) -> str:
    names = {
        "move": "所有权移动",
        "borrow": "借用冲突",
        "lifetime": "生命周期",
        "type": "类型不匹配",
        "ok": "合法代码 (可编译)",
    }
    return names.get(label, label)