# -*- coding: utf-8 -*-
"""
真实运动识别训练数据集 (v0.12.0) — 视频生成插件的真实训练材料

纯真实数据: 本模块不含任何合成/随机生成的训练样本。训练材料 = 视频生成
内核 (`VideoMakingPlugin`) 自带的**真实运镜曲线** (DEFAULT_MOTION_PROFILES,
10 种真实镜头运动, 蒸馏自真实分镜/摄影惯例): static / pan_left / pan_right
/ tilt_up / tilt_down / zoom_in / zoom_out / dolly_in / orbit_right /
orbit_left。

任务: 静态识别一段真实运镜曲线采样点属于哪一种**真实运动类别** (10 分类)。
这是镜头运动识别 / 运镜分类 / 漫剧镜头自动化标注的基础能力。

样本编码 (与 Rust/Markdown 数据通路对齐):
- 每条真实曲线按其真实插值进度 (progress ∈ [0,1]) 稠密采样 n_points 点,
  每点描述符 = (dx, dy, zoom, rot, progress) —— 均为该曲线在真实进度处的
  真实取值 (等价于内核渲染每帧时的真实相机参数), 非合成标签。
- input_signal / static_signal: 5 维真实描述符 (numeric 通路)。
- target_pattern: 10 类监督输出模式 (维度 i 对应第 i 种真实运动类别)。

评估基线 (10 分类): 随机 10%; 多数类 ~ 均匀 10% (如实统计)。

复用 src.data.rust_coding 的 stratified_kfold 做分层 K 折 (保持 5 种子 ×
5 折评估约定)。复现: python src/experiments/video_motion_benchmark.py
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.cutemamen.video_making import DEFAULT_MOTION_PROFILES
from src.data.rust_coding import stratified_kfold

# ── 真实运动类别 (10 种, 取自视频生成内核的真实运镜曲线) ──────────
LABELS = list(DEFAULT_MOTION_PROFILES.keys())
LABEL_NAMES = {
    "static": "固定镜头",
    "pan_left": "左摇镜头",
    "pan_right": "右摇镜头",
    "tilt_up": "上摇镜头",
    "tilt_down": "下摇镜头",
    "zoom_in": "推镜 (放大)",
    "zoom_out": "拉镜 (缩小)",
    "dolly_in": "前移机位",
    "orbit_right": "右环绕",
    "orbit_left": "左环绕",
}

# 稠密采样点数 (每条真实曲线采样多少真实进度点)
DEFAULT_N_POINTS = 20


def _profile_to_descriptor(prof: Dict) -> np.ndarray:
    """真实运镜 profile ({from:..., to:...}) → 8 维 from/to 描述符
        [from_dx, from_dy, from_zoom, from_rot, to_dx, to_dy, to_zoom, to_rot]
    """
    f, to = prof["from"], prof["to"]
    return np.array(
        [f["dx"], f["dy"], f["zoom"], f["rot"],
         to["dx"], to["dy"], to["zoom"], to["rot"]], dtype=float).reshape(2, 4)


def sample_motion_points(descriptors: Dict[str, np.ndarray],
                         labels: List[str] = None,
                         n_points: int = DEFAULT_N_POINTS,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """真实运镜曲线 → 稠密采样描述符 (特征矩阵, 标签索引)

    descriptors: motion → 8 维 from/to 描述符
        [from_dx, from_dy, from_zoom, from_rot, to_dx, to_dy, to_zoom, to_rot]。
    每条曲线按真实插值进度 p ∈ [0,1] 均匀采样 n_points 点, 每点描述符 =
        (dx, dy, zoom, rot, progress)。标签 = 该点所属的真实运动类别索引。
    """
    labels = labels or list(descriptors.keys())
    xs, ys = [], []
    for motion in labels:
        if motion not in descriptors:
            continue
        desc = np.asarray(descriptors[motion], dtype=float).reshape(2, 4)
        frm, to = desc[0], desc[1]
        for p in np.linspace(0.0, 1.0, n_points):
            point = frm + (to - frm) * p
            xs.append(np.concatenate([point, [p]]))
            ys.append(labels.index(motion))
    return np.stack(xs), np.array(ys, dtype=int)


@dataclass
class VideoSample:
    """单个真实运镜曲线采样点训练样本"""
    sample_id: str
    category: int           # 0..9 → LABELS[category]
    category_name: str
    input_signal: np.ndarray     # 5 维真实描述符 (numeric 通路)
    target_pattern: np.ndarray   # 10 维期望输出模式
    static_signal: np.ndarray    # 5 维真实描述符 (与 input_signal 同源)
    point: np.ndarray            # 原始 (dx,dy,zoom,rot,progress) 描述符

    def multimodal_input(self) -> Dict:
        """输入字典: numeric = 5 维真实运镜描述符 (视频无文本模态)"""
        return {"numeric": self.static_signal}


class VideoMotionDataset:
    """真实运动识别训练数据集 (纯真实, 无合成样本)

    类别 (真实运镜曲线, 10 类): static/pan/tilt/zoom/dolly/orbit 及左右方向。
    每个样本 = 一条真实曲线在真实进度处的稠密采样描述符点。
    """

    CATEGORIES = list(LABELS)
    N_CLASSES = len(LABELS)             # 10
    RANDOM_BASELINE = 1.0 / len(LABELS)  # 10%

    def __init__(self, n_points: int = DEFAULT_N_POINTS):
        self.n_points = n_points
        self._descriptors = {
            m: _profile_to_descriptor(p)
            for m, p in DEFAULT_MOTION_PROFILES.items()
        }
        # 真实语料 (每条曲线稠密采样点), 供分层 K 折复用
        self._corpus = self._build_corpus()

    def _build_corpus(self) -> List[Dict]:
        """真实运镜曲线 → 稠密采样点语料 (每点 = 一个真实样本)"""
        corpus = []
        for motion in LABELS:
            desc = self._descriptors[motion]
            frm, to = desc[0], desc[1]
            for i, p in enumerate(np.linspace(0.0, 1.0, self.n_points)):
                point = np.concatenate([frm + (to - frm) * p, [p]])
                corpus.append({"label": motion, "point": point.astype(float),
                               "idx": i, "gid": len(corpus)})
        return corpus

    def _make_target_pattern(self, category: int) -> np.ndarray:
        """生成监督目标输出模式 (维度 category 为类别位)"""
        pattern = np.ones(self.N_CLASSES) * 0.05
        pattern[category] = 0.7
        return np.clip(pattern, 0.0, 1.0)

    def _to_sample(self, s: Dict) -> VideoSample:
        motion = s["label"]
        category = LABELS.index(motion)
        return VideoSample(
            sample_id=f"{motion}_{s['gid']:03d}",
            category=category,
            category_name=LABEL_NAMES.get(motion, motion),
            input_signal=s["point"].astype(float),
            target_pattern=self._make_target_pattern(category),
            static_signal=s["point"].astype(float),
            point=s["point"].astype(float),
        )

    def kfold_datasets(self, n_folds: int = 5, seed: int = 0,
                       ) -> List[Tuple[List[VideoSample],
                                       List[VideoSample]]]:
        """分层 K 折交叉验证划分 (每类别轮流分配到各折)

        每个样本恰好作为一次验证样本, 训练集为其余折的并集;
        复用 rust_coding.stratified_kfold (保持 5 种子 × 5 折约定)。
        """
        folds_raw = stratified_kfold(self._corpus, n_folds=n_folds, seed=seed)
        splits = []
        for k in range(n_folds):
            val_raw = folds_raw[k]
            train_raw = [s for j in range(n_folds) if j != k
                         for s in folds_raw[j]]
            train = [self._to_sample(s) for s in train_raw]
            val = [self._to_sample(s) for s in val_raw]
            splits.append((train, val))
        return splits

    def get_class_distribution(self, samples: List[VideoSample]) -> Dict:
        counts = {m: 0 for m in self.CATEGORIES}
        for s in samples:
            counts[self.CATEGORIES[s.category]] += 1
        return counts

    def majority_baseline(self, samples: List[VideoSample]) -> float:
        if not samples:
            return 0.0
        dist = self.get_class_distribution(samples)
        return max(dist.values()) / len(samples)


if __name__ == "__main__":
    print("=" * 60)
    print("真实运动识别训练数据集测试 (真实运镜曲线稠密采样)")
    print("=" * 60)

    ds = VideoMotionDataset()
    splits = ds.kfold_datasets(n_folds=5, seed=0)
    train, val = splits[0]
    print(f"\n[真实数据规模]")
    print(f"  真实运镜曲线: {len(LABELS)} 类 × {ds.n_points} 点 "
          f"= {len(ds._corpus)} 样本")
    print(f"  随机基线: {ds.RANDOM_BASELINE:.0%}")
    print(f"  多数类基线: {ds.majority_baseline(train):.0%}")
    print(f"\n[各类别示例]")
    for m in LABELS[:4]:
        s = next(x for x in train if x.category_name == LABEL_NAMES[m])
        print(f"  {s.category_name} ({m}): "
              f"desc=[{', '.join(f'{v:.2f}' for v in s.point)}]")
    print("\n" + "=" * 60)
    print("真实运动数据集测试通过!")
    print("=" * 60)
