# -*- coding: utf-8 -*-
"""
真实时序异常检测训练数据集 (v0.13.1) — 框架"流式监控/异常检测"主场的首块真实时序基准

纯真实数据, 无任何合成路径: 语料来自 Numenta Anomaly Benchmark (NAB) 中
**真实运维指标系列** (realAWSCloudwatch / realKnownCause / realTraffic,
均为真实生产系统/传感器采集), 已固化在本包 `raw/` 目录为静态 CSV, 运行时
离线读取 (不联网)。正常/异常窗口标签沿用 NAB 官方 `combined_labels.json`
标注的真实告警时间点, 非程序生成。

接入文件 (均为 NAB `real*`, 排除 artificial* 合成系列):
  - realKnownCause/machine_temperature_system_failure.csv   (真实设备温度, 系统故障)
  - realKnownCause/ambient_temperature_system_failure.csv   (真实环境温度, 系统故障)
  - realKnownCause/cpu_utilization_asg_misconfiguration.csv (真实 ASG CPU 误配置故障)
  - realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv         (真实 EC2 CPU 利用率)
  - realTraffic/TravelTime_387.csv                           (真实路段通行时间)

任务: 窗口级异常检测 (binary) — 把每条真实指标序列切成定长滑动窗口, 窗口若
与官方异常告警时段重叠则标 "anomaly", 否则 "normal"。这是监控/指标巡检的
核心能力 (非分类), 随机基线 50%。

样本编码 (与 Rust/MD/视频数据通路对齐):
- static_signal / input_signal: 窗口内真实统计描述符 (16 维 numeric 通路),
  从真实窗口值计算 (mean/std/zscore/差分/能量等), 非合成样本。
- target_pattern: 2 类监督输出模式 (normal / anomaly)。
- window: 该窗口的原始真实值序列 (供逐窗口度量 / 复核)。

复用 src.data.rust_coding.stratified_kfold 做分层 K 折 (保持 5 种子 × 5 折
评估约定), 分层键 = 二元正常/异常标签, 折间两类比例一致。
"""

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, List, Tuple

import numpy as np

from src.data.rust_coding import stratified_kfold

# ══ 数据目录 (src/data/metrics_ts/) ═══════════════════════════════
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW_DIR = os.path.join(_PKG_DIR, "metrics_ts", "raw")
_LABELS_PATH = os.path.join(_RAW_DIR, "combined_labels.json")

# 类别 (正常/异常)
LABELS = ["normal", "anomaly"]
LABEL_NAMES = {"normal": "正常", "anomaly": "异常"}

# 真实运维时序接入清单: (文件名, NAB labels 键, 上级类别, 中文说明)
METRIC_FILES = [
    ("machine_temperature_system_failure.csv", "realKnownCause",
     "realKnownCause/machine_temperature_system_failure.csv", "设备温度系统故障"),
    ("ambient_temperature_system_failure.csv", "realKnownCause",
     "realKnownCause/ambient_temperature_system_failure.csv", "环境温度系统故障"),
    ("cpu_utilization_asg_misconfiguration.csv", "realKnownCause",
     "realKnownCause/cpu_utilization_asg_misconfiguration.csv", "ASG 误配置 CPU"),
    ("ec2_cpu_utilization_24ae8d.csv", "realAWSCloudwatch",
     "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv", "EC2 实例 CPU 利用率"),
    ("TravelTime_387.csv", "realTraffic",
     "realTraffic/TravelTime_387.csv", "真实路段通行时间"),
]

# ══ 窗口化参数 ═══════════════════════════════════════════════════
DEFAULT_WINDOW = 40      # 每窗口真实采样点数
DEFAULT_STRIDE = 20      # 窗口滑动步长 (50% 重叠, 兼顾时序覆盖与样本量)
DEFAULT_SPAN = 40        # 官方异常点向两端扩展的采样跨度 (形成可疑时段)
DIM = 16                 # 网络维度 (numeric 通路/监督输出模式长度, 与特征维一致)


def _parse_ts(s: str) -> float:
    """真实时间戳 → 数值 epoch (秒)"""
    return (datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
            .timestamp())


def load_series(filename: str) -> List[Dict]:
    """读取单个真实指标 CSV → [{"t": epoch, "v": value}]"""
    path = os.path.join(_RAW_DIR, filename)
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        assert header.strip().lower().startswith("timestamp")
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts, val = line.split(",")
            rows.append({"t": _parse_ts(ts), "v": float(val)})
    return rows


def _load_anomaly_times() -> Dict[str, List[float]]:
    """解析 NAB 官方 combined_labels.json → {labels键: [epoch...]}"""
    import json
    with open(_LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {key: [_parse_ts(t) for t in times]
            for key, times in raw.items()}


def window_features(x: np.ndarray) -> np.ndarray:
    """真实窗口值 → 16 维统计描述符 (numeric 通路, 由真实值计算, 非合成)"""
    x = np.asarray(x, dtype=float)
    d = np.diff(x)
    mean, std = x.mean(), (x.std() + 1e-8)
    z_last = (x[-1] - mean) / std
    hi = float(np.mean(x > mean + 2 * std))
    lo = float(np.mean(x < mean - 2 * std))
    # 自相关 lag-1 (真实窗口内取整信号)
    if len(x) > 2:
        xm = x - mean
        lag1 = float((xm[:-1] * xm[1:]).mean() / (std * std + 1e-8))
    else:
        lag1 = 0.0
    return np.clip(np.array([
        mean, std, x.min(), x.max(), x[-1], x[0],           # 6
        np.median(x), (x.max() - x.min()) / std, z_last,    # 3
        float(np.mean(np.abs(d))), float(np.max(np.abs(d))),# 2
        hi, lo,                                            # 2
        float((x[-1] - x[0]) / std),                        # slop
        float(np.mean(x * x)),                             # energy
        lag1,                                              # lag-1
    ]), -10.0, 10.0)


def _is_anomaly_window(t0: float, t1: float, anom: List[float],
                       span: float, gap: float) -> int:
    """窗口 [t0, t1] 是否与官方异常时段重叠 (重叠→1, 否则→0)"""
    for a in anom:
        lo, hi = a - span * gap, a + span * gap
        if t0 < hi and t1 > lo:
            return 1
    return 0


@dataclass
class MetricSample:
    """单个真实时序窗口训练样本"""
    sample_id: str
    category: int            # 0=normal, 1=anomaly
    category_name: str
    input_signal: np.ndarray     # 16 维窗口统计描述符 (numeric 通路)
    target_pattern: np.ndarray   # 2 维监督输出模式
    static_signal: np.ndarray    # 同 input_signal
    window: np.ndarray           # 窗口原始真实值序列
    source: str                  # 来源指标 (真实运维系列)
    metadata: Dict = None        # 与 trainer 契约: _tokens_of 需 read .metadata (默认无 token)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"source": self.source}

    def multimodal_input(self) -> Dict:
        """输入字典: numeric = 16 维真实窗口描述符 (时序无文本模态)"""
        return {"numeric": self.static_signal}


class OperationalMetricsDataset:
    """真实时序异常检测训练数据集 (纯真实, 无合成样本)

    二元任务: 判断一个定长滑动窗口是否落在真实运维异常时段内。
    类别: 0=normal, 1=anomaly; 随机基线 50%。
    """

    CATEGORIES = list(LABELS)
    N_CLASSES = len(LABELS)               # 2
    RANDOM_BASELINE = 1.0 / len(LABELS)   # 50%

    def __init__(self, window: int = DEFAULT_WINDOW,
                 stride: int = DEFAULT_STRIDE,
                 span: int = DEFAULT_SPAN):
        assert window >= 10 and window > stride
        self.window = window
        self.stride = stride
        self.span = span
        self._corpus = self._build_corpus()

    def _build_corpus(self) -> List[Dict]:
        """全部真实运维序列 → 定长滑动窗口语料 (每窗口一个真实样本)"""
        anom_map = _load_anomaly_times()
        corpus = []
        for (fname, _cat, labels_key, _desc) in METRIC_FILES:
            series = load_series(fname)
            if len(series) < self.window:
                continue
            # 时间步进 (中位采样间隔, 秒) 作为异常跨度换算
            gaps = np.diff([s["t"] for s in series][: min(64, len(series))])
            gap = float(np.median(gaps)) if len(gaps) else 1.0
            anom = anom_map.get(labels_key, [])
            for start in range(0, len(series) - self.window + 1, self.stride):
                seg = series[start: start + self.window]
                vals = np.array([s["v"] for s in seg])
                t0, t1 = seg[0]["t"], seg[-1]["t"]
                label = _is_anomaly_window(t0, t1, anom, self.span, gap)
                corpus.append({
                    "label": "anomaly" if label else "normal",
                    "values": vals.astype(float),
                    "t0": t0, "t1": t1,
                    "gid": len(corpus),
                    "source": labels_key,
                })
        return corpus

    def _make_target_pattern(self, category: int) -> np.ndarray:
        # 监督输出模式长度为 DIM (16), 维度 0=normal, 1=anomaly (与 trainer 契约一致)
        pattern = np.ones(DIM) * 0.05
        pattern[category] = 0.7
        return np.clip(pattern, 0.0, 1.0)

    def _to_sample(self, s: Dict) -> MetricSample:
        category = LABELS.index(s["label"])
        feats = window_features(s["values"])
        return MetricSample(
            sample_id=f"{self.window}w_{s['gid']:05d}",
            category=category,
            category_name=LABEL_NAMES[s["label"]],
            input_signal=feats,
            target_pattern=self._make_target_pattern(category),
            static_signal=feats,
            window=s["values"],
            source=s["source"],
        )

    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """插件接口对齐 (RustCodingPlugin/VideoMakingPlugin.training_data()):
        返回 (特征矩阵 X, 二元正常/异常标签 y)。X 每行 = 一个真实窗口描述符。
        """
        X = np.stack([window_features(s["values"]) for s in self._corpus])
        y = np.array([LABELS.index(s["label"]) for s in self._corpus])
        return X, y

    def kfold_datasets(self, n_folds: int = 5, seed: int = 0,
                       ) -> List[Tuple[List[MetricSample],
                                       List[MetricSample]]]:
        """分层 K 折交叉验证划分 (二元标签 normal/anomaly 分层, 5 种子 × 5 折)"""
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

    def split_stream(self, train_ratio: float = 0.7,
                     seed: int = 0) -> Tuple[List[MetricSample],
                                             List[MetricSample]]:
        """时序在线划分: 按时间顺序前 70% 训练 / 后 30% 验证 (P1 在线持续学习)

        训练 = 时序流前段 (在线看到的是历史), 验证 = 未来段 —— 检验在非平稳
        水库 + 持久输出头在真实时序流上的漂移/稳定性。窗口按全局时间排序。
        """
        ordered = sorted(self._corpus,
                         key=lambda s: (s["source"], s["t0"], s["gid"]))
        n = len(ordered)
        cut = int(n * train_ratio)
        return ([self._to_sample(s) for s in ordered[:cut]],
                [self._to_sample(s) for s in ordered[cut:]])

    def get_class_distribution(self, samples: List[MetricSample]) -> Dict:
        counts = {m: 0 for m in self.CATEGORIES}
        for s in samples:
            counts[self.CATEGORIES[s.category]] += 1
        return counts

    def majority_baseline(self, samples: List[MetricSample]) -> float:
        if not samples:
            return 0.0
        dist = self.get_class_distribution(samples)
        return max(dist.values()) / len(samples)


if __name__ == "__main__":
    print("=" * 60)
    print("真实时序异常检测数据集测试 (NAB 真实运维指标, 纯真实无合成)")
    print("=" * 60)

    ds = OperationalMetricsDataset()
    print(f"\n[真实数据规模]")
    print(f"  真实运维序列: {len(METRIC_FILES)} 条 × 窗口={ds.window}, "
          f"stride={ds.stride} → {len(ds._corpus)} 窗口")
    samples = ds._to_sample(ds._corpus[0])
    tr, va = ds.split_stream(train_ratio=0.7)
    print(f"  在线时序划分: 训练 {len(tr)} / 验证 {len(va)}")
    print(f"  验证分布: {ds.get_class_distribution(va)}")
    print(f"  随机基线: {ds.RANDOM_BASELINE:.0%}")
    print(f"  多数类(验证)基线: {ds.majority_baseline(va):.0%}")
    print(f"  training_data: X={ds.training_data()[0].shape} "
          f"y={ds.training_data()[1].shape}  anomaly占比="
          f"{ds.training_data()[1].mean():.1%}")

    print("\n[窗口特征示例]")
    for s in ([next(x for x in tr if x.category == 0)] +
              [next(x for x in tr if x.category == 1)]):
        print(f"  {s.category_name}: win=[{', '.join('%.1f' % v for v in s.window[:6])}...] "
              f"feat1={s.static_signal[0]:.3f} "
              f"feat={', '.join(f'{v:.2f}' for v in s.static_signal[:6])}")

    splits = ds.kfold_datasets(n_folds=5, seed=0)
    print(f"\n[K 折] 5 种子 × 5 折, 第 0 折训练 {len(splits[0][0])} / "
          f"验证 {len(splits[0][1])}")
    print("\n真实时序数据集测试通过!")