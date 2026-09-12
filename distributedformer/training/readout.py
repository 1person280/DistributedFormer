"""
训练方法学验证: 线性读出层 (reservoir computing 范式)

动机: v5.2 的"注入-前向-恢复"监督规则在消融实验中未超过随机基线。
本模块提供一种可验证的学习路径: 冻结脉冲网络 (SNN 作为固定特征
提取器), 仅训练线性 softmax 读出层。这是 reservoir computing 的
标准做法, 可明确判断网络内部表征是否携带类别信息。

验证协议:
- 合成 4 分类股票数据 (normal/uptrend/downtrend/anomaly)
- 多随机种子, 训练/验证集分离
- 对照随机基线 (25%) 与多数类基线
"""

import time
from typing import Dict, List, Tuple

import numpy as np

from distributedformer.core.distributedformer import DistributedFormer
from distributedformer.training.data_generator import StockTrainingDataset


class PatternExtractor:
    """用冻结的脉冲网络把样本映射为特征向量"""

    def __init__(self, depth: int = 1, dim: int = 16, seed: int = 0):
        self.dim = dim
        rng = np.random.RandomState(seed)
        self.net = DistributedFormer(
            depth=depth, dim=dim, num_think_layers=1, training_mode=True
        )
        # 固定随机性: 关闭自发脉冲噪声对特征的干扰, 保留确定性前向
        self.net.enable_learning(False)
        for u in self.net._all_units:
            u.spontaneous_rate = 0.0
        self.rng = rng

    def features(self, input_signal: np.ndarray, steps: int = 4) -> np.ndarray:
        """多步前向, 从水库内部状态直接读出 (reservoir computing 标准)

        特征 = [皮层各单元状态幅值 | 输出模式 | 思考层聚合模式]。
        皮层状态 (N 维) 保留完整表征; 仅用 16 维输出模式会丢失类别信息
        (v0.5.0 实验确认), 这正是历史消融停留在随机水平的根因之一。
        """
        self.net.reset_state()
        for _ in range(steps):
            self.net.step({"numeric": input_signal})
        out = self.net.get_output_pattern()
        think = self.net.get_think_layer_pattern()
        layer = self.net.think_layers[0]
        if hasattr(layer, "_vec_state") and layer.N > 0:
            reservoir = np.abs(layer._vec_state).mean(axis=1)
        else:
            reservoir = np.array(
                [abs(u.state).mean() for u in layer._all_units_cache]
            )
        return np.concatenate([reservoir, out, think])


class LinearReadout:
    """线性 softmax 读出层 (纯 numpy, 带 L2 正则)"""

    def __init__(self, n_features: int, n_classes: int = 4, lr: float = 0.1,
                 l2: float = 1e-3, epochs: int = 300, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_classes = n_classes
        self.W = rng.randn(n_classes, n_features + 1) * 0.01
        self.lr, self.l2, self.epochs = lr, l2, epochs
        self.mean_, self.scale_ = None, None

    def _fit_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((len(X), 1))])

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return self._fit_bias(X) @ self.W.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearReadout":
        # 标准化 (各维 z-score), 读出层学习更稳定
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        X = (X - self.mean_) / self.scale_
        Xb = self._fit_bias(X)
        n = len(Xb)
        for _ in range(self.epochs):
            logits = Xb @ self.W.T
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            grad = (prob - np.eye(self.n_classes)[y])  # (n, C)
            self.W -= self.lr * (grad.T @ Xb) / n + self.l2 * self.W
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is not None:
            X = (X - self.mean_) / self.scale_
        return self.predict_logits(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())


def run_validation(seed: int = 0, samples_per_class: int = 40,
                   depth: int = 1, verbose: bool = False) -> Dict:
    """单种子验证: 返回读出层/随机基线/多数类基线准确率"""
    dataset = StockTrainingDataset(dim=16, seed=seed)
    train, val = dataset.generate_dataset(
        samples_per_class=samples_per_class, train_ratio=0.75
    )

    extractor = PatternExtractor(depth=depth, seed=seed)
    t0 = time.time()
    X_train = np.stack([extractor.features(s.input_signal) for s in train])
    X_val = np.stack([extractor.features(s.input_signal) for s in val])
    y_train = np.array([s.category for s in train])
    y_val = np.array([s.category for s in val])
    extract_sec = time.time() - t0

    readout = LinearReadout(n_features=X_train.shape[1], seed=seed).fit(X_train, y_train)
    val_acc = readout.accuracy(X_val, y_val)
    train_acc = readout.accuracy(X_train, y_train)
    majority = float(np.bincount(y_train).max() / len(y_train))

    if verbose:
        print(f"  seed={seed} train_acc={train_acc:.2%} val_acc={val_acc:.2%} "
              f"(随机基线 25%, 多数类 {majority:.2%}, 特征提取 {extract_sec:.1f}s)")

    return {
        "seed": seed,
        "samples_per_class": samples_per_class,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "random_baseline": 0.25,
        "majority_baseline": majority,
        "beats_random": val_acc > 0.25 + 0.05,
        "extract_seconds": extract_sec,
    }


if __name__ == "__main__":
    print("训练方法学验证 (reservoir 读出层)")
    for seed in (0, 1, 2):
        run_validation(seed=seed, verbose=True)
