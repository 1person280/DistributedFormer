"""
DistributedFormer 训练器 v5.3 (v0.8.2 P1: 修端到端权重更新)

改进点 (v5.3):
- 持久输出头: 注入-恢复启发式 (临时改 gain/threshold 再还原,
  学习信号不累积) 替换为持久可训练的线性 softmax 输出头,
  权重跨样本/epoch 持久累积
- w_in 向量化更新: 原规则 `w_in += lr * error * mean(input)`
  的均值池化使所有单元收到无差异更新; 现按每单元感受野投影
  `w_in += lr * error * (receptive @ input)`, 各单元更新分化

保留 (v5.2):
- 网络规模: depth=2 (4,368单元/层) + num_think_layers=1 => ~4,400总单元
- 发放率/抑制/思考层对齐损失作为网络塑形信号
- 训练/验证循环 / 权重保存/加载 / 训练报告生成
"""

import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distributedformer.core.distributedformer import DistributedFormer
from distributedformer.data.real_dataset import TrainingSample, RustCodingTrainingDataset


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    loss: float
    spike_rate_loss: float
    target_activation_loss: float
    inhibition_loss: float
    think_align_loss: float = 0.0
    accuracy: float = 0.0
    avg_spikes: float = 0.0
    stdp_ltp: int = 0
    stdp_ltd: int = 0
    learning_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SpikeSupervisedLoss:
    """脉冲神经网络监督损失函数 (v5.2: 增加思考层对齐损失)"""
    
    def __init__(self, dim: int = 16, alpha: float = 1.0, 
                 beta: float = 2.0, gamma: float = 1.5,
                 delta: float = 0.5,
                 n_classes: int = 5):  # delta: 思考层对齐权重
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.n_classes = n_classes
    
    def compute(self, output_pattern: np.ndarray, 
                target_pattern: np.ndarray,
                think_pattern: np.ndarray,
                actual_spike_count: int,
                target_spike_count: int = 3) -> Tuple[float, Dict]:
        # 1. 发放率损失
        rate_error = (actual_spike_count - target_spike_count) ** 2
        rate_loss = rate_error / max(target_spike_count ** 2, 1)
        
        # 2. 目标维度激活损失 (输出层前 n_classes 维)
        n = self.n_classes
        target_dims = target_pattern[:n]
        actual_dims = output_pattern[:n]
        
        activation_loss = 0.0
        for i in range(n):
            if target_dims[i] > 0.3:
                gap = max(0, target_dims[i] - actual_dims[i])
                activation_loss += gap ** 2
            else:
                excess = max(0, actual_dims[i] - 0.1)
                activation_loss += excess ** 2
        
        # 3. 非目标维度抑制损失
        non_target = output_pattern[4:]
        inhibition_loss = np.mean(np.maximum(0, non_target - 0.15) ** 2)
        
        # 4. 思考层表征对齐损失 (v5.2新增)
        # 希望思考层中对应类别的神经元群体激活更高
        think_align_loss = 0.0
        for i in range(self.n_classes):
            if target_pattern[i] > 0.3:
                # 目标类别对应维度的思考层激活应较高
                think_align_loss += max(0, 0.5 - think_pattern[i]) ** 2
            else:
                # 非目标维度的思考层激活应较低
                think_align_loss += max(0, think_pattern[i] - 0.2) ** 2
        
        total = (self.alpha * rate_loss + 
                 self.beta * activation_loss + 
                 self.gamma * inhibition_loss +
                 self.delta * think_align_loss)
        
        return total, {
            "rate_loss": float(rate_loss),
            "activation_loss": float(activation_loss),
            "inhibition_loss": float(inhibition_loss),
            "think_align_loss": float(think_align_loss),
            "total": float(total)
        }
    
    def compute_accuracy(self, output_pattern: np.ndarray, 
                         target_category: int,
                         threshold: float = 0.3) -> bool:
        top = output_pattern[:self.n_classes]
        predicted = int(np.argmax(top))
        return predicted == target_category and top[predicted] >= threshold


class PersistentOutputHead:
    """持久输出头 (v0.8.2 P1): 在线线性 softmax 头, 权重跨样本/epoch 持久

    替换注入-恢复启发式: 后者每步临时改 gain/threshold 再还原,
    学习信号无法累积; 本头是真正的可训练持久参数, 在线 SGD 更新,
    带 running z-score 标准化与 L2 正则。
    """

    def __init__(self, n_features: int, n_classes: int = 5,
                 lr: float = 0.02, l2: float = 1e-3, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_features = n_features
        self.n_classes = n_classes
        self.W = rng.randn(n_classes, n_features + 1) * 0.01  # +1 偏置
        self.lr = lr
        self.l2 = l2
        # running 标准化统计 (在线, 无需预先扫全数据集)
        self.mean_ = np.zeros(n_features)
        self.m2_ = np.ones(n_features) * 1e-8  # E[x^2]
        self.n_seen = 0

    def _standardize(self, x: np.ndarray, update: bool = False) -> np.ndarray:
        if update:
            self.n_seen += 1
            alpha = max(1.0 / self.n_seen, 0.01)
            self.mean_ += alpha * (x - self.mean_)
            self.m2_ += alpha * (x * x - self.m2_)
        std = np.sqrt(np.maximum(self.m2_ - self.mean_ ** 2, 1e-8))
        return (x - self.mean_) / std

    def _with_bias(self, x: np.ndarray) -> np.ndarray:
        return np.append(x, 1.0)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.W @ self._with_bias(x)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(self._standardize(x))))

    def update(self, x: np.ndarray, y: int) -> float:
        """单样本在线 SGD (交叉熵), 返回该样本损失"""
        xs = self._standardize(x, update=True)
        xb = self._with_bias(xs)
        logits = self.W @ xb
        logits -= logits.max()
        prob = np.exp(logits)
        prob /= prob.sum()
        loss = -np.log(max(prob[y], 1e-12))
        # 梯度: (prob - onehot) ⊗ xb
        grad = prob.copy()
        grad[y] -= 1.0
        self.W -= self.lr * (np.outer(grad, xb) + self.l2 * self.W)
        return float(loss)


class DFTrainer:
    """DistributedFormer 训练器 v5.3"""
    
    def __init__(self, 
                 depth: int = 2,
                 dim: int = 16,
                 learning_rate: float = 0.008,
                 super_modulation: float = 0.25,
                 think_lr: float = 0.003,  # 思考层学习率 (较低)
                 use_think_supervision: bool = True,  # 是否启用思考层监督
                 n_classes: int = 5,  # 真实类别数 (rustc 错误族)
                 head_lr: float = 0.02,  # 持久输出头学习率
                 fwd_steps: int = 4):  # 每样本前向步数 (水库充分演化)
        self.depth = depth
        self.dim = dim
        self.lr = learning_rate
        self.think_lr = think_lr
        self.super_mod = super_modulation  # 保留参数 (v5.3 注入式监督已移除)
        self.use_think_supervision = use_think_supervision
        self.n_classes = n_classes
        self.head_lr = head_lr
        self.fwd_steps = fwd_steps
        self.category_names = list(RustCodingTrainingDataset.CATEGORIES)
        
        # 网络 (训练模式: 跳过KV查询以加速)
        # depth>=2 时用单层大思考层; depth=1 时用3层小思考层
        num_think_layers = 1 if depth >= 2 else 3
        self.df = DistributedFormer(
            depth=depth, 
            dim=dim, 
            kv_capacity=10000, 
            num_think_layers=num_think_layers,
            training_mode=True
        )
        self.df.enable_learning(True)
        
        # 损失函数: 若不用思考层监督，delta=0
        loss_delta = 0.5 if use_think_supervision else 0.0
        self.loss_fn = SpikeSupervisedLoss(dim=dim, delta=loss_delta,
                                           n_classes=self.n_classes)
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # 持久输出头 (v0.8.2 P1): 首个样本时按特征维度惰性构建
        self.head: PersistentOutputHead = None
        
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        self.patience = 15  # 增加耐心值 (大网络需要更多epoch)
        self.patience_counter = 0
    
    def _forward_features(self, inputs: Dict) -> Tuple[np.ndarray, int]:
        """多步前向并提取水库特征 (v5.3: 供持久输出头消费)

        特征 = [思考层各单元状态幅值 | 输出模式 | 思考层聚合模式]。
        """
        self.df.reset_state()
        last_spikes = []
        for _ in range(self.fwd_steps):
            last_spikes = self.df.step(inputs)
        out = self.df.get_output_pattern()
        think = self.df.get_think_layer_pattern()
        layer = self.df.think_layers[0]
        if hasattr(layer, "_vec_state") and layer.N > 0:
            reservoir = np.abs(layer._vec_state).mean(axis=1)
        else:
            reservoir = np.array([abs(u.state).mean()
                                  for u in layer._all_units_cache])
        return np.concatenate([reservoir, out, think]), len(last_spikes)
    
    def _sample_inputs(self, sample: TrainingSample) -> Dict:
        """P0 双模态: 语法特征走 numeric 通路, 代码原文走 text 通路"""
        return (sample.multimodal_input()
                if sample.static_signal is not None
                else {"numeric": sample.input_signal})
    
    @staticmethod
    def _unit_receptive(unit) -> np.ndarray:
        """单元感受野投影 (惰性构建, 与 SpikingUnit.step 同种子)"""
        if unit._receptive is None:
            import zlib
            seed = zlib.crc32(unit.unit_id.encode("utf-8"))
            unit._receptive = (np.random.RandomState(seed)
                               .randn(unit.dim) / np.sqrt(unit.dim))
        return unit._receptive
    
    def _update_weights_supervised(self, layer_input: np.ndarray, 
                                    target_pattern: np.ndarray,
                                    output_pattern: np.ndarray,
                                    think_pattern: np.ndarray,
                                    lr: float = 0.005) -> None:
        """发放率监督学习 (v5.3: w_in 按每单元感受野投影向量化)

        原缺陷: `w_in += lr * error * np.mean(layer_input)` 的均值池化
        使所有单元收到无差异更新 (error 相同的单元增量和完全一致)。
        现按 `receptive @ layer_input` 投影, 各单元对同一输入信号
        收到方向/幅度分化的梯度。
        """
        # 1. 输出层权重更新
        for i, unit in enumerate(self.df.output_units):
            error = target_pattern[i] - output_pattern[i]
            
            if abs(error) > 0.05:
                eff_input = float(self._unit_receptive(unit) @ layer_input)
                unit.w_in += lr * error * eff_input
                unit.w_in = np.clip(unit.w_in, -1.0, 1.0)
                
                unit.threshold = np.clip(
                    unit.threshold - lr * error * 0.05, 0.2, 1.5
                )
                
                unit.b_in += lr * error * 0.1
                unit.b_in = np.clip(unit.b_in, -0.5, 0.5)
        
        # 2. 思考层权重更新 (仅当启用时)
        if self.use_think_supervision:
            for layer in self.df.think_layers:
                all_units = layer._all_units_cache
                n_units = len(all_units)
                group_size = max(1, n_units // self.n_classes)
                
                for cat in range(self.n_classes):
                    error = target_pattern[cat] - think_pattern[cat]
                    if abs(error) > 0.05:
                        start_idx = cat * group_size
                        end_idx = min((cat + 1) * group_size, n_units)
                        for idx in range(start_idx, end_idx):
                            unit = all_units[idx]
                            eff_input = float(
                                self._unit_receptive(unit) @ layer_input)
                            unit.w_in += self.think_lr * error * eff_input
                            unit.w_in = np.clip(unit.w_in, -1.0, 1.0)
    
    def train_step(self, sample: TrainingSample) -> Tuple[float, bool, Dict]:
        """单步训练 (v5.3: 持久输出头 + 向量化 w_in 更新)

        v5.2 的注入-恢复启发式 (临时改 gain/threshold 再还原,
        学习信号不累积) 已移除; 分类学习由持久输出头承担,
        网络内部 w_in 仍按感受野投影监督更新。
        """
        # 前向传播 (P0 双模态) + 水库特征提取
        features, n_spikes = self._forward_features(self._sample_inputs(sample))
        
        # 惰性构建持久输出头
        if self.head is None:
            self.head = PersistentOutputHead(
                n_features=len(features), n_classes=self.n_classes,
                lr=self.head_lr, seed=0)
        
        # 先预测 (更新前), 再在线更新 — 诚实反映当前泛化状态
        pred = self.head.predict(features)
        is_correct = (pred == sample.category)
        ce_loss = self.head.update(features, sample.category)
        
        # 网络塑形损失 (发放率/激活/抑制/思考层对齐, 仅作监控与 w_in 更新依据)
        output_pattern = self.df.get_output_pattern()
        think_pattern = self.df.get_think_layer_pattern()
        shape_loss, loss_details = self.loss_fn.compute(
            output_pattern=output_pattern,
            target_pattern=sample.target_pattern,
            think_pattern=think_pattern,
            actual_spike_count=n_spikes,
            target_spike_count=2 + int(np.max(sample.target_pattern) * 3)
        )
        
        # 监督权重更新 (输出层 + 思考层, P0: 用双模态有效特征)
        update_signal = (sample.static_signal if sample.static_signal is not None
                         else sample.input_signal)
        self._update_weights_supervised(
            update_signal,
            sample.target_pattern,
            output_pattern,
            think_pattern,
            lr=self.lr
        )
        
        self.global_step += 1
        
        loss = ce_loss + 0.1 * shape_loss
        return loss, is_correct, {
            "loss": loss,
            "ce_loss": ce_loss,
            **loss_details,
            "output_spikes": n_spikes,
            "predicted": pred,
            "is_correct": is_correct
        }
    
    def train_epoch(self, samples: List[TrainingSample]) -> Dict:
        """训练一个epoch"""
        import random
        random.shuffle(samples)
        
        total_loss = 0.0
        correct = 0
        total_spikes = 0
        
        for sample in samples:
            loss, is_correct, details = self.train_step(sample)
            total_loss += loss
            if is_correct:
                correct += 1
            total_spikes += details["output_spikes"]
        
        n = len(samples)
        stdp_stats = self.df.get_stdp_stats()
        
        metrics = {
            "epoch": self.epoch,
            "loss": total_loss / n,
            "accuracy": correct / n,
            "avg_spikes": total_spikes / n,
            "stdp_ltp": stdp_stats["total_ltp"],
            "stdp_ltd": stdp_stats["total_ltd"],
            "stdp_weight_change": stdp_stats["total_weight_change"],
            "samples": n
        }
        
        self.train_history.append(metrics)
        return metrics
    
    def evaluate(self, samples: List[TrainingSample]) -> Dict:
        """验证集评估 (v5.3: 持久输出头预测, 双模态前向, 不更新任何权重)"""
        was_learning = self.df.learning_enabled
        self.df.enable_learning(False)
        
        total_loss = 0.0
        correct = 0
        total_spikes = 0
        
        per_class_correct = {c: 0 for c in range(self.n_classes)}
        per_class_total = {c: 0 for c in range(self.n_classes)}
        
        for sample in samples:
            features, n_spikes = self._forward_features(self._sample_inputs(sample))
            
            if self.head is not None:
                pred = self.head.predict(features)
                is_correct = (pred == sample.category)
            else:
                is_correct = False
            
            output_pattern = self.df.get_output_pattern()
            think_pattern = self.df.get_think_layer_pattern()
            loss, _ = self.loss_fn.compute(
                output_pattern=output_pattern,
                target_pattern=sample.target_pattern,
                think_pattern=think_pattern,
                actual_spike_count=n_spikes,
                target_spike_count=2 + int(np.max(sample.target_pattern) * 3)
            )
            total_loss += loss
            total_spikes += n_spikes
            
            if is_correct:
                correct += 1
                per_class_correct[sample.category] += 1
            per_class_total[sample.category] += 1
        
        self.df.enable_learning(was_learning)
        
        n = len(samples)
        class_acc = {}
        for cat in range(self.n_classes):
            if per_class_total[cat] > 0:
                class_acc[cat] = per_class_correct[cat] / per_class_total[cat]
            else:
                class_acc[cat] = 0.0
        
        metrics = {
            "epoch": self.epoch,
            "loss": total_loss / n,
            "accuracy": correct / n,
            "avg_spikes": total_spikes / n,
            "class_accuracy": class_acc,
            "samples": n
        }
        
        self.val_history.append(metrics)
        return metrics
    
    def train(self, train_samples: List[TrainingSample],
              val_samples: List[TrainingSample],
              epochs: int = 30,
              save_dir: str = "training/checkpoints") -> Dict:
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 计算总单元数
        total_units = len(self.df._all_units)
        
        print(f"\n{'='*70}")
        print(f"  DistributedFormer v5.3 全面训练 (持久输出头 + 向量化 w_in)")
        print(f"{'='*70}")
        print(f"  网络配置: 深度={self.depth}, 思考层={self.df.num_think_layers}")
        print(f"  总单元数: {total_units:,} (目标~4,400)")
        print(f"  训练样本: {len(train_samples)}")
        print(f"  验证样本: {len(val_samples)}")
        print(f"  Epochs: {epochs}")
        print(f"  学习率: {self.lr} (输出层) / {self.think_lr} (思考层)")
        print(f"  监督调制: {self.super_mod}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            train_metrics = self.train_epoch(train_samples)
            val_metrics = self.evaluate(val_samples)
            
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2%} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2%} | "
                  f"Avg Spikes: {val_metrics['avg_spikes']:.1f}")
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["accuracy"]
                self.save_weights(f"{save_dir}/best_model.npz")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\n  早停触发 (patience={self.patience})")
                break
        
        elapsed = time.time() - start_time
        
        final_val = self.val_history[-1]
        
        self.save_weights(f"{save_dir}/final_model.npz")
        
        summary = {
            "total_epochs": self.epoch + 1,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_acc,
            "final_val_accuracy": final_val["accuracy"],
            "final_val_loss": final_val["loss"],
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "elapsed_seconds": elapsed,
            "class_accuracy": final_val["class_accuracy"],
            "total_units": total_units
        }
        
        print(f"\n{'='*70}")
        print(f"  训练完成")
        print(f"{'='*70}")
        print(f"  耗时: {elapsed:.1f}秒")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")
        print(f"  最佳验证准确率: {self.best_val_acc:.2%}")
        print(f"  最终验证准确率: {final_val['accuracy']:.2%}")
        print(f"  各类别准确率:")
        for cat, acc in final_val["class_accuracy"].items():
            name = self.category_names[cat] if cat < len(self.category_names) else str(cat)
            print(f"    {name:12s}: {acc:.2%}")
        
        return summary
    
    def save_weights(self, path: str) -> None:
        """保存网络权重 (标量权重版本)"""
        weights = {}
        all_units = self.df._all_units
        
        for unit in all_units:
            weights[unit.unit_id] = {
                "w_in": unit.w_in,
                "b_in": unit.b_in,
                "w_state": unit.w_state,
                "w_out": unit.w_out,
                "b_out": unit.b_out,
                "w_attn": unit.w_attn,
                "b_attn": unit.b_attn,
                "threshold": unit.threshold,
                "gain": unit.gain,
                "outgoing": dict(unit.outgoing)
            }
        
        weights["_kv_query_w"] = self.df.kv_stack.query_w
        weights["_kv_key_w"] = self.df.kv_stack.key_w
        weights["_kv_value_w"] = self.df.kv_stack.value_w
        
        # 持久输出头 (v0.8.2)
        if self.head is not None:
            weights["_head"] = {
                "W": self.head.W, "mean": self.head.mean_,
                "m2": self.head.m2_, "n_seen": int(self.head.n_seen),
            }
        
        np.savez(path, **{k: json.dumps(v, default=self._json_serialize) 
                          for k, v in weights.items()})
        
    def load_weights(self, path: str) -> None:
        """加载网络权重 (标量权重版本)"""
        data = np.load(path, allow_pickle=True)
        units_map = {u.unit_id: u for u in self.df._all_units}
        
        for key in data.files:
            if key.startswith("_"):
                if key == "_kv_query_w":
                    self.df.kv_stack.query_w = data[key].item()
                elif key == "_kv_key_w":
                    self.df.kv_stack.key_w = data[key].item()
                elif key == "_kv_value_w":
                    self.df.kv_stack.value_w = data[key].item()
                elif key == "_head":
                    h = json.loads(data[key].item())
                    if self.head is None:
                        n_feat = len(h["W"][0]) - 1
                        self.head = PersistentOutputHead(
                            n_features=n_feat, n_classes=self.n_classes)
                    self.head.W = np.array(h["W"])
                    self.head.mean_ = np.array(h["mean"])
                    self.head.m2_ = np.array(h["m2"])
                    self.head.n_seen = h["n_seen"]
                continue
            
            if key in units_map:
                unit = units_map[key]
                w = json.loads(data[key].item())
                unit.w_in = w["w_in"]
                unit.b_in = w["b_in"]
                unit.w_state = w["w_state"]
                unit.w_out = w["w_out"]
                unit.b_out = w["b_out"]
                unit.w_attn = w["w_attn"]
                unit.b_attn = w.get("b_attn", unit.b_attn)
                unit.threshold = w["threshold"]
                unit.gain = w["gain"]
                unit.outgoing = w["outgoing"]
    
    def _json_serialize(self, obj):
        """JSON序列化辅助"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        raise TypeError(f"不可序列化: {type(obj)}")
    
    def generate_training_report(self) -> str:
        """生成训练报告"""
        n = min(len(self.train_history), len(self.val_history))
        total_units = len(self.df._all_units)
        report = f"""# DistributedFormer v5.3 全面训练报告

## 训练配置
| 参数 | 值 |
|------|-----|
| 分形深度 | {self.depth} |
| 信号维度 | {self.dim} |
| 思考层数 | {self.df.num_think_layers} |
| 总单元数 | {total_units:,} |
| 总参数 | ~{total_units * 16 // 1000}K |
| 学习率 (输出层) | {self.lr} |
| 学习率 (思考层) | {self.think_lr} |
| 监督调制 | {self.super_mod} |

## 训练结果
| 指标 | 值 |
|------|-----|
| 总Epoch数 | {n} |
| 最佳验证损失 | {self.best_val_loss:.4f} |
| 最佳验证准确率 | {self.best_val_acc:.2%} |
| 最终验证准确率 | {self.val_history[-1]['accuracy']:.2%} |

## 训练曲线
"""
        if self.train_history:
            report += "\n| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |\n"
            report += "|-------|------------|-----------|----------|---------|\n"
            for i in range(n):
                t = self.train_history[i]
                v = self.val_history[i]
                report += f"| {i+1:5d} | {t['loss']:10.4f} | {t['accuracy']:9.2%} | {v['loss']:8.4f} | {v['accuracy']:7.2%} |\n"
        
        report += f"\n## STDP 学习统计\n"
        stdp = self.df.get_stdp_stats()
        report += f"- LTP (增强): {stdp['total_ltp']} 次\n"
        report += f"- LTD (抑制): {stdp['total_ltd']} 次\n"
        report += f"- 总权重变化: {stdp['total_weight_change']:.4f}\n"
        report += f"- 平均权重变化: {stdp['avg_weight_change']:.6f}\n"
        
        report += f"\n---\n*训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report


if __name__ == "__main__":
    print("=" * 60)
    print("训练器自测试")
    print("=" * 60)
    
    dataset = RustCodingTrainingDataset(dim=16)
    train, val = dataset.generate_dataset(train_ratio=0.75)
    
    # v5.2: depth=2, num_think_layers=1 => ~4400单元
    trainer = DFTrainer(depth=2, dim=16, learning_rate=0.008)
    
    print(f"\n网络总单元数: {len(trainer.df._all_units):,}")
    
    print("\n[快速训练2个epoch测试]")
    for epoch in range(2):
        metrics = trainer.train_epoch(train)
        val_metrics = trainer.evaluate(val)
        print(f"  Epoch {epoch+1}: Train Acc={metrics['accuracy']:.2%}, "
              f"Val Acc={val_metrics['accuracy']:.2%}")
    
    print("\n[测试权重保存/加载]")
    os.makedirs("training/checkpoints", exist_ok=True)
    trainer.save_weights("training/checkpoints/test_model.npz")
    trainer.load_weights("training/checkpoints/test_model.npz")
    print("  权重保存/加载成功")
    
    print("\n" + "=" * 60)
    print("训练器测试通过!")
    print("=" * 60)
