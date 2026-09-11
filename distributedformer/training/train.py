"""
DistributedFormer 第一次全面训练 - 入口脚本

执行完整训练流程:
1. 生成训练/验证数据集
2. 初始化训练器
3. 执行多epoch训练
4. 保存权重和报告
5. 生成可视化图表
"""

import sys
import os
import time
import json
from typing import Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributedformer.training.data_generator import StockTrainingDataset
from distributedformer.training.trainer import DFTrainer
import numpy as np


def run_first_training(
    samples_per_class: int = 200,
    epochs: int = 30,
    depth: int = 2,
    dim: int = 16,
    learning_rate: float = 0.008,
    super_modulation: float = 0.25,
    seed: int = 42
) -> Dict:
    """
    执行第一次全面训练
    
    Args:
        samples_per_class: 每类训练样本数 (4类: normal, uptrend, downtrend, anomaly)
        epochs: 训练轮数
        depth: 分形深度 (2 = 4,368单元 / 69K参数)
        dim: 信号维度
        learning_rate: 监督学习率
        super_modulation: 监督信号调制强度
        seed: 随机种子
    
    Returns:
        训练总结字典
    """
    
    # ═══════════════════════════════════════════════════════════════
    # 1. 生成数据集
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [1/5] 生成训练数据集")
    print("="*70)
    
    dataset = StockTrainingDataset(dim=dim, seed=seed)
    train_samples, val_samples = dataset.generate_dataset(
        samples_per_class=samples_per_class,
        train_ratio=0.8
    )
    
    train_dist = dataset.get_class_distribution(train_samples)
    val_dist = dataset.get_class_distribution(val_samples)
    
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  训练集分布: {train_dist}")
    print(f"  验证集分布: {val_dist}")
    
    # ═══════════════════════════════════════════════════════════════
    # 2. 初始化训练器
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [2/5] 初始化训练器")
    print("="*70)
    
    trainer = DFTrainer(
        depth=depth,
        dim=dim,
        learning_rate=learning_rate,
        super_modulation=super_modulation
    )
    
    # 网络规模信息
    from distributedformer.core.distributedformer import calculate_scale
    scale = calculate_scale(depth)
    print(f"  分形深度: {depth}")
    print(f"  基础单元数: {scale['base_units']:,}")
    print(f"  总参数: {scale['total_params']:,}")
    print(f"  学习率: {learning_rate}")
    print(f"  监督调制: {super_modulation}")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. 执行训练
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [3/5] 开始训练")
    print("="*70)
    
    save_dir = "training/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)
    
    summary = trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=epochs,
        save_dir=save_dir
    )
    
    # ═══════════════════════════════════════════════════════════════
    # 4. 保存结果
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [4/5] 保存训练结果")
    print("="*70)
    
    # 保存训练总结
    summary_path = "reports/training_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"  训练总结: {summary_path}")
    
    # 保存训练报告
    report = trainer.generate_training_report()
    report_path = "reports/training_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  训练报告: {report_path}")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. 生成可视化
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [5/5] 生成可视化图表")
    print("="*70)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs_range = range(1, len(trainer.train_history) + 1)
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        train_loss = [m['loss'] for m in trainer.train_history]
        val_loss = [m['loss'] for m in trainer.val_history[:len(trainer.train_history)]]
        ax1.plot(epochs_range, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = axes[0, 1]
        train_acc = [m['accuracy'] for m in trainer.train_history]
        val_acc = [m['accuracy'] for m in trainer.val_history[:len(trainer.train_history)]]
        ax2.plot(epochs_range, train_acc, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs_range, val_acc, 'r-', label='Val Accuracy', linewidth=2)
        ax2.axhline(y=0.85, color='g', linestyle='--', label='Target (85%)', alpha=0.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 平均脉冲数
        ax3 = axes[1, 0]
        train_spikes = [m['avg_spikes'] for m in trainer.train_history]
        val_spikes = [m['avg_spikes'] for m in trainer.val_history[:len(trainer.train_history)]]
        ax3.plot(epochs_range, train_spikes, 'b-', label='Train', linewidth=2)
        ax3.plot(epochs_range, val_spikes, 'r-', label='Val', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Avg Output Spikes')
        ax3.set_title('Average Output Spikes per Step')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. STDP 学习统计
        ax4 = axes[1, 1]
        if trainer.train_history:
            ltp_counts = [m['stdp_ltp'] for m in trainer.train_history]
            ltd_counts = [m['stdp_ltd'] for m in trainer.train_history]
            ax4.plot(epochs_range, ltp_counts, 'g-', label='LTP (Enhance)', linewidth=2)
            ax4.plot(epochs_range, ltd_counts, 'm-', label='LTD (Depress)', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('STDP Events')
            ax4.set_title('STDP Learning: LTP vs LTD')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = "visualization/training_curves.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  训练曲线: {viz_path}")
        
        # 额外: 类别准确率柱状图
        if trainer.val_history and 'class_accuracy' in trainer.val_history[-1]:
            fig, ax = plt.subplots(figsize=(8, 5))
            classes = ['Normal', 'Uptrend', 'Downtrend', 'Anomaly']
            accs = [trainer.val_history[-1]['class_accuracy'].get(i, 0) for i in range(4)]
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            bars = ax.bar(classes, accs, color=colors, edgecolor='black')
            ax.set_ylabel('Accuracy')
            ax.set_title('Per-Class Validation Accuracy (Final Epoch)')
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.85, color='r', linestyle='--', label='Target (85%)', alpha=0.5)
            
            for bar, acc in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
            
            ax.legend()
            plt.tight_layout()
            class_viz_path = "visualization/class_accuracy.png"
            plt.savefig(class_viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  类别准确率: {class_viz_path}")
        
    except Exception as e:
        print(f"  可视化生成失败: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 最终输出
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  [DONE] 第一次全面训练完成!")
    print("="*70)
    print(f"  最佳验证准确率: {summary['best_val_accuracy']:.2%}")
    print(f"  最佳验证损失: {summary['best_val_loss']:.4f}")
    print(f"  训练耗时: {summary['elapsed_seconds']:.1f}秒")
    print(f"  权重文件: {save_dir}/best_model.npz")
    print(f"  报告文件: {report_path}")
    print("="*70)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DistributedFormer 第一次全面训练')
    parser.add_argument('--samples', type=int, default=80, help='每类样本数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--depth', type=int, default=2, help='分形深度')
    parser.add_argument('--lr', type=float, default=0.008, help='学习率')
    parser.add_argument('--mod', type=float, default=0.25, help='监督调制强度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    run_first_training(
        samples_per_class=args.samples,
        epochs=args.epochs,
        depth=args.depth,
        learning_rate=args.lr,
        super_modulation=args.mod,
        seed=args.seed
    )
