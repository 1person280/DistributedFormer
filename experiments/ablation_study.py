"""
DistributedFormer 消融对比实验 (快速版)

实验设计:
┌─────┬─────────────┬────────────┬─────────────────────┬─────────────────┐
│ 编号 │ 名称         │ 网络深度    │ 每类样本数           │ 思考层监督       │
├─────┼─────────────┼────────────┼─────────────────────┼─────────────────┤
│ E1  │ Baseline     │ depth=1    │ 20                  │ 关闭             │
│ E2  │ +MoreData    │ depth=1    │ 20                  │ 关闭             │
│ E3  │ +BiggerNet   │ depth=2    │ 20                  │ 关闭             │
│ E4  │ +ThinkSuper  │ depth=1    │ 20                  │ 开启             │
│ E5  │ v5.2 Full    │ depth=2    │ 20                  │ 开启             │
└─────┴─────────────┴────────────┴─────────────────────┴─────────────────┘

快速配置: epochs=3, 相同数据量以隔离网络深度和监督策略变量
"""

import sys, os
import numpy as np
import time
import json

def main(ctx):
    workspace = r'C:\Users\yxhcf\Documents\Kimi\Workspaces\DistributedFormer'
    os.chdir(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    
    from distributedformer.training.trainer import DFTrainer
    from distributedformer.data.real_dataset import RustCodingTrainingDataset
    from distributedformer.core.distributedformer import calculate_scale
    
    EPOCHS = 3
    SEED = 42  # 分层划分种子 (真实语料 75/25)
    
    experiments = [
        {"name": "E1_Baseline",    "depth": 1, "think_supervision": False, "color": "#3498db"},
        {"name": "E3_BiggerNet",   "depth": 2, "think_supervision": False, "color": "#e74c3c"},
        {"name": "E4_ThinkSuper",  "depth": 1, "think_supervision": True,  "color": "#f39c12"},
        {"name": "E5_v52_Full",    "depth": 2, "think_supervision": True,  "color": "#9b59b6"},
    ]
    
    results = []
    
    print("=" * 70)
    print("  DistributedFormer 消融对比实验 (快速版)")
    print("=" * 70)
    print(f"  固定: epochs={EPOCHS}, 真实语料 75/25 分层划分, seed={SEED}")
    print("=" * 70)
    
    for exp in experiments:
        print(f"\n{'─' * 60}")
        print(f"  [{exp['name']}] depth={exp['depth']}, think_super={'ON' if exp['think_supervision'] else 'OFF'}")
        print(f"{'─' * 60}")
        
        dataset = RustCodingTrainingDataset(dim=16)
        train, val = dataset.generate_dataset(train_ratio=0.75, seed=SEED)
        
        trainer = DFTrainer(
            depth=exp["depth"],
            dim=16,
            learning_rate=0.008,
            super_modulation=0.25,
            use_think_supervision=exp["think_supervision"]
        )
        
        total_units = len(trainer.df._all_units)
        print(f"  总单元数: {total_units:,} | 训练: {len(train)} | 验证: {len(val)}")
        
        t0 = time.time()
        summary = trainer.train(train, val, epochs=EPOCHS, save_dir=f"training/checkpoints/{exp['name']}")
        elapsed = time.time() - t0
        
        result = {
            "name": exp["name"],
            "depth": exp["depth"],
            "think_supervision": exp["think_supervision"],
            "total_units": total_units,
            "best_val_acc": summary["best_val_accuracy"],
            "final_val_acc": summary["final_val_accuracy"],
            "best_val_loss": summary["best_val_loss"],
            "final_val_loss": summary["final_val_loss"],
            "elapsed_seconds": elapsed,
            "class_accuracy": summary.get("class_accuracy", {}),
            "val_history": trainer.val_history,
        }
        results.append(result)
        print(f"  ✓ 完成 | BestValAcc={result['best_val_acc']:.2%} | 耗时={elapsed:.1f}s")
    
    # 汇总表
    print(f"\n{'=' * 70}")
    print("  结果汇总")
    print(f"{'=' * 70}")
    print(f"  {'实验':<14} {'深度':>4} {'思考监督':>8} {'单元数':>8} {'BestValAcc':>10} {'耗时(s)':>8}")
    print(f"  {'─' * 14} {'─' * 4} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 8}")
    for r in results:
        print(f"  {r['name']:<14} {r['depth']:>4} {'开' if r['think_supervision'] else '关':>8} {r['total_units']:>8,} {r['best_val_acc']:>9.2%} {r['elapsed_seconds']:>8.1f}")
    
    # 保存
    os.makedirs("experiments", exist_ok=True)
    save_results = [{k: v for k, v in r.items() if k != "val_history"} for r in results]
    with open("experiments/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 可视化 (简化版)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        names = [r['name'] for r in results]
        colors = [exp['color'] for exp in experiments]
        
        # 1. Best Val Acc
        ax = axes[0]
        vals = [r['best_val_acc'] for r in results]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('E', '').split('_')[0] for n in names], rotation=0)
        ax.set_ylabel('Accuracy')
        ax.set_title('Best Validation Accuracy')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{v:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Val Acc Curves
        ax = axes[1]
        for i, r in enumerate(results):
            val_acc = [m['accuracy'] for m in r['val_history']]
            ax.plot(range(1, len(val_acc)+1), val_acc, marker='o', 
                   label=r['name'].split('_')[0], color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Accuracy')
        ax.set_title('Validation Accuracy Curves')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # 3. Per-Class Heatmap
        ax = axes[2]
        class_names = ['Normal', 'Uptrend', 'Downtrend', 'Anomaly']
        acc_matrix = []
        for r in results:
            ca = r['class_accuracy']
            row = [ca.get(str(c), ca.get(c, 0)) for c in range(4)]
            acc_matrix.append(row)
        acc_matrix = np.array(acc_matrix)
        im = ax.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(class_names, rotation=30, ha='right')
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([n.replace('E', '').split('_')[0] for n in names])
        ax.set_title('Per-Class Accuracy')
        for i in range(len(results)):
            for j in range(4):
                ax.text(j, i, f'{acc_matrix[i, j]:.0%}', ha="center", va="center", 
                       color="black" if acc_matrix[i, j] < 0.5 else "white", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.7)
        
        plt.suptitle('DF Ablation Study (3 epochs, 20 samples/class)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("experiments/ablation_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  图表已保存: experiments/ablation_comparison.png")
    except Exception as e:
        print(f"  可视化失败: {e}")
    
    # Markdown报告
    report = f"""# DistributedFormer 消融对比实验报告

## 实验配置
- Epochs: {EPOCHS}
- Samples per class: {SAMPLES}
- Seed: {SEED}

## 结果汇总

| 实验 | 深度 | 思考监督 | 单元数 | Best ValAcc | 耗时(s) |
|------|------|---------|--------|------------|---------|
"""
    for r in results:
        report += f"| {r['name']} | {r['depth']} | {'✅' if r['think_supervision'] else '❌'} | {r['total_units']:,} | {r['best_val_acc']:.2%} | {r['elapsed_seconds']:.1f} |\n"
    
    report += "\n## 各类别准确率\n\n| 实验 | Normal | Uptrend | Downtrend | Anomaly |\n|------|--------|---------|-----------|---------|\n"
    for r in results:
        ca = r['class_accuracy']
        report += f"| {r['name']} | {ca.get('0', ca.get(0, 0)):.0%} | {ca.get('1', ca.get(1, 0)):.0%} | {ca.get('2', ca.get(2, 0)):.0%} | {ca.get('3', ca.get(3, 0)):.0%} |\n"
    
    # 分析
    baseline = next(r for r in results if r['name'] == 'E1_Baseline')
    full = next(r for r in results if r['name'] == 'E5_v52_Full')
    bigger = next(r for r in results if r['name'] == 'E3_BiggerNet')
    think = next(r for r in results if r['name'] == 'E4_ThinkSuper')
    
    report += f"\n## 关键发现\n\n"
    report += f"- **Baseline (E1)**: depth=1, 无思考监督 → BestValAcc = {baseline['best_val_acc']:.2%}\n"
    report += f"- **网络规模效应 (E3)**: depth=2 vs depth=1 → {baseline['best_val_acc']:.2%} → {bigger['best_val_acc']:.2%}\n"
    report += f"- **思考监督效应 (E4)**: 开启思考监督 → {baseline['best_val_acc']:.2%} → {think['best_val_acc']:.2%}\n"
    report += f"- **v5.2 Full (E5)**: depth=2 + 思考监督 → {baseline['best_val_acc']:.2%} → {full['best_val_acc']:.2%}\n"
    report += f"\n*实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    with open("experiments/ablation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: experiments/ablation_report.md")
    
    print(f"\n{'=' * 70}")
    print("  消融实验全部完成!")
    print(f"{'=' * 70}")
    
    return {"ok": True, "results": len(results)}

if __name__ == "__main__":
    main({})
