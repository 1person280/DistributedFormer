"""
DistributedFormer 消融对比实验 (扩展版: 10 epochs × 50 samples/类)

实验设计:
┌─────┬─────────────┬────────────┬─────────────────────┬─────────────────┐
│ 编号 │ 名称         │ 网络深度    │ 每类样本数           │ 思考层监督       │
├─────┼─────────────┼────────────┼─────────────────────┼─────────────────┤
│ E1  │ Baseline     │ depth=1    │ 50                  │ OFF             │
│ E3  │ +BiggerNet   │ depth=2    │ 50                  │ OFF             │
│ E4  │ +ThinkSuper  │ depth=1    │ 50                  │ ON             │
│ E5  │ v5.2 Full    │ depth=2    │ 50                  │ ON             │
└─────┴─────────────┴────────────┴─────────────────────┴─────────────────┘

扩展配置: epochs=10, samples=50/类
"""

import sys, os
import numpy as np
import time
import json
import warnings
warnings.filterwarnings('ignore')

def main(ctx):
    workspace = r'C:\Users\yxhcf\Documents\Kimi\Workspaces\DistributedFormer'
    os.chdir(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    
    from distributedformer.training.trainer import DFTrainer
    from distributedformer.training.data_generator import StockTrainingDataset
    from distributedformer.core.distributedformer import calculate_scale
    
    EPOCHS = 10
    SEED = 42
    SAMPLES = 50  # 每类样本数
    
    experiments = [
        {"name": "E1_Baseline",    "depth": 1, "think_supervision": False, "color": "#3498db"},
        {"name": "E3_BiggerNet",   "depth": 2, "think_supervision": False, "color": "#e74c3c"},
        {"name": "E4_ThinkSuper",  "depth": 1, "think_supervision": True,  "color": "#f39c12"},
        {"name": "E5_v52_Full",    "depth": 2, "think_supervision": True,  "color": "#9b59b6"},
    ]
    
    results = []
    
    print("=" * 70)
    print("  DistributedFormer 消融对比实验 (扩展版)")
    print("=" * 70)
    print(f"  固定: epochs={EPOCHS}, samples_per_class={SAMPLES}, seed={SEED}")
    print("=" * 70)
    
    for exp in experiments:
        print(f"\n{'─' * 60}")
        print(f"  [{exp['name']}] depth={exp['depth']}, think_super={'ON' if exp['think_supervision'] else 'OFF'}")
        print(f"{'─' * 60}")
        
        dataset = StockTrainingDataset(dim=16, seed=SEED)
        train, val = dataset.generate_dataset(samples_per_class=SAMPLES, train_ratio=0.8)
        
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
            "train_history": trainer.train_history,
        }
        results.append(result)
        print(f"  [OK] Done | BestValAcc={result['best_val_acc']:.2%} | 耗时={elapsed:.1f}s")
    
    # 汇总表
    print(f"\n{'=' * 70}")
    print("  结果汇总")
    print(f"{'=' * 70}")
    print(f"  {'实验':<14} {'深度':>4} {'思考监督':>8} {'单元数':>8} {'BestValAcc':>10} {'FinalValAcc':>11} {'耗时(s)':>8}")
    print(f"  {'─' * 14} {'─' * 4} {'─' * 8} {'─' * 8} {'─' * 10} {'─' * 11} {'─' * 8}")
    for r in results:
        print(f"  {r['name']:<14} {r['depth']:>4} {'ON' if r['think_supervision'] else 'OFF':>8} {r['total_units']:>8,} {r['best_val_acc']:>9.2%} {r['final_val_acc']:>10.2%} {r['elapsed_seconds']:>8.1f}")
    
    # 保存
    os.makedirs("experiments", exist_ok=True)
    save_results = [{k: v for k, v in r.items() if k not in ("val_history", "train_history")} for r in results]
    with open("experiments/ablation_results_extended.json", "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2, default=str)
    
    # 可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        names = [r['name'] for r in results]
        colors = [exp['color'] for exp in experiments]
        
        # 1. Best Val Acc 柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        vals = [r['best_val_acc'] for r in results]
        bars = ax1.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.replace('E', '').split('_')[0] for n in names], rotation=0, fontsize=9)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Best Validation Accuracy', fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{v:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Final Val Acc 柱状图
        ax2 = fig.add_subplot(gs[0, 1])
        vals = [r['final_val_acc'] for r in results]
        bars = ax2.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=1.2, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels([n.replace('E', '').split('_')[0] for n in names], rotation=0, fontsize=9)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.set_title('Final Validation Accuracy', fontsize=11, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{v:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. 耗时对比
        ax3 = fig.add_subplot(gs[0, 2])
        vals = [r['elapsed_seconds'] for r in results]
        bars = ax3.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=1.2)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels([n.replace('E', '').split('_')[0] for n in names], rotation=0, fontsize=9)
        ax3.set_ylabel('Time (s)', fontsize=10)
        ax3.set_title('Training Time', fontsize=11, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{v:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 4. Val Acc Curves (全epoch)
        ax4 = fig.add_subplot(gs[1, :])
        for i, r in enumerate(results):
            val_acc = [m['accuracy'] for m in r['val_history']]
            epochs_x = range(1, len(val_acc)+1)
            ax4.plot(epochs_x, val_acc, marker='o', 
                   label=r['name'], color=colors[i], linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Val Accuracy', fontsize=10)
        ax4.set_title('Validation Accuracy Curves (10 Epochs)', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9, loc='lower right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.0)
        ax4.set_xticks(range(1, EPOCHS+1))
        
        # 5. Train Loss Curves
        ax5 = fig.add_subplot(gs[2, 0])
        for i, r in enumerate(results):
            train_loss = [m['loss'] for m in r['train_history']]
            epochs_x = range(1, len(train_loss)+1)
            ax5.plot(epochs_x, train_loss, marker='o', 
                   label=r['name'], color=colors[i], linewidth=2, markersize=4)
        ax5.set_xlabel('Epoch', fontsize=10)
        ax5.set_ylabel('Train Loss', fontsize=10)
        ax5.set_title('Training Loss', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8, loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(range(1, EPOCHS+1))
        
        # 6. Val Loss Curves
        ax6 = fig.add_subplot(gs[2, 1])
        for i, r in enumerate(results):
            val_loss = [m['loss'] for m in r['val_history']]
            epochs_x = range(1, len(val_loss)+1)
            ax6.plot(epochs_x, val_loss, marker='o', 
                   label=r['name'], color=colors[i], linewidth=2, markersize=4)
        ax6.set_xlabel('Epoch', fontsize=10)
        ax6.set_ylabel('Val Loss', fontsize=10)
        ax6.set_title('Validation Loss', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8, loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xticks(range(1, EPOCHS+1))
        
        # 7. Per-Class Heatmap
        ax7 = fig.add_subplot(gs[2, 2])
        class_names = ['Normal', 'Uptrend', 'Downtrend', 'Anomaly']
        acc_matrix = []
        for r in results:
            ca = r['class_accuracy']
            row = [ca.get(str(c), ca.get(c, 0)) for c in range(4)]
            acc_matrix.append(row)
        acc_matrix = np.array(acc_matrix)
        im = ax7.imshow(acc_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax7.set_xticks(range(4))
        ax7.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
        ax7.set_yticks(range(len(results)))
        ax7.set_yticklabels([n.replace('E', '').split('_')[0] for n in names], fontsize=9)
        ax7.set_title('Per-Class Accuracy', fontsize=11, fontweight='bold')
        for i in range(len(results)):
            for j in range(4):
                ax7.text(j, i, f'{acc_matrix[i, j]:.0%}', ha="center", va="center", 
                       color="black" if acc_matrix[i, j] < 0.5 else "white", fontsize=9)
        plt.colorbar(im, ax=ax7, shrink=0.7)
        
        plt.suptitle('DF Ablation Study (10 epochs × 50 samples/class)', fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("experiments/ablation_comparison_extended.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  图表已保存: experiments/ablation_comparison_extended.png")
    except Exception as e:
        import traceback
        print(f"  可视化失败: {e}")
        traceback.print_exc()
    
    # Markdown报告
    report = f"""# DistributedFormer 消融对比实验报告 (扩展版)

## 实验配置
- Epochs: {EPOCHS}
- Samples per class: {SAMPLES}
- Seed: {SEED}
- 总训练样本: {SAMPLES * 4} (4类)

## 结果汇总

| 实验 | 深度 | 思考监督 | 单元数 | Best ValAcc | Final ValAcc | 耗时(s) |
|------|------|---------|--------|------------|-------------|---------|
"""
    for r in results:
        report += f"| {r['name']} | {r['depth']} | {'YES' if r['think_supervision'] else 'NO'} | {r['total_units']:,} | {r['best_val_acc']:.2%} | {r['final_val_acc']:.2%} | {r['elapsed_seconds']:.1f} |\n"
    
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
    report += f"- **Baseline (E1)**: depth=1, 无思考监督 → BestValAcc = {baseline['best_val_acc']:.2%}, FinalValAcc = {baseline['final_val_acc']:.2%}\n"
    report += f"- **网络规模效应 (E3)**: depth=2 vs depth=1 → {baseline['best_val_acc']:.2%} → {bigger['best_val_acc']:.2%} (Δ = {bigger['best_val_acc'] - baseline['best_val_acc']:.2%})\n"
    report += f"- **思考监督效应 (E4)**: ON思考监督 → {baseline['best_val_acc']:.2%} → {think['best_val_acc']:.2%} (Δ = {think['best_val_acc'] - baseline['best_val_acc']:.2%})\n"
    report += f"- **v5.2 Full (E5)**: depth=2 + 思考监督 → {baseline['best_val_acc']:.2%} → {full['best_val_acc']:.2%} (Δ = {full['best_val_acc'] - baseline['best_val_acc']:.2%})\n"
    
    report += f"\n## 效应分解\n\n"
    net_effect = bigger['best_val_acc'] - baseline['best_val_acc']
    think_effect = think['best_val_acc'] - baseline['best_val_acc']
    combined = full['best_val_acc'] - baseline['best_val_acc']
    report += f"- 纯网络规模增益: {net_effect:.2%}\n"
    report += f"- 纯思考监督增益: {think_effect:.2%}\n"
    report += f"- 联合增益 (Full): {combined:.2%}\n"
    report += f"- 协同效应 (Full - 单效应之和): {combined - net_effect - think_effect:.2%}\n"
    
    report += f"\n*实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    with open("experiments/ablation_report_extended.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: experiments/ablation_report_extended.md")
    
    print(f"\n{'=' * 70}")
    print("  消融实验全部完成!")
    print(f"{'=' * 70}")
    
    return {"ok": True, "results": len(results), "workspace": workspace}

if __name__ == "__main__":
    main({})
