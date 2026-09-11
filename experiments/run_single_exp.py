"""
Single experiment runner for ablation study.
Usage: python run_single_exp.py <exp_index>
Exp indices: 0=E1_Baseline, 1=E3_BiggerNet, 2=E4_ThinkSuper, 3=E5_v52_Full
"""
import sys, os
import numpy as np
import time
import json
import warnings
warnings.filterwarnings('ignore')

EXPERIMENTS = [
    {"name": "E1_Baseline",    "depth": 1, "think_supervision": False, "color": "#3498db"},
    {"name": "E3_BiggerNet",   "depth": 2, "think_supervision": False, "color": "#e74c3c"},
    {"name": "E4_ThinkSuper",  "depth": 1, "think_supervision": True,  "color": "#f39c12"},
    {"name": "E5_v52_Full",    "depth": 2, "think_supervision": True,  "color": "#9b59b6"},
]

def run_exp(exp_idx, workspace):
    os.makedirs("experiments", exist_ok=True)
    
    exp = EXPERIMENTS[exp_idx]
    print(f"Running experiment {exp_idx}: {exp['name']}")
    
    os.chdir(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    
    from distributedformer.training.trainer import DFTrainer
    from distributedformer.training.data_generator import StockTrainingDataset
    
    EPOCHS = 10
    SEED = 42
    SAMPLES = 50
    
    print(f"[{exp['name']}] depth={exp['depth']}, think_super={'ON' if exp['think_supervision'] else 'OFF'}")
    
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
    print(f"  Total units: {total_units:,} | Train: {len(train)} | Val: {len(val)}")
    
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
        "color": exp["color"],
    }
    
    # Save partial result
    partial_path = f"experiments/partial_{exp['name']}.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  [OK] Done | BestValAcc={result['best_val_acc']:.2%} | Time={elapsed:.1f}s")
    print(f"  Saved to {partial_path}")
    return result

def main(ctx):
    workspace = r'C:\Users\yxhcf\Documents\Kimi\Workspaces\DistributedFormer'
    os.chdir(workspace)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    
    exp_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    result = run_exp(exp_idx, workspace)
    return {"ok": True, "result": result}

if __name__ == "__main__":
    main({})
