"""
Script d'expérimentation pour trouver les meilleurs hyperparamètres.
Teste plusieurs configurations et garde la meilleure.
"""

import json
import subprocess
import sys
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Configurations to test
configs = [
    # Config 1: Transformer compact (baseline)
    {
        "name": "transformer_compact",
        "args": "--mode transformer --embed_dim 32 --hidden_dim 64 --num_layers 2 --nhead 4 --dropout 0.3 --learning_rate 0.001 --weight_decay 1e-4 --batch_size 64 --epochs 80 --patience 12",
    },
    # Config 2: Transformer with more heads
    {
        "name": "transformer_8head",
        "args": "--mode transformer --embed_dim 32 --hidden_dim 128 --num_layers 2 --nhead 8 --dropout 0.25 --learning_rate 0.0008 --weight_decay 1e-4 --batch_size 64 --epochs 80 --patience 12",
    },
    # Config 3: Deeper transformer, smaller dims
    {
        "name": "transformer_deep",
        "args": "--mode transformer --embed_dim 32 --hidden_dim 64 --num_layers 4 --nhead 4 --dropout 0.35 --learning_rate 0.0005 --weight_decay 1e-4 --batch_size 64 --epochs 80 --patience 15",
    },
    # Config 4: Wider embeddings
    {
        "name": "transformer_wide_embed",
        "args": "--mode transformer --embed_dim 48 --hidden_dim 96 --num_layers 2 --nhead 8 --dropout 0.3 --learning_rate 0.0005 --weight_decay 1e-4 --batch_size 64 --epochs 80 --patience 12",
    },
    # Config 5: LSTM bidirectional for comparison
    {
        "name": "lstm_bidir",
        "args": "--mode lstm --embed_dim 32 --hidden_dim 64 --num_layers 2 --dropout 0.3 --bidirectional --learning_rate 0.001 --weight_decay 1e-4 --batch_size 64 --epochs 80 --patience 12",
    },
    # Config 6: Transformer with less dropout (more capacity)
    {
        "name": "transformer_low_dropout",
        "args": "--mode transformer --embed_dim 32 --hidden_dim 128 --num_layers 2 --nhead 4 --dropout 0.15 --learning_rate 0.0005 --weight_decay 5e-4 --batch_size 64 --epochs 80 --patience 12",
    },
    # Config 7: Transformer with larger batch
    {
        "name": "transformer_large_batch",
        "args": "--mode transformer --embed_dim 32 --hidden_dim 64 --num_layers 3 --nhead 4 --dropout 0.3 --learning_rate 0.001 --weight_decay 1e-4 --batch_size 128 --epochs 80 --patience 12",
    },
]


def parse_best_acc(output: str) -> float:
    """Extract best validation accuracy from training output."""
    for line in output.split('\n'):
        if 'Meilleure accuracy validation' in line:
            # "Meilleure accuracy validation: 91.70%"
            pct = line.split(':')[-1].strip().rstrip('%')
            return float(pct)
    return 0.0


def parse_params(output: str) -> int:
    """Extract parameter count from training output."""
    for line in output.split('\n'):
        if 'Paramètres:' in line:
            return int(line.split(':')[-1].strip().replace(',', ''))
    return 0


def parse_category_results(output: str) -> dict:
    """Extract per-category results."""
    results = {}
    in_category = False
    for line in output.split('\n'):
        if 'Analyse par catégorie' in line:
            in_category = True
            continue
        if in_category and ':' in line and '%' in line:
            parts = line.strip().split(':')
            cat = parts[0].strip()
            acc = parts[1].strip().split('%')[0].strip()
            results[cat] = float(acc)
        if '!' * 10 in line:
            in_category = False
    return results


if __name__ == "__main__":
    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}] Running: {config['name']}")
        print(f"{'='*70}")

        cmd = f"python3 train_dungeon_logs.py {config['args']} --plot"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600
        )

        output = result.stdout + result.stderr
        best_acc = parse_best_acc(output)
        params = parse_params(output)
        cat_results = parse_category_results(output)

        results.append({
            "name": config["name"],
            "best_val_acc": best_acc,
            "params": params,
            "categories": cat_results,
            "args": config["args"],
        })

        print(f"  -> Val Acc: {best_acc:.2f}% | Params: {params:,}")
        for cat, acc in sorted(cat_results.items()):
            print(f"     {cat:35s}: {acc:.2f}%")

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    results.sort(key=lambda x: (-x['best_val_acc'], x['params']))
    for r in results:
        print(f"  {r['name']:30s}: {r['best_val_acc']:.2f}% ({r['params']:,} params)")

    best = results[0]
    print(f"\nBEST: {best['name']} with {best['best_val_acc']:.2f}% ({best['params']:,} params)")

    # Save results
    with open(CHECKPOINT_DIR / "experiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {CHECKPOINT_DIR / 'experiment_results.json'}")
