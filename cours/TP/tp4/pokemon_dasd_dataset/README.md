---
license: apache-2.0
language:
  - fr
size_categories:
  - n<1K
task_categories:
  - question-answering
  - text-generation
tags:
  - pokemon
  - reasoning
  - chain-of-thought
  - dasd
  - distillation
  - french
pretty_name: Pokemon DASD Dataset
---

# Pokemon DASD Dataset

Dataset de questions/réponses sur l'univers Pokémon avec raisonnement structuré, conçu pour l'entraînement de modèles via la méthode **DASD** (Distribution-Aligned Sequence Distillation).

## Description

Ce dataset contient **600 exemples** de questions Pokémon avec des réponses détaillées incluant un raisonnement étape par étape dans des balises `<reasoning>...</reasoning>`.

Les données ont été générées via l'API Infomaniak AI (modèle teacher) à deux températures différentes pour implémenter la méthodologie DASD :
- **Stage 1** (300 exemples) : Température basse (tau=0.3) pour des réponses stables et cohérentes
- **Stage 2** (300 exemples) : Température haute (tau=0.9) pour des réponses plus diversifiées

## Structure des fichiers

```
pokemon_dasd_dataset/
├── README.md
├── stage1.json    # 300 exemples, tau=0.3
└── stage2.json    # 300 exemples, tau=0.9
```

## Format des données

Le dataset utilise le format **Alpaca** compatible avec LLaMA-Factory :

```json
{
  "instruction": "Quelle est l'efficacité d'une attaque de type Combat contre un Pokémon de type Normal ?",
  "input": "",
  "output": "<reasoning>\n1. **Identifier les types en jeu**\n   - Attaque : type **Combat**\n   - Défenseur : type **Normal**\n...\n</reasoning>\n\n**Réponse :** Une attaque de type Combat contre un Pokémon de type Normal a un multiplicateur de **×2**."
}
```

| Champ | Description |
|-------|-------------|
| `instruction` | La question posée sur l'univers Pokémon |
| `input` | Contexte additionnel (vide dans ce dataset) |
| `output` | Réponse complète avec raisonnement structuré |

## Catégories de questions

| Catégorie | Description |
|-----------|-------------|
| `efficacite_types` | Multiplicateurs de dégâts entre types |
| `stab_et_degats` | Bonus STAB et calculs de dégâts |
| `evolution` | Chaînes d'évolution et conditions |
| `stats_et_comparaisons` | Analyse de statistiques de base |
| `mecaniques_combat` | Rochers Furtifs, météo, terrain, etc. |
| `strategies_specifiques` | Sets compétitifs et movesets |
| `team_building` | Synergies et compositions d'équipe |
| `formats_et_tiers` | Little Cup, OU, UU, VGC, etc. |
| `calculs_avances` | Formules de dégâts, probabilités |

## Utilisation avec LLaMA-Factory

### 1. Ajouter au `dataset_info.json`

```json
{
  "pokemon_stage1": {
    "file_name": "stage1.json"
  },
  "pokemon_stage2": {
    "file_name": "stage2.json"
  }
}
```

### 2. Configuration d'entraînement (stage1_train.yaml)

```yaml
model_name_or_path: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
dataset: pokemon_stage1
template: qwen3_nothink
cutoff_len: 2048
num_train_epochs: 5
learning_rate: 1.0e-4
```

### 3. Configuration stage 2 (charge l'adapter du stage 1)

```yaml
model_name_or_path: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
adapter_name_or_path: saves/pokemon-stage1
dataset: pokemon_stage2
num_train_epochs: 3
learning_rate: 5.0e-5
```

## Utilisation avec Hugging Face Datasets

```python
from datasets import load_dataset

# Charger le dataset
dataset = load_dataset("simoncharr/pokemon-dasd-dataset")

# Accéder aux splits
stage1 = dataset["stage1"]
stage2 = dataset["stage2"]

# Exemple
print(stage1[0]["instruction"])
print(stage1[0]["output"])
```

## Méthodologie DASD

Ce dataset a été créé selon la méthodologie **Distribution-Aligned Sequence Distillation** :

1. **Génération multi-température** : Les réponses sont générées à différentes températures pour capturer à la fois la stabilité (basse temp) et la diversité (haute temp)

2. **Divergence-Aware Sampling (DAS)** : Les exemples sont filtrés en comparant les log-probabilités du modèle teacher vs student pour identifier les "Teacher Sentences" (où le teacher apporte le plus de valeur)

3. **Entraînement en 2 stages** :
   - Stage 1 : Établir une base stable avec les données basse température
   - Stage 2 : Enrichir avec les données haute température en chargeant l'adapter du stage 1

## Statistiques

| Métrique | Valeur |
|----------|--------|
| Nombre total d'exemples | 600 |
| Exemples Stage 1 | 300 |
| Exemples Stage 2 | 300 |
| Langue | Français |
| Longueur moyenne des réponses | ~500-2000 tokens |

## Limitations

- Les réponses peuvent contenir des informations sur des Pokémon fictifs ou mal nommés (hallucinations du modèle teacher)
- Certaines mécaniques de combat peuvent être simplifiées ou légèrement inexactes
- Le dataset est en français uniquement

## Citation

```bibtex
@misc{pokemon-dasd-dataset-2025,
  title={Pokemon DASD Dataset},
  author={[Votre nom]},
  year={2025},
  howpublished={HuggingFace Hub}
}
```

## Références

- [DASD Paper - Distribution-Aligned Sequence Distillation](https://github.com/D2I-ai/dasd-thinking)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Alibaba DASD Reference Dataset](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b)
