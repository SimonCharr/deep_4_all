---
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
license: apache-2.0
language:
  - fr
  - en
pipeline_tag: text-generation
tags:
  - lora
  - transformers
  - qwen3
  - pokemon
  - dasd
  - distillation
  - reasoning
datasets:
  - pokemon-dasd-dataset
---

# Pokemon DASD LoRA Adapter

Adaptateur LoRA pour Qwen3-4B fine-tuné sur un dataset de questions/réponses Pokémon avec raisonnement structuré, utilisant la méthode **DASD** (Distribution-Aligned Sequence Distillation).

## Description du modèle

Ce modèle est un adaptateur LoRA entraîné en 2 stages selon la méthodologie DASD :
- **Stage 1** : Entraînement sur des données générées à basse température (tau=0.3) pour la stabilité
- **Stage 2** : Entraînement sur des données générées à haute température (tau=0.9) pour la diversité

Le modèle génère des réponses avec un raisonnement structuré dans des balises `<reasoning>...</reasoning>`.

## Détails techniques

| Paramètre | Valeur |
|-----------|--------|
| Modèle de base | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Type de fine-tuning | LoRA |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Modules ciblés | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Framework | LLaMA-Factory + PEFT 0.18.1 |

### Hyperparamètres d'entraînement

| Stage | Epochs | Learning Rate | Batch Size | Loss finale |
|-------|--------|---------------|------------|-------------|
| Stage 1 (tau=0.3) | 5 | 1e-4 | 1 (grad_accum=4) | 0.772 |
| Stage 2 (tau=0.9) | 3 | 5e-5 | 1 (grad_accum=4) | 0.882 |

## Utilisation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    trust_remote_code=True
)

# Charger l'adaptateur LoRA
model = PeftModel.from_pretrained(base_model, "simoncharr/pokemon-dasd-lora-adapter")

# Générer une réponse
messages = [
    {"role": "system", "content": "Tu es un expert Pokémon. Raisonne étape par étape dans des balises <reasoning>...</reasoning> avant de donner ta réponse finale."},
    {"role": "user", "content": "Quelle est l'efficacité d'une attaque de type Feu contre un Pokémon de type Eau/Sol ?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Dataset d'entraînement

Ce modèle a été entraîné sur le dataset [pokemon-dasd-dataset](https://huggingface.co/datasets/simoncharr/pokemon-dasd-dataset) contenant 600 exemples (300 par stage) de questions Pokémon avec réponses détaillées et raisonnement structuré.

### Catégories couvertes
- Efficacité des types
- STAB et calcul de dégâts
- Évolutions et conditions
- Analyse de statistiques
- Stratégie compétitive (Little Cup, OU, VGC)

## Méthodologie DASD

La méthode DASD (Distribution-Aligned Sequence Distillation) consiste à :
1. Générer des réponses avec un modèle "teacher" à différentes températures
2. Calculer les log-probabilités pour identifier les "Teacher Sentences" (où le teacher est confiant mais l'étudiant ne l'est pas)
3. Filtrer les données via Divergence-Aware Sampling (DAS)
4. Entraîner en 2 stages : stabilité (basse temp) puis diversité (haute temp)

## Limitations

- Le modèle peut générer des informations incorrectes sur des Pokémon fictifs ou mal nommés
- Les réponses sont optimisées pour le français
- Le raisonnement peut parfois être redondant

## Citation

```bibtex
@misc{pokemon-dasd-2025,
  title={Pokemon DASD LoRA Adapter},
  author={[Votre nom]},
  year={2025},
  howpublished={HuggingFace Hub}
}
```

## Références

- [DASD Paper - Distribution-Aligned Sequence Distillation](https://github.com/D2I-ai/dasd-thinking)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3](https://huggingface.co/Qwen)
