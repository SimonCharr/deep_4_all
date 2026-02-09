# TP4 : Distillation de Modeles de Raisonnement (DASD)

## Theme : Pokemon

Distillation des capacites de raisonnement d'un LLM enseignant vers un modele etudiant compact, appliquee a l'univers Pokemon.

---

## Contexte Scientifique

Les grands LLMs (GPT-4, Qwen-235B...) raisonnent bien mais sont couteux et impossibles a deployer localement. Le papier **DASD** (Distribution-Aligned Sequence Distillation) propose de transferer ces capacites vers un modele compact via :

| Technique | Role |
|-----------|------|
| **Temperature-Scheduled Learning** | Stage 1 (tau=0.3) pour la stabilite, Stage 2 (tau=0.9) pour la diversite |
| **Divergence-Aware Sampling (DAS)** | Filtrer les donnees ou l'etudiant a le plus a apprendre |

Le DAS analyse chaque phrase de la reponse du teacher et la classe :
- **Teacher Sentence** (P_teacher >> P_student) : l'etudiant ignore cette connaissance → a garder
- **Shared** (P_teacher ~ P_student) : connaissance deja acquise → neutre
- **Student Sentence** (P_student > P_teacher) : hallucination probable → a rejeter

---

## Stack Technique

| Composant | Choix | Justification |
|-----------|-------|---------------|
| **Teacher** | API Infomaniak | API compatible OpenAI, supporte les logprobs |
| **Student** | Qwen3-4B (4-bit) | Compact, performant, tient sur un T4 (16GB VRAM) |
| **Fine-tuning** | LoRA (rank=8) | Efficace en memoire, evite de modifier tous les poids |
| **Framework** | Llama-Factory | Simplifie l'entrainement LoRA, supporte les templates Qwen |
| **Execution** | Google Colab (T4) | GPU gratuit suffisant pour le 4-bit + LoRA |

---

## Choix du theme Pokemon

Le theme Pokemon a ete choisi car il offre un domaine riche pour tester le raisonnement :

- **Interactions de types** : raisonnement logique avec doubles faiblesses, immunites, STAB
- **Calculs de degats** : raisonnement mathematique avec la formule officielle
- **Team building** : raisonnement strategique multi-etapes
- **Chaines d'evolution** : connaissances factuelles avec conditions variees
- **Stats et tiers** : comparaison quantitative et analyse

Les 28 questions couvrent 5 categories pour une evaluation diversifiee.

---

## Structure du TP

```
tp4/
├── README.md                  # Ce fichier
├── enonce_tp4.md              # Enonce officiel du TP
├── simple_dasd.py             # Code de reference DAS
└── tp4_dasd_pokemon.ipynb     # Notebook principal (tout-en-un)
```

Fichiers generes a l'execution sur Colab :
```
LLaMA-Factory/
├── data/
│   ├── pokemon_stage1.json    # Dataset Alpaca Stage 1 (apres filtrage DAS)
│   └── pokemon_stage2.json    # Dataset Alpaca Stage 2 (apres filtrage DAS)
├── stage1_raw.json            # Donnees brutes + logprobs Stage 1
├── stage2_raw.json            # Donnees brutes + logprobs Stage 2
├── stage1_train.yaml          # Config Llama-Factory Stage 1
├── stage2_train.yaml          # Config Llama-Factory Stage 2
├── das_analysis.png           # Visualisations DAS
├── loss_curves.png            # Courbes de loss
└── saves/
    ├── pokemon-stage1/        # Adapter LoRA Stage 1
    └── pokemon-stage2/        # Adapter LoRA Stage 2 (final)
```

---

## Pipeline (10 phases)

### Phase 1-2 : Setup et etude du dataset de reference
- Installation de Llama-Factory sur Colab
- Configuration de l'API Infomaniak via `google.colab.userdata`
- Exploration du dataset Alibaba DASD pour comprendre le format `<think>...</think>`

### Phase 3 : Generation du dataset Pokemon
- 28 questions hardcodees en francais, reparties en 5 categories
- Generation a deux temperatures via l'API enseignant :
  - **Stage 1** (tau=0.3) : reponses stables et coherentes
  - **Stage 2** (tau=0.9) : reponses plus diversifiees et creatives
- Logprobs recuperees et serialisees pour le DAS
- Retry avec backoff exponentiel + filtre qualite (longueur minimale)

### Phase 4 : Divergence-Aware Sampling (DAS)
- Chargement de Qwen3-4B en 4-bit (BitsAndBytesConfig)
- Forward pass etudiant sur chaque reponse du teacher
- Alignement phrase par phrase avec curseur caractere (adapte de `simple_dasd.py`)
- Score par phrase : P = exp(mean(logprobs)), divergence = P_teacher - P_student
- Filtrage : on garde les exemples a divergence moyenne >= 0
- Visualisations : histogrammes, scatter plots P_teacher vs P_student

### Phase 5 : Configuration de l'entrainement
- Enregistrement des datasets dans `dataset_info.json` (format Alpaca)
- **Stage 1** : `lora_rank=8`, `lr=1e-4`, 5 epochs, `template=qwen3_nothink`
- **Stage 2** : charge l'adapter Stage 1 (`adapter_name_or_path`), `lr=5e-5`, 3 epochs
- `gradient_checkpointing=true` et `cutoff_len=2048` pour eviter les OOM

### Phase 6 : Entrainement
- Liberation memoire GPU (suppression du modele etudiant charge pour le DAS)
- `llamafactory-cli train stage1_train.yaml` puis `stage2_train.yaml`

### Phase 7-9 : Evaluation
- Courbes de loss (parsing de `trainer_log.jsonl`)
- Chargement du modele distille (base + LoRA Stage 2 via `peft`)
- Comparaison qualitative sur 3 questions test (base vs distille)
- Scoring automatique : presence `<reasoning>`, longueur, mots-cles Pokemon, etapes numerotees
- Tableau recapitulatif des metriques

### Phase 10 : Conclusion et limites

---

## Choix techniques detailles

### Pourquoi Qwen3-4B ?
Le modele `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` est pre-quantifie en 4-bit, ce qui permet de le charger sur un T4 (16GB) avec de la marge pour l'entrainement LoRA. La famille Qwen3 est reconnue pour ses bonnes performances en raisonnement.

### Pourquoi LoRA rank=8 ?
Avec seulement 28 exemples d'entrainement, un rank eleve (16, 32) risquerait l'overfitting. Le rank 8 offre un bon compromis entre capacite d'apprentissage et regularisation implicite.

### Pourquoi template qwen3_nothink ?
Le template `qwen3_nothink` desactive le mode "thinking" natif de Qwen3 pour laisser le modele apprendre le format `<reasoning>...</reasoning>` du teacher plutot que d'utiliser son propre `<think>...</think>`.

### Pourquoi deux stages de temperature ?
Suivant le papier DASD :
- **Stage 1 (tau=0.3)** : reponses deterministes et fiables pour ancrer les bases
- **Stage 2 (tau=0.9)** : reponses plus variees pour enrichir la comprehension

Le Stage 2 charge l'adapter LoRA du Stage 1, ce qui permet un apprentissage progressif.

### Pourquoi le filtrage DAS a divergence >= 0 ?
Un score DAS negatif signifie que l'etudiant est deja plus confiant que le teacher sur cette reponse : soit le teacher hesite, soit l'etudiant hallucine. Dans les deux cas, cet exemple n'est pas pedagogiquement utile.

---

## Execution

### Prerequis
- Compte Google (Colab)
- Cle API Infomaniak stockee dans les secrets Colab sous `INFOMANIAK_API_KEY`

### Lancement
1. Ouvrir `tp4_dasd_pokemon.ipynb` dans Google Colab
2. Selectionner le runtime **GPU T4**
3. Ajouter la cle API dans les secrets Colab (icone cle dans le panneau lateral)
4. Executer toutes les cellules sequentiellement

### Duree estimee
- Generation dataset (Phases 1-3) : ~15 min (depend du rate limit API)
- DAS (Phase 4) : ~10 min
- Entrainement Stage 1 + 2 (Phase 6) : ~20-30 min
- Evaluation (Phases 7-9) : ~5 min

---

## Limites connues

- **Taille du dataset** : 28 questions est le minimum pour un TP ; un dataset de 100+ exemples ameliorerait les resultats
- **Alignement token** : l'alignement par curseur caractere entre tokenizers teacher/student est approximatif
- **Evaluation** : le scoring automatique est un proxy ; une evaluation humaine serait plus fiable
- **Overfitting** : risque reel avec un petit dataset malgre LoRA
- **Dependance API** : la qualite des reponses depend du modele disponible sur Infomaniak

---

## Ressources

- [Papier DASD](https://github.com/D2I-ai/dasd-thinking) - Distribution-Aligned Sequence Distillation
- [Dataset de reference](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b)
- [Llama-Factory Documentation](https://llamafactory.readthedocs.io/en/latest/)
- [API Infomaniak AI](https://api.infomaniak.com/2/ai/48/openai/v1)
