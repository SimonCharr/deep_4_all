# TP4 - Distribution-Aligned Sequence Distillation (DASD)

**Thème** : Pokémon
**Auteur** : Simon Charrier / JBILOU Adam

---

## Objectif

Implémenter la méthode DASD pour distiller les capacités de raisonnement d'un modèle "teacher" (API Infomaniak) vers un modèle "étudiant" compact (Qwen3-4B) via fine-tuning LoRA.

---

## Ressources produites

| Ressource | Lien |
|-----------|------|
| **Modèle (adapter LoRA)** | [simoncharr/pokemon-dasd-lora-adapter](https://huggingface.co/simoncharr/pokemon-dasd-lora-adapter) |
| **Dataset** | [simoncharr/pokemon-dasd-dataset](https://huggingface.co/datasets/simoncharr/pokemon-dasd-dataset) |
| **Notebook** | `tp4_dasd_pokemon.ipynb` |

---

## Méthodologie

### 1. Génération des questions (local)

Plutôt que d'utiliser un dataset existant, nous avons généré **1000 questions Pokémon** programmatiquement à partir de templates.

**Pourquoi ce choix ?**
- Contrôle total sur la diversité et la qualité des questions
- Couverture exhaustive des mécaniques Pokémon (types, STAB, stats, stratégie)
- Évite les biais d'un dataset pré-existant

**Implémentation :**
- Base de données de ~200 Pokémon (Gen 1-9) avec leurs types
- ~60 attaques avec type, puissance et catégorie
- 9 catégories de questions avec templates paramétrables :
  - `efficacite_types` : multiplicateurs de dégâts
  - `stab_et_degats` : calculs STAB
  - `stats_et_comparaisons` : analyse BST
  - `evolution` : chaînes d'évolution
  - `mecaniques_combat` : Rochers Furtifs, météo, etc.
  - `strategies_specifiques` : sets compétitifs
  - `team_building` : synergies d'équipe
  - `formats_et_tiers` : LC, OU, VGC, etc.
  - `calculs_avances` : formules de dégâts

### 2. Génération des réponses (API Infomaniak)

Les réponses sont générées via l'API Infomaniak avec le modèle `qwen3` à deux températures :

| Stage | Température | Objectif | Exemples générés |
|-------|-------------|----------|------------------|
| Stage 1 | 0.3 | Stabilité, réponses cohérentes | 500 |
| Stage 2 | 0.9 | Diversité, exploration | 500 |

**System prompt utilisé :**
```
Tu es un expert Pokemon competitif. Pour chaque question, raisonne etape par etape
a l'interieur de balises <reasoning>...</reasoning> avant de donner ta reponse finale.
```

**Filtrage qualité :**
- Longueur minimale (>100 caractères)
- Présence obligatoire des balises `<reasoning>...</reasoning>`
- Retry avec backoff exponentiel en cas d'erreur API

### 3. Divergence-Aware Sampling (DAS)

Le DAS permet d'identifier les exemples où le modèle étudiant a le plus à apprendre.

**Calcul des scores :**
1. Charger le modèle étudiant (Qwen3-4B en 4-bit)
2. Pour chaque réponse, calculer les log-probabilités token par token
3. Comparer avec les logprobs du teacher (fournis par l'API)
4. Classifier chaque phrase :
   - **Teacher Sentence** : P_teacher >> P_student (valeur pédagogique)
   - **Shared Sentence** : P_teacher ≈ P_student (neutre)
   - **Student Sentence** : P_student > P_teacher (bruit à rejeter)

**Filtrage final :** Conservation des exemples avec une densité suffisante de Teacher Sentences (divergence > 0.1).

**Pourquoi ne garder que ~60% des données ?**

L'objectif du DASD n'est pas d'entraîner sur tout le dataset, mais sur les exemples les plus **pédagogiquement utiles** :

| Données | % gardé | Justification |
|---------|---------|---------------|
| **Teacher Sentences** (divergence élevée) | 100% | Le teacher sait, l'étudiant ignore → apprentissage maximal |
| **Shared Sentences** (divergence faible) | Partiel | Connaissances déjà acquises → peu de valeur ajoutée |
| **Student Sentences** (divergence négative) | 0% | L'étudiant est trop confiant → risque de renforcer des erreurs |

En filtrant les 40% de données à faible valeur pédagogique, on obtient un entraînement plus efficace : le modèle se concentre sur ce qu'il ne sait pas encore plutôt que de "réviser" des connaissances triviales ou d'apprendre du bruit.

### 4. Entraînement (Google Colab - T4 GPU)

**Configuration LoRA :**
- `rank` : 8
- `alpha` : 16
- `target_modules` : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Stage 1 (données tau=0.3) :**
```yaml
dataset: pokemon_stage1
num_train_epochs: 5
learning_rate: 1.0e-4
cutoff_len: 2048
```

**Stage 2 (données tau=0.9) :**
```yaml
adapter_name_or_path: saves/pokemon-stage1  # Charge l'adapter du stage 1
dataset: pokemon_stage2
num_train_epochs: 3
learning_rate: 5.0e-5  # LR réduit pour affiner
```

---

## Résultats

### Courbes de loss

![Loss curves](LLaMA-Factory/loss_curves.png) -> Voir git pour l'image

| Stage | Loss initiale | Loss finale | Steps |
|-------|---------------|-------------|-------|
| Stage 1 | ~1.5 | 0.55 | 375 |
| Stage 2 | ~0.95 | 0.82 | 225 |

**Observations :**
- **Stage 1** : Convergence régulière et stable, le modèle apprend bien les patterns de raisonnement structuré
- **Stage 2** : Loss plus volatile (attendu avec haute température), mais tendance descendante

### Analyse DAS

![DAS Analysis](LLaMA-Factory/das_analysis.png) -> Voir git pour l'image

**Distribution des divergences :**
- Stage 1 (tau=0.3) : divergence moyenne ~0.35, distribution centrée
- Stage 2 (tau=0.9) : divergence moyenne ~0.45, distribution plus étalée

**Classification des phrases :**
| Type | Stage 1 | Stage 2 |
|------|---------|---------|
| Teacher | ~8000 | ~14000 |
| Shared | ~3500 | ~12000 |
| Student | ~1000 | ~500 |

**Interprétation :**
- Le stage 2 contient plus de "Teacher Sentences", ce qui confirme que les données haute température apportent plus de diversité pédagogique
- Très peu de "Student Sentences" (bruit), indiquant une bonne qualité des données générées

### Dataset final

Après filtrage DAS, **600 exemples** conservés (300 par stage) au format Alpaca.

---

## Conclusions

### Ce qui a fonctionné

1. **Génération programmatique des questions** : Permet un contrôle précis sur la diversité et évite les biais
2. **Approche 2 stages** : Le stage 1 stabilise l'apprentissage, le stage 2 enrichit avec de la diversité
3. **Filtrage DAS** : Élimine efficacement le bruit (Student Sentences faibles)
4. **Format de raisonnement structuré** : Les balises `<reasoning>` forcent une réponse explicable

### Limitations observées

1. **Hallucinations du teacher** : Certaines réponses contiennent des Pokémon ou mécaniques inventés
2. **Taille du dataset** : 600 exemples reste modeste pour un fine-tuning robuste
3. **Évaluation quantitative** : Pas de benchmark formel (ex: GSM8K) pour mesurer l'amélioration

### Améliorations possibles

- Augmenter le nombre de questions (2000-5000)
- Ajouter une étape de validation manuelle des réponses
- Implémenter une évaluation sur un benchmark de raisonnement
- Tester avec d'autres modèles étudiants (Llama, Mistral)
