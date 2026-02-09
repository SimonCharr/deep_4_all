# Compte Rendu TP2 : Du Scalaire au Tenseur - Le Tournoi de la Guilde

---

## Partie 1 : Introduction a PyTorch

### Points a aborder

1. **Tenseurs vs NumPy** : PyTorch = NumPy + GPU + autograd
   - `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`
   - Memes operations (addition, multiplication, reshaping) mais sur GPU

2. **Autograd** : le gradient se calcule automatiquement
   - `requires_grad=True` sur un tenseur active le tracking
   - `.backward()` calcule les gradients, `.grad` les stocke
   - Exemple : `x=2, y=x^2` -> `x.grad = 4` (derivee 2x)

3. **nn.Module** : classe de base pour tout modele PyTorch
   - Enregistre les parametres automatiquement (poids, biais)
   - Gere `model.train()` / `model.eval()` (active/desactive Dropout)
   - Permet `torch.save()` / `torch.load()` pour sauvegarder

4. **nn.Sequential** : empile des couches sans ecrire `forward()`
   - `nn.Linear(8, 6)` -> couche dense, `nn.ReLU()` -> activation

---

## Partie 2 : Le Tournoi de Generalisation

### Points a aborder

#### A. Le probleme
- Classification binaire : predire la survie d'un aventurier (8 features -> 0 ou 1)
- Le code baseline dans `train_oracle.py` est **volontairement mauvais** (dit en commentaire lignes 7-13)

#### B. Architecture du modele (`baseline_model.py:35-40`)
- MLP a 1 couche cachee : `Linear(8->6) -> ReLU -> Dropout -> Linear(6->1)`
- La sortie est un **logit** (pas une proba), d'ou l'utilisation de `BCEWithLogitsLoss` qui applique sigmoid + cross-entropy en interne (`train_oracle.py:196`)
- 61 parametres seulement (8x6+6 + 6x1+1)

#### C. Problemes du baseline et corrections

| Probleme | Ou dans le code | Correction |
|----------|----------------|------------|
| Pas de normalisation | `train_oracle.py:54-57` : le flag `normalize` est `False` par defaut | `--normalize` : ramene chaque feature a moyenne=0, std=1. Sans ca, `force` (0-100) ecrase `niveau_quete` (1-10) |
| Pas de shuffle | `train_oracle.py:174` : `shuffle=args.shuffle` passe au DataLoader | `--shuffle` : melange les donnees a chaque epoch, sinon le modele apprend l'ordre |
| Pas de dropout | `baseline_model.py:37` : `nn.Dropout(dropout)` avec `dropout=0.0` par defaut dans le CLI | `--dropout 0.5` : desactive 50% des neurones aleatoirement pendant le train, force la redondance |
| Pas de weight decay | `train_oracle.py:202` : `weight_decay=args.weight_decay` dans Adam | `--weight_decay 1e-4` : penalite L2 sur les poids, empeche les valeurs extremes |
| LR trop haut (0.1) | `train_oracle.py:363` : defaut a 0.1 | `--learning_rate 0.001` : valeur standard pour Adam, converge sans osciller |
| hidden_dim trop grand (256) | `train_oracle.py:345` : defaut a 256 | `--hidden_dim 6` : passe de 2561 a 61 parametres, le modele ne memorise plus |
| Pas d'early stopping | `train_oracle.py:267-269` : desactive par defaut | `--early_stopping` : arrete quand val_acc ne monte plus depuis 10 epochs, sauvegarde le meilleur modele |

#### D. La boucle d'entrainement (`train_oracle.py:90-115`)
- `model.train()` active le Dropout, `model.eval()` le desactive
- A chaque batch : `zero_grad()` -> forward -> loss -> `backward()` -> `step()`
- Le scheduler (`train_oracle.py:215-217`) divise le LR par 2 si val_acc stagne pendant 5 epochs

#### E. Resultat
- Baseline : Train 95%, Val 72% (gap 23% = overfitting)
- Optimise : Val ~90%, gap < 5%
- Commande finale :
```bash
uv run train_oracle.py --normalize --shuffle --early_stopping \
    --dropout 0.5 --weight_decay 1e-4 --learning_rate 0.001 --hidden_dim 6
```

#### F. Le twist des Terres Maudites
- Le test secret inverse les regles (force elevee = penalite au lieu de bonus)
- Les modeles qui ont memorise "force haute = survie" echouent
- Seuls les modeles regularises (petit, avec dropout/weight decay) generalisent car ils n'ont pas sur-appris une seule feature

#### G. Lecons cles
1. **Regularisation** (dropout, weight decay, petit modele) = essentiel contre l'overfitting
2. **Plus petit = souvent meilleur** : 61 params > 2561 params sur le test
3. **Analyser les donnees avant** : echelles differentes -> normaliser, donnees ordonnees -> shuffler
4. **Un parametre a la fois** : pour isoler l'effet de chaque changement
