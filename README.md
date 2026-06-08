# 🤖 Reddit Success Predictor

> Pipeline Machine Learning complet pour prédire le niveau d'engagement d'un post Reddit et générer des recommandations personnalisées — développé en L3 Informatique à l'Université Paris Cité.

[![App Live](https://img.shields.io/badge/🚀_App_Live-Streamlit-FF4B4B)](https://ibn425-reddit-engagement-prediction-app-uubkfl.streamlit.app)

---

## 🎯 Problématique

> *"Est-il possible, à partir des seules caractéristiques textuelles et contextuelles d'une publication Reddit, de prédire automatiquement son niveau d'engagement potentiel et de générer des recommandations d'amélioration exploitables ?"*

---

## 📊 Résultats

| Subreddit | Modèle Champion | F1-score |
|-----------|----------------|----------|
| r/relationships | XGBoost | **0.51** |
| r/football | XGBoost | 0.45 |
| r/unpopularopinion | XGBoost | 0.42 |
| r/confession | XGBoost | 0.41 |
| r/histoire | Régression Logistique | 0.41 |
| r/fanTheories | XGBoost | 0.39 |
| r/france | XGBoost / RF | 0.36 |
| r/LifeProTips | SVM | 0.34 |

> Baseline aléatoire sur 4 classes équilibrées : **0.25** — nos modèles atteignent jusqu'à **×2** cette référence.

---

## 🏗️ Architecture du pipeline

```
Post Reddit (titre + contenu)
          │
          ▼
┌─────────────────────────────────────────┐
│           TextProcessor.py              │
│  ┌─────────────────┐  ┌──────────────┐  │
│  │ Features méta   │  │  BERT        │  │
│  │ (10 dimensions) │  │  (384 dims)  │  │
│  │ heure, jour,    │  │  all-MiniLM  │  │
│  │ sentiment,      │  │  -L6-v2      │  │
│  │ majuscules...   │  │              │  │
│  └────────┬────────┘  └──────┬───────┘  │
│           └────────┬─────────┘          │
│                    ▼                    │
│         Vecteur final (394 dims)        │
└────────────────────┬────────────────────┘
                     ▼
          EngagementPredictor.py
          RandomizedSearchCV (15 iter, cv=5)
          4 modèles en compétition :
          SVM · XGBoost · Random Forest · Régression Logistique
                     │
                     ▼
          Modèle champion par subreddit (.joblib)
                     │
                     ▼
          ┌──────────────────────┐
          │     app.py           │
          │  Interface Streamlit │
          │  Prédiction + Reco   │
          └──────────────────────┘
```

---

## 📁 Structure du projet

```
📦 Reddit-Engagement-Prediction
 ┣ 📄 app.py                          # Interface Streamlit — pipeline d'inférence temps réel
 ┣ 📄 EngagementPredictor.py          # Entraînement, optimisation, sélection du modèle champion
 ┣ 📄 RedditFetcher.py                # Collecte via endpoints JSON Reddit (sans auth)
 ┣ 📄 TextProcessor.py                # Feature engineering + encodage BERT (394 dims)
 ┣ 📄 test_projet.py                  # Tests unitaires PyTest (fetcher, sentiment, recommandations)
 ┣ 📄 Analyse_Exploratoire_des_Donnees.ipynb  # EDA complète par subreddit
 ┣ 📂 modeles/
 │   ┣ 📄 modele_confession_export.joblib
 │   ┣ 📄 modele_unpopularopinion_export.joblib
 │   ┣ 📄 modele_football_export.joblib
 │   ┣ 📄 modele_france_export.joblib
 │   ┣ 📄 modele_histoire_export.joblib
 │   ┣ 📄 modele_fanTheories_export.joblib
 │   ┗ 📄 scaler_*.joblib              # StandardScaler par subreddit
 ┣ 📂 recommandations/                # Module de conseils personnalisés par subreddit
 ┗ 📂 mon_modele_local/               # Sentence-BERT all-MiniLM-L6-v2 (stocké localement)
```

---

## 🧠 Approche technique

### Features (394 dimensions par post)

**Méta-données (10 dims)** — issues de l'EDA :
- Temporelles : heure UTC, jour de semaine, mois
- Textuelles : nb mots titre/contenu, nb exclamations/interrogations
- Stylistiques : ratio majuscules (feature la plus négativement corrélée sur r/unpopularopinion : r = −0.112)
- Sentiment : score VADER titre + contenu

**Embeddings BERT (384 dims)** — modèle `all-MiniLM-L6-v2` :
- Encodage sémantique du titre + contenu
- Stocké localement (zéro dépendance réseau à l'inférence)
- Contribue à **plus de 70 %** du pouvoir prédictif total

### Variable cible : 4 classes par quartiles

| Classe | Label | Description |
|--------|-------|-------------|
| 0 | FLOP | Score + commentaires < Q1 |
| 1 | FAIBLE | Entre Q1 et médiane |
| 2 | BON | Entre médiane et Q3 |
| 3 | VIRAL | Score + commentaires > Q3 |

> Discrétisation **indépendante par subreddit** pour garantir 25 % par classe.

### Modélisation

- **Découpage** : 80 % entraînement / 20 % test, stratifié
- **Validation** : cross-validation stratifiée à 5 plis
- **Optimisation** : RandomizedSearchCV (15 itérations × 5 plis = 75 entraînements par modèle)
- **Métrique** : F1-score pondéré
- **Infrastructure** : CPU uniquement, machines personnelles, `n_jobs=-1`

---

## 🚀 Installation et lancement

### 1. Cloner le repo
```bash
git clone https://github.com/Ibn425/Reddit-Engagement-Prediction.git
cd Reddit-Engagement-Prediction
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

### 4. Lancer les tests
```bash
pytest test_projet.py -v
```

---

## 🌐 Subreddits supportés

r/confession · r/unpopularopinion · r/france · r/histoire · r/football · r/fanTheories · r/relationships · r/LifeProTips

---

## 🔧 Dépendances principales

| Bibliothèque | Version min. | Rôle |
|---|---|---|
| streamlit | 1.30.0 | Interface web |
| sentence-transformers | 2.2.0 | Encodage BERT |
| vaderSentiment | 3.3.0 | Analyse de sentiment |
| scikit-learn | 1.2.0 | ML + pipeline |
| xgboost | 1.7.0 | Modèle champion |
| pandas / numpy | 1.5.0 / 1.23.0 | Traitement données |
| requests | 2.28.0 | API Reddit |

---

## 👥 Équipe

Projet tutoré de L3 Informatique — Université Paris Cité (2025–2026), encadré par M. Elie El Debs.

- Abbas Muhammad
- Chergui Abdelhakim
- Mansouri Ghiles
- **Thierno Ibrahima Bah** — [LinkedIn](https://linkedin.com/in/thierno-ibrahimabah) · [GitHub](https://github.com/Ibn425)

---

## 📄 Licence

MIT
