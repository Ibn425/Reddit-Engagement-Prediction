# =============================================================================
# OPTIMISATION DES HYPERPARAMÈTRES PAR RANDOMIZED SEARCH
# =============================================================================
# Objectif : Trouver le meilleur modèle de classification parmi SVM,
#            Random Forest, Régression Logistique et XGBoost en utilisant
#            RandomizedSearchCV comme unique méthode d'optimisation.
#
# Pourquoi RandomizedSearch et non GridSearch ?
# ─────────────────────────────────────────────
# GridSearchCV teste TOUTES les combinaisons d'une grille fixe. Dès que
# l'espace de paramètres est grand (distributions continues ou nombreuses
# valeurs discrètes), le nombre de combinaisons explose et rend GridSearch
# impraticable en temps de calcul.
# RandomizedSearchCV est son ALTERNATIVE : il tire aléatoirement n_iter
# combinaisons depuis des distributions (continues ou discrètes), évalue
# chacune par validation croisée, et retourne la meilleure. Il est prouvé
# empiriquement qu'un bon résultat est trouvé avec bien moins d'itérations
# qu'un GridSearch exhaustif (Bergstra & Bengio, 2012).


# 0. IMPORTS

import pandas as pd                                    # Manipulation de données tabulaires
import numpy as np                                     # Calculs matriciels
import os                                              # Gestion du système de fichiers
import joblib                                          # Sérialisation des modèles sur disque

# Distributions continues et discrètes pour RandomizedSearch.
# loguniform(a, b) : tirage log-uniforme dans [a, b] - adapté aux paramètres
#                    dont l'effet est multiplicatif (C, learning_rate).
# uniform(a, b)    : tirage uniforme dans [a, a+b] - adapté aux fractions [0,1].
# randint(a, b)    : entier aléatoire dans [a, b[ - adapté aux paramètres discrets.
from scipy.stats import loguniform, randint, uniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# 1. PRÉPARATION DES DONNÉES

NOM_SUBREDDIT = "fanTheories"   # Identifiant du subreddit ciblé (adapter si besoin)

# Chargement des deux sources de données complémentaires :
#   - df     : données tabulaires (métadonnées de chaque post Reddit :
#              heure, score, nb de commentaires, etc.)
#   - X_bert : matrice d'embeddings BERT - chaque post est représenté par un
#              vecteur de ~768 dimensions encodant son sens sémantique.
df     = pd.read_pickle(f"fichier_IA/df_final_ia_{NOM_SUBREDDIT}.pkl")
X_bert = np.load(f"fichier_IA/matrice_X_semantique_{NOM_SUBREDDIT}.npy")

# Alignement des deux sources pour éviter tout décalage d'indices
# en cas de différence de taille entre les deux fichiers.
taille = min(len(df), len(X_bert))

# Suppression de la colonne texte brut et de la variable cible avant fusion.
# On ne doit jamais inclure la cible dans les features d'entrée du modèle.
X_meta = df.iloc[:taille].drop(columns=['vecteur_semantique', 'classe_engagement']).values

# Fusion horizontale : on colle les métadonnées et les embeddings côte à côte.
# Si X_meta a N colonnes et X_bert en a 768, X aura N+768 colonnes.
# Chaque ligne = la représentation complète d'un post.
X = np.hstack([X_meta, X_bert[:taille]])

# Variable cible : niveau d'engagement du post.
# Classe 0 = faible | 1 = moyen | 2 = fort | 3 = viral
y = df.iloc[:taille]['classe_engagement'].values

# ── Découpage Train / Test ────────────────────────────────────────────────────
# On réserve 20% des données pour l'évaluation finale (X_test).
# Ces données ne seront jamais utilisées pendant l'optimisation.
# stratify=y : garantit que chaque classe est représentée dans les mêmes
#              proportions dans le train et le test - indispensable quand
#              les classes sont déséquilibrées (ex: peu de posts 'viral').
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # Fixe le hasard -> reproductibilité du découpage
    stratify=y
)

# ── Normalisation ─────────────────────────────────────────────────────────────
# StandardScaler ramène chaque colonne à moyenne=0 et écart-type=1.
# Indispensable pour SVM et Régression Logistique, dont les algorithmes
# d'optimisation interne sont sensibles à l'échelle des valeurs.
#
# RÈGLE CRITIQUE - éviter le data leakage (fuite de données) :
#   fit_transform sur X_train : le scaler APPREND les statistiques du train
#                               (moyenne, écart-type) ET transforme.
#   transform sur X_test      : le scaler APPLIQUE les statistiques apprises
#                               sans les recalculer. Le test reste "inconnu".
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# 2. FONCTION CŒUR - OPTIMISATION PAR RANDOMIZED SEARCH

def executer_optimisation(clf, distributions, nom, n_iterations=100):
    """
    Optimise les hyperparamètres d'un classifieur scikit-learn via
    RandomizedSearchCV avec validation croisée stratifiée à 5 plis.

    Principe de fonctionnement
    --------------------------
    À chaque appel, la fonction tire aléatoirement ``n_iterations``
    combinaisons d'hyperparamètres depuis l'espace défini par
    ``distributions``. Chaque combinaison est évaluée par validation
    croisée à 5 plis sur ``X_train``. La combinaison ayant obtenu le
    meilleur score F1-weighted moyen est retenue, et le modèle
    correspondant - déjà entraîné sur l'intégralité de X_train - est
    retourné.

    Pourquoi RandomizedSearch et non GridSearch ?
    ---------------------------------------------
    Lorsque ``distributions`` contient des distributions continues
    (ex : ``loguniform``), l'espace de recherche est infini et un
    GridSearch exhaustif est impossible par définition. Même pour des
    espaces discrets larges, RandomizedSearch converge vers de bonnes
    solutions en un nombre d'itérations bien inférieur à GridSearch,
    pour un coût computationnel drastiquement réduit.

    Parameters
    ----------
    clf : estimateur scikit-learn
        Classifieur vierge à optimiser (ex : ``SVC(probability=True)``).
        Il ne doit contenir aucun hyperparamètre imposé, sauf contraintes
        techniques (probability=True, max_iter=2000, eval_metric…).
    distributions : dict
        Dictionnaire associant le nom de chaque hyperparamètre à sa
        distribution de tirage :
          - distribution scipy.stats (loguniform, uniform, randint) :
            tirage dans un espace continu ou discret.
          - liste Python (ex : ['rbf', 'linear']) :
            tirage uniforme parmi les valeurs de la liste.
    nom : str
        Nom du modèle utilisé pour les messages de log.
    n_iterations : int, optional (default=100)
        Budget de combinaisons à tester. Plus il est élevé, plus
        l'exploration est large, mais plus le temps de calcul augmente.
        Avec cv=5, chaque valeur de n_iterations correspond à
        ``n_iterations * 5`` entraînements du modèle.

    Returns
    -------
    best_estimator : estimateur scikit-learn
        Le meilleur estimateur trouvé, déjà entraîné sur X_train avec
        les hyperparamètres optimaux identifiés par RandomizedSearchCV.
    """
    print(f"\n{'='*60}")
    print(f"  Optimisation : {nom}  ({n_iterations} combinaisons | cv=5)")
    print(f"{'='*60}")

    # RandomizedSearchCV orchestre l'ensemble du processus :
    #   1. Tire n_iterations combinaisons depuis distributions.
    #   2. Pour chaque combinaison, entraîne et évalue le modèle
    #      par validation croisée à 5 plis sur X_train.
    #   3. Conserve la combinaison ayant le meilleur F1-weighted moyen.
    #   4. Ré-entraîne le modèle gagnant sur l'intégralité de X_train
    #      (refit=True par défaut).
    search = RandomizedSearchCV(
        estimator           = clf,
        param_distributions = distributions,
        n_iter              = n_iterations,   # Nombre de combinaisons testées
        cv                  = 5,              # Validation croisée à 5 plis
        scoring             = 'f1_weighted',  # Métrique d'évaluation
        n_jobs              = -1,             # Parallélisation sur tous les cœurs CPU
        random_state        = 42,             # Reproductibilité des tirages aléatoires
        verbose             = 1              # Affiche la progression
    )

    search.fit(X_train, y_train)

    # Affichage du meilleur résultat trouvé
    print(f"\n  Meilleurs hyperparamètres :")
    for param, val in search.best_params_.items():
        print(f"    {param:25s} = {val}")
    print(f"  Meilleur F1-weighted (CV) : {search.best_score_:.4f}")

    # best_estimator_ : modèle déjà entraîné avec les meilleurs hyperparamètres
    return search.best_estimator_


# 3. ESPACES DE DISTRIBUTIONS PAR MODÈLE
# Pour chaque modèle, on définit un dictionnaire d'espaces de distributions.
# L'usage de distributions continues (loguniform, uniform) plutôt que de
# simples listes fixes est fondamental : il permet à RandomizedSearch de tirer
# des valeurs PARTOUT dans un intervalle réel, pas uniquement aux quelques
# points choisis arbitrairement dans une liste. Cela rend l'exploration
# beaucoup plus riche et représentative de l'espace réel des paramètres.


# ── SVM (Support Vector Machine) ─────────────────────────────────────────────
# Principe : cherche un hyperplan séparateur à marge maximale.
# Le kernel projette les données dans un espace de dimension supérieure
# pour rendre les classes séparables linéairement.
#
# Justification de RandomizedSearch :
#   C sur loguniform = espace continu -> GridSearch impossible.
param_dist_svm = {
    # C : compromis régularisation / tolérance aux erreurs.
    'C':      loguniform(0.1, 10),

    # kernel : transformation de l'espace des données.
    # linear=droite, rbf=gaussienne (le plus polyvalent),
    # poly=polynôme, sigmoid=en S.
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],

    # gamma : rayon d'influence de chaque point (kernels non-linéaires).
    # scale=1/(n_features × var(X))
    'gamma':  ['scale'],

    # degree : degré du polynôme. Utilisé UNIQUEMENT si kernel='poly'.
    'degree': [2, 3]
}


# ── Random Forest ─────────────────────────────────────────────────────────────
# Principe : vote majoritaire de N arbres entraînés sur des sous-échantillons
# aléatoires des données ET des features (double aléatoire = bagging).
param_dist_rf = {
    # n_estimators : nombre d'arbres. Plus il y en a, plus le modèle est
    # stable, mais au-delà d'un seuil les gains marginaux disparaissent.
    'n_estimators':      randint(50, 150),

    # max_depth : profondeur maximale de chaque arbre.
    # None = croissance libre jusqu'à pureté complète -> risque de sur-apprentissage.
    # Valeurs finies = régularisation implicite.
    'max_depth':         [10, 15, 20],

    # min_samples_split : nombre minimum d'exemples pour diviser un nœud.
    # Plus élevé = arbres plus simples et plus régularisés.
    'min_samples_split': [2, 5],

    # max_features : nombre de features examinées à chaque division.
    'max_features':      ['sqrt', 'log2', None],

    # criterion : mesure de qualité d'une division.
    # gini = impureté de Gini.
    'criterion':         ['gini']
}


# ── Régression Logistique ─────────────────────────────────────────────────────
# Principe : modèle linéaire probabiliste. Combine les features par une
# équation linéaire transformée en probabilités via softmax (multiclasse).
# Contrainte : les couples (penalty, solver) ne sont pas tous compatibles.
# L-BFGS est un algorithme de la famille "Quasi-Newton". Au lieu d'avancer à l'aveugle, il analyse la courbure de tes données pour deviner où se trouve le point parfait (l'optimum).
# Mathématiquement, sa formule a besoin que la courbe soit "lisse" (dérivable partout). La pénalité L1 crée des "angles" brusques dans les mathématiques, c'est pour cela que lbfgs refuse catégoriquement de travailler avec L1.
param_dist_log = {
    'C':       loguniform(0.1, 10),
    
    # lbfgs est rapide et gère le multiclasse (4 classes Reddit), 
    # mais il n'accepte mathématiquement que la pénalité 'l2'.
    'penalty': ['l2'],
    
    # lbfgs rapide et efficace pour les problèmes de taille moyenne.
    'solver':  ['lbfgs']
}


# ── XGBoost (Gradient Boosting eXtrême)
# Principe : construit des arbres EN SÉRIE. Chaque arbre corrige les erreurs
# du précédent (boosting). Plus complexe que Random Forest -> plus d'hyperparamètres
param_dist_xgb = {
    # n_estimators : nombre de rounds de boosting (= nombre d'arbres).
    # Trop peu = sous-apprentissage. Trop = sur-apprentissage.
    'n_estimators':     randint(50, 150),

    # learning_rate : poids de la contribution de chaque arbre.
    # Petit = progression lente et précise (besoin de plus d'arbres).
    # Grand = risque de dépasser l'optimum.
    # Effet multiplicatif -> loguniform justifiée.
    'learning_rate':    loguniform(0.05, 0.2),

    # max_depth : profondeur des arbres faibles. Dans XGBoost, des arbres
    # peu profonds (3-6) suffisent généralement car ils se combinent.
    'max_depth':        randint(3, 6),

    # subsample : fraction des données utilisée à chaque round.
    'subsample':        uniform(0.8, 0.2),

    # colsample_bytree : fraction des features utilisée par arbre.
    'colsample_bytree': uniform(0.8, 0.2),

    # gamma : pénalité sur la création de nouvelles feuilles.
    'gamma':            uniform(0, 0.2)
}


# 4. EXÉCUTION - OPTIMISATION DES 4 MODÈLES
# Chaque modèle est passé avec ses distributions à executer_optimisation().
# L'ordre n'a aucune importance : les optimisations sont indépendantes.
#
# Notes techniques par modèle :
#   SVC(probability=True)         : active predict_proba() via calibration de Platt.
#   RandomForestClassifier(rs=42) : fixe l'aléatoire interne pour la reproductibilité.
#   LogisticRegression(max_iter)  : augmente la limite de convergence (défaut=100).
#   XGBClassifier(eval_metric)    : spécifie la métrique interne pour le multiclasse,
#                                   évite un avertissement à l'exécution.

modeles_a_tester = [
    (SVC(probability=True),
     param_dist_svm, "SVM"),

    (RandomForestClassifier(random_state=42),
     param_dist_rf, "Random Forest"),

    (LogisticRegression(max_iter=2000),
     param_dist_log, "Régression Logistique"),

    (XGBClassifier(eval_metric='mlogloss', random_state=42),
     param_dist_xgb, "XGBoost"),
]

# Dictionnaire de résultats : nom du modèle -> meilleur estimateur entraîné
resultats = {}


for clf, dist, nom in modeles_a_tester:
    resultats[nom] = executer_optimisation(clf, dist, nom, n_iterations=15)


# 5. ÉVALUATION FINALE SUR LE JEU DE TEST
# On évalue chaque modèle optimisé sur X_test - les données mises de côté
# dès le début et JAMAIS utilisées pendant l'optimisation ni l'entraînement.
# C'est le seul score qui mesure la vraie capacité de généralisation.
#
# Deux métriques sont affichées :
#   - F1-weighted global : permet de classer les modèles entre eux.
#   - classification_report : détaille précision, rappel et F1 PAR CLASSE
#     (0, 1, 2, 3), indispensable pour identifier les classes mal prédites.

print("\n" + "="*60)
print("  ÉVALUATION FINALE - DONNÉES DE TEST ")
print("="*60)

scores_finaux = {}

for nom, modele in resultats.items():
    # Prédiction sur les données de test
    predictions = modele.predict(X_test)

    # Score F1-weighted global sur le test
    f1 = f1_score(y_test, predictions, average='weighted')
    scores_finaux[nom] = f1

    print(f"\n  Modèle : {nom}")
    print(f"  F1-weighted (test) : {f1:.4f}")
    # Précision, Rappel et F1 pour chaque classe (0, 1, 2, 3)
    print(classification_report(y_test, predictions))


# 6. SÉLECTION DU CHAMPION ET SAUVEGARDE

# Le champion est le modèle avec le meilleur F1-weighted sur X_test
champion_nom    = max(scores_finaux, key=scores_finaux.get)
modele_champion = resultats[champion_nom]

print(f"\n{'='*60}")
print(f"  CHAMPION : {champion_nom} - F1 = {scores_finaux[champion_nom]:.4f}")
print(f"{'='*60}")

# ── Sauvegarde sur disque 
os.makedirs("modeles", exist_ok=True)

joblib.dump(modele_champion, f"modeles/modele_{NOM_SUBREDDIT}_export.joblib")
joblib.dump(scaler,          f"modeles/scaler_{NOM_SUBREDDIT}_export.joblib")

print(f"\n  Modèle sauvegardé -> modeles/modele_{NOM_SUBREDDIT}_export.joblib")
print(f"  Scaler sauvegardé -> modeles/scaler_{NOM_SUBREDDIT}_export.joblib")