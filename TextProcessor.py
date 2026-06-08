# PIPELINE DE PREPARATION DES DONNEES
# Objectif principal : Transformer un corpus textuel brut issu de Reddit en un 
# espace de caracteristiques (Feature Space) numerique, dense et standardise.

import pandas as pd
import numpy as np
import re
import os
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def supprimer_valeurs_aberrantes(df):
    """
    Filtrage statistique conditionnel base sur la methode non parametrique 
    de l'Intervalle Interquartile (Tukey's IQR).
    
    Justification : Les reseaux sociaux presentent des distributions d'engagement 
    fortement asymetriques (Loi de puissance). Ce filtre isole le signal utile 
    en eliminant les donnees extremes, tout en evitant de penaliser les posts 
    trop recents dont le cycle d'engagement n'est pas encore acheve (censure a droite).
    """
    df_clean = df.copy()
    
    # 1. Agregation de la metrique d'engagement cible
    # L'imputation par 0 garantit que les donnees manquantes ne generent pas de NaN lors de la somme.
    score = df_clean['score'].fillna(0)
    commentaires = df_clean['nb_commentaires'].fillna(0)
    df_clean['engagement_brut'] = score + commentaires

    # 2. Evaluation de la maturite temporelle de l'observation
    # La mediane sert de separateur robuste pour distinguer les publications 
    # "matures" des publications "recentes".
    if 'created_utc' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['created_utc'], unit='s')
        date_max = df_clean['date'].max()
        df_clean['age_jours'] = (date_max - df_clean['date']).dt.days
        age_mediane = df_clean['age_jours'].median()
    else:
        # Fallback conservateur si les metadonnees temporelles sont corrompues
        age_mediane = 0 
        df_clean['age_jours'] = 0

    # 3. Calcul des limites de distribution
    # Le multiplicateur 1.5 sur l'IQR est le standard empirique pour detecter les outliers severes.
    Q1 = df_clean['engagement_brut'].quantile(0.25)
    Q3 = df_clean['engagement_brut'].quantile(0.75)
    IQR = Q3 - Q1
    seuil_outlier_haut = Q3 + 1.5 * IQR

    # 4. Definition des regles d'exclusion (Masques booleens)
    # Suppression de l'hyper-viralite mature (Biais de modele)
    condition_outliers_anciens = (df_clean['engagement_brut'] > seuil_outlier_haut) & (df_clean['age_jours'] > age_mediane)
    # Suppression du bruit statistique (Flop confirme)
    condition_flops_anciens = (df_clean['engagement_brut'] == 0) & (df_clean['age_jours'] > age_mediane)
    
    masque_suppression = condition_outliers_anciens | condition_flops_anciens

    # 5. Application du filtrage et tracabilite
    taille_initiale = len(df_clean)
    df_clean = df_clean[~masque_suppression].copy()
    
    print("\nRESULTATS DU FILTRAGE STATISTIQUE (OUTLIERS)")
    print(f"Seuil de tolerance superieur (IQR) : > {seuil_outlier_haut:.2f} points d'engagement")
    print(f"Observations hyper-virales retirees : {condition_outliers_anciens.sum()}")
    print(f"Observations non-engageantes retirees : {condition_flops_anciens.sum()}")
    print(f"Taux de retention : {(len(df_clean) / taille_initiale) * 100:.1f}% des donnees initiales.\n")
    
    return df_clean


def pipeline_preparation_nlp(df_brut, modele_bert=None, analyseur_sentiment=None):
    """
    Orchestrateur de transformation des descripteurs (Feature Engineering).
    Convertit la structure heterogene d'une publication en une serie de 
    caracteristiques exploitables par des algorithmes d'apprentissage supervise.
    """
    print("[PIPELINE] Demarrage du Feature Engineering...")
    df = df_brut.copy()
    
    # AXE 1 : NETTOYAGE STRUCTUREL ET FUSION TEXTUELLE
    # L'analyse semantique necessite l'integralite du contexte. Le titre 
    # et le contenu sont concatenes pour former le document final.
    df = df.dropna(subset=['titre', 'contenu'])
    df['texte_complet'] = df['titre'] + ". " + df['contenu']

    # AXE 2 : DESCRIPTEURS TEMPORELS
    # Extraction de la cyclicite pour capter les comportements d'audience.
    if 'created_utc' in df.columns:
        dt = pd.to_datetime(df['created_utc'], unit='s')
        df['heure'] = dt.dt.hour
        df['jour_semaine'] = dt.dt.dayofweek
        df['mois'] = dt.dt.month
        
    # AXE 3 : DESCRIPTEURS MORPHOLOGIQUES (Proxy d'Effort et d'Emotion)
    # La verbosite et l'usage de la ponctuation agissent comme des proxys 
    # pour mesurer l'investissement de l'auteur et la nature du discours.
    df['longueur_titre'] = df['titre'].apply(lambda x: len(str(x).split()))
    df['longueur_contenu'] = df['contenu'].apply(lambda x: len(str(x).split()))
    
    df['nb_exclamation'] = df['texte_complet'].str.count('!')
    df['nb_interrogation'] = df['texte_complet'].str.count(r'\?')
    
    def calculer_ratio_majuscules(texte):
        """Quantifie l'intensite typographique."""
        texte_str = str(texte)
        return sum(1 for c in texte_str if c.isupper()) / len(texte_str) if len(texte_str) > 0 else 0.0

    df['ratio_majuscules'] = df['titre'].apply(calculer_ratio_majuscules)

    # AXE 4 : ANALYSE DE SENTIMENT LEXICALE (VADER)
    # Approche par dictionnaire pour extraire la polarite 
    # continue (Compound Score de -1.0 a 1.0) sans necessiter d'entrainement.
    if analyseur_sentiment is not None:
        df['sentiment_titre'] = df['titre'].apply(lambda x: analyseur_sentiment.polarity_scores(x)['compound'])
        df['sentiment_contenu_score'] = df['contenu'].apply(lambda x: analyseur_sentiment.polarity_scores(x)['compound'])

    # AXE 5 : DISCRETISATION DE LA CIBLE (TARGET VARIABLE)
    # Transformation du probleme de regression en classification via quantiles,
    # garantissant des classes parfaitement equilibrees (Qcut).
    if 'classe_engagement' not in df.columns and 'engagement_brut' in df.columns:
        df['classe_engagement'] = pd.qcut(df['engagement_brut'], q=4, labels=[0, 1, 2, 3], duplicates='drop')

    # AXE 6 : PLONGEMENT SEMANTIQUE DENSE (SENTENCE-BERT)
    # Transformation bidirectionnelle du texte en un vecteur dense de 384 dimensions,
    # permettant aux modeles de capter le sens latent au-dela de la syntaxe.
    if modele_bert is not None and 'vecteur_semantique' not in df.columns:
        print("[PIPELINE] Inference du modele de langage (S-BERT) en cours...")
        vecteurs = modele_bert.encode(df['texte_complet'].tolist(), show_progress_bar=True)
        df['vecteur_semantique'] = list(vecteurs)

    # AXE 7 : PROJECTION FINALE
    # Isolation stricte des seules variables legitimes pour la modelisation,
    # prevenant toute fuite de donnees (Data Leakage).
    variables_requises = [
        "heure", "jour_semaine", "mois", "longueur_titre", "longueur_contenu",
        "nb_exclamation", "nb_interrogation", "ratio_majuscules", 
        "classe_engagement", "sentiment_titre", "sentiment_contenu_score", 
        "vecteur_semantique"
    ]
    
    colonnes_finales = [col for col in variables_requises if col in df.columns]
    df_propre = df[colonnes_finales]
    
    print("Generation de la Feature Matrix terminee.")
    return df_propre


# ROUTINE D'EXECUTION BATCH (MAIN PROCESS)
# Ce bloc agit comme le point d'entree du programme. Il securise le workflow 
# ETL en gerant les chemins de fichiers, les erreurs de dependances et les 
# sauvegardes specifiques aux types de donnees generes.
if __name__ == "__main__":
    
    # Parametrage global de l'espace de travail
    NOM_SUBREDDIT = "NOM DU SUBRREDDIT"  # A personnaliser selon le corpus cible
    FICHIER_BRUT = f"{NOM_SUBREDDIT}.csv"
    DOSSIER_SORTIE = "fichier_IA"
    
    # Definition des artefacts cibles pour l'interface de modelisation
    CHEMIN_DF_FINAL = f"{DOSSIER_SORTIE}/df_final_ia_{NOM_SUBREDDIT}.pkl"
    CHEMIN_MATRICE_BERT = f"{DOSSIER_SORTIE}/matrice_X_semantique_{NOM_SUBREDDIT}.npy"
    

    # Programation defensive : verification de la disponibilite de la source
    if not os.path.exists(FICHIER_BRUT):
        print(f"Le fichier source '{FICHIER_BRUT}' n'a pas ete detecte.")
    else:
        # Creation du repertoire de livraison s'il n'existe pas
        os.makedirs(DOSSIER_SORTIE, exist_ok=True)

        print("[ETAPE 1] Importation du dataset brut en memoire...")
        df_donnees_brutes = pd.read_csv(FICHIER_BRUT)
        
        print("[ETAPE 2] Execution du filtrage des valeurs aberrantes...")
        df_sans_outliers = supprimer_valeurs_aberrantes(df_donnees_brutes)
        
        print("[ETAPE 3] Instanciation des modeles heuristiques (VADER)...")
        analyseur_vader = SentimentIntensityAnalyzer()
        
        # Gestion intelligente des dependances lourdes (Modele local vs Distant)
        dossier_modele_local = "mon_modele_local"
        try:
            modele_encodeur = SentenceTransformer(dossier_modele_local)
            print(f"          Resolution reussie : Modele local '{dossier_modele_local}' charge.")
        except Exception as e:
            print(f"[ATTENTION] Modele local introuvable ou corrompu : {e}")
            print("          Fallback deploie : Telechargement du depot HuggingFace (all-MiniLM-L6-v2)...")
            modele_encodeur = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("[ETAPE 4] Application des transformations vectorielles (Pipeline)...")
        df_nettoye = pipeline_preparation_nlp(
            df_brut=df_sans_outliers,
            modele_bert=modele_encodeur,
            analyseur_sentiment=analyseur_vader
        )
        
        print("[ETAPE 5] Serialization des embeddings (Format Numpy)...")
        # Transformation de la Serie Pandas en matrice Numpy pour des performances accrues lors du ML
        matrice_bert = np.stack(df_nettoye['vecteur_semantique'].values)
        np.save(CHEMIN_MATRICE_BERT, matrice_bert)
        print(f"          Artefact genere : {CHEMIN_MATRICE_BERT}")
        
        print("[ETAPE 6] Serialization des metadonnees (Format Pickle)...")
        # L'utilisation du format Pickle preserve l'integrite des types de donnees complexes
        df_nettoye.to_pickle(CHEMIN_DF_FINAL)
        print(f"          Artefact genere : {CHEMIN_DF_FINAL}")
        
        
        print("Le pipeline s'est cloture sans erreur. Les artefacts sont prets pour l'apprentissage.")