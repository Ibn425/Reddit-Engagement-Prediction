"""
app.py — Interface principale du Reddit Success Predictor

Ce module constitue le point d'entrée de l'application Streamlit.
Il orchestre l'ensemble du pipeline de prédiction :

    1. Chargement des outils IA (VADER + Sentence-BERT)
    2. Extraction des features textuelles et méta-données
    3. Encodage sémantique via Sentence-BERT
    4. Prédiction de la classe d'engagement (FLOP / FAIBLE / BON / VIRAL)
       pour chaque subreddit sélectionné par l'utilisateur
    5. Génération de conseils personnalisés selon les règles métiers
       spécifiques à chaque communauté Reddit
    6. Affichage des résultats dans une interface visuelle claire

Dépendances principales :
    - streamlit          : interface web interactive
    - vaderSentiment     : analyse de sentiment (polarité du texte)
    - sentence_transformers : encodage sémantique (BERT)
    - joblib             : chargement des modèles ML entraînés
    - numpy              : manipulation des vecteurs de features
"""

import streamlit as st
import os
import joblib
import numpy as np
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer


# 0. IMPORT DES RÈGLES MÉTIERS (CONSEILS SPÉCIFIQUES)
# Chaque module contient une fonction qui analyse le titre et le contenu
# du post selon les tendances statistiques observées lors de l'EDA
# (Exploratory Data Analysis) sur les données du subreddit correspondant.
# Ces fonctions retournent une liste de conseils personnalisés (strings).
from recommandations.conseils_rfrance        import generer_conseils_rfrance
from recommandations.conseils_rhistoire      import generer_conseils_rhistoire
from recommandations.Reco_fanTheories        import generer_conseils_fantheories
from recommandations.Reco_football           import generer_conseils_football
from recommandations.conseil_LifeProTips_relationships import conseil_LifeProTips, conseils_relationships
from recommandations.conseils_confession     import generer_conseils_confession
from recommandations.conseils_unpopularopinion import generer_conseils_unpopularopinion


# CONFIGURATION DE LA PAGE & CSS GLOBAL
# set_page_config() doit impérativement être le 1er appel Streamlit
# du script — toute instruction st.* avant lui lève une erreur.
st.set_page_config(
    page_title="Reddit Success Predictor",
    page_icon="📈",
    layout="centered"       # centré sur ~700px, adapté à la lecture de résultats
)

# Injection du CSS global via st.markdown(unsafe_allow_html=True).
# On ne définit ici que les classes réutilisées par les blocs st.markdown
# statiques (titre, header, bouton). Tous les blocs HTML dynamiques
# (cartes de résultats) utilisent des styles inline pour éviter tout
# conflit de guillemets dans les f-strings Python.
st.markdown("""
<style>
    /* Fond général de l'application */
    .stApp { background-color: #f4f6fb; }

    /* Titre principal centré */
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 4px;
    }

    /* Icône décorative au-dessus du titre */
    .header-icon {
        text-align: center;
        font-size: 40px;
        margin-bottom: 10px;
    }

    /* En-tête de la section résultats */
    .output-header {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 16px;
    }

    /* Bouton principal dégradé violet */
    .stButton > button {
        background: linear-gradient(90deg, #7c5cbf, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 0;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    /* Labels des champs de saisie en gras */
    label { font-weight: 600 !important; color: #333 !important; }
</style>
""", unsafe_allow_html=True)


# 1. CHARGEMENT DES OUTILS IA (CACHE)
@st.cache_resource
def charger_outils_ia():
    """
    Charge et met en cache les deux outils d'analyse textuelle.

    Utilise @st.cache_resource pour ne charger ces ressources qu'une
    seule fois au démarrage de l'application, indépendamment du nombre
    de rechargements de page ou d'interactions utilisateur.
    Cela évite de recharger le modèle BERT (plusieurs centaines de Mo)
    à chaque interaction, ce qui serait rédhibitoire en production.

    Returns
    -------
    analyseur : SentimentIntensityAnalyzer
        Outil VADER pour l'analyse de polarité (compound score ∈ [-1, 1]).
        Rapide, basé sur un lexique, ne nécessite pas de GPU.

    modele_bert : SentenceTransformer
        Modèle BERT local pour l'encodage sémantique dense (vecteur 384-d).
        Chargé depuis le dossier local './mon_modele_local' pour éviter
        toute dépendance réseau en production.
    """
    analyseur   = SentimentIntensityAnalyzer()
    modele_bert = SentenceTransformer('./mon_modele_local')
    return analyseur, modele_bert

# Instanciation globale — ces variables sont utilisées dans
# preparer_donnees_brutes() et dans le calcul des features d'affichage.
analyseur_vader, modele_semantique = charger_outils_ia()


# 2. FONCTIONS DE TRAITEMENT DES FEATURES

def calculer_ratio_majuscules(texte):
    """
    Calcule la proportion de lettres majuscules dans un texte.

    Cette feature capture le style de rédaction : un ratio élevé
    peut indiquer un titre "clickbait" ou agressif, ce qui influence
    l'engagement différemment selon les communautés Reddit.

    Parameters
    ----------
    texte : str
        Le texte à analyser (généralement le titre du post).

    Returns
    -------
    float
        Ratio ∈ [0.0, 1.0] — proportion de caractères en majuscule
        sur le total des caractères. Retourne 0.0 si le texte est vide.
    """
    texte_str = str(texte)
    if len(texte_str) == 0:
        return 0.0
    return sum(1 for c in texte_str if c.isupper()) / len(texte_str)


def decouper_texte(texte, max_mots=200):
    """
    Découpe un texte long en morceaux de taille fixe (en nombre de mots).

    Sentence-BERT a une limite de tokens en entrée (généralement 256 ou 512
    selon le modèle). Pour les posts très longs, on découpe le texte en
    segments de max_mots mots afin d'encoder chaque segment séparément,
    puis on moyenne les vecteurs résultants dans encoder_et_moyenner().

    Parameters
    ----------
    texte    : str  — texte à découper
    max_mots : int  — nombre maximum de mots par segment (défaut : 200)

    Returns
    -------
    list[str]
        Liste de segments textuels. Contient au moins [""] si le texte
        est vide, pour éviter une erreur lors de l'encodage.
    """
    mots = texte.split()
    if not mots:
        return [""]
    return [" ".join(mots[i : i + max_mots]) for i in range(0, len(mots), max_mots)]


def encoder_et_moyenner(texte, modele):
    """
    Encode un texte en vecteur sémantique dense via Sentence-BERT.

    Pour les textes dépassant la limite de tokens du modèle, le texte
    est d'abord découpé en segments (via decouper_texte), chaque segment
    est encodé indépendamment, puis les vecteurs sont moyennés.
    Cette stratégie de pooling préserve l'information sémantique globale
    du document sans tronquer arbitrairement le contenu.

    Parameters
    ----------
    texte  : str               — texte complet à encoder
    modele : SentenceTransformer — modèle BERT local chargé en cache

    Returns
    -------
    np.ndarray
        Vecteur 1D de dimension égale à la taille de sortie du modèle
        (ex. 384 dimensions pour 'all-MiniLM-L6-v2').
    """
    morceaux = decouper_texte(texte, max_mots=200)
    vecteurs = modele.encode(morceaux)       # shape : (n_segments, dim)
    return np.mean(vecteurs, axis=0)         # shape : (dim,)


def preparer_donnees_brutes(titre, contenu):
    """
    Construit le vecteur de features complet à partir du texte brut.

    Ce vecteur est la concaténation de deux parties :
      - Meta-features (10 dimensions) : features numériques extraites
        manuellement du texte et du contexte temporel de publication.
      - Vecteur BERT (384 dimensions) : représentation sémantique dense
        du texte complet (titre + contenu).

    L'ordre et le nombre exact des meta-features doit correspondre
    EXACTEMENT à ce qui a été utilisé lors de l'entraînement des modèles,
    sans quoi les prédictions seront incohérentes.

    Meta-features (dans l'ordre) :
        [0]  heure de publication (0-23)
        [1]  jour de la semaine   (0=lundi … 6=dimanche)
        [2]  mois de publication  (1-12)
        [3]  nombre de mots dans le titre
        [4]  nombre de mots dans le contenu
        [5]  nombre de points d'exclamation (titre + contenu)
        [6]  nombre de points d'interrogation (titre + contenu)
        [7]  ratio de majuscules dans le titre (∈ [0, 1])
        [8]  score de polarité VADER du titre  (∈ [-1, 1])
        [9]  score de polarité VADER du contenu (∈ [-1, 1])

    Parameters
    ----------
    titre   : str — titre du post Reddit saisi par l'utilisateur
    contenu : str — corps du post Reddit saisi par l'utilisateur

    Returns
    -------
    np.ndarray
        Matrice de shape (1, 394) prête à être passée au scaler
        puis au modèle de classification.
    """
    texte_complet = titre + ". " + contenu
    maintenant    = datetime.now()

    # Construction des 10 meta-features dans l'ordre d'entraînement
    meta_features = [
        maintenant.hour,                                          # [0] heure
        maintenant.weekday(),                                     # [1] jour semaine
        maintenant.month,                                         # [2] mois
        len(titre.split()),                                       # [3] mots titre
        len(contenu.split()),                                     # [4] mots contenu
        texte_complet.count('!'),                                 # [5] exclamations
        len(re.findall(r'\?', texte_complet)),                    # [6] interrogations
        calculer_ratio_majuscules(titre),                         # [7] ratio majuscules
        analyseur_vader.polarity_scores(titre)['compound'],       # [8] polarité titre
        analyseur_vader.polarity_scores(contenu)['compound']      # [9] polarité contenu
    ]

    # Encodage sémantique dense du texte complet (384 dimensions)
    vecteur_bert = encoder_et_moyenner(texte_complet, modele_semantique)

    # Concaténation : [10 meta-features | 384 dims BERT] → shape (394,)
    X_brut = np.concatenate([meta_features, vecteur_bert])
    return X_brut.reshape(1, -1)   # reshape en (1, 394) pour sklearn


# 3. CONFIGURATION DES SUBREDDITS ET CLASSES

# Mapping entier → label de classe (correspond aux labels d'entraînement)
# 0 = très faible engagement  /  3 = engagement viral
CLASSES_MAP   = {0: "FLOP", 1: "FAIBLE", 2: "BON", 3: "VIRAL"}

# Ordre fixe des classes pour l'affichage des pastilles (du pire au meilleur)
ORDRE_CLASSES = ["FLOP", "FAIBLE", "BON", "VIRAL"]

# Liste des 8 subreddits supportés par l'application.
# Chacun doit avoir un modèle .joblib et un scaler .joblib dans modeles/
SUBREDDITS = [
    "r/fanTheories",
    "r/unpopularopinion",
    "r/confession",
    "r/relationships",
    "r/france",
    "r/football",
    "r/LifeProTips",
    "r/histoire"
]

# Chaque classe a sa propre couleur sémantique :
#   FLOP  -> rouge   (mauvais résultat attendu)
#   FAIBLE -> orange (résultat moyen)
#   BON   -> vert    (bon résultat)
#   VIRAL -> bleu    (résultat exceptionnel)
STYLES_ACTIFS = {
    "FLOP":   "background:#fde8e8; color:#c0392b; border:1.5px solid #c0392b;",
    "FAIBLE": "background:#fff3e0; color:#c87000; border:1.5px solid #e67e22;",
    "BON":    "background:#e8f5e9; color:#1e7e34; border:1.5px solid #27ae60;",
    "VIRAL":  "background:#e8eaf6; color:#2c3db0; border:1.5px solid #3949ab;",
}

# Couleur du point indicateur (dot) dans la pastille active.
# Doit correspondre à la couleur de texte de STYLES_ACTIFS.
COULEURS_DOT = {
    "FLOP":   "#c0392b",
    "FAIBLE": "#e67e22",
    "BON":    "#27ae60",
    "VIRAL":  "#3949ab",
}

# Dictionnaire de routage : associe chaque subreddit à sa fonction
# de génération de conseils spécifiques.
# La clé doit correspondre EXACTEMENT aux valeurs de la liste SUBREDDITS.
ROUTAGE_CONSEILS = {
    "r/france":           generer_conseils_rfrance,
    "r/histoire":         generer_conseils_rhistoire,
    "r/fanTheories":      generer_conseils_fantheories,
    "r/football":         generer_conseils_football,
    "r/LifeProTips":      conseil_LifeProTips,
    "r/relationships":    conseils_relationships,
    "r/confession":       generer_conseils_confession,
    "r/unpopularopinion": generer_conseils_unpopularopinion,
}


# 4. GÉNÉRATEUR DE CONSEILS

def generer_conseils(titre, contenu, subreddit, classe):
    """
    Génère une liste de conseils personnalisés pour améliorer un post Reddit.

    Combine trois niveaux de recommandations :
      1. Règles universelles : vérifications basiques valables pour
         tous les subreddits (longueur du titre, etc.).
      2. Règles spécifiques au subreddit : issues de l'EDA réalisée
         sur les données historiques de chaque communauté. Ces règles
         sont définies dans les modules du dossier recommandations/.
      3. Message lié à la prédiction IA : rappel contextuel si la
         classe prédite est FLOP pour inciter à retravailler le post.

    Le nombre de conseils retournés varie selon la classe prédite :
      - VIRAL  → 1 message de félicitation, aucun conseil (post optimal)
      - BON    → max 3 conseils (post déjà bien, ajustements mineurs)
      - FAIBLE → max 4 conseils
      - FLOP   → max 5 conseils (post à retravailler en profondeur)

    Parameters
    ----------
    titre     : str — titre du post saisi par l'utilisateur
    contenu   : str — corps du post saisi par l'utilisateur
    subreddit : str — identifiant du subreddit (ex. "r/france")
    classe    : str — classe prédite par le modèle ML ("FLOP", "FAIBLE", "BON", "VIRAL")

    Returns
    -------
    list[str]
        Liste de conseils textuels, chacun prêt à être affiché
        directement dans l'interface (sans formatage supplémentaire).
    """
    conseils_finaux = []

    # Cas VIRAL : le post est déjà optimal, on félicite et on s'arrête
    if classe == "VIRAL":
        return ["Votre post est déjà parfaitement optimisé pour cette communauté, ne changez rien !"]

    # --- Règle universelle 1 : titre trop court ---
    # Un titre de moins de 15 caractères est généralement insuffisant
    # pour capter l'attention sur Reddit, quelle que soit la communauté.
    if len(titre) < 15:
        conseils_finaux.append("Votre titre est très court. Développez-le pour susciter l'intérêt.")

    # --- Règles spécifiques au subreddit (issues de l'EDA) ---
    # On cherche la fonction correspondante dans le dictionnaire de routage.
    # Si le subreddit n'a pas encore de module dédié, on affiche un conseil générique.
    if subreddit in ROUTAGE_CONSEILS:
        fonction_cible       = ROUTAGE_CONSEILS[subreddit]
        conseils_specifiques = fonction_cible(titre, contenu)
        conseils_finaux.extend(conseils_specifiques[:5])   # max 5 conseils spécifiques
    else:
        conseils_finaux.append(f"Astuce : Adaptez toujours votre ton à l'ADN de {subreddit}.")

    # --- Message de renforcement pour les FLOP ---
    # Rappel explicite que l'IA a prédit un échec, pour inciter
    # l'utilisateur à prendre les conseils au sérieux.
    if classe == "FLOP":
        conseils_finaux.append(
            "L'IA prédit un Flop. Relisez bien les recommandations ci-dessus pour modifier votre approche."
        )

    # Nombre maximum de conseils selon la classe prédite.
    # Plus le résultat est mauvais, plus on donne de conseils détaillés.
    limites = {"FLOP": 5, "FAIBLE": 4, "BON": 3, "VIRAL": 5}
    return conseils_finaux[:limites.get(classe, 5)]


# 5. HELPER — LABEL POLARITÉ

def label_polarite(score):
    """
    Convertit un score de polarité VADER en label HTML coloré.

    Le score VADER compound est un float normalisé entre -1 (très négatif)
    et +1 (très positif). Les seuils ±0.05 sont les seuils conventionnels
    recommandés par les auteurs de VADER pour classer le sentiment.

    Le résultat est un fragment HTML inline (span coloré) destiné à être
    injecté directement dans une chaîne HTML via une f-string.
    Les styles utilisent des single quotes pour compatibilité avec
    les f-strings Python à double guillemets.

    Parameters
    ----------
    score : float - score compound VADER [-1.0, 1.0]

    Returns
    -------
    str
        Fragment HTML, ex :
        "<span style='color:#1e7e34; font-weight:600;'>positif (+0.421)</span>"

    Seuils VADER (références) :
        score >= +0.05  -> sentiment positif
        score <= -0.05  -> sentiment négatif
        -0.05 < score < +0.05 -> neutre
    """
    if score >= 0.05:
        return f"<span style='color:#1e7e34; font-weight:600;'>positif ({score:+.3f})</span>"
    elif score <= -0.05:
        return f"<span style='color:#c0392b; font-weight:600;'>négatif ({score:+.3f})</span>"
    else:
        return f"<span style='color:#888; font-weight:600;'>neutre ({score:+.3f})</span>"


# 6. INTERFACE — FORMULAIRE DE SAISIE

# En-tête visuel de l'application
st.markdown('<div class="header-icon">📈</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Reddit Success Predictor</div>', unsafe_allow_html=True)

# Champs de saisie utilisateur
titre = st.text_input(
    "Titre du post",
    placeholder="Entrez votre titre...",
    max_chars=300          # limite cohérente avec les contraintes Reddit
)

contenu = st.text_area(
    "Contenu du post",
    placeholder="Rédigez votre texte ici...",
    height=180,
    max_chars=40000        # Reddit autorise ~40 000 caractères par post
)

subreddits_selectionnes = st.multiselect(
    "Subreddits cibles",
    options=SUBREDDITS,
    placeholder="Sélectionnez un ou plusieurs subreddits..."
)

bouton = st.button("🚀 Lancer l'analyse")


# 7. RÉSULTATS — DÉCLENCHÉS AU CLIC SUR LE BOUTON

if bouton:

    # --- Validation des entrées ---
    # On vérifie que les trois champs obligatoires sont remplis
    # avant de lancer le pipeline de prédiction (coûteux en calcul).
    if not titre.strip() or not contenu.strip() or not subreddits_selectionnes:
        st.error("Veuillez remplir le titre, le contenu et choisir au moins un subreddit.")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="output-header">Résultats de l\'analyse</div>', unsafe_allow_html=True)

    # Avertissement légal / disclaimer affiché systématiquement
    # pour contextualiser la nature probabiliste des prédictions.
    st.info(
        "⚠️ **Disclaimer :** Ces prédictions et recommandations sont données à titre purement indicatif. "
        "Notre modèle d'IA se base sur des tendances statistiques passées, mais il n'est pas infaillible. "
        "Appliquer ces conseils optimise vos chances de succès, mais ne garantit pas un engagement viral absolu."
    )

    # CARTE D'ANALYSE DU POST
    # Affiche les features extraites du texte de façon lisible,
    # permettant à l'utilisateur de comprendre ce que le modèle
    # a "vu" avant de rendre sa prédiction.

    maintenant    = datetime.now()
    texte_complet = titre + ". " + contenu

    # Calcul des features textuelles et temporelles utilisées dans la carte d'analyse.
    nb_mots_titre   = len(titre.split())
    nb_mots_contenu = len(contenu.split())
    nb_chars_titre  = len(titre) 
    nb_chars_contenu = len(contenu)
    nb_exclamation  = texte_complet.count('!')
    nb_question     = len(re.findall(r'\?', texte_complet))
    ratio_maj       = round(calculer_ratio_majuscules(titre) * 100, 1)
    score_titre     = round(analyseur_vader.polarity_scores(titre)['compound'], 3)
    score_contenu   = round(analyseur_vader.polarity_scores(contenu)['compound'], 3)

    # Formatage de la date et de l'heure de soumission
    jours_fr     = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    jour_semaine = jours_fr[maintenant.weekday()]
    date_str     = maintenant.strftime("%d/%m/%Y")
    heure_str    = maintenant.strftime("%Hh%M")

    # Accord grammatical du pluriel pour "mot(s)"
    s_titre   = "s" if nb_mots_titre > 1 else ""
    s_contenu = "s" if nb_mots_contenu > 1 else ""

    # Construction des lignes de la grille (HTML inline, single quotes obligatoires)
    # Chaque ligne : label grisé à gauche + valeur en gras à droite
    lignes_analyse = (
        f"<div>📅 <span style='color:#888;'>Date de publication :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{date_str} — {jour_semaine} à {heure_str}</b></div>"

        f"<div>📝 <span style='color:#888;'>Mots dans le titre :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_mots_titre} mot{s_titre}</b></div>"

        f"<div>📄 <span style='color:#888;'>Mots dans le contenu :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_mots_contenu} mot{s_contenu}</b></div>"

        f"<div>🔤 <span style='color:#888;'>Caractères dans le titre:</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_chars_titre}</b></div>"

        f"<div>🔤 <span style='color:#888;'>Caractères dans le contenu:</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_chars_contenu}</b></div>"

        f"<div>❗ <span style='color:#888;'>Points d'exclamation :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_exclamation}</b></div>"

        f"<div>❓ <span style='color:#888;'>Points d'interrogation :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{nb_question}</b></div>"

        f"<div>🔠 <span style='color:#888;'>Ratio majuscules (titre) :</span>&nbsp;"
        f"<b style='color:#1a1a2e;'>{ratio_maj} %</b></div>"

        # label_polarite() retourne un <span> coloré directement inséré dans la grille
        f"<div>😊 <span style='color:#888;'>Ton du titre :</span>&nbsp;{label_polarite(score_titre)}</div>"

        f"<div>💬 <span style='color:#888;'>Ton du contenu :</span>&nbsp;{label_polarite(score_contenu)}</div>"
    )

    # Assemblage final de la carte d'analyse (concaténation de strings,
    # pas de triple guillemets, pour garder le contrôle total des quotes)
    carte_analyse = (
        "<div style='background:white; border-radius:14px; padding:22px 26px; margin-bottom:24px; "
        "border:1.5px solid #e0e0f0; box-shadow:0 2px 10px rgba(0,0,0,0.05);'>"
            "<div style='font-size:15px; font-weight:700; color:#1a1a2e; margin-bottom:16px;'>"
                "🔍 Analyse de votre post"
            "</div>"
            # Grille CSS 2 colonnes pour une lecture rapide des features
            "<div style='display:grid; grid-template-columns:1fr 1fr; gap:10px 28px; font-size:13px; color:#444;'>"
                + lignes_analyse +
            "</div>"
        "</div>"
    )

    st.markdown(carte_analyse, unsafe_allow_html=True)

    # ENCODAGE SÉMANTIQUE — effectué UNE SEULE FOIS avant la boucle.
    # Le vecteur BERT est identique pour tous les subreddits puisqu'il
    # ne dépend que du texte, pas du subreddit cible. Calculer cet
    # encodage dans la boucle serait inutilement redondant et coûteux.
    with st.spinner("Analyse sémantique en cours..."):
        X_utilisateur_brut = preparer_donnees_brutes(titre, contenu)

    # BOUCLE PAR SUBREDDIT
    # Pour chaque subreddit sélectionné, on charge le modèle dédié,
    # on normalise les features avec son scaler, on prédit la classe
    # d'engagement, puis on génère et affiche les résultats.
    for sub in subreddits_selectionnes:

        # Construction du chemin vers les fichiers du modèle.
        # Convention de nommage : modele_{subreddit_en_minuscules}_export.joblib
        # ex: "r/fanTheories" → sub_id = "fantheories"
        #     → cherche "modeles/modele_fantheories_export.joblib"
        # ⚠️ Si vos fichiers ont une casse différente, ajustez le .lower()
        sub_id        = sub.replace("r/", "").lower()
        chemin_modele = f"modeles/modele_{sub_id}_export.joblib"
        chemin_scaler = f"modeles/scaler_{sub_id}_export.joblib"

        # Vérification de l'existence du fichier modèle avant chargement.
        # Si le modèle est absent, on affiche un avertissement et on passe
        # au subreddit suivant sans planter l'application.
        if not os.path.exists(chemin_modele):
            st.warning(f"Modèle en cours de déploiement pour {sub}...")
            continue

        # Chargement du modèle de classification et du scaler associé.
        # Le scaler doit impérativement être celui entraîné sur les mêmes
        # données que le modèle (même split, même ordre de features).
        modele      = joblib.load(chemin_modele)
        scaler      = joblib.load(chemin_scaler)

        # Normalisation des features puis prédiction de la classe.
        # Le scaler transforme les valeurs brutes pour correspondre à la
        # distribution vue à l'entraînement (standardisation ou min-max).
        X_normalise = scaler.transform(X_utilisateur_brut)
        prediction  = modele.predict(X_normalise)[0]     # entier 0–3
        classe      = CLASSES_MAP[prediction]             # ex: "BON"

        # Génération des conseils personnalisés pour ce subreddit et cette classe
        conseils = generer_conseils(titre, contenu, sub, classe)

        # ---- Construction des pastilles des 4 classes ----
        # On itère sur ORDRE_CLASSES dans l'ordre fixe [FLOP, FAIBLE, BON, VIRAL].
        # La classe prédite reçoit son style coloré + un dot indicateur.
        # Les autres classes sont affichées en gris atténué (opacity 0.55).
        pastilles_html = ""
        for c in ORDRE_CLASSES:
            if c == classe:
                # Pastille active : couleur sémantique + ombre + dot coloré
                dot   = COULEURS_DOT[c]
                style = STYLES_ACTIFS[c]
                pastilles_html += (
                    f"<div style='{style} border-radius:20px; padding:7px 18px; font-weight:700; "
                    f"font-size:13px; display:inline-flex; align-items:center; gap:7px; "
                    f"margin-right:6px; box-shadow:0 2px 8px rgba(0,0,0,0.10);'>"
                    f"<span style='width:8px; height:8px; border-radius:50%; background:{dot}; "
                    f"display:inline-block; flex-shrink:0;'></span>{c}</div>"
                )
            else:
                # Pastilles inactives : gris neutre, opacité réduite
                pastilles_html += (
                    f"<div style='background:#f2f2f2; color:#bbb; border:1.5px solid #e0e0e0; "
                    f"border-radius:20px; padding:7px 18px; font-weight:600; font-size:13px; "
                    f"display:inline-flex; align-items:center; margin-right:6px; opacity:0.55;'>"
                    f"{c}</div>"
                )

        # ---- Construction des items de conseil ----
        # Chaque conseil est un <div> avec une puce "•" et un style inline.
        # On utilise une variable de boucle nommée "conseil" (et non "c")
        # pour éviter toute confusion avec la variable "c" de la boucle
        # des pastilles ci-dessus.
        conseils_html = "".join(
            f"<div style='font-size:13px; color:#444; margin-bottom:6px; padding-left:10px;'>• {conseil}</div>"
            for conseil in conseils
        )

        # ---- Assemblage et rendu de la carte subreddit ----
        # La carte contient : nom du subreddit / rangée de pastilles / boîte de conseils.
        # Tout est en styles inline (single quotes) pour éviter les conflits
        # avec les double guillemets de la concaténation Python.
        carte_sub = (
            "<div style='background:white; border-radius:14px; padding:22px 26px; "
            "margin-bottom:20px; border:1.5px solid #e8e8f0; "
            "box-shadow:0 2px 10px rgba(0,0,0,0.05);'>"

                # Nom du subreddit en titre de carte
                f"<div style='font-size:18px; font-weight:700; color:#1a1a2e; margin-bottom:12px;'>{sub}</div>"

                # Label de section en petites capitales grises
                "<div style='font-size:11px; font-weight:600; color:#aaa; letter-spacing:0.6px; "
                "text-transform:uppercase; margin-bottom:10px;'>Potentiel d'engagement</div>"

                # Rangée des 4 pastilles (active + 3 grises)
                f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:18px;'>"
                f"{pastilles_html}</div>"

                # Boîte de recommandations avec bordure rouge
                "<div style='background:#fff8f8; border:1.5px solid #e74c3c; "
                "border-radius:10px; padding:16px 20px; margin-top:4px;'>"
                    "<div style='font-weight:700; font-size:14px; color:#c0392b; margin-bottom:10px;'>"
                    "Recommandations</div>"
                    f"{conseils_html}"
                "</div>"

            "</div>"
        )

        st.markdown(carte_sub, unsafe_allow_html=True)
