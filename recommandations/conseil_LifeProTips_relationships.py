"""
Module de recommandations spécifiques pour les subreddits r/LifeProTips et r/relationships.

Ce module contient la logique métier permettant d'analyser le contenu d'un post 
et de suggérer des améliorations basées sur les tendances de succès observées 
historiquement sur ces deux communautés.
"""

from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialisation de l'analyseur de sentiment pour r/relationships
analyzer = SentimentIntensityAnalyzer()

# ==========================================================
# FONCTION : CONSEILS POUR r/LifeProTips
# ==========================================================
def conseil_LifeProTips(titre, contenu):
    """
    Génère des conseils d'optimisation pour le subreddit r/LifeProTips.
    
    L'analyse porte sur la fenêtre de publication idéale, la thématique 
    (mots-clés viraux vs flops) et la profondeur du contenu.

    Args:
        titre (str): Le titre du post Reddit.
        contenu (str): Le corps du texte du post.

    Returns:
        list: Une liste de chaînes de caractères contenant les recommandations.
    """
    conseils = []
    
    # On crée le texte complet pour l'analyse des mots-clés
    texte_complet = titre + " " + contenu
    texte_lower = texte_complet.lower()

    # 1. CONSEIL SUR L'HEURE (Basé sur l'heure actuelle)
    heure_post = datetime.now().hour
    if heure_post < 18 or heure_post > 23:
        conseils.append("Publier entre 18h et 23h augmente généralement la visibilité du post.")

    # 2. CONSEIL SUR LA THÉMATIQUE (Détection de patterns de succès/échec)
    mots_flops = ['shower', 'water', 'box', 'ice', 'car']
    mots_viraux = ['job', 'work', 'manager', 'friend', 'people', 'example', 'life', 'brain']

    if any(mot in texte_lower for mot in mots_flops):
        conseils.append("Les astuces trop basiques ou matérielles performent moins bien. Privilégier un conseil utile au quotidien ou professionnel.")
    elif not any(mot in texte_lower for mot in mots_viraux):
        conseils.append("Ajouter des termes liés au travail, aux relations humaines ou à l'amélioration personnelle peut capter davantage l’attention.")

    # 3. CONSEIL SUR LA LONGUEUR
    if len(contenu) < 1500:
        conseils.append("Développer davantage l’astuce. Les posts détaillés performent mieux.")

    if not conseils:
        conseils.append("Le post respecte déjà les principales caractéristiques observées.")

    return conseils

# ==========================================================
# FONCTION : CONSEILS POUR r/relationships
# ==========================================================
def conseils_relationships(titre, contenu):
    """
    Génère des conseils d'optimisation pour le subreddit r/relationships.
    
    Analyse multi-critères incluant le timing hebdomadaire, la saisonnalité, 
    la précision du vocabulaire relationnel et l'intensité émotionnelle (VADER).

    Args:
        titre (str): Le titre du post Reddit.
        contenu (str): Le corps du texte du post.

    Returns:
        list: Une liste de recommandations pour maximiser l'engagement.
    """
    conseils = []
    texte_complet = titre + " " + contenu
    texte_lower = texte_complet.lower()
    maintenant = datetime.now()

    # 1. TIMING (Jours et heures stratégiques)
    if maintenant.weekday() not in [0, 6] or maintenant.hour < 19:
        conseils.append("Publier le dimanche ou le lundi soir semble favoriser l’engagement.")

    # 2. SAISONNALITÉ
    if maintenant.month == 5:
        conseils.append("Le mois de mai présente historiquement de bons résultats.")
    elif maintenant.month == 3:
        conseils.append("Le mois de mars génère en moyenne moins d’engagement.")

    # 3. LONGUEURS (Analyse de la granularité du récit)
    if len(titre) < 200:
        conseils.append("Un titre plus détaillé permet souvent de mieux contextualiser la situation.")
    if len(contenu) < 2500:
        conseils.append("Ajouter davantage de contexte et de détails peut améliorer l’intérêt du post.")

    # 4. VOCABULAIRE (Spécificité des relations)
    mots_viraux = ['marriage', 'husband', 'wife', 'partner', 'dating']
    if not any(mot in texte_lower for mot in mots_viraux):
        conseils.append("Préciser les relations entre les personnes (partner, husband, wife...) améliore la compréhension.")

    # 5. ÉMOTION (Analyse VADER en direct)
    # On vérifie si le texte n'est pas trop "plat" émotionnellement
    score = analyzer.polarity_scores(contenu)['compound']
    if -0.1 < score < 0.1:
        conseils.append("Exprimer davantage le ressenti ou les émotions peut générer plus de réactions.")

    if not conseils:
        conseils.append("Le post respecte déjà les principales caractéristiques performantes.")

    return conseils