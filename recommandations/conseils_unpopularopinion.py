import random
from datetime import datetime


def generer_conseils_unpopularopinion(titre, contenu):
    """
    Génère des conseils hiérarchisés pour r/unpopularopinion.
    Basés sur l'EDA réelle du subreddit :
      - Meilleure heure  : 17h UTC (engagement moyen 1 167)
      - Fenêtre d'or     : 9h-18h UTC
      - Zone morte       : 4h-8h UTC
      - Meilleur jour    : Dimanche (996) > Jeudi (978) > Mardi (908)
      - Pire jour        : Lundi (619)
      - Meilleur mois    : Septembre (1 659) >> Juillet (1 529) >> Juin (1 231)
      - Mois creux       : Mars (546) et Février (566)
      - Longueur titre   : corrélation +0.066 — sweet spot 80-143 caractères
      - ratio_majuscules : corrélation -0.112 (feature la plus impactante !)
      - sentiment_titre  : corrélation -0.074 — les titres négatifs/clivants performent mieux
      - contient_negatif : corrélation +0.037
      - Pas de ? dans le titre : les affirmations > les questions
    """

    prioritaires = []
    secondaires  = []

    maintenant    = datetime.now()
    heure         = maintenant.hour
    jour          = maintenant.weekday()   # 0=Lundi … 6=Dimanche
    mois          = maintenant.month

    texte_complet = (titre + " " + contenu).lower()

    len_titre   = len(titre)
    len_contenu = len(contenu)

    # 1. RÈGLES STRUCTURELLES — PRIORITAIRES
    # Ratio majuscules (feature N°1 la plus impactante, corrélation -0.112) 
    ratio_maj = sum(1 for c in titre if c.isupper()) / max(len_titre, 1)
    if ratio_maj > 0.15:
        prioritaires.append(
            f"Majuscules excessives (ratio {ratio_maj:.0%}) : "
            f"C'est la feature la plus négativement corrélée à l'engagement sur ce subreddit. "
            f"Réécrivez votre titre en minuscules — seule la première lettre doit être majuscule."
        )
    elif ratio_maj > 0.05:
        prioritaires.append(
            f"Majuscules ({ratio_maj:.0%} du titre) : "
            f"L'excès de majuscules est fortement pénalisant sur r/unpopularopinion. "
            f"Limitez-les au maximum pour ne pas paraître agressif."
        )

    # Longueur du titre 
    if len_titre < 40:
        prioritaires.append(
            f"Titre trop court ({len_titre} car.) : "
            f"Les titres qui performent bien font entre 80 et 143 caractères. "
            f"Développez votre opinion pour qu'elle soit immédiatement compréhensible. "
            f"Ex : au lieu de 'Pineapple pizza is good', écrivez pourquoi vous le pensez."
        )
    elif len_titre > 176:
        prioritaires.append(
            f"Titre trop long ({len_titre} car.) : "
            f"Au-delà de 176 caractères, l'engagement chute. "
            f"Condensez votre opinion principale en 80-143 caractères maximum."
        )

    # Point d'interrogation 
    if "?" in titre:
        prioritaires.append(
            "Affirmez votre opinion : Les titres sous forme de question sont moins performants. "
            "r/unpopularopinion valorise les affirmations directes et tranchées. "
            "Reformulez : 'Is pineapple pizza good?' → 'Pineapple pizza is genuinely delicious'."
        )

    #  Mots négatifs / clivants 
    mots_negatifs = ["don't", "dont", "hate", "not", "never", "wrong", "stop", "overrated", "terrible"]
    if not any(mot in titre.lower() for mot in mots_negatifs):
        prioritaires.append(
            "Tonalité : les titres contenant des mots négatifs ou clivants "
            "(hate, don't, overrated, wrong, never) génèrent plus de débats et d'engagement "
            "(corrélation +0.037). Assumez votre opinion sans modération !"
        )

    #  Longueur du contenu 
    if len_contenu < 500:
        secondaires.append(
            f"Argumentation trop courte ({len_contenu} car.) : "
            f"La communauté apprécie qu'on argumente son opinion. "
            f"Développez votre raisonnement — expliquez pourquoi c'est une unpopular opinion "
            f"et quels arguments vous défendez."
        )
    elif len_contenu > 7970:
        secondaires.append(
            f"Contenu très long ({len_contenu} car.) : "
            f"Au-delà de 7 970 caractères, l'engagement commence à stagner sur ce subreddit. "
            f"Synthétisez vos arguments les plus percutants."
        )

    # 2. RÈGLES TEMPORELLES — SECONDAIRES

    #  Heure 
    if 4 <= heure <= 8:
        secondaires.append(
            f"Zone morte horaire ({heure}h UTC) : "
            f"4h-8h UTC est la période de mort statistique sur ce subreddit. "
            f"Attendez la fenêtre d'or : 9h-18h UTC (pic à 17h UTC)."
        )
    elif not (9 <= heure <= 18):
        secondaires.append(
            f"Timing ({heure}h UTC) : "
            f"La fenêtre d'or pour r/unpopularopinion est 9h-18h UTC, "
            f"avec un pic à 17h UTC. "
            f"Si possible, décalez votre publication."
        )
    else:
        secondaires.append(
            f"Bon timing ({heure}h UTC) : Vous êtes dans la fenêtre d'or "
            f"(9h-18h UTC). Le pic est à 17h UTC — vous maximisez votre visibilité !"
        )

    #  Jour 
    jours_top    = {6: "Dimanche (996)", 3: "Jeudi (978)", 1: "Mardi (908)"}
    jours_faibles = {0: "Lundi (619)", 4: "Vendredi (801)"}
    if jour in jours_faibles:
        nom_jour = jours_faibles[jour]
        secondaires.append(
            f"Jour défavorable ({nom_jour.split(' ')[0]}) : "
            f"Le Lundi est le pire jour sur ce subreddit . "
            f"Préférez le Dimanche ou le Jeudi pour un meilleur impact."
        )
    elif jour not in jours_top:
        secondaires.append(
            "Jour : Le Dimanche et le Jeudi sont les jours records. "
            "Si votre post peut attendre, programmez-le pour l'un de ces jours."
        )

    #  Mois 
    if mois in [2, 3]:
        secondaires.append(
            f"Mois creux ({'Février' if mois == 2 else 'Mars'}) : "
            f"L'engagement est au plus bas en Février et Mars. "
            f"Si possible, attendez Avril ou l'été — Septembre est le meilleur mois."
        )
    elif mois == 9:
        secondaires.append(
            "Excellent mois ! Septembre est le mois record de r/unpopularopinion "
            "— votre post a un avantage saisonnier !"
        )
    elif mois in [6, 7, 8, 10, 11]:
        secondaires.append(
            f"Bonne période : La seconde moitié de l'année est favorable sur ce subreddit "
            f". Vous êtes dans une bonne fenêtre."
        )

    # ASSEMBLAGE FINAL
    random.shuffle(secondaires)
    resultat = list(dict.fromkeys(prioritaires + secondaires))
    return resultat[:5]
