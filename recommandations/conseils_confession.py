import random
from datetime import datetime


def generer_conseils_confession(titre, contenu):
    """
    Génère des conseils hiérarchisés pour r/confession.
    Basés sur l'EDA réelle du subreddit :
      - Meilleure heure  : 1h UTC (engagement moyen 3222)
      - Meilleur jour    : Jeudi (2414) > Lundi (1880)
      - Meilleur mois    : Avril (8002) >> Septembre (3714) >> Août (3173)
      - Période du mois  : Début (1-10) = 2231 >> Milieu (1602) >> Fin (1546)
      - Longueur titre   : zone idéale 150-165 caractères (pic EDA)
      - Longueur contenu : corrélation positive jusqu'à ~8200 caractères
      - commence_par_i   : corrélation +0.068 avec l'engagement
      - contient_chiffre : corrélation +0.081 avec l'engagement (2ème feature)
      - Flesch titre     : 76.91 en moyenne — titres lisibles performent mieux
      - Mots intensifs   : légèrement négatifs sur engagement
    """

    prioritaires  = []
    secondaires   = []

    maintenant    = datetime.now()
    heure         = maintenant.hour
    jour          = maintenant.weekday()   # 0=Lundi … 6=Dimanche
    mois          = maintenant.month
    jour_mois     = maintenant.day

    texte_complet = (titre + " " + contenu).lower()

    len_titre    = len(titre)
    len_contenu  = len(contenu)

    # 1. RÈGLES STRUCTURELLES — PRIORITAIRES

    # Longueur du titre 
    if len_titre < 80:
        prioritaires.append(
            f"Titre trop court ({len_titre} car.) : Les titres entre 150 et 165 caractères "
            f"sont dans la zone de performance idéale sur r/confession. "
            f"Développez davantage votre accroche pour contextualiser votre confession."
        )
    elif len_titre > 200:
        prioritaires.append(
            f"Titre trop long ({len_titre} car.) : Au-delà de 200 caractères, "
            f"l'engagement chute. Condensez votre titre en visant 150-165 caractères."
        )
    elif not (150 <= len_titre <= 165):
        prioritaires.append(
            f"Titre ({len_titre} car.) : La zone de performance maximale se situe "
            f"entre 150 et 165 caractères. Vous êtes proche, affinez encore un peu !"
        )

    #  Longueur du contenu 
    if len_contenu < 300:
        prioritaires.append(
            f"Contenu trop court ({len_contenu} car.) : r/confession valorise les récits "
            f"développés. L'engagement monte jusqu'à ~8 200 caractères — racontez votre "
            f"histoire avec plus de détails pour captiver la communauté."
        )
    elif len_contenu > 8200:
        prioritaires.append(
            f"Contenu très long ({len_contenu} car.) : Au-delà de 8 200 caractères, "
            f"l'engagement commence à chuter. Pensez à synthétiser les parties secondaires."
        )

    #  Première personne 
    titre_lower = titre.strip().lower()
    if not (titre_lower.startswith("i ") or
            titre_lower.startswith("i've") or
            titre_lower.startswith("i'm") or
            titre_lower.startswith("i had") or
            titre_lower.startswith("i was")):
        prioritaires.append(
            "Identité narrative : les titres commençant par 'I' "
            "(première personne) génèrent +6,8 % d'engagement en plus. "
            "Exemple : 'I finally told my family the truth about…'"
        )

    #  Chiffres dans le titre 
    if not any(c.isdigit() for c in titre):
        prioritaires.append(
            "Chiffres : Les titres contenant un chiffre sont corrélés à +8 % "
            "d'engagement sur ce subreddit (2ème feature la plus impactante). "
            "Ex : 'I hid this secret for 12 years' est plus accrocheur."
        )

    #  Mots intensifs 
    mots_intensifs = ["very", "extremely", "really", "so ", "absolutely", "totally"]
    if any(mot in texte_complet for mot in mots_intensifs):
        secondaires.append(
            "Tonalité : Limitez les intensifs (very, extremely, really…). "
            "Les confessions sobres et directes inspirent davantage confiance "
            "et génèrent plus d'empathie de la communauté."
        )

    # 2. RÈGLES TEMPORELLES — SECONDAIRES
    #  Heure 
    heures_optimales = {1: "pic principal (engagement ×2)",
                        12: "pic secondaire midi",
                        13: "pic secondaire midi",
                        19: "pic secondaire soirée",
                        20: "pic secondaire soirée",
                        21: "pic secondaire soirée"}
    if heure not in heures_optimales:
        secondaires.append(
            f"Heure actuelle ({heure}h UTC) hors des pics : "
            f"r/confession présente trois fenêtres d'or — 1h UTC (pic majeur, "
            f"engagement moyen 3 222), 12h-13h UTC et 19h-21h UTC. "
            f"Programmez votre post pour l'un de ces créneaux."
        )
    else:
        info = heures_optimales[heure]
        secondaires.append(
            f"Excellent timing ! {heure}h UTC correspond au {info} "
            f"de r/confession — vous êtes dans la bonne fenêtre."
        )

    #  Jour 
    jours_top     = {3: "Jeudi (engagement moyen 2 414)", 0: "Lundi (engagement moyen 1 880)"}
    jours_faibles = {5: "Samedi", 1: "Mardi"}
    if jour in jours_faibles:
        secondaires.append(
            f"Jour défavorable ({list(jours_faibles.values())[list(jours_faibles.keys()).index(jour)]}) : "
            f"Préférez le Jeudi ou le Lundi qui sont statistiquement les "
            f"jours les plus actifs sur ce subreddit."
        )
    elif jour not in jours_top:
        secondaires.append(
            "Jour : Le Jeudi est le jour record, "
            "suivi du Lundi. Si possible, décalez votre publication."
        )

    #  Mois 
    if mois in [2, 3]:
        secondaires.append(
            f"Saisonnalité : Février et Mars sont les mois les plus creux de l'année "
            f". Si votre confession peut attendre, "
            f"Avril est de loin le meilleur mois (engagement moyen 8 002 !)."
        )
    elif mois == 4:
        secondaires.append(
            f"Mois idéal : Vous publiez en Avril, le mois record de r/confession "
            f" — profitez-en !"
        )
    elif mois in [8, 9, 7]:
        secondaires.append(
            f"Bonne période : L'été (Juillet-Septembre) est très actif sur ce subreddit "
            f"(engagement moyen 3 173-3 714). Votre confession a de bonnes chances d'être vue."
        )

    #  Période du mois 
    if not (1 <= jour_mois <= 10):
        secondaires.append(
            f"Période du mois ({jour_mois}e jour) : Les 10 premiers jours du mois "
            f"génèrent un engagement moyen de 2 231, contre 1 546 en fin de mois. "
            f"Si possible, anticipez votre prochaine publication."
        )

    # ASSEMBLAGE FINAL
    random.shuffle(secondaires)
    resultat = list(dict.fromkeys(prioritaires + secondaires))
    return resultat[:5]
