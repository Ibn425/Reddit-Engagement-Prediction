import random
from datetime import datetime

def generer_conseils_rfrance(titre, contenu):
    """
    Genere des conseils hierarchises pour r/france :
    1. PRIORITE : Longueurs, sentiment, mots-cles forts
    2. SECONDAIRE : Timing (heure, jour, mois) et style d'ecriture
    """
    prioritaires = []
    secondaires = []
    
    texte_complet = (titre + " " + contenu).lower()
    maintenant = datetime.now()
    heure = maintenant.hour
    jour = maintenant.weekday()  # 4=Vendredi, 5=Samedi
    mois = maintenant.month

    #  1. REGLES PRIORITAIRES 

    # Longueur du titre (cible : ~78 caracteres)
    longueur_titre = len(titre)
    if longueur_titre < 60:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Trop court. "
                            f"Exemple mauvais : 'Retraites' | Exemple bon : 'Pourquoi la reforme des retraites va surtout penaliser les carrieres longues' (78 car.)")
    elif longueur_titre > 90:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Trop long. "
                            f"Exemple mauvais : titre de 120+ caracteres coupe sur mobile | Exemple bon : environ 78 caracteres max")
    else:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Dans la zone ideale (60-90 car.)")

    # Longueur du contenu (cible : 5000-15000 caracteres)
    longueur_contenu = len(contenu)
    if longueur_contenu < 2000:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Trop court. "
                            f"Exemple mauvais : 800 caracteres, pas assez d'arguments | Exemple bon : 5000-15000 car., developpe et structure")
    elif longueur_contenu > 20000:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Tres long. "
                            f"Risque : les lecteurs decrochent | Conseil : aerez avec des sous-titres et paragraphes")
    elif 5000 <= longueur_contenu <= 15000:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Ideale (5000-15000 car.)")

    # Sentiment : legerement negatif / critique moderee
    mots_negatifs_moderes = ["pas", "ne", "ni", "aucun", "jamais", "contre", "manque", "insuffisant"]
    count_neg = sum(1 for mot in mots_negatifs_moderes if mot in contenu.lower())
    if count_neg < 2:
        prioritaires.append("Sentiment : Les posts trop neutres ou positifs performent moins bien. "
                            "Exemple mauvais : 'Tout va bien en France' | Exemple bon : 'Le systeme educatif ne prepare pas assez au marche du travail'")
    elif count_neg >= 3:
        prioritaires.append("Sentiment : Bonne dose de critique moderee, comme les posts viraux.")

    # Mots-cles forts dans le titre
    mots_forts_titre = ["france", "french", "macron", "contre", "gouvernement", "loi"]
    if not any(mot in titre.lower() for mot in mots_forts_titre):
        prioritaires.append("Mots-cles titre : Vous n'utilisez pas les mots qui buzzent. "
                            "Exemple mauvais : 'Une question sur les impots' | Exemple bon : 'Pourquoi la reforme Macron ne reduit pas la dette francaise'")

    #  2. REGLES SECONDAIRES 

    # Heure (5h du matin)
    if heure != 5:
        secondaires.append(f"Heure de publication ({heure}h) : La meilleure heure est 5h du matin (UTC). "
                           f"Exemple : 14h donne moins d'engagement | 5h donne environ +30 pour cent d'engagement moyen")

    # Jour (Vendredi ou Samedi)
    if jour not in [4, 5]:  # 4=Vendredi, 5=Samedi
        secondaires.append("Jour de publication : Les meilleurs jours sont vendredi et samedi. "
                           "Exemple : Lundi matin est un creux d'engagement | Samedi a 5h est un pic historique")

    # Mois (Novembre a Mars)
    if mois not in [11, 12, 1, 2, 3]:
        secondaires.append(f"Mois ({mois}) : Periode creuse. "
                           f"Exemple : Juillet-aout a un engagement bas | Decembre-mars a un engagement maximal (+30 pour cent)")

    # Formulations gagnantes (bigrammes)
    bigrammes_gagnants = ["je ne", "que je", "je suis", "ce nest pas"]
    if not any(bi in contenu.lower() for bi in bigrammes_gagnants):
        secondaires.append("Style d'ecriture : Utilisez des formulations personnelles. "
                           "Exemple mauvais : 'On constate que' | Exemple bon : 'Je ne suis pas sur que cette politique fonctionne'")

    # Titre interrogatif
    if "?" in titre:
        secondaires.append("Titre interrogatif : A utiliser avec moderation. "
                           "Exemple mauvais : 'Est-ce que la reforme est bonne ?' | Exemple bon : 'La reforme des retraites va echouer pour trois raisons'")

    # Majuscules
    if titre.isupper():
        secondaires.append("Evitez les majuscules dans le titre : correlation negative avec l'engagement. "
                           "Exemple mauvais : 'URGENT ALERTE' | Exemple bon : 'Analyse : ce qui change avec la nouvelle loi'")

    # Melange et assemblage
    random.shuffle(secondaires)
    resultat_final = list(dict.fromkeys(prioritaires + secondaires))
    
    return resultat_final[:6]