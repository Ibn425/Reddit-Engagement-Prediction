import random
from datetime import datetime

def generer_conseils_rhistoire(titre, contenu):
    """
    Genere des conseils hierarchises pour r/histoire :
    1. PRIORITE : Longueurs, mots-cles historiques, sources
    2. SECONDAIRE : Timing, trigrammes, lisibilite
    """
    prioritaires = []
    secondaires = []
    
    texte_complet = (titre + " " + contenu).lower()
    maintenant = datetime.now()
    heure = maintenant.hour
    jour = maintenant.weekday()  # 0=Lundi (bon), 6=Dimanche (mauvais)
    mois = maintenant.month

    #  1. REGLES PRIORITAIRES 

    # Longueur du titre (cible : 50-100 caracteres)
    longueur_titre = len(titre)
    if longueur_titre < 30:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Trop court. "
                            f"Exemple mauvais : 'Verdun 1916' | Exemple bon : 'Pourquoi la bataille de Verdun a-t-elle dure 10 mois ?' (58 car.)")
    elif longueur_titre > 100:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Trop long. "
                            f"Exemple : titre coupe sur mobile | Exemple bon : 50-100 caracteres")
    else:
        prioritaires.append(f"Longueur du titre ({longueur_titre} car.) : Dans la zone ideale (50-100 car.)")

    # Longueur du contenu (cible : 1000-5000 caracteres)
    longueur_contenu = len(contenu)
    if longueur_contenu < 500:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Trop court pour une analyse historique. "
                            f"Exemple mauvais : 300 car. sans preuves | Exemple bon : 1000-5000 car. avec sources")
    elif longueur_contenu > 8000:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Tres long. "
                            f"Exemple : mur de texte | Conseil : structurez avec sous-titres")
    elif 1000 <= longueur_contenu <= 5000:
        prioritaires.append(f"Longueur du contenu ({longueur_contenu} car.) : Ideale (1000-5000 car.)")

    # Mots-cles forts dans le titre (d'apres TF-IDF)
    mots_forts_titre = ["histoire", "guerre", "bataille", "france", "seconde guerre mondiale", "republique", "verdun"]
    if not any(mot in titre.lower() for mot in mots_forts_titre):
        prioritaires.append("Mots-cles titre : Ajoutez un mot fort pour attirer les clics. "
                            "Exemple mauvais : 'Un evenement oublie' | Exemple bon : 'La bataille de Verdun : mythes et realites'")

    # Sources (indispensable sur r/histoire)
    if "source" not in contenu.lower() and "reference" not in contenu.lower() and "bibliographie" not in contenu.lower():
        prioritaires.append("Absence de sources : Sur r/histoire, citer ses sources est crucial. "
                            "Exemple mauvais : 'Selon moi...' | Exemple bon : 'D'apres les archives militaires (Service Historique de la Defense)'")

    #  2. REGLES SECONDAIRES 

    # Heure (tot le matin)
    if not (0 <= heure <= 6):
        secondaires.append(f"Heure ({heure}h) : Les posts qui buzzent sont publies tres tot (0h-6h UTC). "
                           f"Exemple : 14h donne moins d'engagement | 4h donne un pic d'engagement")

    # Jour (Lundi = meilleur)
    if jour != 0:
        secondaires.append("Jour de publication : Le lundi est le meilleur jour sur r/histoire. "
                           "Exemple : Dimanche donne un faible engagement | Lundi matin donne un engagement multiplie par 1.5")
    else:
        secondaires.append("Jour de publication : Vous publiez un lundi, excellent choix !")

    # Mois (Mars a Mai = pic)
    if mois not in [3, 4, 5]:
        secondaires.append(f"Mois ({mois}) : Periode creuse. "
                           f"Exemple : Juillet-aout a un engagement divise par 2 | Mars-avril a un engagement maximal (90 contre 50 en ete)")

    # Trigrammes viraux
    trigrammes_viraux = ["de la guerre", "seconde guerre mondiale", "bataille de la", "mort de la"]
    if not any(tri in contenu.lower() for tri in trigrammes_viraux):
        secondaires.append("Formulations gagnantes : Utilisez des trigrammes typiques des posts viraux. "
                           "Exemple mauvais : 'Le conflit de 14-18' | Exemple bon : 'La seconde guerre mondiale a bouleverse...'")

    # Eviter la premiere personne
    if "je " in contenu.lower() or "mon " in contenu.lower() or "j'ai " in contenu.lower():
        secondaires.append("Style : Evitez la premiere personne sur r/histoire. "
                           "Exemple mauvais : 'Je pense que...' | Exemple bon : 'Les archives montrent que...'")

    # Ajout de media
    if "video" not in contenu.lower() and "image" not in contenu.lower():
        secondaires.append("Medias : Les posts avec video ou image performent mieux. "
                           "Conseil : Ajoutez une carte, une photo d'archive ou un lien vers un documentaire")

    # Sentiment (tres legerement positif)
    mots_trop_negatifs = ["honte", "scandale", "horreur", "degout"]
    if any(mot in titre.lower() for mot in mots_trop_negatifs):
        secondaires.append("Sentiment : Les titres trop negatifs floppent. "
                           "Exemple mauvais : 'L'horreur oubliee de...' | Exemple bon : 'Ce que la bataille de Verdun nous apprend sur la strategie militaire'")

    # Melange et assemblage
    random.shuffle(secondaires)
    resultat_final = list(dict.fromkeys(prioritaires + secondaires))
    
    return resultat_final[:6]