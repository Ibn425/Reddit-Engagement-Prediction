import random
from datetime import datetime

def generer_conseils_fantheories(titre, contenu):
    """
    Génère des conseils hiérarchisés pour r/fanTheories :
    1. PRIORITÉ : Structure et accroche (Taille, Questions)
    2. SECONDAIRE : Stratégie de franchise et Timing
    """
    prioritaires = []
    secondaires = []
    
    texte_complet = (titre + " " + contenu).lower()
    maintenant = datetime.now()
    heure = maintenant.hour
    jour = maintenant.weekday() # 3 = Jeudi, 6 = Dimanche
    mois = maintenant.month

    #1. RÈGLES STRUCTURELLES (PRIORITAIRES)
    longueur_titre = len(titre)
    if longueur_titre < 25 or longueur_titre > 100:
        prioritaires.append("Longueur du titre : Les titres qui buzzent font entre 25 et 100 caractères. "
                            "Exemple : Au lieu de 'Théorie Joker', essayez 'Pourquoi le Joker n'est pas mort à la fin de The Dark Knight'.")

    if "?" in titre:
        prioritaires.append("Affirmez votre théorie : Les titres interrogatifs sont souvent perçus comme des questions d'amateurs. "
                            "Présentez votre idée comme une certitude pour susciter le débat.")

    longueur_contenu = len(contenu)
    if longueur_contenu < 500:
        prioritaires.append("Développez votre argumentaire : Une théorie convaincante nécessite des preuves. Votre texte est un peu court pour r/fanTheories.")
    elif longueur_contenu > 5000:
        prioritaires.append("Texte très dense : Vous dépassez les 5000 caractères. Pensez à résumer.")

    # 2. RÈGLES DE THÉMATIQUES (SECONDAIRES)
    franchises_prouvees = ["marvel", "mcu", "avengers", "star wars", "breaking bad", "batman", "dark knight", "joker"]
    franchise_cachee = [mot for mot in franchises_prouvees if mot in contenu.lower() and mot not in titre.lower()]
    
    if franchise_cachee:
        nom = "MCU" if franchise_cachee[0] == "mcu" else franchise_cachee[0].title()
        secondaires.append(f"Vous parlez de {nom} dans le texte. Ajoutez le dans votre titre pour attirer les fans de la franchise !")
    elif not any(mot in texte_complet for mot in franchises_prouvees):
        secondaires.append("Nos stats montrent que les univers étendus (Marvel, Star Wars, DC) dominent l'engagement. Liez votre théorie à une œuvre culte si possible.")

    #3. RÈGLES TEMPORELLES (SECONDAIRES)
    if not (12 <= heure <= 18):
        secondaires.append("Optimisation Horaire : Il est actuellement hors de la fenêtre 'Gold' (12h-18h UTC). Programmez votre post pour l'après-midi pour viser le public américain.")
        
    if jour not in [3, 6]: # Jeudi et Dimanche sont les meilleurs jours
        # Ce n'est pas le jour idéal, mais on peut atténuer avec un conseil de timing
        secondaires.append("Aujourd'hui n'est pas statistiquement le jour le plus actif. Pour un succès maximal, le Jeudi et le Dimanche restent les jours records sur ce subreddit.")

    if mois in [7, 8, 9]:
        secondaires.append("Période Estivale : L'engagement baisse en été. Soyez particulièrement percutant car l'audience est plus difficile à captiver qu'en Mars/Avril.")

    #ASSEMBLAGE ET VARIÉTÉ
    # Mélange des conseils secondaires pour ne pas être monotone
    random.shuffle(secondaires)
    
    # Fusion (Prioritaires d'abord, puis Secondaires)
    resultat_final = list(dict.fromkeys(prioritaires + secondaires))
    
    # On retourne les 5 plus pertinents
    return resultat_final[:5]