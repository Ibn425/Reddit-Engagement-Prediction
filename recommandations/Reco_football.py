import random
from datetime import datetime

def generer_conseils_football(titre, contenu):
    """
    Génère des conseils hiérarchisés : 
    1. Priorité aux erreurs de structure (Taille, Questions)
    2. Variété sur les conseils stratégiques (Timing, Mots-clés, Sentiment)
    """
    prioritaires = []
    secondaires = []
    
    texte_complet = (titre + " " + contenu).lower()
    maintenant = datetime.now()
    heure = maintenant.hour
    mois = maintenant.month

    #CATEGORIE 1 : PRIORITAIRES (STRUCTURE)
    if len(titre) > 100:
        prioritaires.append("Titre trop long : Les utilisateurs sur mobile coupent souvent leur lecture après 80 caractères. Exemple : Au lieu de tout raconter, mettez juste le fait marquant : 'L'incroyable retourné de Ronaldo'.")
    
    if "?" in titre:
        prioritaires.append("Transformez votre question : Sur r/football, les affirmations lancent mieux le débat. Exemple : Au lieu de 'Qui est le meilleur ?', préférez 'Pourquoi Messi est statistiquement supérieur'.")

    if len(contenu) > 2500:
        prioritaires.append("Contenu trop dense : Votre texte dépasse 2500 caractères. Pour ne pas décourager l'utilisateur, aérez vos paragraphes ou utilisez des listes à puces.")

    #CATEGORIE 2 : SECONDAIRES (STRATÉGIE & TIMING)
    # Timing
    if 21 <= heure <= 22:
        secondaires.append("Timing Critique : Il est entre 21h et 22h. C'est le moment où l'audience est la moins active. Attendez demain matin pour publier.")
    elif not (5 <= heure <= 7):
        secondaires.append("Optimisation du Timing : Publiez idéalement entre 5h et 7h du matin pour toucher les fans dès leur réveil.")
        
    if mois not in [5, 6, 7]:
        secondaires.append("Note Saisonnière : Nous sommes hors période Mai-Juillet. L'engagement sur r/football explose durant l'été avec les tournois majeurs.")

    # Mots-clés
    mots_cles_or = ["world cup", "real madrid", "cristiano", "ronaldo", "messi", "ballon d'or", "euro", "champions league"]
    mot_cache = [mot for mot in mots_cles_or if mot in contenu.lower() and mot not in titre.lower()]
    
    if mot_cache:
        nom_sujet = mot_cache[0].title()
        secondaires.append(f"Mise en avant : Vous mentionnez '{nom_sujet}' dans le texte mais pas dans le titre. Ajoutez-le pour booster le taux de clic !")
    elif not any(mot in texte_complet for mot in mots_cles_or):
        secondaires.append("Manque de piliers : Les sujets sur les icônes (Messi, Ronaldo) ou les grands clubs génèrent plus d'intérêt.")

    # Sources et Sentiment
    sources_sociales = ["twitter.com", "x.com", "talksport", "instagram.com"]
    if not any(source in texte_complet for source in sources_sociales):
        secondaires.append("Source manquante : Les liens vers X/Twitter ou TalkSport provoquent souvent plus de réactions émotionnelles.")

    secondaires.append("Ton du message : Ne soyez pas trop neutre. Une analyse un peu critique ou 'piquante' génère souvent 2x plus de débats.")
    
    # Flairs
    secondaires.append("Choix du Flair : N'oubliez pas d'assigner un Flair. Les plus performants : 'Stats', 'News' ou 'Discussion'.")

    # ASSEMBLAGE
    # 1. On mélange les conseils secondaires pour avoir de la variété
    random.shuffle(secondaires)
    
    # 2. On fusionne (Prioritaires d'abord, puis Secondaires)
    # On utilise dict.fromkeys pour éviter tout doublon par erreur
    resultat_total = list(dict.fromkeys(prioritaires + secondaires))
    
    # 3. On retourne les 5 meilleurs
    return resultat_total[:5]