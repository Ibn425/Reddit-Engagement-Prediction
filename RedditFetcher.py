import requests
import time
import pandas as pd
import os
from collections import Counter

# CONFIGURATION UTILISATEUR
# Seule section à modifier avant de lancer le script

SUBREDDIT         = "Histoire"  # Nom du subreddit cible 
TRI               = "top"       # Méthode de tri : "top", "new", "hot", "rising"
PERIODE           = "all"       # Fenêtre temporelle : "all", "year", "month", "week", "day"
                                # Ignoré si TRI = "new" ou "hot" vu que reddit n'en tient pas compte
NB_POSTS        = 900           # Nombre maximum de posts à récupérer par appel
RECHERCHE_MOTS  = True         # Active la recherche par mots-clés fréquents 
TOP_MOTS        = 30         # Nombre de mots-clés à extraire pour la recherche itérative


# CONSTANTE INTERNE - ne pas modifier
# User-Agent requis par Reddit pour éviter un blocage immédiat (HTTP 403)
HEADERS = {"User-Agent": "Mozilla/5.0 (reddit-scraper-scolaire/2.0)"}

def scraper_subreddit(subreddit, tri="top", periode="all", nb_posts=900, query=None):
    """
    Récupère des posts d'un subreddit via l'API publique Reddit (format .json).

    Deux modes selon le paramètre `query` :
      - query=None  → parcourt le flux principal du subreddit (top, new, hot…)
      - query="mot" → effectue une recherche par mot-clé dans le subreddit

    Reddit limite les résultats à 100 posts par requête, donc la fonction
    pagine automatiquement via le curseur `after` jusqu'à atteindre nb_posts.

    Args:
        subreddit (str) : nom du subreddit (ex: "Histoire")
        tri       (str) : méthode de tri ("top", "new", "hot", "rising")
        periode   (str) : fenêtre temporelle ("all", "year", "month", "week", "day")
        nb_posts  (int) : nombre maximum de posts à retourner
        query     (str) : mot-clé de recherche (None = pas de recherche)

    Returns:
        list[dict] : liste de posts, chacun représenté par un dictionnaire de champs
    """
    posts = []   # Accumule tous les posts récupérés
    after = None # Curseur de pagination Reddit (token du dernier post vu)
    label = f"'{query}'" if query else "general"  # Utilisé uniquement dans les print de suivi

    while len(posts) < nb_posts:

        #  Construction de l'URL et des paramètres selon le mode 
        if query:
            # Mode recherche : endpoint /search.json avec restrict_sr=1
            # pour limiter les résultats au subreddit ciblé
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": query,
                "restrict_sr": 1,   # 1 = cherche uniquement dans ce subreddit
                "sort": tri,
                "t": periode,
                "limit": 100        # Maximum autorisé par l'API Reddit
            }
        else:
            # Mode flux : endpoint /{tri}.json (top, new, hot…)
            url = f"https://www.reddit.com/r/{subreddit}/{tri}.json"
            params = {"limit": 100, "t": periode}

        # Ajout du curseur de pagination si on n'est pas à la première page
        if after:
            params["after"] = after

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)

            # Reddit renvoie 429 quand on dépasse la limite de requêtes
            if response.status_code == 429:
                print("  Rate limit atteint - pause 60 sec...")
                time.sleep(60)
                continue  # Relance la même requête après la pause

            if response.status_code != 200:
                print(f"  Erreur HTTP {response.status_code}")
                break  # Erreur non récupérable → on arrête la boucle

            data = response.json()
            children = data["data"]["children"]  # Liste des posts de la page courante

            # Aucun résultat retourné = on a atteint la fin du subreddit
            if not children:
                print(f"  Plus de posts disponibles pour {label}.")
                break

            #  Extraction des champs utiles pour chaque post 
            for child in children:
                p = child["data"]

                # PROBLEME : cette pause est déclenchée une seule fois
                # quand len(posts) == 450 exactement, mais si un batch fait
                # passer de 400 à 500 d'un coup, la condition n'est jamais vraie.
                # Mieux vaudrait vérifier en dehors de la boucle `for`.
                if len(posts) == 450:
                    print("  Pause 1 min apres 450 posts...")
                    time.sleep(60)

                posts.append({
                        "id"             : p["id"],
                        "titre"          : p["title"],
                       "score"          : p["score"],
                       "upvote_ratio"   : p["upvote_ratio"],
                       "nb_commentaires": p["num_comments"],
                        "created_utc"    : p["created_utc"],
                        "flair"          : p.get("link_flair_text", ""),
                        "est_texte"      : p["is_self"],
                        "contenu"        : p.get("selftext", ""),
                        "url"            : p.get("url", ""),
                        "subreddit"      : p["subreddit"],
                    })

            # Mise à jour du curseur de pagination pour la prochaine page
            after = data["data"]["after"]
            print(f"  {len(posts)} posts recuperes ({label})...")

            # after est None quand Reddit n'a plus de page suivante
            if after is None:
                print(f"  Fin de la pagination pour {label}.")
                break

            time.sleep(2)  # Délai poli entre chaque requête pour ne pas se faire bannir

        except requests.exceptions.Timeout:
            print("  Timeout - nouvelle tentative dans 10 sec...")
            time.sleep(10)
            # Pas de `continue` explicite, mais la boucle while repart
            # naturellement → comportement correct
        except Exception as e:
            print(f"  Erreur inattendue : {e}")
            break  # Erreur inconnue → on abandonne pour ne pas boucler indéfiniment

    # Garantit qu'on ne dépasse jamais nb_posts même si un batch en ajoute trop
    return posts[:nb_posts]


def sauvegarder_donnees(nom_fichier, nouveaux_posts):
    """
    Sauvegarde une liste de posts dans un fichier CSV.

    Si le fichier existe déjà, fusionne les nouvelles données avec l'existant
    et supprime les doublons en se basant sur le champ `id` (identifiant unique Reddit).
    Le premier exemplaire d'un doublon est conservé (les nouvelles données ont la priorité).

    Args:
        nom_fichier       (str)       : chemin du fichier CSV de sortie
        nouveaux_posts (list[dict]): posts à sauvegarder
    """
    if not nouveaux_posts:
        print("  Aucun post a sauvegarder.")
        return

    df_nouveau = pd.DataFrame(nouveaux_posts)

    if os.path.isfile(nom_fichier):
        # Fichier existant : on charge, on concatène, on déduplique
        df_ancien = pd.read_csv(nom_fichier)
        df_total  = pd.concat([df_nouveau, df_ancien], ignore_index=True)
        df_total  = df_total.drop_duplicates(subset=["id"], keep="first")
        nb_ajoutes = len(df_total) - len(df_ancien)
        print(f"  {nb_ajoutes} nouveaux posts uniques ajoutes -> {nom_fichier}  (total : {len(df_total)})")
    else:
        # Première écriture
        df_total = df_nouveau
        print(f"  Fichier cree avec {len(df_total)} posts -> {nom_fichier}")

    # encoding="utf-8-sig" : ajoute un BOM UTF-8 pour que Excel ouvre
    # correctement les accents sans manipulation manuelle
    df_total.to_csv(nom_fichier, index=False, encoding="utf-8-sig")


def extraire_mots_frequents(posts, top_n=30, longueur_min=4):
    """
    Extrait les mots les plus fréquents dans les titres et contenus des posts.

    Utilisé pour alimenter la recherche itérative par mots-clés (étape 3).
    Filtre les mots trop courts et une liste de mots vides (stopwords) basique.

    LIMITE : la liste de stopwords est très réduite. Des mots fréquents
    mais sans intérêt (articles, pronoms…) peuvent apparaître dans les résultats.
    Pour un usage sérieux, utiliser une bibliothèque dédiée comme `nltk` ou `spacy`.

    Args:
        posts       (list[dict]) : posts dont on analyse les textes
        top_n       (int)         : nombre de mots à retourner
        longueur_min (int)       : longueur minimale d'un mot pour être conservé

    Returns:
        list[str] : liste des top_n mots les plus fréquents
    """
    STOPWORDS = {
        "avec", "pour", "dans", "sont", "cette", "comme", "plus",
        "tout", "mais", "bien", "tres", "aussi", "quand", "meme",
        "peut", "fait", "that", "this", "with", "from", "have",
        "they", "their", "about", "what", "your", "will", "just",
        "been", "some", "than", "then", "were", "there",
    }

    # Concatène titre + contenu de chaque post en une seule chaîne
    textes = " ".join([
        str(p.get("titre", "")) + " " + str(p.get("contenu", ""))
        for p in posts
    ])

    # Tokenisation naïve par espace + nettoyage de la ponctuation
    # Ne gère pas les apostrophes ("l'histoire" → "l" et "histoire")
    # ni la casse mixte de manière robuste
    mots = [
        mot.lower().strip(".,!?;:\"'()[]")
        for mot in textes.split()
        if len(mot) >= longueur_min and mot.lower() not in STOPWORDS
    ]

    compteur = Counter(mots)
    # most_common(top_n) retourne une liste de tuples (mot, fréquence)
    # on ne garde que le mot
    return [mot for mot, _ in compteur.most_common(top_n)]




# POINT D'ENTRÉE PRINCIPAL
if __name__ == "__main__":

    # Nom du fichier CSV dynamique incluant le contexte du scraping
    dossier_script = os.path.dirname(os.path.abspath(__file__))
    nom_csv = os.path.join(dossier_script, f"reddit_{SUBREDDIT}_{TRI}_{PERIODE}.csv")

    print("=" * 60)
    print(f"  Reddit Scraper - r/{SUBREDDIT}")
    print(f"  Tri : {TRI} | Periode : {PERIODE} | Nb posts : {NB_POSTS}")
    print(f"  Fichier de sortie : {nom_csv}")
    print("=" * 60)

    #  ÉTAPE 1 : Récupération générale 
    # Récupère les NB_POSTS premiers posts selon le tri choisi
    print("\n[1/3] Recuperation generale des posts...")
    posts_init = scraper_subreddit(SUBREDDIT, tri=TRI, periode=PERIODE, nb_posts=NB_POSTS)
    sauvegarder_donnees(nom_csv, posts_init)

    # Si la recherche par mots-clés est désactivée ou si l'étape 1 n'a rien retourné,
    # on s'arrête là
    if not RECHERCHE_MOTS or not posts_init:
        print("\nScraping termine !")
        print(f"Fichier : {nom_csv}")
        exit()  # `exit()` fonctionne mais `sys.exit()` est la pratique recommandée

    #  ÉTAPE 2 : Extraction des mots-clés fréquents 
    # Analyse les textes collectés à l'étape 1 pour identifier les sujets dominants
    print(f"\n[2/3] Extraction des {TOP_MOTS} mots-cles les plus frequents...")
    mots_cles = extraire_mots_frequents(posts_init, top_n=TOP_MOTS)
    print(f"  Mots-cles : {mots_cles}")

    #  ÉTAPE 3 : Recherche itérative par mot-clé 
    # Pour chaque mot-clé, lance une nouvelle recherche dans le subreddit
    # L'objectif est de trouver des posts qui n'apparaissaient pas dans le flux général
    print(f"\n[3/3] Recherche iterative sur {len(mots_cles)} mots-cles...")
    for i, mot in enumerate(mots_cles, 1):
        print(f"\n  [{i}/{len(mots_cles)}] Mot-cle : '{mot}'")
        posts_mot = scraper_subreddit(
            SUBREDDIT, tri=TRI, periode=PERIODE, nb_posts=NB_POSTS, query=mot
        )
        sauvegarder_donnees(nom_csv, posts_mot)

        # Pause entre chaque mot-clé pour éviter un ban temporaire
        # 2 minutes x 30 mots-clés = 1 heure d'attente minimum
        # À réduire si le subreddit est peu actif
        if i < len(mots_cles):
            print("  Pause 2 min avant le prochain mot-cle...")
            time.sleep(120)

    print("\n" + "=" * 60)
    print(f"  Scraping termine !")
    print(f"  Fichier final : {nom_csv}")
    print("=" * 60)