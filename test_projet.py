"""
Module de tests unitaires pour le projet Reddit Success Predictor.

Ce script utilise Pytest pour valider les trois piliers du projet :
1. La collecte de données (API Reddit via RedditFetcher).
2. Le traitement visuel de l'interface (Labels de sentiment).
3. La logique métier des recommandations (Génération de conseils).

Usage:
    pytest test/test_projet.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

# Imports des fonctions à tester
from RedditFetcher import scraper_subreddit
from app import label_polarite, generer_conseils, calculer_ratio_majuscules

# =====================================================================
# TEST 1 : RÉCUPÉRATION DE DONNÉES VIA L'API
# =====================================================================

@patch('RedditFetcher.requests.get')
def test_api_recuperation_donnees(mock_get):
    """
    Vérifie la robustesse du scraper face aux données de l'API Reddit.
    
    Ce test simule (Mock) une réponse HTTP de Reddit pour vérifier que 
    la fonction 'scraper_subreddit' extrait correctement les champs 
    nécessaires (titre, score, etc.) sans effectuer de vraie requête réseau.
    
    Args:
        mock_get: Objet simulé remplaçant requests.get via unittest.mock.
    """
    # Configuration de la fausse réponse JSON
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "12345",
                        "title": "Un titre de test",
                        "score": 250,
                        "upvote_ratio": 0.98,
                        "num_comments": 45,
                        "created_utc": 1620000000,
                        "link_flair_text": "Discussion",
                        "is_self": True,
                        "selftext": "Voici le contenu de mon post.",
                        "url": "https://reddit.com",
                        "subreddit": "Histoire"
                    }
                }
            ],
            "after": None
        }
    }
    mock_get.return_value = mock_response

    # Exécution
    posts = scraper_subreddit("Histoire", nb_posts=1)

    # Assertions : On vérifie la transformation des données
    assert len(posts) == 1
    assert posts[0]["titre"] == "Un titre de test"
    assert posts[0]["score"] == 250
    assert posts[0]["subreddit"] == "Histoire"


# =====================================================================
# TEST 2 : INTERFACE - AFFICHAGE DES RÉSULTATS
# =====================================================================

def test_label_polarite():
    """
    Valide la logique d'affichage HTML des scores de sentiment.
    
    Vérifie que pour chaque plage de score VADER, la fonction retourne 
    le code couleur HTML et le texte de label appropriés pour l'interface.
    """
    # Test Positif : score >= 0.05
    html_positif = label_polarite(0.500)
    assert "positif" in html_positif and "#1e7e34" in html_positif

    # Test Négatif : score <= -0.05
    html_negatif = label_polarite(-0.300)
    assert "négatif" in html_negatif and "#c0392b" in html_negatif

    # Test Neutre : entre -0.05 et 0.05
    html_neutre = label_polarite(0.020)
    assert "neutre" in html_neutre and "#888" in html_neutre


# =====================================================================
# TEST 3 : INTERFACE - LOGIQUE MÉTIER DES CONSEILS
# =====================================================================

def test_generer_conseils_logique_affichage():
    """
    Teste le système expert de génération de conseils personnalisés.
    
    Vérifie deux scénarios critiques :
    1. Un post VIRAL doit recevoir un message de félicitations unique.
    2. Un post FLOP doit recevoir des alertes de correction (ex: titre trop court).
    """
    # Cas VIRAL
    conseils_viral = generer_conseils("Titre parfait et assez long", "Contenu", "r/france", "VIRAL")
    assert len(conseils_viral) == 1
    assert "parfaitement optimisé" in conseils_viral[0]

    # Cas FLOP (sur un subreddit générique pour éviter les limites d'affichage)
    conseils_flop = generer_conseils("Court", "Contenu", "r/test_sub", "FLOP")
    
    assert len(conseils_flop) > 1
    assert any("très court" in conseil for conseil in conseils_flop)
    assert any("flop" in conseil.lower() for conseil in conseils_flop)


def test_calculer_ratio_majuscules():
    """
    Vérifie la précision du calcul du ratio de majuscules (Feature Engineering).
    
    Teste les cas limites comme les chaînes vides pour prévenir les 
    erreurs de division par zéro.
    """
    assert calculer_ratio_majuscules("HELLO") == 1.0
    assert calculer_ratio_majuscules("Hi") == 0.5
    assert calculer_ratio_majuscules("") == 0.0