# NLP News Analyzer 📰

## Présentation du projet

NLP News Analyzer est une application interactive développée avec **Streamlit** qui permet d’extraire, nettoyer, analyser et visualiser des articles de presse. Ce projet couvre plusieurs étapes essentielles du traitement automatique du langage naturel (NLP) :

- Extraction d’articles via une API (exemple : The Guardian)
- Nettoyage des textes pour préparer les données
- Calcul d’embeddings TF-IDF pour représenter les textes
- Visualisations interactives, notamment projection t-SNE
- Filtrage dynamique des datasets dans l’interface
- Génération et affichage de visualisations basées sur les embeddings

L’application vise à simplifier l’exploration et l’analyse de corpus d’articles de presse grâce à une interface utilisateur intuitive.

---

## Fonctionnalités principales

- **Extraction de données** : Récupération d’articles depuis une API, avec gestion des fichiers locaux.
- **Nettoyage des textes** : Prétraitement avec tokenisation, suppression de stopwords, etc.
- **Embeddings** : Calcul de représentations TF-IDF des textes nettoyés.
- **Visualisations** : Graphiques interactifs avec Seaborn et matplotlib, dont des projections t-SNE.
- **Interface utilisateur** : Sélection des datasets (uniquement fichiers `*_cleaned.csv`), lancement du traitement, affichage des résultats.

---

## Tests réalisés ✅

Les tests automatisés inclus dans ce projet sont principalement des tests unitaires classiques, couvrant des fonctions comme :

- Chargement et prétraitement des données
- Calcul des embeddings
- Fonctionnalités de visualisation (ex. : t-SNE)

Ces tests conduisent généralement à deux réflexions :

- ⚠️ **Je perds peut-être mon temps à écrire des tests unitaires trop basiques ou inutiles**, qui ne couvrent pas assez les cas réels ou ne détectent pas les vrais bugs.

- ✅ **Ces tests sont utiles et peuvent servir de base pour intégrer des fonctionnalités de contrôle qualité directement dans l’application.** Par exemple, ces tests peuvent évoluer en checks automatiques que l’utilisateur pourrait déclencher à tout moment.

Ainsi, ces tests représentent un premier socle vers un code plus robuste et une app plus fiable.

---

## Installation et utilisation

1. Cloner le dépôt
2. Installer les dépendances (via `pip install -r requirements.txt`)
3. Lancer l’application Streamlit avec `streamlit run streamlit_app/app.py`
4. Naviguer dans le menu pour extraire des données, nettoyer, générer des embeddings et visualiser

---

## Contributeurs

- Eric COSTEROUSSE

---

Merci pour votre intérêt et vos contributions !
