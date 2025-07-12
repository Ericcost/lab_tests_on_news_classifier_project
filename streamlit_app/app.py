import os
import pandas as pd
import streamlit as st
from data_extraction.manager import load_or_fetch_articles
from streamlit_app.utils import ask_for_filename
from nlp.preprocessing import Preprocessing
from nlp.embeddings import Embeddings

st.set_page_config(page_title="NLP News Analyzer", layout="centered")
st.title("📰 Analyse d'articles de presse")

menu = st.sidebar.selectbox(
    "Menu",
    ["Accueil", "Extraire les données", "Nettoyage", "Embeddings"]
)

st.write(f"## {menu}")

if menu == "Accueil":
    st.write("Bienvenue sur l'application NLP News Analyzer ! Choisissez une action dans le menu.")

elif menu == "Extraire les données":
    st.subheader("Extraction via The Guardian API")
    filename = ask_for_filename("data/articles.csv")

    if os.path.exists(filename):
        st.info(f"📄 Le fichier `{filename}` existe déjà.")
        df = pd.read_csv(filename)
        st.write(f"🔢 Nombre total d'articles : **{len(df)}**")

        if "section" in df.columns:
            sections = df["section"].dropna().unique().tolist()
            selected_section = st.selectbox("Filtrer par section :", ["Toutes"] + sections)
            if selected_section != "Toutes":
                df = df[df["section"] == selected_section]

        limit = st.slider("Nombre d'articles à afficher", min_value=5, max_value=min(100, len(df)), value=10)
        st.dataframe(df.head(limit))

        st.markdown("---")
        st.subheader("📦 Créer un subset équilibré (1 article par section)")

        if "section" not in df.columns or "text" not in df.columns:
            st.error("❌ Colonnes attendues manquantes : `section`, `text`.")
        else:
            unique_sections = df["section"].dropna().unique()

            if len(unique_sections) < 6:
                st.warning(f"⚠️ Le fichier contient seulement {len(unique_sections)} section(s). Impossible de créer un subset de 6 articles ayant des sections différentes.")
            else:
                if st.button("Créer un subset de 6 articles (1 par section)"):
                    df_clean = df[df["text"].notnull() & (df["text"].str.strip() != "")]
                    selected_sections = unique_sections[:6]
                    subset_df = pd.concat([
                        df_clean[df_clean["section"] == section].head(1)
                        for section in selected_sections
                    ])

                    subset_filename = filename.replace(".csv", "_subset.csv")
                    subset_df.to_csv(subset_filename, index=False)

                    st.success(f"✅ Subset créé avec 6 articles (1 par section) dans `{subset_filename}`")
                    st.dataframe(subset_df[["section", "title"]])
    else:
        st.warning(f"⚠️ Aucun fichier trouvé à `{filename}`. Lance une extraction ci-dessous.")
        if st.button("Lancer l'extraction"):
            load_or_fetch_articles(filename)

elif menu == "Nettoyage":
    st.subheader("Nettoyage des textes")

    csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    if not csv_files:
        st.warning("⚠️ Aucun fichier CSV trouvé dans le dossier `data/`.")
    else:
        selected_file = st.selectbox("Choisissez un fichier CSV :", csv_files)
        if st.button("Lancer le nettoyage"):
            filepath = os.path.join("data", selected_file)
            df = pd.read_csv(filepath)

            if "text" not in df.columns:
                st.error("❌ Colonne `text` manquante dans le fichier.")
            else:
                preprocessor = Preprocessing()
                df["cleaned_text"] = df["text"].fillna("").map(preprocessor.preprocess)

                cleaned_filename = selected_file.replace(".csv", "_cleaned.csv")
                cleaned_filepath = os.path.join("data", cleaned_filename)
                df.to_csv(cleaned_filepath, index=False)

                st.success(f"✅ Nettoyage terminé et sauvegardé dans `{cleaned_filename}`")
                st.write("Exemple de textes originaux et nettoyés :")
                st.dataframe(df[["text", "cleaned_text"]].head(10))


elif menu == "Embeddings":
    st.subheader("Création des embeddings TF-IDF")

    # 🔍 Lister uniquement les fichiers *_cleaned.csv
    csv_files = [f for f in os.listdir("data") if f.endswith("_cleaned.csv")]

    if not csv_files:
        st.warning("⚠️ Aucun fichier *_cleaned.csv trouvé dans le dossier `data/`.")
    else:
        selected_file = st.selectbox("Choisissez un fichier nettoyé :", csv_files)
        filepath = os.path.join("data", selected_file)

        try:
            df = pd.read_csv(filepath)
        except pd.errors.EmptyDataError:
            st.error("❌ Le fichier est vide ou corrompu.")
        else:
            if "cleaned_text" not in df.columns:
                st.error("❌ Colonne `cleaned_text` manquante dans le fichier.")
            else:
                emb = Embeddings()
                corpus = df["cleaned_text"].fillna("").tolist()
                index = df["section"].fillna("").tolist() if "section" in df.columns else None

                tfidf_df = emb.fit_transform(corpus, index=index)
                st.write("🔢 Matrice TF-IDF (extrait)")
                st.dataframe(tfidf_df.round(3).head(10))

                # ✅ Ajout de la visualisation t-SNE
                st.markdown("---")
                st.subheader("🌀 Visualisation t-SNE des embeddings")

                if tfidf_df.shape[0] < 2:
                    st.warning("⚠️ Pas assez de données pour appliquer t-SNE.")
                elif "section" not in df.columns:
                    st.warning("⚠️ Colonne `section` manquante — impossible de colorer par classe.")
                else:
                    from vis.tsne_plot import plot_tsne
                    if st.button("Afficher la projection t-SNE"):
                        fig = plot_tsne(tfidf_df, df["section"])
                        st.pyplot(fig)

