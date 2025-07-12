# streamlit_app/utils.py

import streamlit as st

def ask_for_filename(default="data/articles.csv"):
    """
    Affiche un champ texte pour demander le nom du fichier à l'utilisateur.
    """
    return st.text_input("Nom du fichier :", value=default)

