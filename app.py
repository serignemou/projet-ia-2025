import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("poubelle_model.h5")

st.title("🗑️ Détection de Poubelle Pleine / Vide")

def predict(image):
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    if prob >= 0.5:
        return "🟦 Poubelle vide", prob
    else:
        return "🟩 Poubelle pleine", prob

uploaded_file = st.file_uploader("Uploader une image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image Uploadée", use_column_width=True)

    label, prob = predict(image)

    st.subheader(f"Résultat : **{label}**")
    st.write(f"Probabilité : `{prob:.4f}`")

with open("poubelle_model.h5", "rb") as f:
    st.download_button("⬇️ Télécharger le modèle", f, file_name="poubelle_model.h5")
