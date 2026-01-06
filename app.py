# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
import os

# Disable safetensors
os.environ['SAFETENSORS_FAST_GPU'] = '0'

st.set_page_config(page_title="Lung Cancer Fusion Model", layout="wide")
st.title("🫁 Lung Cancer Prediction (Image + Symptoms)")
st.write("Upload a chest X-ray / CT scan and enter patient symptoms to get prediction.")

# -----------------------------
# 1️⃣ Load CNN + BERT models + fusion architecture
# -----------------------------
@st.cache_resource
def load_models():
    # Load CNN
    cnn_model = load_model("models/lung_cancer_cnn_keras3.keras", compile=False)
    cnn_model.trainable = False

    # Load BERT
    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = TFBertModel.from_pretrained(bert_model_name, use_safetensors=False, from_pt=True)

    # Fusion model architecture
    cnn_input = Input(shape=(224,224,3), name="cnn_input")
    cnn_output = cnn_model(cnn_input)

    text_input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    text_attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    bert_outputs = bert_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
    bert_cls_output = bert_outputs.last_hidden_state[:,0,:]

    fusion = Concatenate(name="fusion_concat")([cnn_output, bert_cls_output])
    fusion = Dense(128, activation="relu", name="fusion_dense_1")(fusion)
    fusion_output = Dense(1, activation="sigmoid", name="fusion_output")(fusion)

    fusion_model = Model(inputs=[cnn_input, text_input_ids, text_attention_mask], outputs=fusion_output)

    # Load trained weights
    weights_path = "models/fusion_weights.h5"
    if os.path.exists(weights_path):
        fusion_model.load_weights(weights_path)

    return fusion_model, tokenizer

fusion_model, tokenizer = load_models()
st.success("Models loaded successfully!")

# -----------------------------
# 2️⃣ Upload image + input text
# -----------------------------
uploaded_file = st.file_uploader("Upload Chest X-ray / CT scan image", type=["jpg","jpeg","png"])
symptoms = st.text_area("Enter patient symptoms", "Persistent cough, chest pain, difficulty breathing")

if st.button("Predict"):
    if uploaded_file is not None and symptoms.strip() != "":
        # Process image
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224,224))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)

        # Tokenize text
        encoded = tokenizer([symptoms], padding="max_length", truncation=True, max_length=128, return_tensors="tf")

        # Predict
        pred = fusion_model.predict([img_array, encoded["input_ids"], encoded["attention_mask"]], verbose=0)
        prob = pred[0][0]
        pred_class = "Malignant (Cancer)" if prob > 0.5 else "Benign (Normal)"
        confidence = abs(prob - 0.5) * 200

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Prediction:** {pred_class}")
        st.write(f"**Probability:** {prob:.4f}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.image(img, caption="Uploaded Image", use_column_width=True)
    else:
        st.error("Please upload an image and enter symptoms to predict!")
