import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd

model = load_model("road_condition_model.h5")

class_names = ['Blocked_Road','Clean_Road','Potholes','Smooth_Road','Traffic_Roads']

risk_map = {
    "Potholes": "High",
    "Blocked_Road": "High",
    "Traffic_Roads": "Medium",
    "Clean_Road": "Low",
    "Smooth_Road": "Low"
}

st.title(" Smart Road Condition Analysis System")

uploaded_file = st.file_uploader("Upload Road Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    risk = risk_map[predicted_class]

    st.subheader("Prediction Result")
    st.write("Road Condition:", predicted_class)
    st.write("Risk Level:", risk)
