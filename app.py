import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import onnxruntime as ort

# Load class labels
with open("class_indices.json") as f:
    class_names = json.load(f)

# Load ONNX model
@st.cache_resource
def load_model():
    session = ort.InferenceSession("eye_disease_model.onnx",providers=["CPUExecutionProvider"])
    return session



session = load_model()

input_name = session.get_inputs()[0].name

st.title("👁️ Eye Disease Detection — MobileNetV2 Transfer Learning")

st.title("Classes: Normal · Diabetic Retinopathy · Cataract · Glaucoma 4 classes")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run ONNX inference
    outputs = session.run(None, {input_name: img})
    prediction = outputs[0]

    # Get predicted class
    pred_index = np.argmax(prediction)
    pred_label = class_names[str(pred_index)]
    confidence = float(np.max(prediction)) * 100

    st.subheader("Prediction Result")
    st.success(f"{pred_label} ({confidence:.2f}% confidence)")
