# 👁️ Eye Disease Detection using Deep Learning

## 🚀 Project Overview

This project is a **Deep Learning–based web application** that detects eye diseases from retinal images.
The model is trained using a **Convolutional Neural Network (CNN)** based on the **MobileNetV2 architecture** and deployed using **Streamlit**.

The application allows users to **upload an eye image 🖼️** and get a prediction of the possible eye condition.

---

## 🔍 Classes Predicted

The model classifies images into the following four categories:

* 🟡 **Cataract**
* 🔴 **Diabetic Retinopathy**
* 🟠 **Glaucoma**
* 🟢 **Normal**

---

## 🤖 Model Details

* **Architecture:** MobileNetV2 (Transfer Learning)
* **Exported Model Format:** ONNX
* **Inference Engine:** ONNX Runtime
* **Deployment Framework:** Streamlit

---

## 🌐 Live Application

You can try the deployed application here:

👉 **Live Demo:**
https://eyediseasedetector1234.streamlit.app/

Upload an eye image and the model will predict the condition in a few seconds ⚡.

---

## 📂 Project Structure

```
eye_disease_detector/
│
├── app.py
├── model.onnx
├── classes.json
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🛠️ Requirements

* **Python 3.11 recommended**
* **pip** package manager

Main dependencies:

* streamlit
* onnxruntime
* numpy
* pillow

---

## 💻 Installation & Running Locally

### 1️⃣ Clone the Repository

```
git clone https://github.com/kundetivamsi2001/eye_disease_detector.git
cd eye_disease_detector
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

```
python -m venv venv
```

Activate the environment:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application

```
streamlit run app.py
```

---

### 5️⃣ Open the Application

Once the app starts, open your browser and go to:

```
http://localhost:8501
```

Upload a retinal image and the system will display the **predicted eye disease along with confidence score 📊**.

---

## ⚙️ How the Model Works

1️⃣ User uploads an eye image
2️⃣ Image is resized to **224 × 224 pixels**
3️⃣ Image is normalized and converted into a tensor
4️⃣ The **ONNX model performs inference**
5️⃣ The predicted class and confidence score are displayed

---

## 🌟 Future Improvements

* 🔬 Add **Grad-CAM visualization** for model explainability
* 📊 Add **probability charts** for predictions
* 🎨 Improve UI with better medical interpretation messages
* 📦 Support
## 📸 Application Screenshot

![App Screenshot](assets/app_screenshot.png)
