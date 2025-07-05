# Group62-AI-Tools-Assignment-Week-3

# AI Tools Assignment - Week 3 (Group 62)

This project demonstrates foundational AI and machine learning techniques across classical ML, deep learning, and NLP using Python. It includes model training, evaluation, and ethical analysis, with optional web deployment.

---

## 📁 Contents

- **Task 1:** Classical ML with Scikit-learn (Iris dataset)
- **Task 2:** CNN for MNIST Digit Classification (TensorFlow)
- **Task 3:** NLP with spaCy and TextBlob (NER + Sentiment)
- **Part 4:** Ethics & Optimization
- **Bonus:** Streamlit Web App Deployment (optional)

---

## 🧠 Task 1: Classical ML with Scikit-learn

**Goal:** Predict iris species using a Decision Tree Classifier.

- 📊 Dataset: Iris
- 🧹 Preprocessing: Label encoding, missing value handling
- ✅ Evaluation: Accuracy, Precision, Recall

### Files:
- `task1_iris_classifier.py`
- `task1_iris_classifier.ipynb`

---

## 🤖 Task 2: Deep Learning with TensorFlow

**Goal:** Build a CNN to classify handwritten digits.

- 🧠 Dataset: MNIST
- 🧱 Model: 2 Conv layers + Dense layers
- 🎯 Accuracy: >95% on test set
- 🖼 Visuals: Shows prediction on 5 sample images

### Files:
- `task2_mnist_cnn.py`
- `task2_mnist_cnn.ipynb`

---

## 🗣️ Task 3: NLP with spaCy & TextBlob

**Goal:** Perform Named Entity Recognition (NER) and Sentiment Analysis.

- 📦 Tools: spaCy (`en_core_web_sm`), TextBlob
- 🧠 Features: Extract product/brand entities, detect positive/negative sentiment

### Files:
- `task3_nlp_sentiment.py`
- `task3_nlp_sentiment.ipynb`

---

## 🔍 Ethics & Optimization

### 1. Bias Analysis
- **MNIST:** Risk of bias due to over-representation of certain digit styles.
- **Amazon Reviews:** Sentiment may misclassify sarcasm, informal slang, or underrepresented dialects.
- **Mitigation:** Use tools like [TensorFlow Fairness Indicators](https://www.tensorflow.org/responsible_ai/fairness_indicators) and improve rule-based models in spaCy.

### 2. Troubleshooting Challenge
- Debugged TensorFlow code with:
  - Shape mismatches
  - Incorrect loss functions
  - Missing activation or wrong input dimensions

---

## 🚀 Bonus Task: Streamlit Web App (Optional)

**Goal:** Deploy MNIST CNN as a simple web app.

- **Framework:** Streamlit
- **Features:** Upload image or select from MNIST test set for prediction

### File:
- `mnist_streamlit_app.py`

To run:
```bash
streamlit run mnist_streamlit_app.py
