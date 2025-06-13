# 🎓 Student Exam Performance Indicator

A web application to predict students' **Math scores** based on their **demographic data and previous scores** using a machine learning model.

This project is built with:
- Flask (backend API)
- HTML/CSS (frontend UI)
- Scikit-learn / CatBoost (model pipeline)
- Pandas, NumPy (data preprocessing)

---

## 🚀 Demo

<img src="templates/static/demo-form.png" width="400">
<img src="templates/static/demo-project-structure.png" width="400">

---

## 📂 Project Structure

├── application.py # Flask entry point
├── artifacts/ # Trained model and preprocessor
├── catboost_info/ # CatBoost training logs
├── notebook/ # Jupyter notebooks for EDA & model development
├── src/ # Source code package
│ ├── components/ # Data ingestion, transformation, model trainer
│ ├── pipeline/ # Prediction pipeline classes
│ ├── utils.py # Helper functions
│ ├── logger.py # Logging configuration
│ └── exception.py # Custom exception handling
├── templates/ # HTML frontend templates
│ ├── index.html
│ └── home.html
├── requirements.txt # Project dependencies
├── setup.py # Package setup
└── README.md # Project documentation



---

## 🧠 ML Model

The prediction pipeline uses:
- **Preprocessing**: OneHotEncoder, StandardScaler
- **Model**: CatBoostRegressor (or any saved model in `artifacts/model.pkl`)

---

## 💡 Features

- Predict student's **Math score** based on:
  - Gender
  - Ethnicity
  - Parental level of education
  - Lunch type
  - Test preparation
  - Reading score
  - Writing score

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/StudentExamPerformance.git
cd StudentExamPerformance
pip install -r requirements.txt
