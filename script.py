import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Performance Prediction using ML Models")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your student CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Step 1: Create binary label
    df['Result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    features = ['studytime', 'failures', 'absences', 'internet', 'schoolsup', 'famsup', 'paid', 'G1', 'G2']

    for col in ['internet', 'schoolsup', 'famsup', 'paid']:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df[features]
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            eval_metric='logloss',
            random_state=42
        )
    }

    st.header("📊 Model Evaluation Results")

    accuracies = {}
    xgb_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        st.subheader(f"{name}")
        st.write(f"**Accuracy:** {acc:.4f}")
        st.text("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        if name == "XGBoost":
            xgb_model = model
            y_pred_xgb = y_pred

    # 📉 Accuracy Comparison
    st.header("📈 Accuracy Comparison Chart")
    fig_acc = plt.figure(figsize=(6, 4))
    plt.bar(accuracies.keys(), accuracies.values(), color=["red", "green", "blue"])
    plt.ylim(0.6, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    st.pyplot(fig_acc)

    # 📌 Confusion Matrix for XGBoost
    st.header("🧾 XGBoost Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_xgb)
    fig_cm = plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("XGBoost Confusion Matrix")
    st.pyplot(fig_cm)

    # 🔍 Feature Importance
    st.header("⭐ Feature Importance - XGBoost")
    importances = xgb_model.feature_importances_
    fig_feat = plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color="orange")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance - XGBoost")
    plt.gca().invert_yaxis()
    st.pyplot(fig_feat)

    # 🧮 Custom Input
    st.header("🧠 Make Custom Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        studytime = st.slider("Study Time", 1, 4, 2)
        failures = st.slider("Failures", 0, 3, 1)
        absences = st.slider("Absences", 0, 30, 3)
    with col2:
        internet = st.selectbox("Internet Access", ["No", "Yes"])
        schoolsup = st.selectbox("School Support", ["No", "Yes"])
        famsup = st.selectbox("Family Support", ["No", "Yes"])
    with col3:
        paid = st.selectbox("Paid Classes", ["No", "Yes"])
        G1 = st.slider("G1 Grade", 0, 20, 12)
        G2 = st.slider("G2 Grade", 0, 20, 13)

    if st.button("🔮 Predict Result"):
        encode = lambda val: 1 if val == "Yes" else 0
        sample = pd.DataFrame([[studytime, failures, absences,
                                encode(internet), encode(schoolsup),
                                encode(famsup), encode(paid),
                                G1, G2]], columns=features)
        prediction = xgb_model.predict(sample)[0]
        st.success("🎉 Prediction: PASS ✅" if prediction == 1 else "❌ Prediction: FAIL")
else:
    st.warning("Please upload a dataset to begin.")
