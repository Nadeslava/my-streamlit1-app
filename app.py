import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 🔝 CONFIG
st.set_page_config(page_title="AI Model Dashboard", layout="wide")

# 🎨 STYLE
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #1f4e79;}
</style>
""", unsafe_allow_html=True)

# 🎯 HEADER
st.title("🤖 AI Model Comparison Dashboard")
st.caption("Compare Machine Learning Models in Real-Time")

st.markdown("---")

# 🎛️ SIDEBAR
st.sidebar.title("⚙️ Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
max_k = st.sidebar.slider("Max K (for graph)", 1, 20, 10)

# 📂 FILE UPLOAD
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    # 🧠 MODELS
    knn = KNeighborsClassifier(n_neighbors=3)
    logreg = LogisticRegression(max_iter=1000)
    svm = SVC()

    knn.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    acc_knn = knn.score(X_test, y_test)
    acc_log = logreg.score(X_test, y_test)
    acc_svm = svm.score(X_test, y_test)

    # 📊 MODEL COMPARISON
    st.markdown("## 📊 Model Comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("KNN Accuracy", f"{acc_knn:.2f}")
    col2.metric("Logistic Regression", f"{acc_log:.2f}")
    col3.metric("SVM Accuracy", f"{acc_svm:.2f}")

    # 📈 BAR CHART
    st.markdown("### 📈 Model Accuracy Chart")

    fig, ax = plt.subplots()
    models = ["KNN", "Logistic", "SVM"]
    scores = [acc_knn, acc_log, acc_svm]
    ax.bar(models, scores)
    ax.set_ylabel("Accuracy")
    st.pyplot(fig)

    # 📈 KNN GRAPH
    st.markdown("### 📉 Accuracy vs K (KNN)")

    k_values = list(range(1, max_k + 1))
    accuracies = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        accuracies.append(model.score(X_test, y_test))

    fig2, ax2 = plt.subplots()
    ax2.plot(k_values, accuracies, marker='o')
    ax2.set_xlabel("K Value")
    ax2.set_ylabel("Accuracy")
    st.pyplot(fig2)

    # 🔮 PREDICTION
    st.markdown("### 🔮 Make Prediction")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(col, value=float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    pred_knn = knn.predict(input_df)[0]
    pred_log = logreg.predict(input_df)[0]
    pred_svm = svm.predict(input_df)[0]

    st.write(f"KNN Prediction: {pred_knn}")
    st.write(f"Logistic Prediction: {pred_log}")
    st.write(f"SVM Prediction: {pred_svm}")

else:
    st.info("Upload a dataset to begin.")


