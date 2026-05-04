import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 🔝 CONFIG
st.set_page_config(page_title="KNN Classifier App", layout="wide")

# 🎯 HEADER
st.title("KNN Classifier App")
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning Demo")

# 🎛️ SIDEBAR
st.sidebar.title("Settings")
k = st.sidebar.slider("Select K value", 1, 15, 3)
test_size = st.sidebar.slider("Test size (%)", 10, 50, 20)

# 📂 FILE UPLOAD
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 🎯 TARGET
    target_column = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ⚙️ TRAIN
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # 📊 LAYOUT
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        st.success(f"Accuracy: {accuracy:.2f}")

        # 📊 CONFUSION MATRIX
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Make a Prediction")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.success(f"Prediction: {prediction}")

    # 📥 DOWNLOAD
    st.subheader("Download Predictions")

    df["Prediction"] = model.predict(X)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV with Predictions",
        csv,
        "predictions.csv",
        "text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")

