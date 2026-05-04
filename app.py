import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 🔝 MUST BE FIRST (only once!)
st.set_page_config(
    page_title="KNN Classifier App",
    layout="wide"
)

# 🎯 HEADER
st.title("KNN Classifier App")
st.markdown("---")
st.caption("Built with Streamlit | Machine Learning Demo")
st.write("Upload your dataset and train a KNN model interactively.")

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

    # 🎯 SELECT TARGET
    target_column = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ⚙️ TRAIN MODEL
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42
    )

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    st.subheader("Model Performance")
    st.success(f"Accuracy: {accuracy:.2f}")

    # 📊 VISUALIZATION (only if 2 features)
    if X.shape[1] == 2:
        st.subheader("Feature Visualization")
        fig, ax = plt.subplots()
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")

