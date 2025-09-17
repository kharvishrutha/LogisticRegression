import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.title("🚢 Titanic Survival Prediction (Logistic Regression)")

# File upload
train_file = st.file_uploader("Upload Titanic Training CSV", type=["csv"])
test_file = st.file_uploader("Upload Titanic Test CSV", type=["csv"])

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    st.subheader("📊 Training Data Preview")
    st.write(train_df.head())

    # Handle missing values
    train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
    test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
    train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
    test_df["Embarked"].fillna(test_df["Embarked"].mode()[0], inplace=True)
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    # Label Encoding
    label_enc = LabelEncoder()
    for col in ["Sex", "Embarked"]:
        train_df[col] = label_enc.fit_transform(train_df[col])
        test_df[col] = label_enc.transform(test_df[col])

    # One-Hot Encoding
    categorical_cols = ["Sex", "Embarked", "Pclass"]
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

    # Align columns
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

    # Define X, y
    X = train_encoded.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    y = train_encoded['Survived']

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_val)
    y_proba = log_reg.predict_proba(X_val)[:, 1]

    st.subheader("✅ Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_val, y_pred):.3f}")
    st.write(f"**Precision:** {precision_score(y_val, y_pred):.3f}")
    st.write(f"**Recall:** {recall_score(y_val, y_pred):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_val, y_pred):.3f}")
    st.write(f"**ROC-AUC Score:** {roc_auc_score(y_val, y_proba):.3f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0,1], [0,1], 'k--', label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Show feature importance
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": log_reg.coef_[0],
        "Odds_Ratio": np.exp(log_reg.coef_[0])
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    st.subheader("📌 Feature Importance")
    st.dataframe(coefficients)

else:
    st.info("👆 Please upload both training and test CSV files to continue.")
