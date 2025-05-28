import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Fair ML Model Evaluator", layout="wide", page_icon="🤖")

# --- App Title ---
st.markdown(
    """
    <h1 style='text-align:center; color:#4B8BBE;'>
        🤖 Fairness-Aware ML Classifier
    </h1>
    <p style='text-align:center; font-size:16px;'>
        Upload a dataset, select a model, and evaluate performance with fairness metrics.
    </p>
    """, unsafe_allow_html=True
)
st.write("---")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Expected columns
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']

    df = pd.read_csv(uploaded_file, names=columns, na_values=" ?", skipinitialspace=True)

    st.subheader("📋 Raw Dataset Preview")
    st.dataframe(df.head())

    # Preprocessing
    df.dropna(inplace=True)

    # Label Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns='income')
    y = df['income']
    sensitive = df['sex']  # Sensitive attribute

    X_scaled = StandardScaler().fit_transform(X)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Model Selection & Parameters")
    model_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Random Forest"))

    if model_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, step=10)
        max_depth = st.sidebar.slider("Max Depth", 1, 50, 10, step=1)

    st.sidebar.markdown("---")
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 30, step=5)
    random_state = 42

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X_scaled, y, sensitive, test_size=test_size / 100, random_state=random_state, stratify=sensitive
    )

    # --- Model Training ---
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_proba) * 100
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=s_test) * 100
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=s_test) * 100

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Model Performance Metrics")
        st.metric("Accuracy", f"{acc:.2f}%")
        st.metric("ROC-AUC Score", f"{auc:.2f}%")
        st.markdown(
            f"""
            <div style='margin-top:10px;'>
                <b>Demographic Parity Difference:</b> <span style='color:#e63946;'>{dp:.2f}%</span><br>
                <b>Equalized Odds Difference:</b> <span style='color:#e63946;'>{eo:.2f}%</span>
            </div>
            """, unsafe_allow_html=True
        )

        st.subheader("👥 Group-wise Fairness Metrics")
        mf = MetricFrame(
            metrics={"Accuracy": accuracy_score, "Selection Rate": selection_rate},
            y_true=y_test, y_pred=y_pred, sensitive_features=s_test
        )
        st.dataframe(mf.by_group.style.format("{:.2f}"))

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax,
                    annot_kws={"size": 12}, cbar=False, linewidths=0.7, linecolor='gray')
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.tick_params(labelsize=10)
        st.pyplot(fig)

    # --- Group-wise bar chart ---
    st.subheader("Group-wise Metrics Visualization")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5), dpi=110)
    mf.by_group.plot.bar(ax=ax2, width=0.5)
    ax2.set_title(f"Group-wise Metrics by Sensitive Attribute: Sex", fontsize=12)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', labelsize=10, rotation=0)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.legend(fontsize=9, loc='upper right', frameon=True)
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Classification Report ---
    st.subheader("Classification Report")
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict).transpose()
    st.dataframe(class_report_df.style.format("{:.2f}"))

    # --- Download Report ---
    csv = class_report_df.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download Classification Report as CSV",
        data=csv,
        file_name=f'{model_name.replace(" ", "_").lower()}_classification_report.csv',
        mime='text/csv',
        key='download-report'
    )

    # --- Final Verdict ---
    st.subheader("🔍 Final Fairness Verdict")

    fairness_summary = []

   
    if abs(dp) <= 10:
        fairness_summary.append("Demographic Parity Difference is within fair range (≤ 10%).")
   

    if abs(eo) <= 10:
        fairness_summary.append("Equalized Odds Difference is within fair range (≤ 10%).")
   

    min_accuracy = mf.by_group["Accuracy"].min()
    if min_accuracy >= 0.7:
        fairness_summary.append("All groups have decent accuracy (≥ 70%).")
    

   
    for item in fairness_summary:
        st.markdown(f"- {item}")

   
    st.warning("🔍 The model  shows signs of unbias ")


else:
    st.info("Please upload a CSV file to start the evaluation.")

# --- Footer ---
st.write("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>Developed using Streamlit & Fairlearn</p>",
    unsafe_allow_html=True
)
