"""
Streamlit Web Application for ML-Based Credit Card Fraud Detection System.
Provides interactive dashboards, model comparisons, real-time single inference,
batch CSV scoring, and financial cost-threshold calibration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fraud-badge-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        border: 1px solid #ef9a9a;
    }
    .fraud-badge-safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        border: 1px solid #a5d6a7;
    }
    .fraud-badge-review {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        border: 1px solid #ffe082;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path Helpers & Caching
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_resource
def load_models_and_preprocessor():
    """Loads all serialized models, preprocessor, and metrics."""
    models = {}
    preprocessor = None
    metrics = {}

    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    model_files = {
        "XGBoost (Cost-Sensitive Weighted)": "xgboost.joblib",
        "Random Forest (Class Weighted)": "random_forest.joblib",
        "Logistic Regression (Class Weighted)": "logistic_regression.joblib",
        "XGBoost + SMOTE": "xgboost_smote.joblib",
        "Logistic Regression + SMOTE": "logistic_regression_smote.joblib",
    }

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if os.path.exists(best_path):
        models["⭐ Best Overall Model"] = joblib.load(best_path)

    return models, preprocessor, metrics


@st.cache_data
def load_sample_dataset():
    """Loads sample or benchmark dataset for EDA and batch demos."""
    sample_path = os.path.join(DATA_DIR, "creditcard_sample.csv")
    full_path = os.path.join(DATA_DIR, "creditcard.csv")

    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    elif os.path.exists(full_path):
        df = pd.read_csv(full_path)
        return df.sample(n=min(10000, len(df)), random_state=42)
    return None


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=64)
st.sidebar.title("Fraud Guard ML")
st.sidebar.caption("Intelligent Credit Card Transaction Screening")

models, preprocessor, metrics = load_models_and_preprocessor()
sample_df = load_sample_dataset()

# Model selection
model_options = list(models.keys()) if models else ["No trained models found"]
selected_model_name = st.sidebar.selectbox(
    "Select ML Algorithm",
    model_options,
    index=0 if model_options else None,
    help="Compare different models trained with Class Weighting vs SMOTE."
)

# Threshold slider
threshold = st.sidebar.slider(
    "Decision Threshold (θ)",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
    help="Transactions with predicted fraud probability >= θ are flagged as Suspicious/Fraud."
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
if sample_df is not None:
    st.sidebar.write(f"• Dataset Sample: **{len(sample_df):,} rows**")
    fraud_pct = (sample_df['Class'].mean()) * 100
    st.sidebar.write(f"• Baseline Fraud Rate: **{fraud_pct:.3f}%**")
st.sidebar.write("• Imbalance Handling: **SMOTE & Cost Weighting**")
st.sidebar.write("• Primary Metric: **PR-AUC & Recall**")

# ---------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------
st.title("💳 ML-Based Credit Card Fraud Detection System")
st.markdown("""
An intelligent, machine-learning-driven financial transaction screening platform designed to detect fraudulent credit card activity
in highly imbalanced transaction streams using advanced classification algorithms, SMOTE resampling, and risk threshold optimization.
""")

tab_eda, tab_benchmark, tab_single, tab_batch, tab_threshold = st.tabs([
    "📊 EDA & Fraud Patterns",
    "🏆 Model Benchmarking",
    "🔍 Real-Time Fraud Screener",
    "📁 Batch Transaction Screening",
    "⚖️ Threshold & Financial Simulator"
])

# ---------------------------------------------------------
# TAB 1: EDA & FRAUD PATTERNS
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis & Fraud Dynamics")

    if sample_df is None:
        st.warning("⚠️ No dataset found. Please run training pipeline first (`python -m src.train`).")
    else:
        total_tx = len(sample_df)
        total_fraud = int(sample_df['Class'].sum())
        total_genuine = total_tx - total_fraud
        fraud_ratio = (total_fraud / total_tx) * 100

        avg_genuine_amt = sample_df[sample_df['Class'] == 0]['Amount'].mean()
        avg_fraud_amt = sample_df[sample_df['Class'] == 1]['Amount'].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_tx:,}")
        col2.metric("Genuine Transactions", f"{total_genuine:,}")
        col3.metric("Fraudulent Cases", f"{total_fraud:,}", delta=f"{fraud_ratio:.3f}% rate", delta_color="inverse")
        col4.metric("Avg Fraud Amount", f"€{avg_fraud_amt:.2f}", delta=f"vs €{avg_genuine_amt:.2f} genuine")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Extreme Class Imbalance")
            fig_pie = px.pie(
                names=["Genuine (0)", "Fraudulent (1)"],
                values=[total_genuine, total_fraud],
                color=["Genuine (0)", "Fraudulent (1)"],
                color_discrete_map={"Genuine (0)": "#2b5c8f", "Fraudulent (1)": "#e53935"},
                hole=0.45,
                title="Class Distribution (Fraud represents ~0.17%)"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Transaction Amount Distribution (Log-Scale)")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=np.log1p(sample_df[sample_df['Class'] == 0]['Amount']),
                name="Genuine",
                opacity=0.6,
                marker_color="#1E88E5",
                nbinsx=40
            ))
            fig_hist.add_trace(go.Histogram(
                x=np.log1p(sample_df[sample_df['Class'] == 1]['Amount']),
                name="Fraudulent",
                opacity=0.75,
                marker_color="#E53935",
                nbinsx=40
            ))
            fig_hist.update_layout(
                barmode='overlay',
                title="Log(Amount + 1) Distribution by Class",
                xaxis_title="Log Transformed Amount",
                yaxis_title="Count",
                legend=dict(x=0.8, y=0.95)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Top Discriminative PCA Features for Fraud Detection")
        st.markdown("""
        Empirical studies demonstrate that features **V14, V17, V12, and V10** exhibit pronounced negative deviations during fraud,
        while **V4, V11, and V2** exhibit significant positive spikes.
        """)

        feature_cols = ['V14', 'V17', 'V12', 'V10', 'V4', 'V11']
        valid_cols = [c for c in feature_cols if c in sample_df.columns]
        if valid_cols:
            means = sample_df.groupby('Class')[valid_cols].mean().T
            means.columns = ['Genuine', 'Fraudulent']
            means['Shift'] = means['Fraudulent'] - means['Genuine']

            fig_bar = px.bar(
                means.reset_index(),
                x='index',
                y='Shift',
                color='Shift',
                color_continuous_scale='RdBu_r',
                title="Feature Value Shift in Fraud vs Genuine Transactions",
                labels={'index': 'Feature', 'Shift': 'Mean Difference (Fraud - Genuine)'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: MODEL BENCHMARKING
# ---------------------------------------------------------
with tab_benchmark:
    st.subheader("Algorithm Comparison & Performance Metrics")
    st.markdown("""
    Because fraud datasets have extreme class imbalance, traditional **Accuracy is misleading** (a dummy model predicting 100% genuine achieves 99.8% accuracy but catches zero fraud).
    We evaluate models primarily using **PR-AUC (Precision-Recall AUC)**, **Recall (Sensitivity)**, and **F2-Score**.
    """)

    if not metrics:
        st.info("Run the training script to generate and save comparative model benchmarks.")
    else:
        # Create summary table
        rows = []
        for name, m in metrics.items():
            if name.startswith("_"):
                continue
            rows.append({
                "Algorithm": name,
                "PR-AUC (Primary)": f"{m.get('pr_auc', 0.0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
                "Recall (Detection Rate)": f"{m.get('recall', 0.0):.4f}",
                "Precision": f"{m.get('precision', 0.0):.4f}",
                "F1-Score": f"{m.get('f1_score', 0.0):.4f}",
                "F2-Score": f"{m.get('f2_score', 0.0):.4f}",
                "Accuracy": f"{m.get('accuracy', 0.0):.4%}",
                "Optimal F1 Thresh": f"{m.get('optimal_thresholds', {}).get('best_f1_threshold', 0.5):.2f}"
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df.sort_values(by="PR-AUC (Primary)", ascending=False), use_container_width=True)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Metric Comparison Across Models")
            plot_df = pd.DataFrame([
                {
                    "Algorithm": name,
                    "Recall": m.get('recall', 0),
                    "Precision": m.get('precision', 0),
                    "PR-AUC": m.get('pr_auc', 0),
                    "F1-Score": m.get('f1_score', 0)
                }
                for name, m in metrics.items() if not name.startswith("_")
            ])
            fig_compare = px.bar(
                plot_df,
                x="Algorithm",
                y=["Recall", "Precision", "PR-AUC"],
                barmode="group",
                title="Recall vs Precision vs PR-AUC",
                color_discrete_sequence=["#E53935", "#1E88E5", "#43A047"]
            )
            fig_compare.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig_compare, use_container_width=True)

        with col_c2:
            st.markdown(f"#### Confusion Matrix ({selected_model_name})")
            cur_metrics = metrics.get(selected_model_name, {})
            cm = cur_metrics.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})

            cm_matrix = [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=['Genuine (0)', 'Fraud (1)'],
                y=['Genuine (0)', 'Fraud (1)'],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Test Set Confusion Matrix @ θ={threshold}"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: REAL-TIME TRANSACTION FRAUD SCREENER
# ---------------------------------------------------------
with tab_single:
    st.subheader("Single Transaction Fraud Screener")
    st.markdown("Enter transaction parameters or load pre-configured test presets to simulate real-time risk assessment.")

    # Preset selection
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("🟢 Preset: Genuine Coffee & Grocery (€18.50)", use_container_width=True):
        preset_choice = "genuine"
    if p_col2.button("🔴 Preset: Suspicious Overnight Drain (€1,890.00)", use_container_width=True):
        preset_choice = "fraud"
    if p_col3.button("🎲 Preset: Random Sample from Test Data", use_container_width=True):
        preset_choice = "random"

    # Default values
    default_amount = 45.00
    default_time = 36000.0
    v_defaults = {f"V{i}": 0.0 for i in range(1, 29)}

    if preset_choice == "genuine":
        default_amount = 18.50
        default_time = 42000.0  # midday
        v_defaults["V14"] = 0.12
        v_defaults["V17"] = -0.05
        v_defaults["V12"] = 0.20
        v_defaults["V10"] = -0.10
        v_defaults["V4"] = 0.05
    elif preset_choice == "fraud":
        default_amount = 1890.00
        default_time = 8200.0   # 2:15 AM
        v_defaults["V14"] = -7.45
        v_defaults["V17"] = -6.10
        v_defaults["V12"] = -5.30
        v_defaults["V10"] = -4.80
        v_defaults["V4"] = 4.90
        v_defaults["V11"] = 3.80
        v_defaults["V2"] = 3.50
    elif preset_choice == "random" and sample_df is not None:
        sample_row = sample_df.sample(1).iloc[0]
        default_amount = float(sample_row['Amount'])
        default_time = float(sample_row['Time'])
        for i in range(1, 29):
            if f"V{i}" in sample_row:
                v_defaults[f"V{i}"] = float(sample_row[f"V{i}"])

    with st.form("single_tx_form"):
        st.markdown("##### Core Transaction Metadata")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            inp_amount = st.number_input("Transaction Amount (€)", min_value=0.01, max_value=50000.0, value=float(default_amount), step=1.0)
        with t_col2:
            inp_time = st.number_input("Transaction Time (Seconds Elapsed, 0 to 172800)", min_value=0.0, max_value=200000.0, value=float(default_time), step=100.0)

        with st.expander("Key Discriminative Components (V14, V17, V12, V10, V4, V11)", expanded=True):
            k_col1, k_col2, k_col3 = st.columns(3)
            with k_col1:
                v14 = st.slider("V14 (High Negative Fraud Indicator)", -15.0, 10.0, float(v_defaults["V14"]), step=0.1)
                v17 = st.slider("V17 (High Negative Fraud Indicator)", -15.0, 10.0, float(v_defaults["V17"]), step=0.1)
            with k_col2:
                v12 = st.slider("V12 (Negative Fraud Indicator)", -15.0, 10.0, float(v_defaults["V12"]), step=0.1)
                v10 = st.slider("V10 (Negative Fraud Indicator)", -15.0, 10.0, float(v_defaults["V10"]), step=0.1)
            with k_col3:
                v4 = st.slider("V4 (Positive Fraud Indicator)", -10.0, 15.0, float(v_defaults["V4"]), step=0.1)
                v11 = st.slider("V11 (Positive Fraud Indicator)", -10.0, 15.0, float(v_defaults.get("V11", 0.0)), step=0.1)

        with st.expander("Remaining Latent Features (V1 - V28)", expanded=False):
            st.caption("PCA features are normalized and centered near 0.")
            pca_cols = st.columns(4)
            other_vs = {}
            for i in range(1, 29):
                col_name = f"V{i}"
                if col_name in ["V14", "V17", "V12", "V10", "V4", "V11"]:
                    continue
                col_idx = (i % 4)
                with pca_cols[col_idx]:
                    other_vs[col_name] = st.number_input(col_name, value=float(v_defaults.get(col_name, 0.0)), step=0.1, key=f"inp_{col_name}")

        submit_btn = st.form_submit_button("⚡ Analyze Transaction Risk", use_container_width=True)

    if submit_btn:
        active_model = models.get(selected_model_name)
        if active_model is None or preprocessor is None:
            st.error("Model or Preprocessor not available. Please train models first.")
        else:
            # Construct feature row
            row_dict = {"Time": inp_time, "Amount": inp_amount}
            row_dict["V14"] = v14
            row_dict["V17"] = v17
            row_dict["V12"] = v12
            row_dict["V10"] = v10
            row_dict["V4"] = v4
            row_dict["V11"] = v11
            for k, val in other_vs.items():
                row_dict[k] = val

            # Arrange into DataFrame matching original schema
            expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            tx_df = pd.DataFrame([row_dict])[expected_cols]

            # Transform through fitted preprocessor
            tx_scaled = preprocessor.transform(tx_df)

            # Predict
            if hasattr(active_model, "predict_proba"):
                fraud_prob = float(active_model.predict_proba(tx_scaled)[0, 1])
            else:
                fraud_prob = float(active_model.predict(tx_scaled)[0])

            st.markdown("### Risk Assessment Results")
            res_c1, res_c2 = st.columns([1, 1])

            with res_c1:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    number={'suffix': "%", 'valueformat': ".2f"},
                    title={'text': f"Fraud Probability ({selected_model_name})"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#111827"},
                        'steps': [
                            {'range': [0, 25], 'color': "#81C784"},
                            {'range': [25, threshold * 100], 'color': "#FFF176"},
                            {'range': [threshold * 100, 100], 'color': "#E57373"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.8,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_c2:
                st.write("")
                st.write("")
                if fraud_prob >= threshold:
                    st.markdown("""
                    <div class="fraud-badge-alert">
                        <h3>🚨 SUSPICIOUS FRAUD DETECTED</h3>
                        <p>This transaction exhibits anomalous risk vectors exceeding the authorized threshold.</p>
                        <p><b>Recommended Action:</b> Block transaction immediately and initiate automated SMS/2FA verification.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif fraud_prob >= 0.20:
                    st.markdown("""
                    <div class="fraud-badge-review">
                        <h3>⚠️ MANUAL SCREENING RECOMMENDED</h3>
                        <p>Probability is elevated above normal baseline but below hard blocking limit.</p>
                        <p><b>Recommended Action:</b> Route to fraud investigation analyst queue for review.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="fraud-badge-safe">
                        <h3>✅ TRANSACTION APPROVED</h3>
                        <p>Low risk score aligned with standard legitimate consumer spending patterns.</p>
                        <p><b>Recommended Action:</b> Approve transaction seamlessly without customer friction.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write(f"• Evaluated Model: **{selected_model_name}**")
                st.write(f"• Decision Threshold (θ): **{threshold:.2f}**")
                st.write(f"• Computed Score: **{fraud_prob:.4f}** ({fraud_prob*100:.2f}%)")

# ---------------------------------------------------------
# TAB 4: BATCH TRANSACTION SCREENING
# ---------------------------------------------------------
with tab_batch:
    st.subheader("Batch CSV Transaction Screening")
    st.markdown("Upload a CSV file containing transactions (`Time`, `Amount`, `V1`..`V28`) to triage transactions in bulk.")

    col_b1, col_b2, col_b3 = st.columns([2, 1, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload Transaction File (CSV)", type=["csv"])
    with col_b2:
        st.write("")
        st.write("")
        use_sample = st.button("📂 Load Quick Sample (1,000 Txns)", use_container_width=True)
    with col_b3:
        st.write("")
        st.write("")
        test_sample_path = os.path.join(DATA_DIR, "test_batch_sample.csv")
        if os.path.exists(test_sample_path):
            with open(test_sample_path, "rb") as f:
                st.download_button(
                    "📥 Download Test CSV (15 Txns)",
                    data=f.read(),
                    file_name="test_batch_sample.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Click to download a pre-configured 15-transaction test CSV file containing both genuine and fraudulent transactions."
                )

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample and sample_df is not None:
        batch_df = sample_df.sample(min(1000, len(sample_df)), random_state=123).copy()

    if batch_df is not None:
        st.success(f"Loaded {len(batch_df):,} transactions for batch screening.")
        active_model = models.get(selected_model_name)

        if active_model is not None and preprocessor is not None:
            with st.spinner("Scoring batch transactions..."):
                # Clean and transform
                features_only = batch_df[[c for c in batch_df.columns if c != 'Class']]
                scaled_features = preprocessor.transform(features_only)

                if hasattr(active_model, "predict_proba"):
                    probs = active_model.predict_proba(scaled_features)[:, 1]
                else:
                    probs = active_model.predict(scaled_features)

                scored_df = batch_df.copy()
                scored_df['Fraud_Probability'] = probs
                scored_df['Decision'] = np.where(probs >= threshold, '🚨 FLAGGED FRAUD', '✅ APPROVED')
                scored_df['Risk_Level'] = pd.cut(
                    probs,
                    bins=[-0.01, 0.20, threshold, 1.01],
                    labels=['Low', 'Medium (Review)', 'High (Blocked)']
                )

                flagged_count = int((probs >= threshold).sum())
                flagged_volume = scored_df[probs >= threshold]['Amount'].sum()

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Transactions Scanned", f"{len(scored_df):,}")
                k2.metric("Flagged as Fraud", f"{flagged_count:,}", f"{(flagged_count/len(scored_df))*100:.2f}% of batch")
                k3.metric("Prevented Loss Volume", f"€{flagged_volume:,.2f}")
                k4.metric("Approved Count", f"{len(scored_df) - flagged_count:,}")

                st.markdown("#### Scored Batch Sample (Top Highest Risk First)")
                display_cols = ['Transaction_ID', 'Decision', 'Fraud_Probability', 'Risk_Level', 'Amount', 'Time', 'V14', 'V17', 'V12']
                avail_cols = [c for c in display_cols if c in scored_df.columns]

                sorted_scored = scored_df.sort_values(by="Fraud_Probability", ascending=False)
                st.dataframe(sorted_scored[avail_cols].head(50), use_container_width=True)

                # Download CSV
                csv_bytes = sorted_scored.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Scored Batch Report (CSV)",
                    data=csv_bytes,
                    file_name="scored_fraud_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ---------------------------------------------------------
# TAB 5: FINANCIAL COST & THRESHOLD SIMULATOR
# ---------------------------------------------------------
with tab_threshold:
    st.subheader("Financial Impact & Decision Threshold Simulator")
    st.markdown("""
    In financial institutions, misclassification carries asymmetrical costs:
    - **False Negative (Missed Fraud)**: Direct loss of the stolen amount + chargeback fee ($150 - $500).
    - **False Positive (False Alarm)**: Analyst manual investigation cost + consumer friction ($10 - $30).
    Use this simulator to identify the mathematically optimal threshold that minimizes overall financial loss.
    """)

    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        cost_fn = st.slider("Cost of False Negative (Missed Fraud €)", min_value=50, max_value=1000, value=250, step=25)
    with sim_c2:
        cost_fp = st.slider("Cost of False Positive (Manual Review €)", min_value=5, max_value=100, value=20, step=5)

    test_data_path = os.path.join(MODELS_DIR, "test_data.joblib")
    if os.path.exists(test_data_path) and selected_model_name in models:
        test_data = joblib.load(test_data_path)
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        model = models[selected_model_name]

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            y_probs = model.predict(X_test)

        thresh_range = np.linspace(0.02, 0.98, 49)
        sim_results = []
        for t in thresh_range:
            preds = (y_probs >= t).astype(int)
            fn = int(((y_test == 1) & (preds == 0)).sum())
            fp = int(((y_test == 0) & (preds == 1)).sum())
            tp = int(((y_test == 1) & (preds == 1)).sum())
            total_cost = (fn * cost_fn) + (fp * cost_fp)
            sim_results.append({
                "Threshold": t,
                "False Negatives (Missed)": fn,
                "False Positives (Alerts)": fp,
                "Detected Frauds (TP)": tp,
                "Total Loss (€)": total_cost
            })

        sim_df = pd.DataFrame(sim_results)
        best_sim_row = sim_df.loc[sim_df['Total Loss (€)'].idxmin()]

        st.info(f"💡 **Optimal Financial Threshold:** θ = **{best_sim_row['Threshold']:.2f}** with Minimum Total Loss = **€{best_sim_row['Total Loss (€)']:,.2f}**")

        fig_sim = px.line(
            sim_df,
            x="Threshold",
            y="Total Loss (€)",
            title="Total Business Cost vs Decision Threshold",
            labels={"Threshold": "Decision Threshold (θ)", "Total Loss (€)": "Total Cost (€)"}
        )
        fig_sim.add_vline(x=best_sim_row['Threshold'], line_dash="dash", line_color="green", annotation_text="Cost-Optimal Threshold")
        st.plotly_chart(fig_sim, use_container_width=True)
    else:
        st.info("Train models to populate live cost simulation curves.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("ML-Based Credit Card Fraud Detection System | Implemented in Python, Scikit-learn, XGBoost, Imbalanced-learn & Streamlit.")
