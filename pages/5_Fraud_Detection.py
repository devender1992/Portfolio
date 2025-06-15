import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split # Though not strictly used for IF, good for structure
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Fraud Detection Case Study",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Project: Fraud Detection Case Study")
st.markdown("""
This section demonstrates a **Fraud Detection** system using machine learning.
Fraud detection is crucial for financial institutions and businesses to identify
and prevent illicit transactions. We'll explore an **unsupervised anomaly detection**
approach with the **Isolation Forest** algorithm.
""")

st.subheader("1. Data Generation & Overview")
st.write("We'll generate a synthetic dataset simulating transaction data, with a small percentage of fraudulent transactions embedded.")

@st.cache_data
def generate_transaction_data(n_transactions=1000, fraud_ratio=0.01):
    """Generates synthetic transaction data with injected fraud."""
    np.random.seed(42)

    # Legitimate transactions
    n_legit = int(n_transactions * (1 - fraud_ratio))
    legit_data = {
        'Time': np.sort(np.random.randint(0, 7200, n_legit)), # Time in seconds from first transaction
        'Amount': np.random.normal(50, 20, n_legit),
        'V1': np.random.normal(0, 1, n_legit), # Dummy features (like PCA components)
        'V2': np.random.normal(0, 1, n_legit),
        'V3': np.random.normal(0, 1, n_legit),
        'LocationType': np.random.choice(['Online', 'In-Store', 'ATM'], n_legit, p=[0.6, 0.3, 0.1]),
        'Class': 0 # 0 for legitimate
    }
    legit_df = pd.DataFrame(legit_data)
    legit_df['Amount'] = np.maximum(legit_df['Amount'], 1) # Ensure positive amounts

    # Fraudulent transactions
    n_fraud = n_transactions - n_legit
    fraud_data = {
        'Time': np.sort(np.random.randint(0, 7200, n_fraud)), # Can be random, or clustered in certain times
        'Amount': np.random.normal(300, 100, n_fraud), # Higher amounts
        'V1': np.random.normal(-2, 2, n_fraud), # Different distribution for fraud
        'V2': np.random.normal(2, 2, n_fraud),
        'V3': np.random.normal(-1, 2, n_fraud),
        'LocationType': np.random.choice(['Online', 'International_Online'], n_fraud, p=[0.7, 0.3]), # Unusual locations
        'Class': 1 # 1 for fraud
    }
    fraud_df = pd.DataFrame(fraud_data)
    fraud_df['Amount'] = np.maximum(fraud_df['Amount'], 50) # Ensure positive and higher minimum amounts

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset

    return df.round(2)

transaction_df = generate_transaction_data(n_transactions=1000, fraud_ratio=0.02)

st.dataframe(transaction_df.head(10))
st.write(f"Dataset shape: {transaction_df.shape[0]} transactions, {transaction_df.shape[1]} columns")

with st.expander("Explore Raw Data Diagnostics & Imbalance"):
    st.write("#### Data Information (`df.info()`)")
    buf = io.StringIO()
    transaction_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("#### Descriptive Statistics (`df.describe(include='all')`)")
    st.dataframe(transaction_df.describe(include='all'))

    st.write("#### Class Distribution (Imbalance Check)")
    class_counts = transaction_df['Class'].value_counts(normalize=True).reset_index()
    class_counts.columns = ['Class', 'Percentage']
    class_counts['Class_Label'] = class_counts['Class'].map({0: 'Legitimate', 1: 'Fraud'})
    st.dataframe(class_counts)

    fig_imbalance = px.pie(class_counts, names='Class_Label', values='Percentage',
                           title='Distribution of Transaction Classes', hole=0.3,
                           color_discrete_map={'Legitimate': 'lightgreen', 'Fraud': 'salmon'})
    st.plotly_chart(fig_imbalance, key='class_distribution')

    st.warning("Note the significant class imbalance: fraud cases are rare. This is typical for real-world fraud detection and poses a challenge for models.")

# --- 2. Data Preprocessing ---

st.subheader("2. Data Preprocessing")
st.markdown("""
Before applying the anomaly detection model, numerical features need to be scaled, and
categorical features encoded.
""")

df_processed = transaction_df.copy()

# Identify numerical and categorical columns for preprocessing
numerical_cols = ['Time', 'Amount', 'V1', 'V2', 'V3']
categorical_cols = ['LocationType']

# Scale numerical features
st.write("- Scaling numerical features using `StandardScaler`.")
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

# One-hot encode categorical features
st.write("- One-hot encoding categorical features (`LocationType`).")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for dense array
encoded_features = encoder.fit_transform(df_processed[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_processed.index)

# Combine processed numerical and encoded categorical features
X = pd.concat([df_processed[numerical_cols], encoded_df], axis=1)
y = df_processed['Class']

st.write("First 5 rows of preprocessed features (X):")
st.dataframe(X.head())
st.write(f"Shape of preprocessed features: {X.shape}")

st.success("Data preprocessing complete!")

# --- 3. Anomaly Detection Model (Isolation Forest) ---

st.subheader("3. Anomaly Detection with Isolation Forest")
st.markdown("""
The **Isolation Forest** algorithm is effective for anomaly detection because it isolates
anomalies (outliers) rather than profiling normal data points. It does this by
randomly selecting a feature and then randomly selecting a split value between
the maximum and minimum values of the selected feature. This partitioning continues
until all instances are isolated. Anomalies are points that require fewer splits to be isolated.
""")

# We use the 'contamination' parameter to estimate the proportion of outliers in our dataset.
# This guides the model in setting the decision boundary.
contamination_rate = y.value_counts(normalize=True)[1] # Use the actual fraud ratio from generated data
st.write(f"- Initializing Isolation Forest with `contamination={contamination_rate:.3f}` (estimated fraud ratio).")

model = IsolationForest(contamination=contamination_rate, random_state=42)
model.fit(X)

# Get anomaly scores (the lower the score, the more anomalous)
anomaly_scores = model.decision_function(X)
# Predict -1 for outliers (fraud) and 1 for inliers (legitimate)
predictions = model.predict(X)

# Convert Isolation Forest predictions (-1, 1) to (1, 0) for conventional classification report
# 1 (fraud) if original prediction is -1 (outlier/anomaly)
# 0 (legitimate) if original prediction is 1 (inlier)
predicted_classes = np.where(predictions == -1, 1, 0)


st.write("Model trained: **Isolation Forest**")
st.success("Model training complete!")

# --- 4. Model Evaluation ---

st.subheader("4. Model Evaluation")
st.markdown("""
For anomaly detection, evaluation can be tricky, as true labels for all anomalies are often unknown.
However, since we injected fraud, we can assess how well the model identified these cases.
""")

st.write("#### Predicted vs. Actual Fraud Cases")
results_df = pd.DataFrame({'Actual_Class': y, 'Predicted_Anomaly': predicted_classes, 'Anomaly_Score': anomaly_scores})
st.dataframe(results_df.head())

st.write("#### Confusion Matrix")
cm = confusion_matrix(y, predicted_classes)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Legitimate', 'Predicted Fraud'],
            yticklabels=['Actual Legitimate', 'Actual Fraud'], ax=ax_cm)
ax_cm.set_ylabel('Actual Label')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_title('Confusion Matrix (Isolation Forest)')
st.pyplot(fig_cm)

st.write("#### Classification Report")
st.code(classification_report(y, predicted_classes, target_names=['Legitimate (0)', 'Fraud (1)']))

st.info("""
**Interpretation of Metrics in Anomaly Detection:**
* **Precision (for 'Fraud' class):** Out of all transactions predicted as fraud, how many were actually fraudulent? High precision reduces false alarms.
* **Recall (for 'Fraud' class):** Out of all actual fraudulent transactions, how many did the model correctly identify? High recall ensures fewer frauds are missed.
* **F1-Score:** Balances precision and recall.
* **Confusion Matrix:**
    * **True Positives (bottom right):** Correctly identified frauds.
    * **False Positives (top right):** Legitimate transactions incorrectly flagged as fraud (false alarms).
    * **False Negatives (bottom left):** Fraudulent transactions missed by the model.
* In fraud detection, there's often a trade-off. Depending on the business cost of false positives vs. false negatives, you might prioritize recall (to catch most frauds) or precision (to avoid overwhelming investigations with false alarms).
""")

# --- 5. Interactive Fraud Prediction ---

st.subheader("5. Interactive Transaction Fraud Check")
st.markdown("Enter transaction details below to get a real-time fraud prediction from the model.")

with st.form("fraud_prediction_form"):
    st.write("Enter Transaction Details:")

    col1, col2 = st.columns(2)
    with col1:
        input_time = st.slider("Time (seconds from start)", 0, 7500, 1000, 100)
        input_amount = st.slider("Amount", 1.0, 500.0, 75.0, 5.0)
        input_v1 = st.slider("Feature V1", -5.0, 5.0, 0.0, 0.1)
    with col2:
        input_v2 = st.slider("Feature V2", -5.0, 5.0, 0.0, 0.1)
        input_v3 = st.slider("Feature V3", -5.0, 5.0, 0.0, 0.1)
        input_location = st.selectbox("Location Type", ['Online', 'In-Store', 'ATM', 'International_Online'])

    submitted = st.form_submit_button("Check for Fraud")

    if submitted:
        # Create input DataFrame
        user_input_data = {
            'Time': [input_time],
            'Amount': [input_amount],
            'V1': [input_v1],
            'V2': [input_v2],
            'V3': [input_v3],
            'LocationType': [input_location]
        }
        user_input_df = pd.DataFrame(user_input_data)

        # Preprocess user input using the *trained* scaler and encoder
        user_input_scaled = scaler.transform(user_input_df[numerical_cols])
        user_input_encoded = encoder.transform(user_input_df[categorical_cols])

        user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=numerical_cols)
        user_input_encoded_df = pd.DataFrame(user_input_encoded, columns=encoder.get_feature_names_out(categorical_cols))

        final_user_input = pd.concat([user_input_scaled_df, user_input_encoded_df], axis=1)

        # Predict anomaly score
        user_anomaly_score = model.decision_function(final_user_input)[0]
        # Predict if it's an outlier (-1) or inlier (1)
        user_prediction = model.predict(final_user_input)[0]

        st.write("---")
        st.subheader("Prediction Result:")
        st.write(f"**Anomaly Score:** `{user_anomaly_score:.4f}` (lower indicates higher anomaly likelihood)")

        if user_prediction == -1:
            st.error(f"🚨 **Potential Fraud Detected!**")
            st.markdown("💡 *This transaction has characteristics similar to anomalies identified by the model. Further investigation recommended.*")
        else:
            st.success(f"✅ **Transaction Appears Legitimate.**")
            st.markdown("👍 *This transaction falls within the patterns of normal behavior.*")
    else:
        st.info("Adjust the parameters and click 'Check for Fraud' to see a prediction.")


st.markdown("---")
st.subheader("Key Takeaways from Fraud Detection Case Study:")
st.markdown("""
* **Anomaly Detection:** Unsupervised learning algorithms like Isolation Forest are powerful for identifying rare, unusual patterns (anomalies) without requiring labeled fraud data for training.
* **Imbalanced Data:** Fraud datasets are highly imbalanced, meaning legitimate transactions far outnumber fraudulent ones. This requires careful evaluation and potentially specialized techniques (e.g., oversampling/undersampling, different metrics) in production systems.
* **Feature Engineering is Key:** In real-world scenarios, carefully engineered features (e.g., transaction frequency, velocity, network analysis features) significantly improve fraud detection accuracy.
* **Dynamic Thresholding:** The decision boundary for classifying fraud often needs to be tuned based on business risk tolerance (cost of false positives vs. false negatives).
* **Human-in-the-Loop:** Fraud detection systems often serve as a first line of defense, flagging suspicious cases for human analysts to review.
""")
