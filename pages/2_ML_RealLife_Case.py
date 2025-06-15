import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Machine Learning Case Study",
    page_icon="�",
    layout="wide"
)

st.title("🧠 Project: Machine Learning Case Study - Customer Churn Prediction")
st.markdown("""
This section showcases a practical application of machine learning by building a **Customer Churn Prediction** model.
Understanding and predicting customer churn is crucial for businesses to retain valuable customers.
Here, I'll walk through the process of data generation, preprocessing, model training, evaluation, and interactive prediction.
""")

st.subheader("1. Data Generation & Overview")
st.write("We'll start by generating a synthetic dataset that mimics customer behavior related to churn.")

@st.cache_data
def generate_customer_churn_data(n_samples=500):
    """Generates synthetic customer churn data."""
    np.random.seed(42)

    data = {
        'MonthlyCharges': np.random.normal(70, 20, n_samples),
        'TotalCharges': np.random.normal(1500, 800, n_samples),
        'Tenure': np.random.randint(1, 72, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.6, 0.25, 0.15]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.25, 0.65, 0.10]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.20, 0.70, 0.10]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    df = pd.DataFrame(data)

    # Introduce some correlation with churn
    df['Churn'] = 0 # Default to no churn

    # Rule 1: Customers with month-to-month contract, high monthly charges, and low tenure are more likely to churn
    condition1 = (df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 80) & (df['Tenure'] < 24)
    if df[condition1].shape[0] > 0:
        df.loc[condition1, 'Churn'] = np.random.choice([0, 1], size=len(df[condition1]), p=[0.3, 0.7])

    # Rule 2: Customers with no online security or tech support are also slightly more prone to churn
    condition2 = (df['OnlineSecurity'] == 'No') | (df['TechSupport'] == 'No')
    if df[condition2].shape[0] > 0:
        # Generate new churn decisions (0 or 1) for this subset based on probabilities
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition2]), p=[0.7, 0.3])
        # Update 'Churn': if a customer was already marked as 1 by Rule 1, keep it 1. Otherwise, use new_churn_decisions.
        df.loc[condition2, 'Churn'] = np.maximum(df.loc[condition2, 'Churn'], new_churn_decisions)

    # Ensure Churn is binary 0 or 1
    df['Churn'] = df['Churn'].astype(int)

    # Handle potential NaNs in MonthlyCharges or Tenure before calculating TotalCharges to avoid NaN propagation
    df['MonthlyCharges_temp'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())
    df['Tenure_temp'] = df['Tenure'].fillna(df['Tenure'].median())

    # Use masks to ensure operations are only on relevant rows and shapes match
    mask_no_churn = df['Churn'] == 0
    if df[mask_no_churn].shape[0] > 0: # Check if there are any non-churners
        df.loc[mask_no_churn, 'TotalCharges'] = (
            df.loc[mask_no_churn, 'MonthlyCharges_temp'] *
            df.loc[mask_no_churn, 'Tenure_temp'] *
            np.random.uniform(0.9, 1.1, size=len(df.loc[mask_no_churn]))
        )

    mask_churn = df['Churn'] == 1
    if df[mask_churn].shape[0] > 0: # Check if there are any churners
        df.loc[mask_churn, 'TotalCharges'] = (
            df.loc[mask_churn, 'MonthlyCharges_temp'] *
            df.loc[mask_churn, 'Tenure_temp'] *
            np.random.uniform(0.5, 0.9, size=len(df.loc[mask_churn]))
        )

    # Drop temp columns
    df.drop(columns=['MonthlyCharges_temp', 'Tenure_temp'], inplace=True)

    # Add some NaN values to TotalCharges and Tenure *after* initial calculations for the data cleaning step later
    df.loc[df.sample(frac=0.02).index, 'TotalCharges'] = np.nan
    df.loc[df.sample(frac=0.01).index, 'Tenure'] = np.nan

    return df.round(2)

churn_df = generate_customer_churn_data()

st.dataframe(churn_df.head())
st.write(f"Dataset shape: {churn_df.shape[0]} rows, {churn_df.shape[1]} columns")

with st.expander("Show Data Description & Missing Values"):
    st.write("`df.info()`:")
    buf = io.StringIO()
    churn_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("`df.describe(include='all')`:")
    st.dataframe(churn_df.describe(include='all'))

    st.write("Churn Distribution:")
    fig_churn = px.pie(churn_df, names='Churn', title='Distribution of Customer Churn', hole=0.3,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_churn)

# --- 2. Data Preprocessing ---

st.subheader("2. Data Preprocessing")
st.markdown("""
This step involves handling missing values, encoding categorical features, and scaling numerical features
to prepare the data for the machine learning model.
""")

df_processed = churn_df.copy()

# Handle missing values
st.write("- Imputing missing `TotalCharges` and `Tenure` with their respective medians.")
df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
df_processed['Tenure'].fillna(df_processed['Tenure'].median(), inplace=True)

# Encode categorical features
st.write("- Encoding categorical features (`Contract`, `OnlineSecurity`, `TechSupport`, `Gender`) using One-Hot Encoding and Label Encoding.")
categorical_cols_ohe = ['Contract', 'OnlineSecurity', 'TechSupport']
df_processed = pd.get_dummies(df_processed, columns=categorical_cols_ohe, drop_first=True)

# Label encode Gender (binary)
le = LabelEncoder()
df_processed['Gender'] = le.fit_transform(df_processed['Gender']) # Male=1, Female=0 (or vice versa)

st.write("DataFrame after encoding categorical features:")
st.dataframe(df_processed.head())

# Scale numerical features
st.write("- Scaling numerical features (`MonthlyCharges`, `TotalCharges`, `Tenure`) using StandardScaler.")
numerical_cols = ['MonthlyCharges', 'TotalCharges', 'Tenure']
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

st.write("DataFrame after scaling numerical features:")
st.dataframe(df_processed.head())

st.success("Data preprocessing complete!")

# --- 3. Model Training ---

st.subheader("3. Model Training")
st.markdown("""
Now, we'll split the data into training and testing sets and train a **Logistic Regression** model
to predict customer churn.
""")

X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

st.write(f"Training data shape: {X_train.shape}")
st.write(f"Testing data shape: {X_test.shape}")

model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

st.write("Model trained: **Logistic Regression**")
st.success("Model training complete!")

# --- 4. Model Evaluation ---

st.subheader("4. Model Evaluation")
st.markdown("""
Evaluating the model's performance on unseen data is crucial. We'll look at key metrics like
accuracy, precision, recall, F1-score, and the confusion matrix.
""")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"- **Accuracy:** `{accuracy:.4f}`")
st.write(f"- **Precision:** `{precision:.4f}`")
st.write(f"- **Recall:** `{recall:.4f}`")
st.write(f"- **F1-Score:** `{f1:.4f}`")

st.markdown("---")
st.write("#### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted No Churn', 'Predicted Churn'],
            yticklabels=['Actual No Churn', 'Actual Churn'], ax=ax_cm)
ax_cm.set_ylabel('Actual Label')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

st.markdown("---")
st.write("#### Classification Report")
st.code(classification_report(y_test, y_pred))

st.info("""
**Interpretation of Metrics:**
* **Accuracy:** Overall correctness of predictions.
* **Precision:** Of all predicted churners, how many actually churned? (Important for minimizing false positives if churn intervention is costly)
* **Recall:** Of all actual churners, how many did the model correctly identify? (Important for minimizing false negatives if missing churners is costly)
* **F1-Score:** Harmonic mean of precision and recall, balancing both.
* **Confusion Matrix:** Visualizes the counts of true positives, true negatives, false positives, and false negatives.
""")


# --- 5. Interactive Prediction ---

st.subheader("5. Interactive Churn Prediction")
st.markdown("""
Test the model yourself! Adjust the customer attributes below to see how the model predicts churn.
""")

# Get a list of features used by the model after one-hot encoding
feature_columns = X.columns.tolist()

with st.form("churn_prediction_form"):
    st.write("Enter Customer Details:")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'], index=0)
        senior_citizen = st.checkbox("Senior Citizen?", False)
        monthly_charges = st.slider("Monthly Charges", 10.0, 150.0, 70.0, 5.0)
    with col2:
        tenure = st.slider("Tenure (Months)", 1, 72, 24, 1)
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'], index=0)
        online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'], index=1)
        tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'], index=1)

    submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Create a dictionary for the new input
        input_data = {
            'MonthlyCharges': monthly_charges,
            'TotalCharges': monthly_charges * tenure, # Estimate total charges
            'Tenure': tenure,
            'SeniorCitizen': 1 if senior_citizen else 0,
            'Gender': 1 if gender == 'Male' else 0 # Based on label encoder fit earlier
        }

        # Add one-hot encoded columns, ensuring all are present and correctly ordered
        for col in ['Contract_One year', 'Contract_Two year',
                    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                    'TechSupport_No internet service', 'TechSupport_Yes']:
            input_data[col] = 0

        if contract == 'One year':
            input_data['Contract_One year'] = 1
        elif contract == 'Two year':
            input_data['Contract_Two year'] = 1

        if online_security == 'Yes':
            input_data['OnlineSecurity_Yes'] = 1
        elif online_security == 'No internet service':
            input_data['OnlineSecurity_No internet service'] = 1

        if tech_support == 'Yes':
            input_data['TechSupport_Yes'] = 1
        elif tech_support == 'No internet service':
            input_data['TechSupport_No internet service'] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure input_df has all columns that X_train has, in the same order
        # Fill missing columns with 0 (for categorical features not selected)
        missing_cols = set(feature_columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[feature_columns] # Reorder columns

        # Scale numerical features of the input data using the *trained* scaler
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])


        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0, 1]

        st.write("---")
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"This customer is **LIKELY TO CHURN** with a probability of **{prediction_proba:.2f}**.")
            st.markdown("💡 *Consider proactive retention strategies for this customer!*")
        else:
            st.success(f"This customer is **UNLIKELY TO CHURN** with a probability of **{1 - prediction_proba:.2f}**.")
            st.markdown("✅ *Good news! Continue monitoring customer satisfaction.*")

st.markdown("---")
st.subheader("Key Takeaways from Machine Learning Case Study:")
st.markdown("""
* **End-to-End ML Workflow:** This project demonstrates the complete pipeline from data understanding,
    preprocessing, model training, to evaluation and deployment.
* **Importance of Preprocessing:** Proper handling of missing values, categorical encoding, and scaling
    is vital for model performance.
* **Model Selection & Evaluation:** Choosing the right model and evaluating it with appropriate metrics
    (especially for imbalanced datasets like churn) is crucial for business impact.
* **Actionable Insights:** Machine learning models can provide predictive power, enabling businesses
    to take timely actions, e.g., targeted customer retention campaigns.
""")