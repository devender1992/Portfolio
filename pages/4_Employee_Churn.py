import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Employee Churn Prediction Case Study",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Project: Employee Churn Prediction Case Study")
st.markdown("""
This section focuses on predicting **Employee Churn**, a critical challenge for organizations
seeking to retain talent and reduce recruitment costs. We will build a machine learning model
to identify employees at risk of leaving, based on various factors.
""")

st.subheader("1. Data Generation & Overview")
st.write("We'll start by generating a synthetic dataset representing employee attributes and their churn status.")

@st.cache_data
def generate_employee_churn_data(n_employees=1000, churn_rate=0.15):
    """Generates synthetic employee churn data."""
    np.random.seed(42)

    data = {
        'Age': np.random.randint(22, 60, n_employees),
        'YearsAtCompany': np.random.randint(1, 20, n_employees),
        'MonthlyHours': np.random.normal(160, 20, n_employees), # Typical working hours
        'Salary': np.random.normal(5000, 1500, n_employees), # Monthly salary
        'LastPromotion': np.random.randint(0, 5, n_employees), # Years since last promotion
        'SatisfactionScore': np.random.randint(1, 5, n_employees), # 1 (low) to 5 (high)
        'Department': np.random.choice(['Sales', 'HR', 'IT', 'Marketing', 'Operations'], n_employees),
        'JobRole': np.random.choice(['Manager', 'Associate', 'Analyst', 'Specialist', 'Assistant'], n_employees),
        'Education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n_employees),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_employees, p=[0.7, 0.2, 0.1]),
        'Gender': np.random.choice(['Male', 'Female'], n_employees),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees)
    }
    df = pd.DataFrame(data)

    df['Churn'] = 0 # Default to no churn

    # Introduce churn based on conditions using corrected approach
    # Rule 1: Low satisfaction, high hours, long time since promotion
    condition1 = (df['SatisfactionScore'] < 2) & (df['MonthlyHours'] > 180) & (df['LastPromotion'] > 3)
    if df[condition1].shape[0] > 0:
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition1]), p=[0.2, 0.8])
        df.loc[condition1, 'Churn'] = np.maximum(df.loc[condition1, 'Churn'], new_churn_decisions)

    # Rule 2: Low salary, long years at company
    condition2 = (df['Salary'] < 3500) & (df['YearsAtCompany'] > 5)
    if df[condition2].shape[0] > 0:
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition2]), p=[0.5, 0.5])
        df.loc[condition2, 'Churn'] = np.maximum(df.loc[condition2, 'Churn'], new_churn_decisions)

    # Rule 3: Single marital status, low years at company
    condition3 = (df['MaritalStatus'] == 'Single') & (df['YearsAtCompany'] < 2)
    if df[condition3].shape[0] > 0:
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition3]), p=[0.6, 0.4])
        df.loc[condition3, 'Churn'] = np.maximum(df.loc[condition3, 'Churn'], new_churn_decisions)

    # Rule 4: Sales department
    condition4 = df['Department'] == 'Sales'
    if df[condition4].shape[0] > 0:
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition4]), p=[0.7, 0.3])
        df.loc[condition4, 'Churn'] = np.maximum(df.loc[condition4, 'Churn'], new_churn_decisions)

    # Rule 5: Assistant job role
    condition5 = df['JobRole'] == 'Assistant'
    if df[condition5].shape[0] > 0:
        new_churn_decisions = np.random.choice([0, 1], size=len(df[condition5]), p=[0.7, 0.3])
        df.loc[condition5, 'Churn'] = np.maximum(df.loc[condition5, 'Churn'], new_churn_decisions)


    # Ensure the churn rate is roughly as desired
    current_churn_rate = df['Churn'].mean()
    if current_churn_rate < churn_rate:
        # If actual churn is too low, randomly flip some non-churners to churners
        n_to_flip = int((churn_rate - current_churn_rate) * n_employees)
        non_churn_indices = df[df['Churn'] == 0].sample(n=n_to_flip, random_state=42).index
        df.loc[non_churn_indices, 'Churn'] = 1
    elif current_churn_rate > churn_rate:
        # If actual churn is too high, randomly flip some churners to non-churners
        n_to_flip = int((current_churn_rate - churn_rate) * n_employees)
        churn_indices = df[df['Churn'] == 1].sample(n=n_to_flip, random_state=42).index
        df.loc[churn_indices, 'Churn'] = 0

    df['Churn'] = df['Churn'].astype(int)

    # Introduce some missing values
    df.loc[df.sample(frac=0.03).index, 'Salary'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'MonthlyHours'] = np.nan
    df.loc[df.sample(frac=0.01).index, 'SatisfactionScore'] = np.nan

    return df.round(2)

employee_df = generate_employee_churn_data(n_employees=1000, churn_rate=0.18) # Aim for ~18% churn

st.dataframe(employee_df.head(10))
st.write(f"Dataset shape: {employee_df.shape[0]} employees, {employee_df.shape[1]} columns")

with st.expander("Explore Raw Data Diagnostics & Imbalance"):
    st.write("#### Data Information (`df.info()`)")
    buf = io.StringIO()
    employee_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("#### Descriptive Statistics (`df.describe(include='all')`)")
    st.dataframe(employee_df.describe(include='all'))

    st.write("#### Churn Distribution (Imbalance Check)")
    churn_counts = employee_df['Churn'].value_counts(normalize=True).reset_index()
    churn_counts.columns = ['Churn', 'Percentage']
    churn_counts['Churn_Label'] = churn_counts['Churn'].map({0: 'No Churn', 1: 'Churn'})
    st.dataframe(churn_counts)

    fig_imbalance = px.pie(churn_counts, names='Churn_Label', values='Percentage',
                           title='Distribution of Employee Churn', hole=0.3,
                           color_discrete_map={'No Churn': 'lightgreen', 'Churn': 'salmon'})
    st.plotly_chart(fig_imbalance, key='churn_distribution_pie')

    st.warning("Note the class imbalance: churners are fewer than non-churners. This is common in real-world scenarios and requires careful model evaluation.")

# --- 2. Data Preprocessing ---

st.subheader("2. Data Preprocessing")
st.markdown("""
This step involves handling missing values, encoding categorical features into a numerical format,
and scaling numerical features to prepare the data for the machine learning model.
""")

df_processed = employee_df.copy()

# Handle missing values
st.write("- Imputing missing numerical values (`Salary`, `MonthlyHours`, `SatisfactionScore`) with their respective medians.")
for col in ['Salary', 'MonthlyHours', 'SatisfactionScore']:
    df_processed[col].fillna(df_processed[col].median(), inplace=True)

st.write("Missing values after imputation:")
missing_after_imputation = df_processed.isnull().sum().reset_index(name='Missing Count').rename(columns={'index': 'Column'})
st.dataframe(missing_after_imputation[missing_after_imputation['Missing Count'] > 0])
if missing_after_imputation['Missing Count'].sum() == 0:
    st.success("All missing values handled.")
else:
    st.warning("Some missing values still exist, review if further handling is needed.")


# Identify categorical columns for one-hot encoding
categorical_cols = ['Department', 'JobRole', 'Education', 'BusinessTravel', 'MaritalStatus']
# Gender is binary, can use LabelEncoder for simplicity or keep as is if 0/1 for Female/Male
# For consistency with other string categoricals, let's include Gender for OHE as well.
categorical_cols.append('Gender')


st.write("- One-hot encoding categorical features.")
# Create a OneHotEncoder instance
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical data
encoded_features = encoder.fit_transform(df_processed[categorical_cols])

# Get feature names for the encoded columns
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Create a DataFrame from the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df_processed.index)

# Drop original categorical columns and concatenate the encoded ones
df_processed = df_processed.drop(columns=categorical_cols)
df_processed = pd.concat([df_processed, encoded_df], axis=1)

# Numerical columns for scaling
numerical_cols = ['Age', 'YearsAtCompany', 'MonthlyHours', 'Salary', 'LastPromotion', 'SatisfactionScore']

st.write("- Scaling numerical features using `StandardScaler`.")
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])


X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

st.write("First 5 rows of preprocessed features (X):")
st.dataframe(X.head())
st.write(f"Shape of preprocessed features: {X.shape}")

st.success("Data preprocessing complete!")

# --- 3. Model Training ---

st.subheader("3. Model Training")
st.markdown("""
We'll split the data into training and testing sets and train a **Random Forest Classifier**.
Random Forests are powerful ensemble models known for their good performance and ability
to handle both numerical and categorical features.
""")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

st.write(f"Training data shape: {X_train.shape}")
st.write(f"Testing data shape: {X_test.shape}")

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

st.write("Model trained: **Random Forest Classifier**")
st.info("Using `class_weight='balanced'` to handle class imbalance during training.")
st.success("Model training complete!")

# --- 4. Model Evaluation ---

st.subheader("4. Model Evaluation")
st.markdown("""
Evaluating the model's performance on unseen data is crucial. We'll look at key metrics,
the confusion matrix, and feature importances.
""")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of churning

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])


st.write(f"- **Accuracy:** `{accuracy:.4f}`")
st.write(f"- **Precision (Churn):** `{precision:.4f}`")
st.write(f"- **Recall (Churn):** `{recall:.4f}`")
st.write(f"- **F1-Score (Churn):** `{f1:.4f}`")
st.write(f"- **ROC AUC Score:** `{roc_auc:.4f}`")

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
st.code(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

st.markdown("---")
st.write("#### Feature Importances")
st.markdown("""
Understanding which features contribute most to the model's predictions is vital for
actionable insights and business strategies.
""")
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
# Reset index and rename columns for Plotly Express
feature_importances_df = feature_importances.head(10).reset_index()
feature_importances_df.columns = ['Feature', 'Importance'] # Renamed for clarity in plot

st.dataframe(feature_importances_df)

fig_fi = px.bar(feature_importances_df, x='Importance', y='Feature', orientation='h',
                labels={'Importance': 'Importance', 'Feature': 'Feature'},
                title='Top 10 Feature Importances',
                color='Importance', # Changed to refer to the column name
                color_continuous_scale=px.colors.sequential.Viridis)
fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig_fi, key='feature_importance_chart')

st.info("""
**Interpretation of Metrics & Feature Importances:**
* **Precision (Churn):** High precision means fewer non-churners are misidentified as churners, reducing wasted retention efforts.
* **Recall (Churn):** High recall means more actual churners are identified, allowing for targeted interventions.
* **ROC AUC:** Measures the model's ability to distinguish between churners and non-churners.
* **Feature Importances:** Identifies key drivers of churn, guiding HR strategies (e.g., improve satisfaction, address salary concerns, offer promotions).
""")

# --- 5. Interactive Churn Prediction ---

st.subheader("5. Interactive Employee Churn Prediction")
st.markdown("Enter hypothetical employee details to see the model's churn prediction.")

# Get original categories for selectboxes
original_df_categorical = employee_df[['Department', 'JobRole', 'Education', 'BusinessTravel', 'Gender', 'MaritalStatus']]
categorical_unique_values = {col: original_df_categorical[col].unique().tolist() for col in original_df_categorical.columns}

with st.form("employee_churn_form"):
    st.write("Enter Employee Details:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 65, 30)
        years_at_company = st.slider("Years at Company", 0, 30, 3)
        monthly_hours = st.slider("Monthly Hours", 120, 220, 160)
        salary = st.slider("Monthly Salary", 2000, 10000, 5000)
    with col2:
        last_promotion = st.slider("Years Since Last Promotion", 0, 10, 1)
        satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        department = st.selectbox("Department", categorical_unique_values['Department'])
        job_role = st.selectbox("Job Role", categorical_unique_values['JobRole'])
    with col3:
        education = st.selectbox("Education Level", categorical_unique_values['Education'])
        business_travel = st.selectbox("Business Travel", categorical_unique_values['BusinessTravel'])
        gender = st.selectbox("Gender", categorical_unique_values['Gender'])
        marital_status = st.selectbox("Marital Status", categorical_unique_values['MaritalStatus'])

    submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        # Create input dictionary
        input_data = {
            'Age': age,
            'YearsAtCompany': years_at_company,
            'MonthlyHours': monthly_hours,
            'Salary': salary,
            'LastPromotion': last_promotion,
            'SatisfactionScore': satisfaction_score,
            'Department': department,
            'JobRole': job_role,
            'Education': education,
            'BusinessTravel': business_travel,
            'Gender': gender,
            'MaritalStatus': marital_status
        }
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing steps to the new input data
        # Impute missing (though not expected for single user input, good practice)
        for col in ['Salary', 'MonthlyHours', 'SatisfactionScore']:
            if input_df[col].isnull().any():
                input_df[col].fillna(employee_df[col].median(), inplace=True) # Use original df median

        # One-hot encode categorical features (ensure consistent columns with training data)
        input_encoded = encoder.transform(input_df[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=input_df.index)

        input_df_processed_temp = input_df.drop(columns=categorical_cols)
        final_input_df = pd.concat([input_df_processed_temp, input_encoded_df], axis=1)

        # Scale numerical features
        final_input_df[numerical_cols] = scaler.transform(final_input_df[numerical_cols])

        # Ensure all columns are present and in the same order as training data (X)
        # Add missing columns (from training data X) with zeros, then reorder
        missing_cols = set(X.columns) - set(final_input_df.columns)
        for c in missing_cols:
            final_input_df[c] = 0
        final_input_df = final_input_df[X.columns]


        prediction = model.predict(final_input_df)[0]
        prediction_proba = model.predict_proba(final_input_df)[0, 1]

        st.write("---")
        st.subheader("Prediction Result:")
        st.write(f"**Predicted Probability of Churn:** `{prediction_proba:.2f}`")

        if prediction == 1:
            st.error(f"⚠️ This employee is **PREDICTED TO CHURN**.")
            st.markdown("💡 *Consider tailored retention strategies for this individual (e.g., career development, salary review, addressing satisfaction).*")
        else:
            st.success(f"✅ This employee is **PREDICTED TO STAY**.")
            st.markdown("👍 *Continue fostering a positive work environment.*")
    else:
        st.info("Adjust the employee details and click 'Predict Churn Risk' to see a prediction.")


st.markdown("---")
st.subheader("Key Takeaways from Employee Churn Prediction:")
st.markdown("""
* **Proactive Retention:** Predicting churn allows HR and management to intervene early with targeted retention efforts.
* **Understanding Drivers:** Feature importance provides insights into what factors are most strongly associated with employees leaving.
* **Data-Driven HR:** Machine learning enables data-driven decision-making in human resources, moving beyond intuition.
* **Ethical Considerations:** When deploying such models, it's crucial to consider fairness and bias, ensuring predictions are not discriminatory.
""")
