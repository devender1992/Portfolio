import streamlit as st
import pandas as pd
import numpy as np
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Data Cleaning Showcase",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Project: Data Cleaning & Preprocessing")
st.markdown("""
This section demonstrates the critical process of transforming raw, 'dirty' data into a clean,
structured format suitable for analysis and machine learning. I'll showcase techniques for
handling missing values, incorrect data types, duplicates, and outliers.
""")

# --- 1. Generate Synthetic Dirty Data ---
@st.cache_data
def generate_dirty_data():
    """Generates a synthetic DataFrame with various data quality issues."""
    np.random.seed(42)
    data = {
        'CustomerID': range(101, 151),
        'Age': np.random.randint(18, 70, 50).astype(float),
        'Gender': np.random.choice(['Male', 'Female', 'Other', 'unknown'], 50),
        'ProductCategory': np.random.choice(['Electronics', 'Books', 'Home Goods', 'Food', 'Software', 'N/A'], 50),
        'PurchaseAmount': np.random.normal(150, 50, 50),
        'PurchaseDate': pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D').strftime('%Y-%m-%d')),
        'Rating': np.random.choice([1, 2, 3, 4, 5, np.nan], 50, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1]),
        'Email': [f'customer{i}@example.com' for i in range(101, 151)],
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', None], 50)
    }
    df = pd.DataFrame(data)

    # Introduce dirtiness
    # Missing values
    df.loc[df.sample(frac=0.1).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'PurchaseAmount'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'Email'] = None # Example of empty string instead of NaN

    # Duplicates
    df = pd.concat([df, df.sample(n=3, random_state=1)]) # Add 3 duplicate rows

    # Incorrect data types (simulated by having mixed types, or values that should be numeric but are strings)
    df.loc[5, 'Age'] = 'twenty five' # String in numeric column
    df.loc[10, 'PurchaseAmount'] = '$120.50' # String with currency symbol

    # Outliers
    df.loc[20, 'PurchaseAmount'] = 15000.0 # Extreme outlier
    df.loc[25, 'Age'] = 120 # Unrealistic age

    # Inconsistent formatting (Gender)
    df.loc[12, 'Gender'] = 'male'
    df.loc[18, 'Gender'] = 'FEMALE'

    # Introduce some invalid emails
    df.loc[30, 'Email'] = 'invalid_email'
    df.loc[35, 'Email'] = 'customer135@examplecom'

    return df

raw_df = generate_dirty_data()

st.subheader("1. Raw, Dirty Data Overview")
st.write("This is the initial dataset, containing various imperfections:")
st.dataframe(raw_df.head())
st.write(f"Shape of raw data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

with st.expander("Show Full Raw Data Description"):
    st.write("`df.info()`:")
    buf = io.StringIO()
    raw_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("`df.describe(include='all')`:")
    st.dataframe(raw_df.describe(include='all'))

# --- 2. Step-by-Step Cleaning Process ---

st.subheader("2. Step-by-Step Data Cleaning Process")

st.markdown("---")
st.markdown("#### **Step 2.1: Handling Duplicates**")
initial_rows = raw_df.shape[0]
df_cleaned = raw_df.copy()
df_cleaned.drop_duplicates(inplace=True)
duplicate_removed_count = initial_rows - df_cleaned.shape[0]
st.write(f"- Removed **{duplicate_removed_count}** duplicate rows.")
st.dataframe(df_cleaned.head())
if duplicate_removed_count > 0:
    st.success("Duplicates handled.")
else:
    st.info("No duplicates found/removed in this run.")

st.markdown("---")
st.markdown("#### **Step 2.2: Standardizing Categorical Columns (e.g., 'Gender', 'ProductCategory')**")

# Standardize 'Gender'
df_cleaned['Gender'] = df_cleaned['Gender'].replace({'male': 'Male', 'FEMALE': 'Female', 'unknown': np.nan, 'Other':np.nan})
df_cleaned['Gender'] = df_cleaned['Gender'].fillna('Not Specified') # Replace remaining NaNs

# Standardize 'ProductCategory'
df_cleaned['ProductCategory'] = df_cleaned['ProductCategory'].replace({'N/A': np.nan})
df_cleaned['ProductCategory'] = df_cleaned['ProductCategory'].fillna('Uncategorized') # Replace remaining NaNs

st.write("- Standardized 'Gender' and 'ProductCategory' values (e.g., 'Male', 'Female', 'Not Specified', 'Uncategorized').")
st.dataframe(df_cleaned[['Gender', 'ProductCategory']].value_counts().reset_index(name='Count'))

st.markdown("---")
st.markdown("#### **Step 2.3: Correcting Data Types & Handling Non-Numeric Entries**")

st.write("- Converting 'Age' and 'PurchaseAmount' to numeric, coercing errors to NaN.")
# Convert 'Age'
df_cleaned['Age'] = pd.to_numeric(df_cleaned['Age'], errors='coerce')

# Clean 'PurchaseAmount' by removing non-numeric characters and converting
df_cleaned['PurchaseAmount'] = df_cleaned['PurchaseAmount'].astype(str).str.replace('[^0-9.]', '', regex=True)
df_cleaned['PurchaseAmount'] = pd.to_numeric(df_cleaned['PurchaseAmount'], errors='coerce')

st.write("`df.info()` after type conversion:")
buf = io.StringIO()
df_cleaned.info(buf=buf)
st.code(buf.getvalue())

st.markdown("---")
st.markdown("#### **Step 2.4: Handling Missing Values**")

st.write("- Imputing missing 'Age' with the median.")
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)

st.write("- Imputing missing 'PurchaseAmount' with the mean.")
df_cleaned['PurchaseAmount'].fillna(df_cleaned['PurchaseAmount'].mean(), inplace=True)

st.write("- Imputing missing 'Rating' with the mode.")
df_cleaned['Rating'].fillna(df_cleaned['Rating'].mode()[0], inplace=True)

st.write("- Filling missing 'City' with 'Unknown'.")
df_cleaned['City'].fillna('Unknown', inplace=True)

st.write("- Handling missing 'Email' values (e.g., converting None to 'No Email').")
df_cleaned['Email'].fillna('No Email', inplace=True)

st.write("Missing values after imputation:")
st.dataframe(df_cleaned.isnull().sum().reset_index(name='Missing Count').rename(columns={'index': 'Column'}))
if df_cleaned.isnull().sum().sum() == 0:
    st.success("All missing values handled.")
else:
    st.warning("Some missing values still exist, review if further handling is needed.")


st.markdown("---")
st.markdown("#### **Step 2.5: Handling Outliers (e.g., using IQR for 'PurchaseAmount')**")

st.write("- Identifying and capping 'PurchaseAmount' outliers using the IQR method.")
Q1 = df_cleaned['PurchaseAmount'].quantile(0.25)
Q3 = df_cleaned['PurchaseAmount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers
df_cleaned['PurchaseAmount'] = np.where(
    df_cleaned['PurchaseAmount'] < lower_bound,
    lower_bound,
    df_cleaned['PurchaseAmount']
)
df_cleaned['PurchaseAmount'] = np.where(
    df_cleaned['PurchaseAmount'] > upper_bound,
    upper_bound,
    df_cleaned['PurchaseAmount']
)

st.write(f"Purchase Amount outliers capped to range: [{lower_bound:.2f}, {upper_bound:.2f}]")

st.write("- Handling 'Age' outliers (e.g., capping unrealistic ages).")
df_cleaned['Age'] = np.where(df_cleaned['Age'] > 100, df_cleaned['Age'].median(), df_cleaned['Age']) # Cap age at 100
df_cleaned['Age'] = np.where(df_cleaned['Age'] < 18, df_cleaned['Age'].median(), df_cleaned['Age']) # Cap age at 18
st.write("Age outliers capped between 18 and 100, replaced with median.")

st.markdown("---")
st.markdown("#### **Step 2.6: Validating and Cleaning Email Formats**")

st.write("- Basic validation of email formats.")
# Simple regex for email validation (can be more robust)
email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df_cleaned['Email_Valid'] = df_cleaned['Email'].astype(str).str.match(email_regex)

# Identify invalid emails
invalid_emails_count = df_cleaned[df_cleaned['Email_Valid'] == False].shape[0]
st.write(f"- Identified **{invalid_emails_count}** potentially invalid email addresses.")

# Option to remove or mark invalid emails
df_cleaned.loc[df_cleaned['Email_Valid'] == False, 'Email'] = 'Invalid/No Email'
df_cleaned.drop(columns=['Email_Valid'], inplace=True) # Drop helper column

st.markdown("---")

st.subheader("3. Cleaned Data Summary")
st.write("The dataset after all cleaning steps:")
st.dataframe(df_cleaned.head())
st.write(f"Shape of cleaned data: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

with st.expander("Show Full Cleaned Data Description"):
    st.write("`df_cleaned.info()`:")
    buf = io.StringIO()
    df_cleaned.info(buf=buf)
    st.code(buf.getvalue())

    st.write("`df_cleaned.describe(include='all')`:")
    st.dataframe(df_cleaned.describe(include='all'))

st.success("Data Cleaning Process Completed!")

st.markdown("---")
st.subheader("Key Takeaways from Data Cleaning:")
st.markdown("""
* **Importance of Data Quality:** Even seemingly small issues like inconsistent casing or extra characters can severely impact analysis.
* **Iterative Process:** Data cleaning is rarely a one-shot activity; it often requires multiple passes and validations.
* **Domain Knowledge:** Understanding the data's context (e.g., realistic age ranges, valid product categories) is crucial for effective cleaning.
* **Impact on Models:** Clean data directly leads to more accurate and reliable machine learning models and more trustworthy insights.
""")

st.download_button(
    label="Download Cleaned Data as CSV",
    data=df_cleaned.to_csv(index=False).encode('utf-8'),
    file_name='cleaned_data.csv',
    mime='text/csv',
)
