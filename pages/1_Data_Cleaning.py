import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
handling various data quality issues, supported by visualizations and clear explanations.
""")

# --- 1. Generate Synthetic Dirty Data ---
@st.cache_data
def generate_dirty_data():
    """Generates a synthetic DataFrame with various data quality issues."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'CustomerID': range(1001, 1001 + n_samples),
        'Age': np.random.randint(18, 65, n_samples).astype(float),
        'Gender': np.random.choice(['Male', 'Female', 'other ', 'UNKNOWN'], n_samples),
        'ProductCategory': np.random.choice(['Electronics', 'Books', 'Home Goods', 'Food', 'Software', 'N/A'], n_samples),
        'PurchaseAmount': np.random.normal(150, 50, n_samples),
        'PurchaseDate': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples, freq='D').strftime('%Y-%m-%d')),
        'Rating': np.random.choice([1, 2, 3, 4, 5, np.nan], n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1]),
        'Email': [f'customer{i}@example.com' for i in range(1001, 1001 + n_samples)],
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', None, ' '], n_samples),
        'TransactionID': range(10001, 10001 + n_samples),
        'ItemsPurchased': np.random.randint(1, 10, n_samples).astype(float) # For zero/negative value demo
    }
    df = pd.DataFrame(data)

    # Introduce dirtiness
    # Missing values
    df.loc[df.sample(frac=0.1).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.08).index, 'PurchaseAmount'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Email'] = None
    df.loc[df.sample(frac=0.03).index, 'ItemsPurchased'] = np.nan


    # Duplicates
    df = pd.concat([df, df.sample(n=5, random_state=1, replace=True)]) # Add 5 duplicate rows

    # Incorrect data types (simulated by having mixed types, or values that should be numeric but are strings)
    df.loc[5, 'Age'] = 'twenty five' # String in numeric column
    df.loc[10, 'PurchaseAmount'] = '$120.50' # String with currency symbol
    df.loc[20, 'ItemsPurchased'] = 'five' # String in numeric column

    # Outliers
    df.loc[25, 'PurchaseAmount'] = 15000.0 # Extreme outlier
    df.loc[30, 'Age'] = 120 # Unrealistic age
    df.loc[35, 'ItemsPurchased'] = 500 # Unrealistic items purchased

    # Inconsistent formatting (Gender, ProductCategory, City)
    df.loc[12, 'Gender'] = 'male'
    df.loc[18, 'Gender'] = 'FEMALE'
    df.loc[22, 'ProductCategory'] = 'electronics'
    df.loc[28, 'City'] = 'new york'

    # Invalid values (e.g., negative items purchased)
    df.loc[40, 'ItemsPurchased'] = -3
    df.loc[45, 'PurchaseAmount'] = -100

    # Invalid emails
    df.loc[50, 'Email'] = 'invalid_email'
    df.loc[55, 'Email'] = 'customer1055@examplecom'

    # Inconsistent date formats (for explicit date cleaning)
    df.loc[60, 'PurchaseDate'] = '2023/03/01'
    df.loc[65, 'PurchaseDate'] = 'March 15, 2023'

    # Reset index after concatenation for consistent indexing
    df.reset_index(drop=True, inplace=True)
    return df

raw_df = generate_dirty_data()

st.subheader("1. Raw, Dirty Data Overview")
st.write("This is the initial dataset, containing various imperfections:")
st.dataframe(raw_df.head(10))
st.write(f"Shape of raw data: **{raw_df.shape[0]} rows**, **{raw_df.shape[1]} columns**")

with st.expander("Explore Raw Data Diagnostics"):
    st.write("#### Data Information (`df.info()`)")
    buf = io.StringIO()
    raw_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("#### Descriptive Statistics (`df.describe(include='all')`)")
    st.dataframe(raw_df.describe(include='all'))

    st.write("#### Missing Values Before Cleaning")
    missing_data = raw_df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    missing_data['Percentage'] = (missing_data['Missing Count'] / len(raw_df)) * 100
    st.dataframe(missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False))

    fig_missing = px.bar(missing_data, x='Column', y='Missing Count',
                         title='Missing Values by Column (Before Cleaning)',
                         color='Missing Count', color_continuous_scale=px.colors.sequential.Sunset,
                         height=400)
    st.plotly_chart(fig_missing, key='raw_missing_chart') # Added unique key

    st.write("#### Initial Data Distributions (Numerical Columns)")
    for col in ['Age', 'PurchaseAmount', 'ItemsPurchased']:
        if col in raw_df.columns:
            fig = px.histogram(raw_df, x=col, title=f'Distribution of {col} (Raw)', nbins=20)
            st.plotly_chart(fig, key=f'raw_hist_{col}') # Added unique key

    st.write("#### Initial Data Distributions (Categorical Columns)")
    for col in ['Gender', 'ProductCategory', 'City']:
        if col in raw_df.columns:
            st.write(f"**Value Counts for {col} (Raw):**")
            # Corrected: Rename columns for Plotly Express
            value_counts_df = raw_df[col].value_counts(dropna=False).reset_index()
            value_counts_df.columns = [col, 'Count']
            st.dataframe(value_counts_df)
            fig = px.bar(value_counts_df, x=col, y='Count',
                         title=f'Distribution of {col} (Raw)')
            st.plotly_chart(fig, key=f'raw_cat_bar_{col}') # Added unique key


# --- 2. Step-by-Step Cleaning Process ---

st.subheader("2. Step-by-Step Data Cleaning Process")
st.markdown("We will now apply various cleaning techniques systematically.")

df_cleaned = raw_df.copy()

st.markdown("---")
st.markdown("#### **Step 2.1: Handling Duplicates**")
st.markdown("""
Duplicate rows can skew analyses and model training. The first step is to identify and remove them.
""")
initial_rows = df_cleaned.shape[0]
df_cleaned.drop_duplicates(inplace=True)
duplicate_removed_count = initial_rows - df_cleaned.shape[0]
st.write(f"- Removed **{duplicate_removed_count}** duplicate rows.")
if duplicate_removed_count > 0:
    st.success("Duplicates handled.")
    st.write("Dataframe head after removing duplicates:")
    st.dataframe(df_cleaned.head())
else:
    st.info("No duplicates found or removed in this step.")

st.markdown("---")
st.markdown("#### **Step 2.2: Correcting Data Types and Handling Non-Numeric Entries**")
st.markdown("""
Incorrect data types (e.g., numbers stored as strings, or mixed types) can prevent proper calculations
and analysis. We'll convert columns to their appropriate types, coercing errors to `NaN` for later handling.
""")

st.write("- Converting 'Age', 'PurchaseAmount', and 'ItemsPurchased' to numeric, handling non-numeric characters and coercing errors to NaN.")
# Convert 'Age'
df_cleaned['Age'] = pd.to_numeric(df_cleaned['Age'], errors='coerce')

# Clean 'PurchaseAmount' by removing non-numeric characters and converting
df_cleaned['PurchaseAmount'] = df_cleaned['PurchaseAmount'].astype(str).str.replace('[^0-9.]', '', regex=True)
df_cleaned['PurchaseAmount'] = pd.to_numeric(df_cleaned['PurchaseAmount'], errors='coerce')

# Convert 'ItemsPurchased'
df_cleaned['ItemsPurchased'] = pd.to_numeric(df_cleaned['ItemsPurchased'], errors='coerce')


st.write("- Converting 'PurchaseDate' to datetime objects for consistent date operations.")
# Convert 'PurchaseDate' to datetime, handling various formats
# Using infer_datetime_format=True helps with mixed formats
df_cleaned['PurchaseDate'] = pd.to_datetime(df_cleaned['PurchaseDate'], errors='coerce', infer_datetime_format=True)


with st.expander("Show `df.info()` after type conversion"):
    buf = io.StringIO()
    df_cleaned.info(buf=buf)
    st.code(buf.getvalue())
st.success("Data types converted where possible. Errors now represented as NaNs.")


st.markdown("---")
st.markdown("#### **Step 2.3: Standardizing Categorical Columns (Inconsistent Casing & Values)**")
st.markdown("""
Categorical features often suffer from inconsistent casing, leading to multiple representations of the same category.
We'll standardize these to a consistent format and handle 'unknown' or 'N/A' values.
""")

st.write("- Standardizing 'Gender' to 'Male' / 'Female' / 'Not Specified'.")
df_cleaned['Gender'] = df_cleaned['Gender'].astype(str).str.strip().str.capitalize().replace({'Other': 'Not Specified', 'Unknown': 'Not Specified'})
# Ensure 'male' -> 'Male', 'FEMALE' -> 'Female' are handled by capitalize() for consistency

st.write("- Standardizing 'ProductCategory' by cleaning casing and replacing 'N/A' with 'Uncategorized'.")
df_cleaned['ProductCategory'] = df_cleaned['ProductCategory'].astype(str).str.strip().str.capitalize().replace({'N/a': 'Uncategorized'})

st.write("- Standardizing 'City' by cleaning casing and replacing empty strings/spaces with 'Unknown'.")
df_cleaned['City'] = df_cleaned['City'].astype(str).str.strip().str.title().replace({'': 'Unknown', ' ': 'Unknown'}) # Title case and handle empty strings

st.write("Value counts after standardization:")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**Gender:**")
    gender_counts = df_cleaned['Gender'].value_counts(dropna=False).reset_index()
    gender_counts.columns = ['Gender', 'Count'] # Explicitly rename
    st.dataframe(gender_counts)
    fig_gender_cleaned = px.bar(gender_counts, x='Gender', y='Count', title='Distribution of Gender (Cleaned)')
    st.plotly_chart(fig_gender_cleaned, key='cleaned_gender_bar') # Added unique key
with col2:
    st.write("**ProductCategory:**")
    product_counts = df_cleaned['ProductCategory'].value_counts(dropna=False).reset_index()
    product_counts.columns = ['ProductCategory', 'Count'] # Explicitly rename
    st.dataframe(product_counts)
    fig_product_cleaned = px.bar(product_counts, x='ProductCategory', y='Count', title='Distribution of ProductCategory (Cleaned)')
    st.plotly_chart(fig_product_cleaned, key='cleaned_product_bar') # Added unique key
with col3:
    st.write("**City:**")
    city_counts = df_cleaned['City'].value_counts(dropna=False).reset_index()
    city_counts.columns = ['City', 'Count'] # Explicitly rename
    st.dataframe(city_counts)
    fig_city_cleaned = px.bar(city_counts, x='City', y='Count', title='Distribution of City (Cleaned)')
    st.plotly_chart(fig_city_cleaned, key='cleaned_city_bar') # Added unique key

st.success("Categorical data standardized.")


st.markdown("---")
st.markdown("#### **Step 2.4: Handling Missing Values**")
st.markdown("""
Missing data can significantly impact analysis and model performance. We employ various imputation strategies
based on the column type and nature of the missingness.
""")

st.write("- Current missing values count:")
current_missing = df_cleaned.isnull().sum().reset_index()
current_missing.columns = ['Column', 'Missing Count']
st.dataframe(current_missing[current_missing['Missing Count'] > 0])

st.write("- Imputing missing `Age` with the median (for numerical columns with potential outliers).")
df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)

st.write("- Imputing missing `PurchaseAmount` with the mean.")
df_cleaned['PurchaseAmount'].fillna(df_cleaned['PurchaseAmount'].mean(), inplace=True)

st.write("- Imputing missing `ItemsPurchased` with the mode (for discrete numerical data).")
df_cleaned['ItemsPurchased'].fillna(df_cleaned['ItemsPurchased'].mode()[0], inplace=True)

st.write("- Imputing missing `Rating` with the mode (for categorical/ordinal numerical data).")
df_cleaned['Rating'].fillna(df_cleaned['Rating'].mode()[0], inplace=True)

st.write("- Handling missing `Email` values (e.g., converting `None` to 'No Email').")
df_cleaned['Email'].fillna('No Email', inplace=True)

# Post-imputation check
st.write("Missing values after imputation:")
missing_after_imputation = df_cleaned.isnull().sum().reset_index(name='Missing Count').rename(columns={'index': 'Column'})
st.dataframe(missing_after_imputation[missing_after_imputation['Missing Count'] > 0])

if missing_after_imputation['Missing Count'].sum() == 0:
    st.success("All missing values handled.")
else:
    st.warning("Some missing values still exist, review if further handling is needed for specific columns.")

with st.expander("Visualize Distributions After Imputation"):
    st.write("Distributions of `Age`, `PurchaseAmount`, `ItemsPurchased` after imputation:")
    for col in ['Age', 'PurchaseAmount', 'ItemsPurchased']:
        fig = px.histogram(df_cleaned, x=col, title=f'Distribution of {col} (Post-Imputation)', nbins=20)
        st.plotly_chart(fig, key=f'impute_hist_{col}') # Added unique key


st.markdown("---")
st.markdown("#### **Step 2.5: Handling Invalid/Zero/Negative Values**")
st.markdown("""
Some numerical values might be technically present but semantically incorrect (e.g., negative purchase amounts, zero items purchased if that's invalid).
""")

st.write("- Capping `PurchaseAmount` to a minimum of 0 (no negative purchases).")
df_cleaned['PurchaseAmount'] = df_cleaned['PurchaseAmount'].apply(lambda x: max(x, 0))

st.write("- Capping `ItemsPurchased` to a minimum of 1 (no zero or negative items).")
df_cleaned['ItemsPurchased'] = df_cleaned['ItemsPurchased'].apply(lambda x: max(x, 1))

st.success("Invalid zero/negative values adjusted.")

with st.expander("Visualize Distributions After Value Correction"):
    st.write("Distributions of `PurchaseAmount`, `ItemsPurchased` after correcting invalid values:")
    for col in ['PurchaseAmount', 'ItemsPurchased']:
        fig = px.histogram(df_cleaned, x=col, title=f'Distribution of {col} (Post-Value Correction)', nbins=20)
        st.plotly_chart(fig, key=f'value_corr_hist_{col}') # Added unique key


st.markdown("---")
st.markdown("#### **Step 2.6: Outlier Detection and Treatment**")
st.markdown("""
Outliers are extreme values that can disproportionately influence statistical analyses and machine learning models.
We will visualize and treat outliers in numerical columns like `Age` and `PurchaseAmount` using the IQR method.
""")

st.write("#### Outliers in `Age` (Before Treatment)")
fig_age_before = px.box(raw_df.dropna(subset=['Age']), y="Age", title="Age Distribution (Raw Data)", points="all")
st.plotly_chart(fig_age_before, key='raw_age_box') # Added unique key

st.write("- Capping 'Age' outliers (e.g., values below 18 or above 100).")
df_cleaned['Age'] = np.where(df_cleaned['Age'] > 100, df_cleaned['Age'].median(), df_cleaned['Age']) # Cap age at 100
df_cleaned['Age'] = np.where(df_cleaned['Age'] < 18, df_cleaned['Age'].median(), df_cleaned['Age']) # Cap age at 18
st.write("Age outliers capped between 18 and 100, replaced with median.")

st.write("#### Outliers in `PurchaseAmount` (Before Treatment)")
fig_purchase_before = px.box(raw_df.dropna(subset=['PurchaseAmount']), y="PurchaseAmount", title="PurchaseAmount Distribution (Raw Data)", points="all")
st.plotly_chart(fig_purchase_before, key='raw_purchase_box') # Added unique key

st.write("- Identifying and capping 'PurchaseAmount' outliers using the IQR method.")
Q1_pa = df_cleaned['PurchaseAmount'].quantile(0.25)
Q3_pa = df_cleaned['PurchaseAmount'].quantile(0.75)
IQR_pa = Q3_pa - Q1_pa
lower_bound_pa = Q1_pa - 1.5 * IQR_pa
upper_bound_pa = Q3_pa + 1.5 * IQR_pa

# Cap outliers
df_cleaned['PurchaseAmount'] = np.where(
    df_cleaned['PurchaseAmount'] < lower_bound_pa,
    lower_bound_pa,
    df_cleaned['PurchaseAmount']
)
df_cleaned['PurchaseAmount'] = np.where(
    df_cleaned['PurchaseAmount'] > upper_bound_pa,
    upper_bound_pa,
    df_cleaned['PurchaseAmount']
)
st.write(f"Purchase Amount outliers capped to range: `[{lower_bound_pa:.2f}, {upper_bound_pa:.2f}]`")
st.success("Outliers detected and treated.")

with st.expander("Visualize Distributions After Outlier Treatment"):
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.write("#### Age Distribution (After Treatment)")
        fig_age_after = px.box(df_cleaned, y="Age", title="Age Distribution (Cleaned)", points="all")
        st.plotly_chart(fig_age_after, key='cleaned_age_box') # Added unique key
    with col_out2:
        st.write("#### PurchaseAmount Distribution (After Treatment)")
        fig_purchase_after = px.box(df_cleaned, y="PurchaseAmount", title="PurchaseAmount Distribution (Cleaned)", points="all")
        st.plotly_chart(fig_purchase_after, key='cleaned_purchase_box') # Added unique key


st.markdown("---")
st.markdown("#### **Step 2.7: Validating and Cleaning Email Formats**")
st.markdown("""
Email addresses need to be in a valid format for communication or integration purposes.
We'll apply a basic regex validation.
""")

st.write("- Basic validation of email formats.")
# Simple regex for email validation (can be more robust for production)
email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df_cleaned['Email_Valid'] = df_cleaned['Email'].astype(str).str.match(email_regex)

# Identify invalid emails
invalid_emails_count = df_cleaned[df_cleaned['Email_Valid'] == False].shape[0]
st.write(f"- Identified **{invalid_emails_count}** potentially invalid email addresses.")

st.dataframe(df_cleaned[df_cleaned['Email_Valid'] == False][['Email']].head())

# Option to remove or mark invalid emails
df_cleaned.loc[df_cleaned['Email_Valid'] == False, 'Email'] = 'Invalid/No Email Provided'
df_cleaned.drop(columns=['Email_Valid'], inplace=True) # Drop helper column
st.success("Email formats validated and invalid ones marked.")


st.markdown("---")
st.markdown("#### **Step 2.8: Feature Engineering (Creating new features)**")
st.markdown("""
Feature engineering involves creating new features from existing ones to provide more predictive power
or better insights. Here, we'll extract month and day of week from the `PurchaseDate`.
""")

st.write("- Extracting 'PurchaseMonth' and 'PurchaseDayOfWeek' from 'PurchaseDate'.")
df_cleaned['PurchaseMonth'] = df_cleaned['PurchaseDate'].dt.month
df_cleaned['PurchaseDayOfWeek'] = df_cleaned['PurchaseDate'].dt.day_name()

st.write("New features added:")
st.dataframe(df_cleaned[['PurchaseDate', 'PurchaseMonth', 'PurchaseDayOfWeek']].head())
st.success("New features engineered.")


st.subheader("3. Cleaned Data Summary")
st.write("The dataset after all cleaning steps:")
st.dataframe(df_cleaned.head())
st.write(f"Final shape of cleaned data: **{df_cleaned.shape[0]} rows**, **{df_cleaned.shape[1]} columns**")

with st.expander("Explore Cleaned Data Diagnostics"):
    st.write("#### `df_cleaned.info()`:")
    buf = io.StringIO()
    df_cleaned.info(buf=buf)
    st.code(buf.getvalue())

    st.write("#### `df_cleaned.describe(include='all')`:")
    st.dataframe(df_cleaned.describe(include='all'))

    st.write("#### Missing Values After Cleaning")
    final_missing = df_cleaned.isnull().sum().reset_index(name='Missing Count').rename(columns={'index': 'Column'})
    st.dataframe(final_missing[final_missing['Missing Count'] > 0])
    if final_missing['Missing Count'].sum() == 0:
        st.success("All critical missing values successfully handled!")
    else:
        st.info("Some `PurchaseDate` NaNs might remain if original format was unrecognizable and not critical for other steps.")

    st.write("#### Final Data Distributions (Numerical Columns)")
    for col in ['Age', 'PurchaseAmount', 'ItemsPurchased', 'Rating']:
        if col in df_cleaned.columns:
            fig = px.histogram(df_cleaned, x=col, title=f'Distribution of {col} (Cleaned)', nbins=20)
            st.plotly_chart(fig, key=f'final_hist_{col}') # Added unique key

    st.write("#### Final Data Distributions (Categorical Columns)")
    for col in ['Gender', 'ProductCategory', 'City', 'PurchaseDayOfWeek']:
        if col in df_cleaned.columns:
            st.write(f"**Value Counts for {col} (Cleaned):**")
            # Corrected: Rename columns for Plotly Express
            value_counts_df = df_cleaned[col].value_counts(dropna=False).reset_index()
            value_counts_df.columns = [col, 'Count']
            st.dataframe(value_counts_df)
            fig = px.bar(value_counts_df, x=col, y='Count',
                         title=f'Distribution of {col} (Cleaned)')
            st.plotly_chart(fig, key=f'final_cat_bar_{col}') # Added unique key

    st.write("#### Correlation Matrix of Numerical Features (Cleaned Data)")
    numerical_cols_for_corr = ['Age', 'PurchaseAmount', 'ItemsPurchased', 'Rating', 'PurchaseMonth']
    corr_matrix = df_cleaned[numerical_cols_for_corr].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    st.pyplot(plt)


st.success("Data Cleaning Process Completed! The data is now ready for further analysis and modeling.")

st.markdown("---")
st.subheader("Key Takeaways from Data Cleaning:")
st.markdown("""
* **Data Quality is Paramount:** The old adage "garbage in, garbage out" holds true. Clean data is the foundation for reliable insights and accurate models.
* **Systematic Approach:** A structured approach to data cleaning ensures that all potential issues are addressed, from duplicates and incorrect types to missing values and outliers.
* **Visual Validation:** Visualizing data at different stages of the cleaning process is crucial for verifying the effectiveness of transformations and catching subtle issues.
* **Domain Knowledge is Key:** Understanding the business context helps in making informed decisions about how to clean and impute data (e.g., what constitutes an "outlier," which values are "invalid").
* **Feature Engineering Enriches Data:** Creating new features from existing ones can unlock hidden patterns and improve model performance.
""")

st.download_button(
    label="Download Cleaned Data as CSV",
    data=df_cleaned.to_csv(index=False).encode('utf-8'),
    file_name='cleaned_portfolio_data.csv',
    mime='text/csv',
)
