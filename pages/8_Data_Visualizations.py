import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Data Visualization Case Study",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Project: Data Visualization Case Study")
st.markdown("""
This section demonstrates the power of **Data Visualization** in transforming raw data
into actionable insights. Effective visualizations are crucial for exploratory data
analysis (EDA), communicating findings, and understanding complex patterns.

We'll use a synthetic sales dataset to illustrate various chart types and their
applications in uncovering trends, distributions, relationships, and compositions.
""")

st.subheader("1. Data Generation & Overview (Simulated Sales Data)")
st.write("We'll generate a synthetic dataset representing sales transactions across different regions, products, and time periods.")

@st.cache_data
def generate_sales_data(n_records=2000):
    """Generates synthetic sales data."""
    np.random.seed(42)

    product_categories = ['Electronics', 'Clothing', 'Books', 'Home Goods', 'Food & Beverage']
    regions = ['North', 'South', 'East', 'West', 'Central']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Online Transfer']
    customer_segments = ['New Customer', 'Returning Customer', 'Loyal Customer']

    data = {
        'OrderID': range(1, n_records + 1),
        'OrderDate': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_records, freq='H').strftime('%Y-%m-%d %H:%M:%S')),
        'ProductCategory': np.random.choice(product_categories, n_records, p=[0.25, 0.2, 0.15, 0.2, 0.2]),
        'Region': np.random.choice(regions, n_records, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'Sales': np.random.normal(100, 40, n_records),
        'Profit': np.random.normal(20, 10, n_records),
        'UnitsSold': np.random.randint(1, 10, n_records),
        'CustomerAge': np.random.randint(18, 70, n_records),
        'PaymentMethod': np.random.choice(payment_methods, n_records, p=[0.4, 0.3, 0.15, 0.15]),
        'CustomerSegment': np.random.choice(customer_segments, n_records, p=[0.2, 0.5, 0.3])
    }
    df = pd.DataFrame(data)

    # Introduce some variations
    df.loc[df['ProductCategory'] == 'Electronics', 'Sales'] = np.random.normal(150, 50, len(df[df['ProductCategory'] == 'Electronics']))
    df.loc[df['ProductCategory'] == 'Electronics', 'Profit'] = np.random.normal(35, 15, len(df[df['ProductCategory'] == 'Electronics']))
    df.loc[df['Region'] == 'East', 'Sales'] = np.random.normal(120, 30, len(df[df['Region'] == 'East']))
    df.loc[df['CustomerSegment'] == 'Loyal Customer', 'Sales'] = np.random.normal(130, 45, len(df[df['CustomerSegment'] == 'Loyal Customer']))

    df['Sales'] = np.maximum(df['Sales'], 10).round(2) # Ensure positive sales
    df['Profit'] = np.maximum(df['Profit'], 1).round(2) # Ensure positive profit

    # Add a 'YearMonth' column for time series
    df['YearMonth'] = df['OrderDate'].dt.to_period('M').astype(str)

    return df

sales_df = generate_sales_data(n_records=2000)

st.dataframe(sales_df.head())
st.write(f"Dataset shape: {sales_df.shape[0]} records, {sales_df.shape[1]} columns")

with st.expander("Explore Data Statistics"):
    st.write("#### Data Information (`df.info()`)")
    buf = io.StringIO()
    sales_df.info(buf=buf)
    st.code(buf.getvalue())

    st.write("#### Descriptive Statistics (`df.describe(include='all')`)")
    st.dataframe(sales_df.describe(include='all'))

st.markdown("---")
st.subheader("2. Visualizing Distributions")
st.markdown("Understanding the distribution of numerical features is a crucial first step in EDA.")

st.write("#### Histogram: Distribution of Sales Amounts")
fig_hist_sales = px.histogram(sales_df, x='Sales', nbins=50, title='Distribution of Sales Amount',
                              template='plotly_white', marginal='box', opacity=0.7,
                              color_discrete_sequence=[px.colors.sequential.Viridis[3]])
st.plotly_chart(fig_hist_sales, use_container_width=True, key='hist_sales')
st.info("A histogram shows the frequency distribution of a numerical variable. The marginal box plot helps visualize central tendency, spread, and outliers.")

st.write("#### Box Plot: Sales Distribution by Product Category")
fig_box_sales_category = px.box(sales_df, x='ProductCategory', y='Sales',
                                 title='Sales Distribution by Product Category',
                                 color='ProductCategory', points='all', template='plotly_white')
st.plotly_chart(fig_box_sales_category, use_container_width=True, key='box_sales_category')
st.info("Box plots are excellent for comparing distributions across different categories, highlighting medians, quartiles, and potential outliers.")

st.markdown("---")
st.subheader("3. Visualizing Relationships")
st.markdown("Scatter plots are ideal for examining relationships between two numerical variables.")

st.write("#### Scatter Plot: Sales vs. Profit")
fig_scatter_sales_profit = px.scatter(sales_df, x='Sales', y='Profit',
                                      title='Sales vs. Profit Relationship',
                                      hover_data=['ProductCategory', 'Region'],
                                      color='ProductCategory', template='plotly_white', opacity=0.7)
st.plotly_chart(fig_scatter_sales_profit, use_container_width=True, key='scatter_sales_profit')
st.info("A scatter plot helps identify correlations and patterns between two continuous variables. Coloring by a third categorical variable can reveal hidden clusters or trends.")

st.write("#### Scatter Plot: Monthly Hours vs. Salary (from Employee Churn - illustrative)")
# Re-using a concept from Employee Churn for demonstration of relationships
# (If you don't have employee_df loaded, you can generate a small one here or remove this)
if 'MonthlyHours' in sales_df.columns and 'Salary' in sales_df.columns: # Checking if these columns exist in this DF
    fig_scatter_hr_salary = px.scatter(sales_df, x='MonthlyHours', y='Salary',
                                       title='Monthly Hours vs. Salary (Illustrative)',
                                       hover_name='CustomerAge', # Just for demo purpose
                                       color='CustomerSegment', template='plotly_white', opacity=0.6)
    st.plotly_chart(fig_scatter_hr_salary, use_container_width=True, key='scatter_hr_salary')
    st.info("This plot demonstrates how scatter plots can reveal relationships that might not be immediately obvious, like potential compensation fairness issues.")
else:
    st.info("To show Monthly Hours vs. Salary, ensure these columns are part of the generated data or consider loading a relevant dataset here.")


st.markdown("---")
st.subheader("4. Visualizing Trends Over Time")
st.markdown("Line charts are perfect for displaying how a variable changes over a continuous period.")

st.write("#### Line Chart: Total Sales Over Time (Monthly)")
# Aggregate sales by Month-Year
monthly_sales = sales_df.groupby('YearMonth')['Sales'].sum().reset_index()
# Ensure correct sorting for time series
monthly_sales['OrderDate_Sort'] = pd.to_datetime(monthly_sales['YearMonth'])
monthly_sales = monthly_sales.sort_values('OrderDate_Sort')

fig_line_sales_time = px.line(monthly_sales, x='YearMonth', y='Sales',
                              title='Total Sales Trend Over Time (Monthly)',
                              markers=True, template='plotly_white',
                              labels={'Sales': 'Total Sales', 'YearMonth': 'Month-Year'})
st.plotly_chart(fig_line_sales_time, use_container_width=True, key='line_sales_time')
st.info("Line charts effectively show trends, seasonality, and growth over time.")


st.markdown("---")
st.subheader("5. Visualizing Compositions and Proportions")
st.markdown("Pie charts and bar charts can be used to show parts of a whole or compare categorical frequencies.")

st.write("#### Pie Chart: Proportion of Sales by Product Category")
sales_by_category = sales_df.groupby('ProductCategory')['Sales'].sum().reset_index()
fig_pie_category_sales = px.pie(sales_by_category, names='ProductCategory', values='Sales',
                                title='Proportion of Sales by Product Category',
                                hole=0.3, template='plotly_white',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_pie_category_sales, use_container_width=True, key='pie_category_sales')
st.info("Pie charts illustrate the composition of a whole. The 'hole' makes it a donut chart, often preferred for aesthetics.")


st.write("#### Bar Chart: Total Sales by Region")
sales_by_region = sales_df.groupby('Region')['Sales'].sum().reset_index()
fig_bar_region_sales = px.bar(sales_by_region, x='Region', y='Sales',
                              title='Total Sales by Region',
                              color='Sales', template='plotly_white',
                              color_continuous_scale=px.colors.sequential.Plasma)
st.plotly_chart(fig_bar_region_sales, use_container_width=True, key='bar_region_sales')
st.info("Bar charts are excellent for comparing discrete categories. Coloring by the value itself can add emphasis.")

st.markdown("---")
st.subheader("6. Visualizing Correlations")
st.markdown("Heatmaps are useful for displaying the correlation matrix between multiple numerical variables.")

st.write("#### Heatmap: Correlation Matrix of Numerical Features")
numerical_cols_for_corr = ['Sales', 'Profit', 'UnitsSold', 'CustomerAge']
corr_matrix = sales_df[numerical_cols_for_corr].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
st.pyplot(plt) # Removed the 'key' argument here
st.info("A heatmap visualizes the correlation coefficients between pairs of variables. Red indicates negative correlation, blue indicates positive, and white indicates no correlation.")


st.markdown("---")
st.subheader("Key Takeaways from Data Visualization:")
st.markdown("""
* **Exploratory Data Analysis (EDA):** Visualizations are fundamental for initially exploring datasets, identifying patterns, outliers, and potential relationships.
* **Effective Communication:** Well-designed charts simplify complex data, making insights accessible to both technical and non-technical audiences.
* **Choosing the Right Chart:** The choice of visualization depends on the type of data and the message you want to convey (e.g., trend, comparison, distribution, relationship).
* **Interactivity (Plotly):** Tools like Plotly enable interactive charts, allowing users to zoom, pan, and hover for more detailed information, enhancing data exploration.
* **Iterative Process:** Visualization is an iterative process, where initial plots lead to new questions, prompting further analysis and more refined visualizations.
""")
