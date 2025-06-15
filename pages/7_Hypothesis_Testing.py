import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Hypothesis Testing Case Study",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Project: Hypothesis Testing Case Study - A/B Test Analysis")
st.markdown("""
This section demonstrates **Hypothesis Testing**, a fundamental statistical method used in data analysis
to make inferences about population parameters based on sample data. It's crucial for validating
assumptions, comparing groups, and making data-driven decisions (e.g., in A/B testing).

Here, we'll simulate an A/B test scenario to determine if there's a statistically significant
difference between two groups (e.g., two versions of a website, two marketing campaigns).
""")

st.subheader("1. Data Generation & Overview (Simulated A/B Test)")
st.write("We'll generate synthetic data for two groups (Control and Treatment) to simulate an experiment.")

# User inputs for data generation
st.sidebar.header("Data Simulation Parameters")
sample_size_a = st.sidebar.slider("Sample Size Group A (Control)", 50, 1000, 500, 50)
mean_a = st.sidebar.slider("Mean Group A (e.g., Conversion Rate)", 0.0, 1.0, 0.50, 0.01)
std_dev_a = st.sidebar.slider("Std Dev Group A", 0.01, 0.2, 0.05, 0.01)

sample_size_b = st.sidebar.slider("Sample Size Group B (Treatment)", 50, 1000, 500, 50)
mean_b = st.sidebar.slider("Mean Group B (e.g., Conversion Rate)", 0.0, 1.0, 0.52, 0.01)
std_dev_b = st.sidebar.slider("Std Dev Group B", 0.01, 0.2, 0.05, 0.01)

@st.cache_data
def generate_ab_test_data(size_a, mean_a, std_a, size_b, mean_b, std_b):
    """Generates synthetic data for two groups."""
    np.random.seed(42)

    # For conversion rates, typically binomial distribution or clip normal distribution
    # We'll use normal for simplicity and then clip for conversion-like data
    data_a = np.random.normal(mean_a, std_a, size_a)
    data_a = np.clip(data_a, 0, 1) # Ensure values are between 0 and 1 for rates
    df_a = pd.DataFrame({'Value': data_a, 'Group': 'Control'})

    data_b = np.random.normal(mean_b, std_b, size_b)
    data_b = np.clip(data_b, 0, 1)
    df_b = pd.DataFrame({'Value': data_b, 'Group': 'Treatment'})

    combined_df = pd.concat([df_a, df_b], ignore_index=True)
    return combined_df.round(3)

ab_test_df = generate_ab_test_data(sample_size_a, mean_a, std_dev_a, sample_size_b, mean_b, std_dev_b)

st.write("#### Sample Data Overview:")
st.dataframe(ab_test_df.head())
st.write(f"Total data points: {ab_test_df.shape[0]}")

with st.expander("Explore Data Statistics & Distributions"):
    st.write("#### Descriptive Statistics by Group:")
    st.dataframe(ab_test_df.groupby('Group')['Value'].agg(['mean', 'median', 'std', 'min', 'max']))

    st.write("#### Distribution of Values by Group:")
    fig_dist = px.histogram(ab_test_df, x='Value', color='Group', barmode='overlay',
                            title='Distribution of Values by Group',
                            histnorm='probability density', marginal='box',
                            color_discrete_map={'Control': 'skyblue', 'Treatment': 'lightcoral'})
    st.plotly_chart(fig_dist, key='ab_test_distribution')

st.markdown("---")
st.subheader("2. Hypothesis Formulation (Independent Samples T-Test)")
st.markdown("""
We want to test if there is a statistically significant difference between the means of Group A and Group B.
""")

st.markdown("""
* **Null Hypothesis ($H_0$):** There is no significant difference between the means of Group A and Group B.
    $H_0: \\mu_A = \\mu_B$
* **Alternative Hypothesis ($H_1$):** There is a significant difference between the means of Group A and Group B.
    $H_1: \\mu_A \\neq \\mu_B$
""")

st.write("#### Significance Level ($\\alpha$)")
alpha = st.slider("Select Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)
st.write(f"The chosen significance level ($\alpha$) is: **{alpha}**")
st.info(f"""
The significance level (alpha) is the probability of rejecting the null hypothesis when it is actually true (Type I error).
A common choice is {alpha*100}%. If the p-value is less than {alpha}, we reject the null hypothesis.
""")

st.markdown("---")
st.subheader("3. Performing the Hypothesis Test (Independent Samples T-Test)")
st.markdown("""
We will use an independent samples t-test (specifically Welch's t-test, which does not assume equal variances)
to compare the means of the two groups.
""")

# Extract data for each group
group_a_data = ab_test_df[ab_test_df['Group'] == 'Control']['Value']
group_b_data = ab_test_df[ab_test_df['Group'] == 'Treatment']['Value']

# Perform Welch's t-test (equal_var=False) - CORRECTED LINE
t_statistic, p_value = stats.ttest_ind(group_a_data, group_b_data, equal_var=False)

st.write(f"- **T-statistic:** `{t_statistic:.4f}`")
st.write(f"- **P-value:** `{p_value:.4f}`")

st.success("Hypothesis test performed!")

st.markdown("---")
st.subheader("4. Conclusion")
st.markdown("Based on the p-value and the chosen significance level, we can draw a conclusion about the hypothesis.")

if p_value < alpha:
    st.markdown(f"""
    <div style="background-color:#d4edda; color:#155724; padding:10px; border-radius:5px;">
        **Conclusion:** Since the p-value ({p_value:.4f}) is less than the significance level ($\alpha = {alpha}$),
        we **reject the Null Hypothesis ($H_0$)**.
        <br><br>
        This indicates that there is a **statistically significant difference** between the means of Group A (Control)
        and Group B (Treatment).
    </div>
    """, unsafe_allow_html=True)
    if group_b_data.mean() > group_a_data.mean():
        st.success(f"Group B's mean ({group_b_data.mean():.3f}) is significantly higher than Group A's mean ({group_a_data.mean():.3f}).")
    else:
        st.warning(f"Group A's mean ({group_a_data.mean():.3f}) is significantly higher than Group B's mean ({group_b_data.mean():.3f}).")
else:
    st.markdown(f"""
    <div style="background-color:#ffeeba; color:#856404; padding:10px; border-radius:5px;">
        **Conclusion:** Since the p-value ({p_value:.4f}) is greater than or equal to the significance level ($\alpha = {alpha}$),
        we **fail to reject the Null Hypothesis ($H_0$)**.
        <br><br>
        This indicates that there is **no statistically significant difference** between the means of Group A (Control)
        and Group B (Treatment) at the {alpha*100}% significance level.
    </div>
    """, unsafe_allow_html=True)
    st.info(f"Observed means: Group A = {group_a_data.mean():.3f}, Group B = {group_b_data.mean():.3f}. The difference is likely due to random chance.")

st.markdown("---")
st.subheader("Key Takeaways from Hypothesis Testing:")
st.markdown("""
* **Foundation of Inference:** Hypothesis testing allows us to draw conclusions about populations from samples, guiding decisions where full population data is unavailable.
* **Null & Alternative Hypotheses:** Clearly defining these helps structure the statistical question.
* **P-value:** The p-value quantifies the evidence against the null hypothesis. A small p-value (typically < $\alpha$) suggests the observed effect is unlikely due to chance.
* **Significance Level ($\alpha$):** This pre-defined threshold determines how much evidence is needed to reject the null hypothesis.
* **Types of Tests:** The choice of statistical test (e.g., t-test, ANOVA, chi-square) depends on the data type, number of groups, and research question.
* **A/B Testing:** Hypothesis testing is the backbone of A/B testing, enabling businesses to confidently implement changes that truly drive improvements (e.g., higher conversion rates, better engagement).
""")
