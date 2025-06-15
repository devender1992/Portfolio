# Home.py
import streamlit as st

st.set_page_config(
    page_title="Senior Data Analyst Portfolio",
    page_icon="📊",
    layout="wide"
)

st.title("👨‍💻 Senior Data Analyst Portfolio")
st.markdown("---")

st.markdown("""
Welcome to my data analytics portfolio! Here, you'll find a collection of projects
showcasing my expertise in various aspects of data analysis, machine learning,
and data-driven problem-solving. Each section demonstrates a different skill set
and approach to tackling real-world data challenges.

**Navigate through the sidebar to explore different projects:**

* **Data Cleaning:** See how I transform raw, messy data into clean, usable formats.
* **ML Case Study (Customer Churn):** Explore a real-life machine learning application.
* **Sentiment Analysis:** Understand how to extract insights from textual data.
* **Employee Churn Prediction:** Predict and understand factors contributing to employee turnover.
* **Fraud Detection:** Build models to identify anomalous and potentially fraudulent activities.
* **Recommender System:** Develop systems that suggest relevant items to users.
* **Hypothesis Testing:** Conduct statistical tests to validate assumptions and draw conclusions.
* **Data Visualizations:** Various Data visualizations and showing importance of visualizations.
* **Time Series Analysis:** Explore Time series analysis.

---

### About Me

As a Senior Data Analyst, I specialize in transforming complex data into actionable insights.
My passion lies in leveraging statistical methods, machine learning, and robust data
engineering practices to drive strategic decisions and solve critical business problems.
I'm proficient in Python (Pandas, NumPy, Scikit-learn, NLTK, SpaCy), SQL, and data
visualization tools.

Feel free to connect with me via [LinkedIn](https://linkedin.com/in/devenderchaursia7303297848) or [GitHub](https://github.com/devender1992).
""")

st.sidebar.success("Select a page above to begin exploring.")

# Optional: Add an image or logo
# st.image("your_profile_picture.png", width=200)
 
