import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import io

# Set page config for this specific page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Sentiment Analysis Case Study",
    page_icon="💬",
    layout="wide"
)

# Ensure NLTK resources are downloaded (only needs to run once)
# NLTK is smart enough to only download if not already present.
# 'quiet=True' prevents verbose output if already downloaded.
st.info("Ensuring NLTK VADER lexicon is available. This may take a moment on first run.")
nltk.download('vader_lexicon', quiet=True)


st.title("💬 Project: Sentiment Analysis Case Study")
st.markdown("""
This section demonstrates how to perform **Sentiment Analysis** on textual data.
Sentiment analysis is the process of computationally identifying and categorizing
opinions expressed in a piece of text, especially to determine whether the writer's
attitude towards a particular topic, product, etc., is positive, negative, or neutral.
""")

st.subheader("1. Data Generation & Overview")
st.write("We'll start by generating a synthetic dataset of customer reviews for various products.")

@st.cache_data
def generate_reviews_data(n_reviews=200):
    """Generates synthetic product review data with varying sentiments."""
    np.random.seed(42)

    positive_reviews = [
        "This product is amazing! Absolutely love it. Highly recommend.",
        "Excellent quality and fast delivery. Very satisfied with my purchase.",
        "Works perfectly as described. A must-have for everyone.",
        "Couldn't be happier! Great value for money.",
        "Fantastic experience, top-notch customer service.",
        "Best purchase ever, so glad I bought this. Five stars!",
        "Very good product, exceeds expectations.",
        "Seamless experience, highly intuitive.",
        "A game changer, truly revolutionary!",
        "Solid performance, reliable and efficient."
    ]
    negative_reviews = [
        "Very disappointing product. Broke after a week.",
        "Worst customer service ever. Do not buy!",
        "Not worth the money. Poor quality and misleading description.",
        "Received damaged goods. Extremely unhappy.",
        "Terrible experience, would not recommend to anyone.",
        "Awful, just awful. Complete waste of time and money.",
        "Doesn't work at all, very frustrating.",
        "Highly dissatisfied, felt ripped off.",
        "Poor design, not user-friendly.",
        "Regretting this purchase."
    ]
    neutral_reviews = [
        "The product arrived on time. Packaging was standard.",
        "It functions as expected. Nothing special, nothing bad.",
        "A basic item, serves its purpose. No strong feelings.",
        "Received the package. Contents were as described.",
        "It's okay. Not great, not terrible.",
        "Does the job, but could be improved.",
        "No complaints, but not impressed either.",
        "Average quality, nothing stands out.",
        "It works. That's about it.",
        "Fairly standard, meets minimum requirements."
    ]

    reviews = []
    sentiments = []
    for _ in range(n_reviews // 3):
        reviews.append(np.random.choice(positive_reviews))
        sentiments.append('Positive')
        reviews.append(np.random.choice(negative_reviews))
        sentiments.append('Negative')
        reviews.append(np.random.choice(neutral_reviews))
        sentiments.append('Neutral')

    # Add some mixed/complex reviews and some with emojis for cleaning demo
    mixed_reviews = [
        "The delivery was fast 👍 but the product itself is just okay.",
        "I like some features, but other aspects are really bad 👎. Mixed feelings.",
        "This is good 😊 for its price, but the setup was a nightmare.",
        "Customer service was terrible, but the product is solid. 😐",
        "It was expensive yet it works so well! 🎉",
        "This is not bad... actually, it's quite good.",
        "I absolutely HATE this thing! It's so bad. 😡"
    ]
    for rev in mixed_reviews:
        reviews.append(rev)
        # Assign a 'mixed' sentiment for these, to be determined by VADER
        sentiments.append('Mixed')

    # Add some reviews with URLs, special characters for cleaning demo
    dirty_reviews = [
        "Check out this link: http://example.com It's a great product!",
        "What a #mess! Seriously, this is @terrible.",
        "This is an awesome product!!! So happy. 😍😊�",
        "Absolutely amazing!!!! Best ever. Price @ $100.00"
    ]
    for rev in dirty_reviews:
        reviews.append(rev)
        sentiments.append('Dirty')


    df = pd.DataFrame({'ReviewText': reviews, 'AssignedSentiment': sentiments})
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle
    return df

reviews_df = generate_reviews_data()

st.dataframe(reviews_df.head(10))
st.write(f"Dataset shape: {reviews_df.shape[0]} reviews")

with st.expander("Explore Raw Data Examples"):
    st.write("Some examples of the raw review text:")
    for i, row in reviews_df.sample(5).iterrows():
        st.write(f"- **Review:** `{row['ReviewText']}`")
        st.write(f"  **Assigned Sentiment:** `{row['AssignedSentiment']}`")

st.markdown("---")
st.subheader("2. Text Preprocessing")
st.markdown("""
Raw text data often contains noise like punctuation, numbers, special characters, and varying cases.
Preprocessing steps clean the text, making it suitable for analysis.
""")

@st.cache_data
def preprocess_text(text):
    """
    Cleans text by:
    - Lowercasing
    - Removing URLs
    - Removing mentions (@username)
    - Removing hashtags (#tag)
    - Removing special characters and punctuation
    - Removing extra whitespaces
    """
    text = text.lower() # Lowercasing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'#\w+', '', text) # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

# Apply preprocessing and show comparison
st.write("#### Before and After Preprocessing Examples:")
sample_reviews = reviews_df.sample(5, random_state=1)
for i, row in sample_reviews.iterrows():
    original_text = row['ReviewText']
    cleaned_text = preprocess_text(original_text)
    st.code(f"Original: '{original_text}'", language='text')
    st.code(f"Cleaned : '{cleaned_text}'", language='text')
    st.markdown("---")

st.success("Text preprocessing complete!")


st.markdown("---")
st.subheader("3. Sentiment Analysis with VADER")
st.markdown("""
We'll use the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** lexicon and rule-based
sentiment analysis model. VADER is specifically attuned to sentiments expressed in social media.
It provides a `compound` score, which is a normalized, weighted composite score ranging from -1 (most extreme negative) to +1 (most extreme positive).
""")

analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def analyze_sentiment(text):
    """Applies VADER sentiment analysis to text."""
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to the cleaned reviews
reviews_df['CleanedReviewText'] = reviews_df['ReviewText'].apply(preprocess_text)
reviews_df['VADER_Sentiment'] = reviews_df['CleanedReviewText'].apply(analyze_sentiment)
reviews_df['VADER_Compound_Score'] = reviews_df['CleanedReviewText'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

st.write("#### Sample Reviews with Predicted Sentiment:")
st.dataframe(reviews_df[['ReviewText', 'CleanedReviewText', 'VADER_Sentiment', 'VADER_Compound_Score']].head(10))

st.success("Sentiment analysis performed!")

st.markdown("---")
st.subheader("4. Visualization of Sentiment Distribution")
st.markdown("""
Visualizing the distribution of sentiments helps in quickly grasping the overall
tone of the reviews.
""")

sentiment_counts = reviews_df['VADER_Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

fig_sentiment_dist = px.pie(sentiment_counts, names='Sentiment', values='Count',
                             title='Distribution of VADER Sentiments',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             hole=0.3)
st.plotly_chart(fig_sentiment_dist, key='sentiment_distribution_chart')

st.markdown("---")
st.subheader("5. Interactive Sentiment Analyzer")
st.markdown("Try analyzing the sentiment of your own text below!")

user_text = st.text_area("Enter your text here:", "This is a wonderful experience! I am so happy with Streamlit.")

if st.button("Analyze Sentiment"):
    if user_text:
        processed_user_text = preprocess_text(user_text)
        sentiment_result = analyze_sentiment(processed_user_text)
        compound_score_user = analyzer.polarity_scores(processed_user_text)['compound']

        st.write("---")
        st.subheader("Analysis Result:")
        st.write(f"**Processed Text:** `{processed_user_text}`")
        st.write(f"**Predicted Sentiment:** **{sentiment_result}**")
        st.write(f"**VADER Compound Score:** `{compound_score_user:.4f}`")

        if sentiment_result == 'Positive':
            st.success("😊 This text expresses a **Positive** sentiment.")
        elif sentiment_result == 'Negative':
            st.error("😞 This text expresses a **Negative** sentiment.")
        else:
            st.info("😐 This text expresses a **Neutral** sentiment.")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.subheader("Key Takeaways from Sentiment Analysis:")
st.markdown("""
* **Text Preprocessing is Vital:** Cleaning text data (removing noise, standardizing) is a crucial first step for accurate NLP.
* **VADER is Quick & Effective:** For social media-like text, VADER provides a fast and robust sentiment analysis solution without complex model training.
* **Compound Score:** Understanding the compound score helps in gauging the intensity and overall polarity of the sentiment.
* **Actionable Insights:** Sentiment analysis can be used to monitor brand reputation, analyze customer feedback, or understand public opinion on various topics.
""")
