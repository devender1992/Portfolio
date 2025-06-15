import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import io
from spacy.cli import download

download("en_core_web_sm")

# Set page config for this specific page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="NLP Applications Case Study",
    page_icon="✍️",
    layout="wide"
)

# --- NLTK Downloads (ensure these run before any NLTK usage) ---
@st.cache_resource # Use st.cache_resource for heavy objects like NLP models/downloaded data
def download_nltk_data():
    st.info("Downloading NLTK data (stopwords, WordNet, punkt, punkt_tab). This will only happen once.")
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True) # Open Multilingual Wordnet for WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # Added to explicitly download punkt_tab

download_nltk_data()

# --- SpaCy Model Loading ---
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        st.stop() # Stop the app if model is not found
    return nlp

nlp = load_spacy_model()


st.title("✍️ Project: NLP Applications Case Study")
st.markdown("""
This section delves into **Natural Language Processing (NLP)**, demonstrating techniques
to extract meaningful insights from unstructured text data. NLP is vital for understanding
customer feedback, market trends, and automating text-based tasks.

We will explore **Named Entity Recognition (NER)** to identify key entities and
**Topic Modeling** to discover underlying themes within a collection of documents.
""")

st.subheader("1. Data Generation & Overview (Synthetic News Articles)")
st.write("We'll generate a synthetic dataset of short news-like articles covering various domains.")

@st.cache_data
def generate_articles_data(n_articles=100):
    """Generates synthetic article data."""
    np.random.seed(42)

    categories = ['Technology', 'Finance', 'Sports', 'Politics', 'Health']
    article_templates = {
        'Technology': [
            "Tech giant [ORG_NAME] unveiled its new [PRODUCT_NAME] today. [PERSON_NAME] praised its innovative features.",
            "The future of AI discussed at [EVENT_NAME] in [LOCATION]. [PERSON_NAME] gave a keynote speech.",
            "[PRODUCT_NAME] stock soared after [ORG_NAME]'s earnings report. Analysts predict rapid growth.",
            "New cybersecurity threats emerged, impacting users globally. [ORG_NAME] issued a patch.",
            "[PERSON_NAME] from [ORG_NAME] announced a breakthrough in quantum computing.",
            "The latest smartphone, [PRODUCT_NAME], features an incredible new camera. It's built by [ORG_NAME].",
            "Software update released by [ORG_NAME] for popular [PRODUCT_NAME] application. Users in [LOCATION] reported issues.",
        ],
        'Finance': [
            "Stock markets reacted to [COUNTRY]'s inflation data. [ORG_NAME] reported record profits.",
            "[PERSON_NAME], CEO of [ORG_NAME], discussed Q3 earnings during a conference call.",
            "Global economy faces challenges as interest rates rise. [CURRENCY_NAME] gained against the dollar.",
            "Investment firm [ORG_NAME] acquired [ANOTHER_ORG_NAME] for [AMOUNT_VALUE] billion dollars.",
            "Cryptocurrency market experienced volatility. [PERSON_NAME] warned about speculative trading.",
            "Central Bank of [COUNTRY] announced new monetary policies to curb inflation.",
        ],
        'Sports': [
            "[ATHLETE_NAME] scored a hat-trick in the [SPORT_NAME] match against [TEAM_NAME].",
            "The [SPORT_NAME] championship finals will be held in [CITY_NAME], [COUNTRY_NAME].",
            "[TEAM_NAME] signed a new contract with [ATHLETE_NAME]. Fans are excited.",
            "[ATHLETE_NAME] won the [SPORT_NAME] tournament, defeating [OPPONENT_NAME].",
            "Injuries plague [TEAM_NAME] ahead of crucial playoffs. [PERSON_NAME], their coach, is concerned.",
        ],
        'Politics': [
            "[POLITICIAN_NAME] delivered a speech on national security in [LOCATION].",
            "The new bill proposed by [POLITICIAN_NAME] faces opposition from [ORG_NAME].",
            "Elections in [COUNTRY_NAME] are approaching. [POLITICIAN_NAME] leads in recent polls.",
            "International summit held in [CITY_NAME] to discuss climate change.",
            "Government of [COUNTRY_NAME] announced new economic reforms.",
        ],
        'Health': [
            "New study reveals benefits of [FOOD_TYPE] for heart health. Published by [ORG_NAME].",
            "[PERSON_NAME], a doctor, advised on preventing [DISEASE_NAME] outbreaks.",
            "Vaccine development progresses against [DISEASE_NAME]. Clinical trials by [ORG_NAME].",
            "Mental health awareness campaign launched in [COUNTRY]. [PERSON_NAME] supports the initiative.",
            "Researchers at [ORG_NAME] discovered a new treatment for [DISEASE_NAME].",
        ]
    }

    # Placeholder values for entities
    entities = {
        'ORG_NAME': ['Google', 'Apple', 'Microsoft', 'Amazon', 'Meta', 'Tesla', 'IBM', 'JP Morgan', 'Goldman Sachs', 'Fidelity', 'Reuters', 'WHO', 'UN', 'FIFA', 'IOC', 'NASA'],
        'PRODUCT_NAME': ['VisionPro', 'PixelFold', 'GPT-5', 'QuantumX', 'CyberGuard', 'EcoDrive', 'SwiftCharge'],
        'PERSON_NAME': ['Satya Nadella', 'Sundar Pichai', 'Elon Musk', 'Tim Cook', 'Warren Buffett', 'Christine Lagarde', 'Lionel Messi', 'Novak Djokovic', 'Serena Williams', 'Joe Biden', 'Emmanuel Macron', 'Dr. Fauci', 'Dr. Smith'],
        'LOCATION': ['California', 'New York', 'London', 'Tokyo', 'Paris', 'Washington DC', 'Geneva'],
        'EVENT_NAME': ['Tech Summit', 'AI World Expo', 'Climate Conference', 'Financial Forum'],
        'COUNTRY': ['USA', 'Germany', 'France', 'Japan', 'India', 'China', 'Brazil'],
        'CURRENCY_NAME': ['Euro', 'Yen', 'Pound', 'Rupee', 'Yuan'],
        'AMOUNT_VALUE': ['20', '50', '100', '500'], # In billions for finance
        'ATHLETE_NAME': ['LeBron James', 'Megan Rapinoe', 'Roger Federer', 'Cristiano Ronaldo'],
        'SPORT_NAME': ['Basketball', 'Soccer', 'Tennis', 'Cricket'],
        'TEAM_NAME': ['Lakers', 'Real Madrid', 'Manchester United', 'Yankees'],
        'CITY_NAME': ['Doha', 'Paris', 'Los Angeles', 'London', 'Berlin'],
        'POLITICIAN_NAME': ['Angela Merkel', 'Justin Trudeau', 'Jacinda Ardern', 'Rishi Sunak'],
        'FOOD_TYPE': ['blueberries', 'avocado', 'green tea', 'salmon'],
        'DISEASE_NAME': ['flu', 'diabetes', 'cancer', 'COVID-19', 'malaria']
    }

    articles_data = []
    for _ in range(n_articles):
        category = np.random.choice(categories)
        template = np.random.choice(article_templates[category])
        
        # Replace placeholders with random entities
        for placeholder, vals in entities.items():
            if f'[{placeholder}]' in template:
                template = template.replace(f'[{placeholder}]', np.random.choice(vals))
        
        articles_data.append({'Category': category, 'ArticleText': template})

    df = pd.DataFrame(articles_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
    return df

articles_df = generate_articles_data(n_articles=150)

st.dataframe(articles_df.head(10))
st.write(f"Dataset shape: {articles_df.shape[0]} articles")

with st.expander("Explore Raw Data Examples"):
    st.write("Some examples of the raw article text:")
    for i, row in articles_df.sample(5).iterrows():
        st.write(f"- **Category:** `{row['Category']}`")
        st.write(f"  **Article:** `{row['ArticleText']}`")
        st.markdown("---")

st.markdown("---")
st.subheader("2. Text Preprocessing for NLP Tasks")
st.markdown("""
Text preprocessing is essential for NLP to convert raw text into a clean, normalized
format suitable for analysis. This typically involves:
* **Lowercasing:** Standardizes text by converting all characters to lowercase.
* **Punctuation Removal:** Eliminates special characters that don't add semantic value.
* **Tokenization:** Breaking text into words or phrases (tokens).
* **Stop Word Removal:** Eliminating common words (e.g., 'the', 'is', 'a') that carry little meaning.
* **Lemmatization:** Reducing words to their base or dictionary form (e.g., 'running' -> 'run').
""")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@st.cache_data
def preprocess_for_nlp(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2] # Remove stopwords and short words
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmas)

# Apply preprocessing and show comparison
st.write("#### Before and After Preprocessing Examples:")
sample_articles = articles_df.sample(3, random_state=1)
for i, row in sample_articles.iterrows():
    original_text = row['ArticleText']
    cleaned_text = preprocess_for_nlp(original_text)
    st.code(f"Original: '{original_text}'", language='text')
    st.code(f"Cleaned : '{cleaned_text}'", language='text') # Moved this line to be inside the loop
    st.markdown("---")

articles_df['CleanedArticleText'] = articles_df['ArticleText'].apply(preprocess_for_nlp)
st.success("Text preprocessing complete! Cleaned text will be used for downstream NLP tasks.")


st.markdown("---")
st.subheader("3. Named Entity Recognition (NER)")
st.markdown("""
**Named Entity Recognition (NER)** is an NLP technique that identifies and classifies named
entities in text into pre-defined categories such as person names, organizations, locations, dates, etc.
It's crucial for information extraction and structuring unstructured data.
""")

@st.cache_data
def extract_entities(text, _nlp_model): # Changed nlp_model to _nlp_model
    doc = _nlp_model(str(text)) # Explicitly cast text to str
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Extract entities from a sample
st.write("#### Sample Article with Extracted Entities:")
sample_ner_article = articles_df.iloc[0]['CleanedArticleText']
st.write(f"**Cleaned Article:** `{sample_ner_article}`")
entities = extract_entities(articles_df.iloc[0]['ArticleText'], nlp) # Passed nlp (the model) here
st.write("**Extracted Entities:**")
if entities:
    for entity, label in entities:
        st.write(f"- `{entity}` **({label})**")
else:
    st.info("No entities found in this sample.")

# Aggregate entity types across the dataset
all_entities = []
for text in articles_df['ArticleText']: # Use original text for NER
    all_entities.extend(extract_entities(text, nlp)) # Passed nlp here

entity_labels = [label for _, label in all_entities]
entity_counts = Counter(entity_labels)

if entity_counts:
    entity_counts_df = pd.DataFrame(entity_counts.items(), columns=['Entity Type', 'Count']).sort_values(by='Count', ascending=False)
    st.write("#### Distribution of Entity Types Across Dataset:")
    st.dataframe(entity_counts_df)
    fig_entity_dist = px.bar(entity_counts_df, x='Entity Type', y='Count',
                             title='Distribution of Named Entity Types',
                             color='Count', color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig_entity_dist, key='entity_type_distribution')
else:
    st.info("No entities were extracted from the dataset.")

st.success("Named Entity Recognition completed.")


st.markdown("---")
st.subheader("4. Topic Modeling (Latent Dirichlet Allocation - LDA)")
st.markdown("""
**Topic Modeling** is an unsupervised machine learning technique that discovers the abstract
"topics" that occur in a collection of documents. LDA (Latent Dirichlet Allocation) is a
probabilistic model that assumes documents are a mixture of topics and that each topic is
a mixture of words. It helps in understanding the main themes present in a large corpus.
""")

# Create TF-IDF vectorizer
@st.cache_data
def create_tfidf_model(corpus):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

tfidf_matrix, tfidf_vectorizer = create_tfidf_model(articles_df['CleanedArticleText'].dropna())
st.write(f"- TF-IDF Matrix created with {tfidf_matrix.shape[1]} features.")

# Train LDA model
n_topics = st.slider("Select number of topics for LDA:", 2, 8, 3) # Allow user to choose topics
@st.cache_data
def train_lda_model(_tfidf_data, num_topics): # Added underscore to tfidf_data
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42,
                                    n_jobs=-1, # Use all available cores
                                    learning_method='online',
                                    learning_decay=0.7) # Recommended for online learning
    lda.fit(_tfidf_data) # Used _tfidf_data here
    return lda

lda_model = train_lda_model(tfidf_matrix, n_topics)

st.write("- LDA model trained.")

@st.cache_data
def display_topics(_model, feature_names, no_top_words): # Changed 'model' to '_model'
    topics = {}
    for topic_idx, topic in enumerate(_model.components_): # Used '_model' here
        topics[f"Topic {topic_idx + 1}"] = " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics

no_top_words = 10
st.write(f"#### Top {no_top_words} Words Per Topic:")
topic_words = display_topics(lda_model, tfidf_vectorizer.get_feature_names_out(), no_top_words)
for topic, words in topic_words.items():
    st.write(f"- **{topic}:** {words}")
    # Generate and display Word Cloud for each topic
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {topic}')
    st.pyplot(plt) # Removed the 'key' argument here
    plt.close() # Close plot to prevent memory issues

st.success("Topic modeling completed.")


st.markdown("---")
st.subheader("5. Interactive NLP Analyzer")
st.markdown("Enter your own text to see its entities extracted!")

user_text_nlp = st.text_area("Enter text for NLP analysis:", "Elon Musk, CEO of Tesla, spoke at a tech conference in San Francisco about AI's future.")

if st.button("Analyze Text (NLP)"):
    if user_text_nlp:
        st.write("---")
        st.subheader("Extracted Entities (from SpaCy):")
        user_entities = extract_entities(user_text_nlp, nlp)
        if user_entities:
            for entity, label in user_entities:
                st.write(f"- `{entity}` **({label})**")
        else:
            st.info("No named entities found in your text.")
    else:
        st.warning("Please enter some text for analysis.")

st.markdown("---")
st.subheader("Key Takeaways from NLP Case Study:")
st.markdown("""
* **Text as Data:** NLP transforms unstructured text into quantifiable data, unlocking vast analytical potential.
* **Preprocessing is Foundational:** Proper cleaning (lowercasing, tokenization, stop word removal, lemmatization) is crucial for accurate NLP results.
* **Named Entity Recognition (NER):** Automates the identification of key information (people, organizations, locations) from text, valuable for information extraction and knowledge graph construction.
* **Topic Modeling:** Helps in understanding the overarching themes in large text corpora, useful for content categorization, trend analysis, and market research.
* **Applications:** NLP is used in chatbots, sentiment analysis, spam detection, translation, text summarization, and more.
""")
