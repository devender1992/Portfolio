import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io

# Set page config for this specific page
st.set_page_config(
    page_title="Recommender System Case Study",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Project: Recommender System Case Study")
st.markdown("""
This section demonstrates a basic **Recommender System**. Recommender systems are powerful tools
used to predict the 'rating' or 'preference' a user would give to an item. They are widely
used in e-commerce, streaming services, and content platforms to suggest relevant products,
movies, or articles to users, enhancing user experience and driving engagement.

Here, we implement a simplified **item-based collaborative filtering** system.
""")

st.subheader("1. Data Generation & Overview")
st.write("We'll generate a synthetic dataset representing users, items (movies), and their ratings (1-5 stars).")

@st.cache_data
def generate_ratings_data(n_users=50, n_movies=20):
    """Generates a synthetic user-item ratings matrix."""
    np.random.seed(42)

    movies = [f"Movie_{i+1}" for i in range(n_movies)]
    users = [f"User_{i+1}" for i in range(n_users)]

    # Initialize an empty DataFrame for ratings (NaN for unrated)
    ratings_df = pd.DataFrame(index=users, columns=movies)

    # Populate with some random ratings (simulating user preferences)
    for user in users:
        # Each user rates a random number of movies (e.g., 5 to 15)
        num_rated_movies = np.random.randint(5, 16)
        rated_movies = np.random.choice(movies, num_rated_movies, replace=False)
        for movie in rated_movies:
            ratings_df.loc[user, movie] = np.random.randint(1, 6) # Ratings from 1 to 5

    # Introduce some clear patterns for better demo
    # Group 1: Likes Action/SciFi (Movie_1 to Movie_5)
    action_scifi_movies = [f"Movie_{i+1}" for i in range(5)]
    for user_id in np.random.choice(users, 15, replace=False):
        for movie in action_scifi_movies:
            if np.random.rand() > 0.3: # 70% chance to rate
                ratings_df.loc[user_id, movie] = np.random.randint(4, 6) # High ratings

    # Group 2: Likes Comedy/Romance (Movie_6 to Movie_10)
    comedy_romance_movies = [f"Movie_{i+6}" for i in range(5)]
    for user_id in np.random.choice(users, 15, replace=False):
        for movie in comedy_romance_movies:
            if np.random.rand() > 0.3:
                ratings_df.loc[user_id, movie] = np.random.randint(4, 6)

    # Fill some remaining NaNs with 0 to make it dense for similarity calculation,
    # but keep NaNs for display if desired. For collaborative filtering,
    # often unrated items are treated as 0 or not included in similarity.
    # For this demo, we'll work with the sparse matrix for similarity.
    
    return ratings_df.astype(float) # Ensure ratings are float for calculations

# Generate data
ratings_matrix = generate_ratings_data()

st.write("#### Sample of User-Item Ratings Matrix (NaN means unrated):")
st.dataframe(ratings_matrix.head())
st.write(f"Dataset shape: {ratings_matrix.shape[0]} users, {ratings_matrix.shape[1]} movies")

with st.expander("Explore Data Sparsity"):
    rated_count = ratings_matrix.count().sum()
    total_cells = ratings_matrix.size
    sparsity = (1 - (rated_count / total_cells)) * 100
    st.write(f"Number of rated entries: **{rated_count}**")
    st.write(f"Total possible entries: **{total_cells}**")
    st.write(f"Data sparsity: **{sparsity:.2f}%** (Typical for recommender systems)")
    st.info("Sparsity is common in real-world rating datasets, where users rate only a small fraction of available items.")

st.markdown("---")
st.subheader("2. Recommendation Logic: Item-Based Collaborative Filtering")
st.markdown("""
Item-based collaborative filtering works by finding items that are similar to the ones a user has already liked.
The core idea is: "Users who liked X also liked Y."

Here's a simplified breakdown:
1.  **Create an Item-Feature Matrix:** Transform the user-item matrix so that items are rows and users are columns, with ratings as values.
2.  **Calculate Item Similarity:** Compute the similarity between every pair of items. We'll use **Cosine Similarity**, which measures the cosine of the angle between two vectors, indicating how similar they are in terms of user ratings.
3.  **Generate Recommendations:** For a given user, identify items they have already rated positively. Then, find other items that are most similar to those highly-rated items and recommend the unrated ones.
""")

@st.cache_data
def calculate_item_similarity(df):
    """Calculates item-item cosine similarity."""
    # Transpose the matrix to have items as rows and users as columns
    item_user_matrix = df.fillna(0).T # Fill NaNs with 0 for similarity calculation, assuming 0 rating for unrated
    item_similarity = cosine_similarity(item_user_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)
    return item_similarity_df

@st.cache_data
def recommend_items(user_liked_items, ratings_matrix, item_similarity_df, top_n=5):
    """Recommends items based on liked items using item-item similarity."""
    recommendations = {}
    for liked_item in user_liked_items:
        # Get similarity scores for the liked item
        similar_items = item_similarity_df[liked_item].sort_values(ascending=False)

        # Iterate through similar items (excluding itself)
        for item, similarity_score in similar_items.drop(liked_item).items():
            if item not in user_liked_items: # Only recommend items not already liked by the user
                if item not in recommendations:
                    recommendations[item] = 0
                recommendations[item] += similarity_score # Accumulate similarity scores

    # Sort recommendations by aggregated similarity score
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Return top N recommendations
    return [item for item, score in sorted_recommendations[:top_n]]

item_similarity_matrix = calculate_item_similarity(ratings_matrix)

st.write("#### Sample of Item-Item Similarity Matrix (Cosine Similarity):")
st.dataframe(item_similarity_matrix.head().style.background_gradient(cmap='Blues'))
st.write("*(Higher values indicate greater similarity between items)*")

st.success("Item similarity calculated!")

# --- 3. Interactive Recommender ---

st.subheader("3. Interactive Movie Recommender")
st.markdown("Select some movies you've liked (rated highly) and get recommendations!")

all_movies = ratings_matrix.columns.tolist()

selected_liked_movies = st.multiselect(
    "Which movies have you liked?",
    options=all_movies,
    help="Select movies you would typically rate 4 or 5 stars."
)

if st.button("Get Recommendations"):
    if selected_liked_movies:
        with st.spinner("Generating recommendations..."):
            recommendations = recommend_items(selected_liked_movies, ratings_matrix, item_similarity_matrix, top_n=7)
            
            st.write("---")
            st.subheader("🎬 Your Recommended Movies:")
            if recommendations:
                for i, movie in enumerate(recommendations):
                    st.write(f"{i+1}. **{movie}**")
            else:
                st.info("No new recommendations could be generated based on your selections. Try selecting more liked movies.")
    else:
        st.warning("Please select at least one movie you liked to get recommendations.")


st.markdown("---")
st.subheader("Key Takeaways from Recommender System:")
st.markdown("""
* **Personalization:** Recommender systems offer personalized experiences, improving user satisfaction and retention.
* **Collaborative Filtering:** This method leverages the collective behavior of users to find patterns and make suggestions.
* **Item-Based vs. User-Based:**
    * **Item-based:** Finds items similar to what a user liked. "People who liked X, also liked Y." (Used here)
    * **User-based:** Finds users similar to you and recommends items they liked. "Users similar to you liked X."
* **Sparsity Challenge:** Real-world rating data is often sparse, which can impact the accuracy of similarity calculations.
* **Scalability:** For very large datasets, more advanced techniques (e.g., matrix factorization, deep learning) are used to handle scalability.
""")
