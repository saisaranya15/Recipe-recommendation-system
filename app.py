import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load updated JSON dataset
@st.cache_data
def load_data():
    train_df = pd.read_json('dish_names.json')
    train_df['ingredients_str'] = train_df['ingredients'].apply(lambda x: ' '.join(x))
    return train_df

# Load data
train_df = load_data()

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(train_df['ingredients_str'])

# Nearest Neighbors Model
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(tfidf_matrix)

# Streamlit App
st.title("🥗 Recipe Recommender System")

# User Input
user_input = st.text_input("Enter ingredients or a recipe name:")

if user_input:
    user_vec = tfidf.transform([user_input])
    distances, indices = nn_model.kneighbors(user_vec, n_neighbors=5)

    st.subheader("Recommended Recipes:")
    for idx in indices[0]:
        recipe = train_df.iloc[idx]
        st.write(f"### 🍽️ Dish Name: {recipe['dish_name']}")
        st.write(f"**Cuisine:** {recipe['cuisine']}")
        st.write(f"**Ingredients:** {', '.join(recipe['ingredients'])}")
        st.write("---")