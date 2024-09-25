import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import wordcloud
import ast
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    .main {
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .stSelectbox>div>div>div {
        color: #ffffff;
        font-size: 16px;
        font-family: 'Roboto', sans-serif;
    }
    .stText {
        font-family: 'Roboto', sans-serif;
        font-size: 18px;
        color: #ffffff;
        overflow: auto;
    }
    .stHeader {
        font-family: 'Roboto', sans-serif;
        font-size: 42px;
        color: #4CAF50;
        text-align: center;
    }
    .stImage>img {
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .movie-title {
        font-family: 'Roboto', sans-serif;
        font-size: 14px;
        color: #ffffff;
        text-align: center;
        margin-top: 5px;
        overflow: auto;
        white-space: pre;
    }
    </style>
    """, unsafe_allow_html=True)

def load_dataset():
    movies = pd.read_csv("movies.csv")
    credits = pd.read_csv("credits.csv")
    return movies, credits

def merge_dataset(dataset1, dataset2, on='title'):
    merged_dataset = dataset1.merge(dataset2, on=on)
    return merged_dataset

def convert_text(text):
    L = list()
    for t in ast.literal_eval(text):
        name = t['name']
        L.append(name)
    return (L)

def convert_text3(text):
    L = list()
    counter = 0
    for t in ast.literal_eval(text):
        if counter < 3:
            name = t['name']
            L.append(name)
            counter += 1
        else:
            break
    return (L)

def fetch_director(text):
    L = list()
    for t in ast.literal_eval(text):
        if t['job'] == 'Director':
            L.append(t['name'])
    return (L)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ","_"))
    return (L1)

def convert(dataset):
    dataset['genres'] = dataset['genres'].apply(convert_text)
    dataset['cast'] = dataset['cast'].apply(convert_text3)
    dataset['crew'] = dataset['crew'].apply(fetch_director)
    dataset['overview'] = movies_credits['overview'].apply(lambda n: (n.split()))
    dataset['keywords'] = dataset['keywords'].apply(convert_text)
    dataset['genres'] = dataset['genres'].apply(collapse)
    dataset['keywords'] = dataset['keywords'].apply(collapse)
    dataset['cast'] = dataset['cast'].apply(collapse)
    dataset['crew'] = dataset['crew'].apply(collapse)
    return dataset

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        full_path = ""
    return full_path

def recommend(movie, similarity):
    idx = list(movie_list).index(movie)
    distances = similarity[idx]
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances:
        int(i)
        id = movie_id[i]
        recommended_movie_posters.append(fetch_poster(id))
        recommended_movie_names.append(movie_list[i])
    
    return recommended_movie_names, recommended_movie_posters

movies, credits = load_dataset()
movies_credits = merge_dataset(movies, credits)
movies_credits = movies_credits[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies_credits.dropna(inplace=True)
movies_credits.sort_values(by='movie_id')

with st.sidebar:
    st.title("🎬 Movie Recommender")
    option = option_menu(
        menu_title="",
        options=["Dataset", "EDA", "Preprocessing", "Training", "Result"],
        icons=["database", "bar-chart", "cpu", "gear", "window"]
    )

# Dataset Menu
if option == "Dataset":
    st.title("Dataset")
    dataset = st.radio(
        label="Select dataset below",
        options=("Movies", "Credits"),
        horizontal=True
    )

    if st.button("Show"):
        if dataset == "Movies":
            st.write(movies)
        elif dataset == "Credits":
            st.write(credits)
        # elif dataset == "Merged":
        #     st.write(movies_credits)

# EDA Menu
elif option == "EDA":
    st.title("Exploratory Data Analysis")

    all_item = []
    count = {}

    st.markdown('''
        <h3>Frequency Distribution</h3>
        ''', unsafe_allow_html=True)
    selected1 = st.selectbox(
        label="Type or select column below",
        options=["Genre", "Keyword"],
        key='freqdist'
    )

    if selected1 == "Genre":
        new_ = movies_credits.copy()
        new_['genres'] = new_['genres'].apply(convert_text)
        new_['genres'] = new_['genres'].apply(collapse)
        items = new_['genres'].values
    
    if selected1 == "Keyword":
        new_ = movies_credits.copy()
        new_['keywords'] = new_['keywords'].apply(convert_text)
        new_['keywords'] = new_['keywords'].apply(collapse)
        items = new_['keywords'].values

    for item in items:
        all_item.extend(item)

    for item in all_item:
        key = item
        try:
            count[key] += 1
        except Exception:
            count[key] = 1

    num_feat1 = st.slider(
        label="Number of feature to show",
        min_value=1,
        max_value=len(count),
        key='num_feat1'
    )

    x = list(count.keys())
    y = list(count.values())

    sorted = st.radio(
        label="Sorting",
        options=("Ascending", "Descending", "Random")
    )

    if sorted == "Ascending":
        x.sort()
        y.sort()
    elif sorted == "Descending":
        x.sort(reverse=True)
        y.sort(reverse=True)

    if st.button("Show graph"):
        
        fig = plt.figure(figsize=(num_feat1+4, 6))
        plt.bar(x[:num_feat1], y[:num_feat1])
        st.write(fig)
    
    st.markdown('''
        <h3>Wordcloud</h3>
        ''', unsafe_allow_html=True)
    selected2 = st.selectbox(
        label="Type or select column below",
        options=["Genre", "Keyword"],
        key='wordcloud'
    )

    if selected2 == "Genre":
        new_ = movies_credits.copy()
        new_['genres'] = new_['genres'].apply(convert_text)
        new_['genres'] = new_['genres'].apply(collapse)
        items = new_['genres'].values
    
    if selected2 == "Keyword":
        new_ = movies_credits.copy()
        new_['keywords'] = new_['keywords'].apply(convert_text)
        new_['keywords'] = new_['keywords'].apply(collapse)
        items = new_['keywords'].values
        
    for item in items:
        all_item.extend(item)

    for item in all_item:
        key = item
        count[key] = 1
    
    num_feat2 = st.slider(
        label="Number of feature to show",
        min_value=1,
        max_value=len(count),
        key='num_feat2'
    )


    feat = list(count.keys())
    random.shuffle(feat)
    words = ' '.join(feat[:num_feat2])
    wc = wordcloud.WordCloud().generate(words)

    if st.button("Show image"):
        fig = plt.figure(figsize=(19, 6))
        fig = px.imshow(wc)
        st.write(fig)
        
# Preprocessing Menu
elif option == "Preprocessing":
    st.title("Preprocessing")
    st.write("Check any column to be used for training")
    overview = st.checkbox("overview")
    genres = st.checkbox("genres")
    keywords = st.checkbox("keywords")
    cast = st.checkbox("cast")
    crew = st.checkbox("crew")

    bag = np.zeros(shape=(5, 2))
    bag[0] = [1, overview]
    bag[1] = [2, genres]
    bag[2] = [3, keywords]
    bag[3] = [4, cast]
    bag[4] = [5, crew]


    st.write("")
    st.write("Choose the text vectorizer")
    selected = st.selectbox(
        label="Type or select",
        options=["Count Vectorizer", "TF-IDF Vectorizer", "Hashing Vectorizer"],
        key='vectorizer'
    )

    num_feature = st.slider(
        label="Max features",
        min_value=10,
        max_value=5000
    )

    if selected == "Count Vectorizer":
        vectorizer = CountVectorizer(max_features=num_feature, stop_words='english')
    elif selected == "TF-IDF Vectorizer":
        vectorizer = TfidfVectorizer(max_features=num_feature)
    elif selected == "Hashing Vectorizer":
        vectorizer = HashingVectorizer(n_features=num_feature)

    proceed = st.button("Proceed")
    if proceed:
        check = 0
        for i in bag:
            if i[1] == 1:
                check = 1
                break

        if check == 1:
            movies_credits = convert(movies_credits)
            # st.write(len(movies_credits))
            tags = list(np.zeros(shape=(len(movies_credits))))
            # st.write(len(tags))
            tags = [list(str('')) for i in tags]
            movies_credits['tags'] = tags
            for checked in bag:
                if checked[0] == 1 and checked[1] == 1:
                    movies_credits['tags'] += movies_credits['overview']
                if checked[0] == 2 and checked[1] == 1:
                    movies_credits['tags'] += movies_credits['genres']
                if checked[0] == 3 and checked[1] == 1:
                    movies_credits['tags'] += movies_credits['keywords']
                if checked[0] == 4 and checked[1] == 1:
                    movies_credits['tags'] += movies_credits['cast']
                if checked[0] == 5 and checked[1] == 1:
                    movies_credits['tags'] += movies_credits['crew']
                
            new_movies = movies_credits.drop(columns=['overview','genres','keywords','cast','crew'])
            new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))

            with open("movies_dataset.pickle", 'wb') as file:
                pickle.dump(new_movies, file)
            
            with open("vectorizer.pickle", 'wb') as file:
                pickle.dump(vectorizer, file)
            
            st.success("Success. Saved new dataset...")
            
            # if st.button("Show new dataset"):
            st.markdown("""
            <h3><center>New preprocessed dataset</center></h3><br>
            """,
            unsafe_allow_html=True)
            st.write(new_movies)

        else:
            st.error("Must include at least one column or more")

# Training Menu
elif option == "Training":
    st.title("Training")
    try:
        with open("movies_dataset.pickle", 'rb') as file:
            new_movies = pickle.load(file)

        with open("vectorizer.pickle", 'rb') as file:
            vectorizer = pickle.load(file)
        
        X = new_movies.iloc[:, -1].values
        vector = vectorizer.fit_transform(X)

        st.write("Choose training type")
        selected = st.selectbox(
            label="Type or select",
            options=["Collaborative Filtering (Cosine Similarity)", "Content-Based Filtering (Nearest Neighbors)"],
            key='model'
        )

        num_recommend = st.slider(
            label="Number of recommendation shown",
            min_value=1, 
            max_value=30,
            key='num_rec'
        )

        if st.button("Proceed"):

            # Cosine Similarity
            if selected == "Collaborative Filtering (Cosine Similarity)":
                cos_sim = cosine_similarity(vector)
                sim = np.array([np.sort(cos_sim[i])[-(num_recommend+1):][::-1][1:] for i in range(len(cos_sim))])
                similarity = np.array([np.argsort(cos_sim[i])[-(num_recommend+1):][::-1][1:] for i in range(len(cos_sim))]).astype(int)
            
            # Nearest Neighbors
            elif selected == "Content-Based Filtering (Nearest Neighbors)":
                nn = NearestNeighbors(n_neighbors=num_recommend+1)
                nn.fit(vector)

                tags = new_movies['tags'].values
                vec = [vectorizer.transform([tag]) for tag in tags]
                # nearest = np.array([nn.kneighbors(v)for v in vec])
                # sim = nearest[:, 0, 0, 1:]
                # similarity = nearest[:, 1, 0, 1:].astype(int)
                similarity = np.array([nn.kneighbors(v)[1][0][1:] for v in vec]).astype(int)
            
            st.success("Training completed...")
            st.write("Output")
            st.write(similarity)

            with open("similarity.pickle", 'wb') as file:
                pickle.dump(similarity, file)

    except Exception as e:
        st.error("An error occured. Please process the dataset first...")
        st.error(e)

# Result Menu
elif option == "Result":

    with open("movies_dataset.pickle", 'rb') as file:
        movie_dataset = pickle.load(file)
    
    with open("similarity.pickle", 'rb') as file:
        similarity = pickle.load(file)

    movie_id = movie_dataset['movie_id'].values
    movie_list = movie_dataset['title'].values

    st.header('🎬 Movie Recommender System')

    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendations'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie, similarity)
        num_rows = 5
        num_cols = 6

        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                index = row * num_cols + col
                if index < len(recommended_movie_names):
                    with cols[col]:
                        movie_name = f"{recommended_movie_names[index]}"
                        container = f"""<div class='movie-title'>{movie_name}</div>"""
                        st.markdown(container, unsafe_allow_html=True)
                        try:
                            st.image(recommended_movie_posters[index])
                        except:
                            st.empty()