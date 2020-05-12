import pandas as pd
import numpy as np
import json
from subprocess import check_output
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import CountVectorizer

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV

###################################################################
def get_data():
    '''keywords.csv: Contains the movie plot keywords for our MovieLens movies. 
    Available in the form of a stringified JSON Object.
    credits.csv: Consists of Cast and Crew Information for all our movies. 
    Available in the form of a stringified JSON Object.
    links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
    links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
    ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.'''
    
    movies_metadata = pd.read_csv('../data/movies_metadata.csv'
                              ,parse_dates=['release_date']
                              ,encoding="utf8", header='infer')
    keywords = pd.read_csv('../data/keywords.csv')
    credits = pd.read_csv('../data/credits.csv')
    links = pd.read_csv('../data/links.csv')
    links_small = pd.read_csv('../data/links_small.csv')
    ratings_small = pd.read_csv('../data/ratings_small.csv')
    tmdb_5000_credits = pd.read_csv('../data/tmdb_5000_credits.csv')
    tmdb_5000_movies = pd.read_csv('../data/tmdb_5000_movies3.csv',parse_dates=['release_date'])
    
    return tmdb_5000_movies, tmdb_5000_credits, links_small,ratings_small, movies_metadata, keywords, credits, links


###################################################################
def transfer_to_json(col_list, df):
    for col in col_list:
        df[col] = df[col].apply(json.loads)
        
    return df

###################################################################
def safe_access(container, index_values):
    
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    
    except IndexError or KeyError:
        return pd.np.nan

    
###################################################################
def get_job(job,container):
    
    result = container
    try:
        for i in range(len(container)):
            result = container[i]
            if result['job'] == job:
                return result['name']
        
    except IndexError or KeyError:
        return pd.np.nan

    
###################################################################
def get_actor_name(container, order):
    
    try:
        return (container[order]['name'])
    
    except IndexError or KeyError:
        
        return pd.np.nan
   

###################################################################
def pipe_flatten_names(keywords):
    
    return '|'.join([x['name'] for x in keywords])


###################################################################
def pipe_flatten_names2(keywords):
    
    return '|'.join([x['name'] for x in ast.literal_eval(keywords)])


###################################################################
def get_movie_details(movie, credits,renamed_cols):
    
    movie.rename(columns = renamed_cols, inplace = True)
    movie['country'] = movie['production_countries'].apply(lambda x:safe_access(x,[0,'name']))
    movie['language'] = movie['spoken_languages'].apply(lambda x:safe_access(x,[0,'name']))
    movie['director_name'] = credits['crew'].apply(lambda x: get_job('Director',x))
    movie['actor_1_name'] = credits['cast'].apply(lambda x: get_actor_name(x,1))
    movie['actor_2_name'] = credits['cast'].apply(lambda x: get_actor_name(x,2))
    movie['actor_3_name'] = credits['cast'].apply(lambda x: get_actor_name(x,3))
    movie['genres'] = movie['genres'].apply(pipe_flatten_names)
    movie['plot_keywords'] = movie['plot_keywords'].apply(pipe_flatten_names)
    movie['title_year']  = movie['release_date'].dt.year
    
    return movie


###################################################################
def get_all_keywords(df, col, split):
    
    set_keywords = set()
    for liste_keywords in df[col].str.split(split).values:
        #if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
        set_keywords = set_keywords.union(liste_keywords)

    set_keywords.remove('')
    
    return set_keywords


###################################################################
def count_word(df, col, set_keywords, split):
    
    key_words_count = {}

    for word in ((set_keywords)):
        key_words_count[word] = 0    
    for statement in df[col].str.split(split):
        for s in [s for s in statement if s in set_keywords]:
            if pd.notnull(s):
                key_words_count[s] += 1
        
    return {k: v for k, v in sorted(key_words_count.items(), key=lambda item: item[1],reverse=True)}



###################################################################
def IMDB_weighted_rating(df, col_vote_count, col_vote_average):
    
    v = df[col_vote_count]
    m= df[col_vote_count].quantile(0.9)
    R = df[col_vote_average]
    C = df[col_vote_average].mean()
    WR = (v / (v + m) * R) + (m / (v + m) * C)
    
    return WR



###################################################################
def get_recommendations(title, cosine_sim, df):
    
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['original_title'].iloc[movie_indices].reset_index()


###################################################################
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

        

###################################################################        
def group_features(x):
    
    return  (x['actor_1_name']+ ' ' + x['actor_2_name']+ ' ' + x['actor_3_name']
             + ' ' + x['genres']+ ' ' + x['plot_keywords']+ ' ' + x['director_name'])

