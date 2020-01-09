import pandas as pd
import datetime, time
import os
import random
import numpy as np
import scipy.sparse as sp
import json
import requests

DATA_DIR = "static/data"

def update_data(friends_id, ratings, rated_movie_ids, df, df_friends, df_movies):
    df_friends = df_friends.append({"fid": friends_id, "fid_user_avg_age":0}, ignore_index=True)
    print(f"New number of friends features: {df_friends.shape[0]}")
    print(f"New number of movies features: {df_movies.shape[0]}")

    data_new_friends_training = []
    for mid, movie_real_id in enumerate(rated_movie_ids):
        avg_mv_rating = np.median(np.array([user_ratings[mid] for user_ratings in ratings]))
        data_new_friends_training.append([friends_id, movie_real_id, avg_mv_rating]) 

    columns = ["fid", "iid", "rating"]
    # user initial input that will be given to him to rate it before recommendation
    df_new_friends_train = pd.DataFrame(data_new_friends_training, columns=columns)

    df_train = df.copy()
    df_train = pd.concat([df_train, df_new_friends_train], sort=False)

    df_train = df_train[["fid", "iid", "rating"]].astype(np.int64)
    #df_new_friends_train = df_new_friends_train[["fid", "iid", "rating"]].astype(np.int64)

    return df_train, df_friends, df_movies

def onehotencoding2genre(x):
        genres= ['unknown','action','adventure','animation','childrens','comedy','crime','documentary','drama','fantasy','noir','horror','musical','mystery','romance','scifi','thriller','war','western']
        ret_val = []
        for c in genres:
            g = getattr(x, c)
            if g == 1:
                ret_val.append(c)
        return ret_val

def get_trending_movie_ids(k, df):
    df_movie_count_mean = df.groupby(["movie_id_ml", "title"], as_index=False)["rating"].agg(["count", "mean"]).reset_index()
    C = df_movie_count_mean["mean"].mean()
    m = df_movie_count_mean["count"].quantile(0.9)

    def weighted_rating(x, m=m, C=C):
        """Calculation based on the IMDB formula"""
        v = x['count']
        R = x['mean']
        return (v/(v+m) * R) + (m/(m+v) * C)

    df_movies = pd.read_csv(f"{DATA_DIR}/movies_cast_company.csv", encoding='utf8')
    df_movies["cast"] = df_movies["cast"].apply(lambda x: json.loads(x))
    df_movies["company"] = df_movies["company"].apply(lambda x: json.loads(x))
    df_movies["genres"] = df_movies.apply(lambda x: onehotencoding2genre(x), axis=1)


    df_movies_1 = df_movie_count_mean.copy().loc[df_movie_count_mean["count"] > m]
    df = pd.merge(df_movies, df_movies_1, on=["movie_id_ml", "title"])


    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    df['score'] = df.apply(weighted_rating, axis=1)


    #Sort movies based on score calculated above
    df = df.sort_values('score', ascending=False).reset_index()

    df = df.head(200)

    df = df.sample(k)

    return list(df.movie_id_ml)
