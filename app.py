from flask import Flask, request, render_template, session
import numpy as np
import pandas as pd
from surprise import NMF, Dataset, Reader
from scipy.stats import hmean 
import scipy.sparse as sp
import os
import json

from lightfm.data import Dataset as LightFMDataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

app = Flask(__name__, template_folder='templates')
app.secret_key = "super secret key"

DATA_DIR = "static/data"

# Siamese data
movies = json.load(open(f'{DATA_DIR}/movies.json'))
friends = json.load(open(f'{DATA_DIR}/friends.json'))
ratings = json.load(open(f'{DATA_DIR}/ratings.json'))
#soup_movie_features = np.load(f'{DATA_DIR}/soup_movie_features_11.npy')
soup_movie_features = sp.load_npz(f'{DATA_DIR}/soup_movie_features_11.npz').toarray()
df_movies = pd.DataFrame(movies)
movie_ids = np.array(df_movies.movie_id_ml.unique())
new_friend_id = len(friends)

# MF data
df_ratings = pd.read_csv(f'{DATA_DIR}/ratings.csv')
mat = np.zeros((max(df_ratings.user_id), max(df_ratings.movie_id_ml)))
ind = np.array(list(zip(list(df_ratings.user_id-1), list(df_ratings.movie_id_ml-1))))
mat[ind[:,0], ind[:,1]] = 1
movies_ = mat.sum(axis=0).argsort()+1
column_item = ["movie_id_ml", "title", "release", "vrelease", "url", "unknown", 
					"action", "adventure", "animation", "childrens", "comedy",
				   "crime", "documentary", "drama", "fantasy", "noir", "horror",
				   "musical", "mystery", "romance", "scifi", "thriller",
				   "war", "western"]
df_ML_movies = pd.read_csv(f'{DATA_DIR}/u.item.txt', delimiter='|', encoding = "ISO-8859-1", names=column_item) 
df_posters = pd.read_csv(f"{DATA_DIR}/movie_poster.csv", names=["movie_id_ml", "poster_url"])
df_ML_movies = pd.merge(df_ML_movies,df_posters, on="movie_id_ml")




def recommendation_mf(userArray, numUsers, movieIds):

	ratings_dict = {'itemID': list(df_ratings.movie_id_ml) + list(numUsers*movieIds),
					'userID': list(df_ratings.user_id) + [max(df_ratings.user_id)+1+x for x in range(numUsers) for y in range(len(userArray[0]))],
					'rating': list(df_ratings.rating) + [item for sublist in userArray for item in sublist]
				}

	df = pd.DataFrame(ratings_dict)
	reader = Reader(rating_scale=(1, 5))
	data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
	trainset = data.build_full_trainset()

	nmf = NMF()
	nmf.fit(trainset)

	userIds = [trainset.to_inner_uid(max(df_ratings.user_id)+1+x) for x in range(numUsers)]

	mat = np.dot(nmf.pu, nmf.qi.T)

	scores = hmean(mat[userIds, :], axis=0)
	best_movies = scores.argsort()
	best_movies = best_movies[-9:][::-1]
	scores = scores[best_movies]
	movie_ind = [trainset.to_raw_iid(x) for x in best_movies]

	recommendation = list(zip(list(df_ML_movies[df_ML_movies.movie_id_ml.isin(movie_ind)].title), 
					list(df_ML_movies[df_ML_movies.movie_id_ml.isin(movie_ind)].poster_url), 
					list(scores)))

	return recommendation

def recommendation_siamese(top_movies, scores):
	recommendation = list(zip(list(top_movies.title), 
					list(top_movies.poster_url), 
					scores)) 
	return recommendation

def predict_top_k_movies(model, friends_id, k, n_movies, user_features=None, item_features=None, use_features=False):
	if use_features:
		prediction = model.predict(friends_id, np.arange(n_movies), user_features=friends_features, item_features=item_features)
	else:
		prediction = model.predict(friends_id, np.arange(n_movies))
	
	global movie_ids
	#movie_ids = np.arange(data.shape[1])
	# return movie ids
	return movie_ids[np.argsort(-prediction)][:k], prediction[np.argsort(-prediction)][:k]



@app.route('/', methods=['GET', 'POST'])
def main():

	if request.method == 'POST':
		global df_movies
		# global top_trending_ids
		# print(list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].title) )
		print(request.form)
		# Get recommendations!
		if 'run-mf-model' in request.form:
			
			for i, user_rating in enumerate(session['arr']):
				session['arr'][i] = user_rating[:-2]
			session['movieIds'] = session['movieIds'][:-2]
			rated_movies = min(len(session['arr'][0]), len(session['movieIds']))
			for i, user_rating in enumerate(session['arr']):
				session['arr'][i] = user_rating[:rated_movies]
			session['movieIds'] = session['movieIds'][:rated_movies]

			pu = recommendation_mf(session['arr'], session['members'], session['movieIds'])


			session.clear()
			top_trending_ids = list(df_movies.sort_values(by="trending_score").head(200).sample(15).movie_id_ml)
			session['counter'] = 0
			session['members'] = 0
			session['userAges'] = []
			session['userGenders'] = []
			session['movieIds'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].movie_id_ml)
			session['top15'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].title) 
			session['top15_posters'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].poster_url)
			session['arr'] = None
			return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': False, 'people': 0, 'buttonDisable': False,'chooseRecommendation':False, 'recommendation': pu}))
		
		if 'run-siamese-model' in request.form:
			# global df
			global friends
			global ratings
			global new_friend_id
			new_ratings = []
			for mid, movie_real_id in enumerate(session['movieIds']):
				avg_mv_rating = np.median(np.array([user_ratings[mid] for user_ratings in session['arr']]))
				new_ratings.append({'movie_id_ml':movie_real_id, 
									'rating': avg_mv_rating,
									'friend_id': new_friend_id}) 
			new_friend = {'friend_id': new_friend_id, 'friends_age': np.mean(np.array(session['userAges'])), 'friends_gender': np.mean(np.array(session['userGenders']))}	

			friends.append(new_friend)
			ratings.extend(new_ratings)

			dataset = LightFMDataset()
			item_str_for_eval = "x['title'],x['release'], x['unknown'], x['action'], x['adventure'],x['animation'], x['childrens'], x['comedy'], x['crime'], x['documentary'], x['drama'],  x['fantasy'], x['noir'], x['horror'], x['musical'],x['mystery'], x['romance'], x['scifi'], x['thriller'], x['war'], x['western'], *soup_movie_features[x['soup_id']]"
			friend_str_for_eval = "x['friends_age'], x['friends_gender']"

			dataset.fit(users=(int(x['friend_id']) for x in friends),
						items=(int(x['movie_id_ml']) for x in movies),
						item_features=(eval("("+item_str_for_eval+")") for x in movies),
						user_features=((eval(friend_str_for_eval)) for x in friends))
			num_friends, num_items = dataset.interactions_shape()
			print(f'Num friends: {num_friends}, num_items {num_items}.')

			(interactions, weights) = dataset.build_interactions(((int(x['friend_id']), int(x['movie_id_ml']))
													  for x in ratings))
			item_features = dataset.build_item_features(((x['movie_id_ml'], 
											  [eval("("+item_str_for_eval+")")]) for x in movies) )
			user_features = dataset.build_user_features(((x['friend_id'], 
											  [eval(friend_str_for_eval)]) for x in friends) )

			epochs = 50 #150
			lr = 0.015
			max_sampled = 11

			loss_type = "warp"  # "bpr"


			model = LightFM(learning_rate=lr, loss=loss_type, max_sampled=max_sampled)

			model.fit_partial(interactions, epochs=epochs, user_features=user_features, item_features=item_features)
			train_precision = precision_at_k(model, interactions, k=10, user_features=user_features, item_features=item_features).mean()

			train_auc = auc_score(model, interactions, user_features=user_features, item_features=item_features).mean()

			print(f'Precision: {train_precision}, AUC: {train_auc}')

			k = 18
			top_movie_ids, scores = predict_top_k_movies(model, new_friend_id, k, num_items, user_features=user_features, item_features=item_features, use_features = False)
			top_movies = df_movies[df_movies.movie_id_ml.isin(top_movie_ids)]

			pu = recommendation_siamese(top_movies, scores)

			return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': False, 'people': 0, 'buttonDisable': False,'chooseRecommendation':False, 'recommendation': pu}))
		
		# Collect friends info
		elif 'person-select-gender-0' in request.form:
			for i in range(session['members']):
				session['userAges'].append(int(request.form.get(f'age-{i}')))
				session['userGenders'].append(int(request.form.get(f'person-select-gender-{i}')))

			return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': True, 'people': session['members'], 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))

		# Choose number of people in the group
		elif 'people-select' in request.form:
			count = int(request.form.get('people-select'))
			session['members'] = count
			session['arr'] = [[0 for x in range(15)] for y in range(count)] 
			return(render_template('main.html', settings = {'friendsInfo':True, 'showVote': False, 'people': count, 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))

		# All people voting
		elif 'person-select-0' in request.form:
			for i in range(session['members']):
				session['arr'][i][session['counter']] = int(request.form.get(f'person-select-{i}'))
			
			session['counter'] += 1 
			if session['counter'] < 15:     
				return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': True, 'people': len(request.form), 'buttonDisable': True,'chooseRecommendation':False, 'recommendation': None}))
			else:
				return(render_template('main.html', settings = {'friendsInfo':False, 'showVote': False, 'people': len(request.form), 'buttonDisable': True,'chooseRecommendation':True,  'recommendation': None}))

	elif request.method == 'GET':
		session.clear()
		top_trending_ids = list(df_movies.sort_values(by="trending_score").head(200).sample(15).movie_id_ml)
		print(top_trending_ids)
		print(list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].title) )
		session['counter'] = 0
		session['members'] = 0
		session['userAges'] = []
		session['userGenders'] = []
		session['movieIds'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].movie_id_ml) 
		session['top15'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].title) 
		session['top15_posters'] = list(df_movies[df_movies.movie_id_ml.isin(top_trending_ids)].poster_url)
		session['arr'] = None

		return(render_template('main.html', settings = {'showVote': False, 'people': 0, 'buttonDisable': False, 'recommendation': None}))

@app.route('/static/<path:path>')
def serve_dist(path):
	return send_from_directory('static', path)

if __name__ == '__main__':
	# Bind to PORT if defined, otherwise default to 5000.
	port = int(os.environ.get('PORT', 5000))
	host= '0.0.0.0'
	app.run(host=host, port=port)
