from flask import Flask, render_template, request, send_from_directory
import os
from data import initial_data, get_posters, update_data
from siamese_training import training
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
print(TEMPLATES_DIR)

friends_num = 1

app = Flask(__name__, template_folder=TEMPLATES_DIR)

df, df_friends, df_movies, new_fid = initial_data()
df_movie_urls = df[["iid", "poster_url", "title"]].drop_duplicates()


@app.route('/')
def root():
	# TODO: make index,html
	return open('friends_initialize.html').read()

@app.route('/friends', methods=["POST"])
def friends():
	if request.method == 'POST':
		global friends_num
		friends_num = int(request.form['numberOfPeople'])
		print("Number of people ", friends_num)
		
		movie_names_and_posters = []
		for friend_id in range(friends_num):
			data = []
			samples = df_movie_urls.sample(n=12)
			
			for i, t, p in zip(list(samples.iid), list(samples.title), get_posters(samples)):
				data.append({"poster": p, "movie_name": t, "movie_id": i})

			movie_names_and_posters.append(data)

		print(len(movie_names_and_posters))
		print(len(movie_names_and_posters[0]))
		return render_template('friends_info.html', friends_num=friends_num, movie_names_and_posters=movie_names_and_posters)

@app.route('/recommendation-siamese', methods=['POST', 'GET'])
def recommendation():
	global df
	global df_friends
	global df_movies
	global new_fid
	if request.method == 'POST':
		data = request.form
		print(data)
		friends_age, movie_and_avg_rating = handle_friends_input(data)
		df_train, df_friends, df_movies = update_data(new_fid, friends_age, movie_and_avg_rating, df, df_friends, df_movies)

		top_movie_ids = training(df_train, df_friends, df_movies, new_fid)
		top_movies = df_movie_urls[df_movie_urls.iid.isin(top_movie_ids)]

		movie_names_and_posters = []
		for i, t, p in zip(list(top_movies.iid), list(top_movies.title), get_posters(top_movies)):
			movie_names_and_posters.append({"poster": p, "movie_name": t, "movie_id": i})


		return render_template('recommended_movies.html', movie_names_and_posters=movie_names_and_posters)


@app.route('/assets/<path:path>')
def serve_dist(path):
    return send_from_directory('assets', path)


def handle_friends_input(data):
	global friends_num
	friends_age = np.mean(np.array([ int(data[f"age_{i}"]) for i in range(friends_num)]))
	print(friends_age)
	
	movie_and_rating = {}
	for i in range(friends_num):
		user_input_keys = []
		for i in range(friends_num):
			for k in data.keys():
				if k.startswith(f"btn_group_{i}_"):
					user_input_keys.append(k)
		
		for k in user_input_keys:
			movie_id = int(k.split("_")[-1])
			val = movie_and_rating.get(movie_id, None)
			rating = 5 if data[k]=='like' else 2 if data[k]=='dislike' else 0
			if val:
				movie_and_rating[movie_id] += [rating]
			else:
				movie_and_rating[movie_id] = [rating]
	
	movie_and_avg_rating = []
	for k in movie_and_rating.keys():
		movie_and_avg_rating.append([k, np.median(movie_and_rating[k])])
	
	return friends_age, movie_and_avg_rating




if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	host = '127.0.0.1'#'0.0.0.0'
	app.run(host=host, port=port)
