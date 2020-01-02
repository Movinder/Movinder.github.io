from flask import Flask, render_template, request, send_from_directory
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")


app = Flask(__name__, template_folder=TEMPLATES_DIR)

@app.route('/')
def root():
    # TODO: make index,html
	return open('index.html').read()


@app.route('/recommendation', methods=['POST', 'GET'])
def recommendation():
	if request.method == 'POST':
		data = {}
		return render_template('recommended_movies.html', data=data)


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
