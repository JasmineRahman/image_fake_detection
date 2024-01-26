from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deepfake_awareness')
def deepfake_awareness():
    return render_template('deepfake_awareness.html')

@app.route('/deepfake_examples')
def deepfake_examples():
    return render_template('deepfake_examples.html')

@app.route('/tutorial_on_spotting_deepfakes')
def tutorial_on_spotting_deepfakes():
    return render_template('tutorial_on_spotting_deepfakes.html')

@app.route('/community_reports')
def community_reports():
    return render_template('community_reports.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/newsfeed')
def get_newsfeed():
    api_key = "b88af15a44a34695ac9df0190bee8d74"
    base_url = "https://newsapi.org/v2/everything"

    # Parameters for the request
    params = {
        'q': 'deepfake',
        'apiKey': api_key,
        'sortBy': 'publishedAt',
    }

    # Make the request
    response = requests.get(base_url, params=params)
    data = response.json()

    # Check if there are articles
    if data['status'] == 'ok' and data['totalResults'] > 0:
        articles = data['articles']
    else:
        articles = []

    return render_template('newsfeed.html', news_articles=articles)

@app.route('/user_generated_content_analysis')
def user_generated_content_analysis():
    return render_template('user_generated_content_analysis.html')

@app.route('/image_history')
def image_history():
    return render_template('image_history.html')

if __name__ == '__main__':
    app.run(debug=True)
