from flask import Flask, render_template

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

@app.route('/newsfeed')
def newsfeed():
    return render_template('newsfeed.html')

@app.route('/user_generated_content_analysis')
def user_generated_content_analysis():
    return render_template('user_generated_content_analysis.html')

@app.route('/image_history')
def image_history():
    return render_template('image_history.html')

# @app.route('/community_forum')
# def community_forum():
#     return render_template('community_forum.html')

if __name__ == '__main__':
    app.run(debug=True)
