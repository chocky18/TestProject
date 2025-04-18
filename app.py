from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    avatar_url = "https://models.readyplayer.me/6800de4564ce38bc90d3b488.glb"
    return render_template('index.html', avatar_url=avatar_url)

if __name__ == '__main__':
    app.run(debug=True)
