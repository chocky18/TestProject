# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/save-avatar', methods=['POST'])
def save_avatar():
    data = request.get_json()
    avatar_url = data.get('avatarUrl')

    if avatar_url:
        print(f"✅ Avatar URL received: {avatar_url}")
        return jsonify({"status": "success", "url": avatar_url})
    else:
        return jsonify({"status": "error", "message": "No URL received"}), 400

if __name__ == '__main__':
    app.run(debug=True)
