
from flask import Flask, send_from_directory, request, jsonify
import random
# from flask_cors import CORS
from flask_cors import cross_origin

from langchain_app import ask_question


app = Flask(__name__)
# CORS(app)

# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory('client/public', 'index.html')

# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory('client/public', path)

@app.route("/rand", methods=["POST", "GET"])
@cross_origin()
def hello():
    answer_res = ask_question(request.json["question"])
    print(answer_res)

    return jsonify(answer_res)

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8080, debug=True)