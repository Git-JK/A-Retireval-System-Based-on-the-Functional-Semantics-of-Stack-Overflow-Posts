import flask
import os
import pickle
from utils.config import TEST_RESULTS_STORE_PATH
from flask import Flask, jsonify, request
from utils.read.read_utils import load_test_posts, query2id
from get_posts_ranking import get_recommendation_posts

app = Flask(__name__)

def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

app.after_request(after_request)

@app.route('/')
def hello():
    return 'hello, world!'

@app.route('/search/<query>', methods=['GET'])
def search(query):
    q_id = query2id(query)
    data = []
    if q_id != "":
        file_name = q_id + ".pkl"
        data_path = os.path.join(TEST_RESULTS_STORE_PATH, file_name)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as rbf:
                data = pickle.load(rbf)
    else:
        data = get_recommendation_posts(query)
    ret = {
        "success": True,
        'data': data
    }
    response = flask.make_response(jsonify(ret))
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port='3001')