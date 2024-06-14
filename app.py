from flask import Flask, request
import requests

from topics import get_topic_predictions

app = Flask(__name__)


@app.route('/topics', methods=['GET', 'POST'])
def topics():
    title = request.args.get('title') or request.json.get('title')
    abstract = request.args.get('abstract') or request.json.get('abstract')

    topic_predictions = get_topic_predictions(title, abstract)

    topic_ids = [f"T{topic['topic_id']}" for topic in topic_predictions]

    r = requests.get("https://api.openalex.org/topics?filter=id:{0}".format("|".join(topic_ids)))
    topics_from_api = r.json()['results']
    return topics_from_api

if __name__ == '__main__':
    app.run(debug=True)
