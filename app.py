from flask import Flask, request, jsonify
import requests

from topics import get_topic_predictions, TopicsSchema

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/topics', methods=['GET', 'POST'])
def topics():
    title = request.args.get('title') or request.json.get('title')
    abstract = request.args.get('abstract') or request.json.get('abstract')

    if not title or not abstract:
        return jsonify({"error": "Both title and abstract must be provided"}), 400

    topic_predictions = get_topic_predictions(title, abstract)

    topic_ids = [f"T{topic['topic_id']}" for topic in topic_predictions]

    r = requests.get("https://api.openalex.org/topics?filter=id:{0}".format("|".join(topic_ids)))
    topics_from_api = r.json()['results']

    ordered_topics = []
    for topic in topic_predictions:
        for api_topic in topics_from_api:
            if api_topic['id'] == f"https://openalex.org/T{topic['topic_id']}":
                api_topic['score'] = topic['topic_score']
                ordered_topics.append(api_topic)
                break
    topic_schema = TopicsSchema(many=True)
    return topic_schema.dumps(ordered_topics)


if __name__ == '__main__':
    app.run(debug=True)
