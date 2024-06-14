from flask import Flask, request, jsonify
import requests

from topics import get_topic_predictions, TopicsSchema

app = Flask(__name__)


@app.route("/topics", methods=["GET", "POST"])
def topics():
    title = request.args.get("title") or request.json.get("title")
    abstract = request.args.get("abstract") or request.json.get("abstract")

    # error checking
    if not title and not abstract:
        return jsonify({"error": "A title or abstract must be provided"}), 400

    combined_text = title + " " + abstract
    combined_text_limit = 10000
    if len(combined_text) > combined_text_limit:
        return (
            jsonify(
                {
                    "error": f"The combined length of title and abstract must not exceed {combined_text_limit} characters"
                }
            ),
            400,
        )

    topic_predictions = get_topic_predictions(title, abstract)

    # get topics from OpenAlex API, using topic predictions
    topic_ids = [f"T{topic['topic_id']}" for topic in topic_predictions]
    r = requests.get(
        "https://api.openalex.org/topics?filter=id:{0}".format("|".join(topic_ids))
    )
    topics_from_api = r.json()["results"]

    ordered_topics = []
    for topic in topic_predictions:
        for api_topic in topics_from_api:
            if api_topic["id"] == f"https://openalex.org/T{topic['topic_id']}":
                api_topic["score"] = topic["topic_score"]
                ordered_topics.append(api_topic)
                break
    topic_schema = TopicsSchema(many=True)
    return topic_schema.dumps(ordered_topics)


if __name__ == "__main__":
    app.run(debug=True)
