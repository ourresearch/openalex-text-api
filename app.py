from collections import OrderedDict

from flask import Flask, request, jsonify
import requests

from concepts import get_concept_predictions, ConceptsMessageSchema
from topics import get_topic_predictions, TopicsMessageSchema
from validate import validate_input

app = Flask(__name__)


@app.route("/text/topics", methods=["GET", "POST"])
def topics():
    if request.method == "GET":
        title = request.args.get("title")
        abstract = request.args.get("abstract")
    else:
        title = request.json.get("title")
        abstract = request.json.get("abstract")

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

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

    result = OrderedDict()
    result["meta"] = {
        "count": len(ordered_topics),
    }
    result["results"] = ordered_topics
    message_schema = TopicsMessageSchema()
    return message_schema.dumps(result)


@app.route("/text/concepts", methods=["GET", "POST"])
def concepts():
    if request.method == "GET":
        title = request.args.get("title")
        abstract = request.args.get("abstract")
    else:
        title = request.json.get("title")
        abstract = request.json.get("abstract")

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    concept_predictions = get_concept_predictions(title, abstract)

    # get concepts from OpenAlex API, using concept predictions tuples (id, score) as input
    concept_ids = [f"C{concept_id}" for concept_id, _ in concept_predictions]
    r = requests.get(
        "https://api.openalex.org/concepts?filter=ids.openalex:{0}".format("|".join(concept_ids))
    )
    concepts_from_api = r.json()["results"]

    ordered_concepts = []
    for concept_id, concept_score in concept_predictions:
        for api_concept in concepts_from_api:
            if api_concept["id"] == f"https://openalex.org/C{concept_id}":
                api_concept["score"] = concept_score
                ordered_concepts.append(api_concept)
                break

    result = OrderedDict()
    result["meta"] = {
        "count": len(ordered_concepts),
    }
    result["results"] = ordered_concepts
    message_schema = ConceptsMessageSchema()
    return message_schema.dumps(result)


if __name__ == "__main__":
    app.run(debug=True)
