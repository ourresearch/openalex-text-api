from collections import OrderedDict

from flask import Flask, request

from concepts import (
    get_concept_predictions,
    ConceptsMessageSchema,
    format_concepts,
    get_concepts_from_api,
)
from topics import (
    get_topic_predictions,
    TopicsMessageSchema,
    format_topics,
    get_topics_from_api,
)
from validate import validate_input

app = Flask(__name__)


@app.route("/text/topics", methods=["GET", "POST"])
def topics():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    topic_predictions = get_topic_predictions(title, abstract)
    topics_from_api = get_topics_from_api(topic_predictions)

    formatted_topics = format_topics(topic_predictions, topics_from_api)

    result = OrderedDict()
    result["meta"] = {
        "count": len(formatted_topics),
    }
    result["results"] = formatted_topics
    message_schema = TopicsMessageSchema()
    return message_schema.dumps(result)


@app.route("/text/concepts", methods=["GET", "POST"])
def concepts():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    concept_predictions = get_concept_predictions(title, abstract)
    concept_ids = [f"C{concept_id}" for concept_id, _ in concept_predictions]
    concepts_from_api = get_concepts_from_api(concept_ids)
    formatted_concepts = format_concepts(concept_predictions, concepts_from_api)

    result = OrderedDict()
    result["meta"] = {
        "count": len(formatted_concepts),
    }
    result["results"] = formatted_concepts
    message_schema = ConceptsMessageSchema()
    return message_schema.dumps(result)


def get_title_and_abstract():
    if request.method == "GET":
        title = request.args.get("title")
        abstract = request.args.get("abstract")
    else:
        title = request.json.get("title")
        abstract = request.json.get("abstract")
    return title, abstract


if __name__ == "__main__":
    app.run(debug=True)
