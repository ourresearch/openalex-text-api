from collections import OrderedDict

from flask import Flask

from combined import CombinedMessageSchema
from concepts import (
    get_concept_predictions,
    ConceptsMessageSchema,
    format_concepts,
    get_concepts_from_api,
)
from keywords import (
    get_keywords_predictions,
    get_keywords_from_api,
    format_keywords,
    KeywordsMessageSchema,
)
from topics import (
    get_topic_predictions,
    TopicsMessageSchema,
    format_topics,
    get_topics_from_api,
)
from utils import get_title_and_abstract
from validate import validate_input

app = Flask(__name__)
app.json.sort_keys = False


@app.route("/text", methods=["GET", "POST"])
def combined_view():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    concept_predictions = get_concept_predictions(title, abstract)
    concept_ids = [f"C{concept_id}" for concept_id, _ in concept_predictions]
    concepts_from_api = get_concepts_from_api(concept_ids)
    formatted_concepts = format_concepts(concept_predictions, concepts_from_api)

    keywords_predictions = get_keywords_predictions(title, abstract)
    keyword_ids = [
        f"keywords/{keyword['keyword_id']}" for keyword in keywords_predictions
    ]
    keywords_from_api = get_keywords_from_api(keyword_ids)
    formatted_keywords = format_keywords(keywords_predictions, keywords_from_api)

    topic_predictions = get_topic_predictions(title, abstract)
    topic_ids = [f"T{topic['topic_id']}" for topic in topic_predictions]
    topics_from_api = get_topics_from_api(topic_ids)
    formatted_topics = format_topics(topic_predictions, topics_from_api)

    result = OrderedDict()
    result["meta"] = {
        "keywords_count": len(formatted_keywords),
        "topics_count": len(formatted_topics),
        "concepts_count": len(formatted_concepts),
    }
    result["keywords"] = formatted_keywords
    result["topics"] = formatted_topics
    result["concepts"] = formatted_concepts
    message_schema = CombinedMessageSchema()
    print(result)
    return message_schema.dump(result)


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
    return message_schema.dump(result)


@app.route("/text/keywords", methods=["GET", "POST"])
def keywords():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    keyword_predictions = get_keywords_predictions(title, abstract)
    keyword_ids = [
        f"keywords/{keyword['keyword_id']}" for keyword in keyword_predictions
    ]
    keywords_from_api = get_keywords_from_api(keyword_ids)
    formatted_keywords = format_keywords(keyword_predictions, keywords_from_api)

    result = OrderedDict()
    result["meta"] = {
        "count": len(formatted_keywords),
    }
    result["results"] = formatted_keywords
    message_schema = KeywordsMessageSchema()
    return message_schema.dump(result)


@app.route("/text/topics", methods=["GET", "POST"])
def topics():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

    topic_predictions = get_topic_predictions(title, abstract)
    topic_ids = [f"T{topic['topic_id']}" for topic in topic_predictions]
    topics_from_api = get_topics_from_api(topic_ids)
    formatted_topics = format_topics(topic_predictions, topics_from_api)

    result = OrderedDict()
    result["meta"] = {
        "count": len(formatted_topics),
    }
    result["results"] = formatted_topics
    message_schema = TopicsMessageSchema()
    return message_schema.dump(result)


if __name__ == "__main__":
    app.run(debug=True)
