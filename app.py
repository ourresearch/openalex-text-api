from collections import OrderedDict

from flask import Flask

from combined import CombinedMessageSchema
from keywords import (
    get_keywords_predictions,
    get_keywords_from_api,
    format_keywords,
    KeywordsMessageSchema,
    KEYWORDS_UNAVAILABLE_NOTE,
)
from topics import (
    get_topic_predictions,
    TopicsMessageSchema,
    format_topics,
    get_topics_from_api,
)

from oql import(
    get_openai_response
)

from related_to_text import(
    get_similar_works,
    get_similar_authors,
    connect_to_db
)

from utils import get_title_and_abstract, get_natural_language_text, get_related_to_text
from validate import validate_input, validate_natural_language

app = Flask(__name__)
app.json.sort_keys = False


@app.route("/text", methods=["GET", "POST"])
def combined_view():
    title, abstract = get_title_and_abstract()

    invalid_response = validate_input(title, abstract)
    if invalid_response:
        return invalid_response

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
    }
    if not formatted_keywords:
        result["meta"]["note"] = KEYWORDS_UNAVAILABLE_NOTE
    result["keywords"] = formatted_keywords
    result["primary_topic"] = formatted_topics[0] if formatted_topics else None
    result["topics"] = formatted_topics
    message_schema = CombinedMessageSchema()
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
    if not formatted_keywords:
        result["meta"]["note"] = KEYWORDS_UNAVAILABLE_NOTE
    result["keywords"] = formatted_keywords
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
    result["primary_topic"] = formatted_topics[0] if formatted_topics else None
    result["topics"] = formatted_topics
    message_schema = TopicsMessageSchema()
    return message_schema.dump(result)

@app.route("/text/oql", methods=["GET", "POST"])
def get_oql_json_object():
    natural_language_text = get_natural_language_text()

    invalid_response = validate_natural_language(natural_language_text)
    if invalid_response:
        return invalid_response
    
    openai_response = get_openai_response(natural_language_text.strip())
    return openai_response

@app.route("/text/related-works", methods=["GET", "POST"])
def get_works_related_to_text():
    related_to_text = get_related_to_text()

    conn = connect_to_db()
    works_list = get_similar_works(conn, related_to_text, 0.35, topK = 1000)
    conn.close()
    
    return works_list

@app.route("/text/related-authors", methods=["GET", "POST"])
def get_authors_related_to_text():
    related_to_text = get_related_to_text()

    conn = connect_to_db()
    authors_list = get_similar_authors(conn, related_to_text, 0.5, topK = 5000)
    conn.close()
    
    return authors_list


if __name__ == "__main__":
    app.run(debug=True)
