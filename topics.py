import functools
import json
import os

from marshmallow import Schema, fields
import requests

from utils import format_score


# The topic model: the OpenAlex topic classifier (q8b_2m, a Qwen3-8B classifier distilled from Opus 5.5 labels, oxjobs #1485)
# served on Modal from modal/topics_model.py. Same model, input and topic rule as the topics the works carry. It answers with
# the topics already resolved to the (unchanged) vocabulary, so no topics API call is needed. Until TOPICS_MODEL_URL is set
# (release day, oxjobs #1531), the endpoint keeps the previous model on SageMaker below; unset it to roll back.
MODEL_URL = os.getenv("TOPICS_MODEL_URL")
MODEL_TOKEN = os.getenv("TOPICS_MODEL_TOKEN")

TOPICS_UNAVAILABLE_NOTE = "Topic tagging is temporarily unavailable, so topics is empty."
# ?version=1: the previous classifier, kept for people reproducing analyses made with the old topics (Jason, 2026-10-05).
PREVIOUS_MODEL_UNTIL = "2027-01-13"
PREVIOUS_MODEL_NOTE = f"version=1: the previous topic classifier, available until {PREVIOUS_MODEL_UNTIL}."
NOT_CLASSIFIABLE_NOTE = "The topic model found no topic that fits this text, so topics is empty."


def tag_topics(title, abstract, version=None):
    """Formatted topics best first plus a note when topics is empty for a reason the caller should know ("" otherwise).
    version="1" selects the previous classifier (with a note saying so). Never raises: the endpoint does not 500 over topics."""
    if version == "1":
        try:
            return previous_model_topics(title, abstract), PREVIOUS_MODEL_NOTE
        except Exception as e:
            print(f"Error tagging topics with the previous model: {e}")
            return [], f"{PREVIOUS_MODEL_NOTE} {TOPICS_UNAVAILABLE_NOTE}"
    if not MODEL_URL:
        return previous_model_topics(title, abstract), ""
    try:
        r = requests.post(
            MODEL_URL,
            json={"title": title, "abstract": abstract},
            headers={"Authorization": f"Bearer {MODEL_TOKEN}"},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"Error tagging topics: {r.status_code} {r.text[:200]!r}")
            return [], TOPICS_UNAVAILABLE_NOTE
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Error tagging topics: {e}")
        return [], TOPICS_UNAVAILABLE_NOTE
    topics = [{**t, "score": format_score(t["score"])} for t in body.get("topics") or []]
    return topics, (NOT_CLASSIFIABLE_NOTE if body.get("not_classifiable") else "")


# The previous model (multilingual BERT on SageMaker): ?version=1, and the default while TOPICS_MODEL_URL is unset.
def previous_model_topics(title, abstract):
    predictions = get_topic_predictions(title, abstract)
    topic_ids = [f"T{topic['topic_id']}" for topic in predictions]
    return format_topics(predictions, get_topics_from_api(topic_ids))


@functools.lru_cache(maxsize=64)
def get_topic_predictions(title, abstract):
    api_url = "https://5gl84dua69.execute-api.us-east-1.amazonaws.com/api/"
    api_key = os.getenv("SAGEMAKER_API_KEY")
    headers = {"X-API-Key": api_key}
    data = {
        "title": title,
        "abstract_inverted_index": abstract,
        "journal_display_name": "",
        "referenced_works": [],
        "inverted": False,
    }

    r = requests.post(api_url, json=json.dumps([data], sort_keys=True), headers=headers)
    if r.status_code == 200:
        response_json = r.json()
        resp_data = response_json[0]
        # The model answers topic_id -1 / score 0 when it declines to classify
        # the text; that is "no topics", not a topic to look up.
        return [t for t in resp_data if t.get("topic_id", -1) >= 0]
    else:
        print(f"Error tagging topics: {r.status_code} {r.text[:200]!r}")
        return []


def get_topics_from_api(topic_ids):
    if not topic_ids:
        return []
    r = requests.get(
        "https://api.openalex.org/topics?filter=id:{0}".format("|".join(topic_ids)),
        timeout=30,
    )
    if r.status_code != 200:
        print(f"Error fetching topics from API: {r.status_code}")
        return []
    return r.json().get("results", [])


def format_topics(topic_predictions, topics_from_api):
    ordered_topics = []
    for topic in topic_predictions:
        for api_topic in topics_from_api:
            if api_topic["id"] == f"https://openalex.org/T{topic['topic_id']}":
                api_topic["score"] = format_score(topic["topic_score"])
                ordered_topics.append(api_topic)
                break
    return ordered_topics


class TopicHierarchySchema(Schema):
    id = fields.Str()
    display_name = fields.Str()

    class Meta:
        ordered = True


class TopicsSchema(Schema):
    id = fields.Str()
    display_name = fields.Str()
    score = fields.Float()
    subfield = fields.Nested(TopicHierarchySchema)
    field = fields.Nested(TopicHierarchySchema)
    domain = fields.Nested(TopicHierarchySchema)

    class Meta:
        ordered = True


class MetaSchema(Schema):
    count = fields.Int()
    note = fields.Str()

    class Meta:
        ordered = True


class TopicsMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    primary_topic = fields.Nested(TopicsSchema)
    topics = fields.Nested(TopicsSchema, many=True)

    class Meta:
        ordered = True
