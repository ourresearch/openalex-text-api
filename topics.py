import functools
import json
import os

from marshmallow import Schema, fields
import requests

from utils import format_score


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
        return resp_data
    else:
        return []


def get_topics_from_api(topic_ids):
    r = requests.get(
        "https://api.openalex.org/topics?filter=id:{0}".format("|".join(topic_ids))
    )
    topics_from_api = r.json()["results"]
    return topics_from_api


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

    class Meta:
        ordered = True


class TopicsMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    results = fields.Nested(TopicsSchema, many=True)

    class Meta:
        ordered = True
