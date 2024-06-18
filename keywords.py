import json
import os

from marshmallow import Schema, fields
import requests

from topics import get_topic_predictions
from utils import format_score


def get_keywords_predictions(title, abstract):
    api_url = "https://qapir74yac.execute-api.us-east-1.amazonaws.com/api/"
    api_key = os.getenv("SAGEMAKER_API_KEY")
    headers = {"X-API-Key": api_key}

    topic_predictions = get_topic_predictions(title, abstract)
    topic_ids = [topic["topic_id"] for topic in topic_predictions]
    input_data = {
        "title": title,
        "abstract_inverted_index": abstract,
        "inverted": False,
        "topics": topic_ids,
    }

    r = requests.post(api_url, json=json.dumps([input_data]), headers=headers)
    if r.status_code == 200:
        response_json = r.json()
        resp_data = response_json[0]
        return resp_data
    else:
        print(f"Error tagging keywords: {r.status_code}")
        return []


def get_keywords_from_api(keyword_ids):
    r = requests.get(
        "https://api.openalex.org/keywords?filter=id:{0}".format("|".join(keyword_ids))
    )
    keywords_from_api = r.json()["results"]
    return keywords_from_api


def format_keywords(keyword_predictions, keywords_from_api):
    ordered_keywords = []
    for keyword in keyword_predictions:
        for api_keyword in keywords_from_api:
            if (
                api_keyword["id"]
                == f"https://openalex.org/keywords/{keyword['keyword_id']}"
            ):
                api_keyword["score"] = format_score(keyword["score"])
                ordered_keywords.append(api_keyword)
                break
    return ordered_keywords


class KeywordsSchema(Schema):
    id = fields.Str()
    display_name = fields.Str()
    score = fields.Float()

    class Meta:
        ordered = True


class MetaSchema(Schema):
    count = fields.Int()

    class Meta:
        ordered = True


class KeywordsMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    keywords = fields.Nested(KeywordsSchema, many=True)

    class Meta:
        ordered = True
