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

    try:
        r = requests.post(
            api_url, json=json.dumps([input_data]), headers=headers, timeout=30
        )
    except requests.RequestException as e:
        print(f"Error tagging keywords: request failed: {e}")
        return []
    if r.status_code == 200:
        try:
            return r.json()[0] or []
        except (ValueError, IndexError, KeyError, TypeError) as e:
            print(f"Error tagging keywords: bad response: {e}")
            return []
    else:
        print(f"Error tagging keywords: {r.status_code} {r.text[:200]!r}")
        return []


def get_keywords_from_api(keyword_ids):
    if not keyword_ids:
        return []
    r = requests.get(
        "https://api.openalex.org/keywords?filter=id:{0}".format("|".join(keyword_ids)),
        timeout=30,
    )
    if r.status_code != 200:
        print(f"Error fetching keywords from API: {r.status_code}")
        return []
    return r.json().get("results", [])


KEYWORDS_UNAVAILABLE_NOTE = (
    "Keyword tagging is temporarily unavailable, so keywords is empty. "
    "Topics are unaffected. Keywords are being rebuilt."
)


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
    note = fields.Str()

    class Meta:
        ordered = True


class KeywordsMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    keywords = fields.Nested(KeywordsSchema, many=True)

    class Meta:
        ordered = True
