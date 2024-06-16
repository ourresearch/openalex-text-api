import json
import os

from marshmallow import Schema, fields
import requests


def get_concept_predictions(title, abstract):
    api_url = "https://l7a8sw8o2a.execute-api.us-east-1.amazonaws.com/api/"
    api_key = os.getenv("SAGEMAKER_API_KEY")
    headers = {"X-API-Key": api_key}
    data_list = [
        {
            "title": title,
            "doc_type": "",
            "journal": "",
            "abstract": abstract,
            "inverted_abstract": False,
            "paper_id": 1234,
        }
    ]

    r = requests.post(api_url, json=json.dumps(data_list), headers=headers)
    if r.status_code == 200:
        response_json = r.json()
        resp_data = response_json[0]
        concepts_combined = list(zip(resp_data["tag_ids"], resp_data["scores"]))
        concepts_ordered = sorted(concepts_combined, key=lambda x: x[1], reverse=True)
        concepts_without_0 = [x for x in concepts_ordered if x[1] > 0]
        return concepts_without_0
    else:
        print(f"Error tagging concepts: {r.status_code}")
        return []


def get_concepts_from_api(concept_ids):
    r = requests.get(
        "https://api.openalex.org/concepts?filter=ids.openalex:{0}".format(
            "|".join(concept_ids)
        )
    )
    concepts_from_api = r.json()["results"]
    return concepts_from_api


def format_concepts(concept_predictions, concepts_from_api):
    ordered_concepts = []
    for concept_id, concept_score in concept_predictions:
        for api_concept in concepts_from_api:
            if api_concept["id"] == f"https://openalex.org/C{concept_id}":
                api_concept["score"] = concept_score
                ordered_concepts.append(api_concept)
                break
    return ordered_concepts


class AncestorsSchema(Schema):
    id = fields.Str()
    display_name = fields.Str()
    level = fields.Int()

    class Meta:
        ordered = True


class ConceptsSchema(Schema):
    id = fields.Str()
    display_name = fields.Str()
    score = fields.Float()
    level = fields.Int()
    description = fields.Str()
    ancestors = fields.List(fields.Nested(AncestorsSchema))

    class Meta:
        ordered = True


class MetaSchema(Schema):
    count = fields.Int()

    class Meta:
        ordered = True


class ConceptsMessageSchema(Schema):
    meta = fields.Nested(MetaSchema)
    results = fields.Nested(ConceptsSchema, many=True)

    class Meta:
        ordered = True
