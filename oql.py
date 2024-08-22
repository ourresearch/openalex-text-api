import functools
import json
import os
import random

from marshmallow import Schema, fields
import requests


@functools.lru_cache(maxsize=64)
def get_openai_response(prompt):
    api_key = os.getenv("OPENAI_API_KEY")

    random_num = random.randint(0, 3)

    if random_num == 0:
        rand_obj = {
            "filters": [],
            "summarize_by": None,
            "sort_by": {
                "column_id": "display_name",
                "direction": "asc"
            },
            "return_columns": [
                "display_name",
                "publication_year",
                "type",
                "primary_location",
                "authors",
                "institutions",
                "topic",
                "oa_status",
                "cited_by_count"
            ]
        }
    elif random_num == 1:
        rand_obj = {
            "filters": [{
                "id": "br_tjBegE",
                "subjectEntity": "institutions",
                "type": "branch",
                "operator": "and",
                "column_id": "id",
                "value": "i137902535"
                },
                {"id": "br_wjVegT",
                "subjectEntity": "countries",
                "type": "branch",
                "operator": "and",
                "column_id": "id",
                "value": "CA"}],
            "summarize_by": "institutions",
            "sort_by": {
                "column_id": "display_name",
                "direction": "desc"
            },
            "return_columns": [
                "display_name",
                "type",
                "country_code",
                "ror"
            ]
        }
    else:
        rand_obj = {
            "filters": [],
            "summarize_by": "institutions",
            "sort_by": {
                "column_id": "display_name",
                "direction": "asc"
            },
            "return_columns": [
                "display_name",
                "type",
                "country_code",
                "ror"
            ]
        }
    
    return rand_obj

class FiltersSchema(Schema):
    id = fields.Str()
    subjectEntity = fields.Str()
    type = fields.Str()
    operator = fields.Str()
    column_id = fields.Str()
    value = fields.Raw()

    class Meta:
        ordered = True

class SortBySchema(Schema):
    column_id = fields.Str()
    direction = fields.Str()

    class Meta:
        ordered = True

class JsonObjectSchema(Schema):
    filters = fields.Nested(FiltersSchema, many=True)
    summarize_by = fields.Str(nullable=True)
    sort_by = fields.Nested(SortBySchema)
    return_columns = fields.List(fields.Str())

    class Meta:
        ordered = True
