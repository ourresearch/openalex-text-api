import json
import os

import requests


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
