from flask import request


def get_title_and_abstract():
    if request.method == "GET":
        title = request.args.get("title")
        abstract = request.args.get("abstract")
    else:
        title = request.json.get("title")
        abstract = request.json.get("abstract")
    return title, abstract


def format_score(score):
    return round(score, 3)
