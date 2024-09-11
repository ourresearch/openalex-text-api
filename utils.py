from flask import request


def get_title_and_abstract():
    if request.method == "GET":
        title = request.args.get("title")
        abstract = request.args.get("abstract")
    else:
        title = request.json.get("title")
        abstract = request.json.get("abstract")
    return title, abstract

def get_natural_language_text():
    if request.method == "GET":
        natural_language_text = request.args.get("natural_language")
    else:
        natural_language_text = request.json.get("natural_language")
    return natural_language_text

def get_related_to_text():
    if request.method == "GET":
        text_input = request.args.get("text")
    else:
        text_input = request.json.get("text")
    return text_input

def format_score(score):
    return round(score, 3)
