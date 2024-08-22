from flask import jsonify


def validate_input(title, abstract):
    combined_text_minimum = 20
    combined_text_limit = 2000
    if not title:
        return (
            jsonify(
                {
                    "error": "Add a title param or abstract param (optional) to get keywords, topics, etc. Example: "
                    "https://api.openalex.org/text?title=Phosphates%20as%20Assisting%20Groups%20in%20Glycan%20Synthesis"
                }
            ),
            400,
        )

    abstract = abstract or ""

    combined_text = title + " " + abstract
    if len(combined_text) > combined_text_limit:
        return (
            jsonify(
                {
                    "error": f"The combined length of title and abstract must not exceed {combined_text_limit} characters"
                }
            ),
            400,
        )
    elif len(combined_text) < combined_text_minimum:
        return (
            jsonify(
                {
                    "error": f"The combined length of title and abstract must be at least {combined_text_minimum} characters"
                }
            ),
            400,
        )
    return None

def validate_natural_language(natural_language_text):
    text_minimum = 5
    text_limit = 300
    if not natural_language_text:
        return (
            jsonify(
                {
                    "error": "There is not enough information to return OQL. Please give more details"
                }
            ),
            400,
        )

    if len(natural_language_text.strip()) > text_limit:
        return (
            jsonify(
                {
                    "error": f"The length of the input must not exceed {text_limit} characters"
                }
            ),
            400,
        )
    elif len(natural_language_text.strip()) < text_minimum:
        return (
            jsonify(
                {
                    "error": f"The length of the input must be at least {text_minimum} characters"
                }
            ),
            400,
        )
    return None
