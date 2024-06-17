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
