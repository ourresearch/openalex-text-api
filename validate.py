from flask import jsonify


def validate_input(title, abstract, combined_text_limit=10000):
    if not title or not abstract:
        return jsonify({"error": "A title or abstract must be provided"}), 400

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
    return None
