import os
import re
import unicodedata

from marshmallow import Schema, fields
import requests

from utils import format_score


# The keyword model: the OpenAlex keyword tagger (a Qwen3-4B student distilled from Opus 5.5 indexer
# labels, oxjobs #1322) served on Modal from modal/keywords_model.py. Same model and prompt as the
# keywords the works carry, so text submitted here gets the same vocabulary.
MODEL_URL = os.getenv("KEYWORDS_MODEL_URL")
MODEL_TOKEN = os.getenv("KEYWORDS_MODEL_TOKEN")


def get_keywords_predictions(title, abstract):
    """Returns [{keyword, rank, score}] best first, or [] on any failure (the endpoint never 500s over keywords)."""
    if not MODEL_URL:
        print("Error tagging keywords: KEYWORDS_MODEL_URL is not set")
        return []
    try:
        r = requests.post(
            MODEL_URL,
            json={"title": title, "abstract": abstract},
            headers={"Authorization": f"Bearer {MODEL_TOKEN}"},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Error tagging keywords: request failed: {e}")
        return []
    if r.status_code != 200:
        print(f"Error tagging keywords: {r.status_code} {r.text[:200]!r}")
        return []
    try:
        return r.json().get("keywords") or []
    except (ValueError, AttributeError) as e:
        print(f"Error tagging keywords: bad response: {e}")
        return []


GREEK = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda",
                       "μ": "mu", "π": "pi", "σ": "sigma", "τ": "tau", "ω": "omega", "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ω": "omega"})


def keyword_slug(keyword):
    """keywords/<slug> id convention (today's entity: 'Beta-lactamase' -> beta-lactamase): Greek letters spelled out,
    ASCII-folded, lower-case, runs of non-alphanumerics -> '-'."""
    s = unicodedata.normalize("NFKD", keyword.translate(GREEK)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def get_keywords_from_api(keyword_ids):
    if not keyword_ids:
        return []
    try:
        r = requests.get(
            "https://api.openalex.org/keywords?filter=id:{0}&per-page=50".format("|".join(keyword_ids)),
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"Error fetching keywords from API: {e}")
        return []
    if r.status_code != 200:
        print(f"Error fetching keywords from API: {r.status_code}")
        return []
    return r.json().get("results", [])


KEYWORDS_UNAVAILABLE_NOTE = (
    "Keyword tagging is temporarily unavailable, so keywords is empty. "
    "Topics are unaffected."
)
KEYWORDS_PROVISIONAL_NOTE = (
    "Some keyword ids are provisional: they follow the keywords/<slug> convention but do not "
    "resolve in the keywords API until the rebuilt keyword entity ships."
)


def format_keywords(keyword_predictions, keywords_from_api):
    """Model order; id = keywords/<slug>; display_name from the API entity when the id resolves, else the model's string."""
    by_id = {k["id"]: k for k in keywords_from_api}
    ordered_keywords = []
    seen = set()
    for keyword in keyword_predictions:
        slug = keyword_slug(keyword["keyword"])
        if not slug or slug in seen:
            continue
        seen.add(slug)
        full_id = f"https://openalex.org/keywords/{slug}"
        api_keyword = by_id.get(full_id)
        ordered_keywords.append({
            "id": full_id,
            "display_name": api_keyword["display_name"] if api_keyword else keyword["keyword"],
            "score": format_score(keyword["score"]),
            "resolved": api_keyword is not None,
        })
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
