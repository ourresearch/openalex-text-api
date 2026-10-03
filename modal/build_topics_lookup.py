"""Build the topic lookup the topic model service uses: the model's class order (modal/topics_class_order.json, 4,516 topic ids)
with each topic's id, display_name, subfield, field and domain exactly as the topics API serves them, so the service answers in the
shape /text/topics always had without an API call per request. The topics themselves never change with the model.

python3 modal/build_topics_lookup.py            # writes /tmp/topics_q8b_2m.json and uploads it to Volume openalex-text-topics /lookup/
"""
import json, os, subprocess, requests

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "topics_q8b_2m"


def main():
    order = json.load(open(f"{HERE}/topics_class_order.json"))
    by_id, cursor = {}, "*"
    while cursor:
        r = requests.get("https://api.openalex.org/topics", timeout=60, params={
            "per-page": 200, "cursor": cursor, "select": "id,display_name,subfield,field,domain",
            "api_key": os.environ.get("OPENALEX_API_KEY", "")})
        r.raise_for_status(); d = r.json()
        for t in d["results"]:
            by_id[t["id"].rsplit("/", 1)[-1]] = t
        cursor = d["meta"].get("next_cursor") if d["results"] else None
    missing = [t for t in order if t not in by_id]
    assert not missing, f"{len(missing)} class topics missing from the API: {missing[:5]}"
    topics = [{k: by_id[t][k] for k in ("id", "display_name", "subfield", "field", "domain")} for t in order]
    out = f"/tmp/{NAME}.json"
    json.dump({"version": NAME, "topics": topics}, open(out, "w"), ensure_ascii=False)
    print(f"{len(topics)} topics in class order ({len(by_id)} in the API) -> {out}")
    subprocess.run(["modal", "volume", "put", "--force", "openalex-text-topics", out, f"/lookup/{NAME}.json"], check=True)


if __name__ == "__main__":
    main()
