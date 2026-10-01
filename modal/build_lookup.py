"""Build the vocabulary lookup the keyword tagger maps its output through (oxjobs #1465), from the production keyword tables, and put it on the
Modal Volume openalex-text-keywords at /lookup/<name>.json.gz. Rebuild whenever the vocabulary or synonym map changes, then redeploy (or restart) the app.

    uv run --with httpx python modal/build_lookup.py [--name keywords_v2] [--prefix openalex.common.keywords_v2] [--no-upload]

Needs the databricks CLI (auth) on desk. Tables read (all SELECT):
  <prefix>        vocabulary: kid, display_name           <prefix>_fold    kid0 -> kid (plural fold)
  <prefix>_synmap kid -> canon (synonym map)              <prefix>_purge   non-subject kinds (dual_use = false -> always dropped)
  <prefix>_country_kids  kid, cc, rx (Java regex of country names)
Same resolution as #1322 build_vocab.py: heading = COALESCE(synmap(COALESCE(fold(kid0), kid0)), fold(kid0), kid0), kept only if in the vocabulary."""
import argparse, gzip, json, os, re, subprocess, sys, time
import httpx

ap = argparse.ArgumentParser(); ap.add_argument('--name', default='keywords_v2'); ap.add_argument('--prefix', default='openalex.common.keywords_v2')
ap.add_argument('--warehouse', default='3996dc0a9b183ce3'); ap.add_argument('--no-upload', action='store_true'); a = ap.parse_args()
P = a.prefix


def api(method, path, body=None):
    cmd = ["databricks", "api", method, path] + (["--json", json.dumps(body)] if body is not None else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0: raise RuntimeError(out.stderr[:800])
    return json.loads(out.stdout) if out.stdout.strip() else {}


def sql(q):
    st = api("post", "/api/2.0/sql/statements", {"warehouse_id": a.warehouse, "statement": q, "wait_timeout": "30s", "disposition": "EXTERNAL_LINKS", "format": "JSON_ARRAY"})
    while st["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(3); st = api("get", f"/api/2.0/sql/statements/{st['statement_id']}")
    if st["status"]["state"] != "SUCCEEDED": raise RuntimeError(json.dumps(st["status"])[:1000])
    rows = []
    with httpx.Client(timeout=300) as h:
        for i in range(st["manifest"]["total_chunk_count"]):
            ch = api("get", f"/api/2.0/sql/statements/{st['statement_id']}/result/chunks/{i}")
            for link in ch.get("external_links", []):
                for attempt in range(5):   # presigned downloads sometimes reset
                    try: rows += h.get(link["external_link"]).json(); break
                    except httpx.HTTPError:
                        if attempt == 4: raise
                        time.sleep(3 * (attempt + 1))
    return rows


t0 = time.time()
names = {k: n for k, n in sql(f"SELECT kid, display_name FROM {P}")}
alias = {k: h for k, h in sql(f"""WITH k AS (SELECT kid0 AS k FROM {P}_fold WHERE kid0 <> kid UNION SELECT kid FROM {P}_synmap),
    r AS (SELECT k.k, COALESCE(sy.canon, fo.kid, k.k) AS h FROM k LEFT JOIN {P}_fold fo ON fo.kid0 = k.k LEFT JOIN {P}_synmap sy ON sy.kid = COALESCE(fo.kid, k.k))
    SELECT r.k, r.h FROM r JOIN {P} v ON v.kid = r.h WHERE r.k <> r.h""")}
purge = sorted({k for (k,) in sql(f"SELECT DISTINCT kid FROM {P}_purge WHERE NOT dual_use")})


def java_to_py(rx):   # \Q...\E literal blocks -> escaped text (the `regex` module has no \Q)
    return re.sub(r'\\Q(.*?)\\E', lambda m: re.escape(m.group(1)), rx)


country = {}
for k, rx in sql(f"SELECT kid, rx FROM {P}_country_kids"):
    country[k] = f"(?:{country[k]})|(?:{java_to_py(rx)})" if k in country else java_to_py(rx)
# discipline labels, kept only when the phrase is in the title (= build_vocab.py DISC; that rule also exempts books and reference entries)
disc = """medicine biology physics chemistry computer-science engineering psychology economics history education literature literary-criticism epidemiology
theology geography public-health physiology sociology philosophy anthropology archaeology linguistics law political-science statistics ecology geology astronomy botany zoology
genetics immunology microbiology pharmacology neuroscience nursing agriculture management business art music religion humanities social-sciences natural-sciences life-sciences
earth-sciences environmental-science materials-science biochemistry molecular-biology cell-biology oncology cardiology psychiatry pediatrics surgery dentistry veterinary-medicine
architecture finance accounting marketing sciences science technology health research""".split()
out = {'version': f"{a.name}@{time.strftime('%Y-%m-%dT%H:%M')}", 'names': names, 'alias': alias, 'purge': purge, 'disc': disc, 'country': country}
import tempfile
path = os.path.join(tempfile.mkdtemp(), f'{a.name}.json.gz')
with gzip.open(path, 'wt') as f: json.dump(out, f, ensure_ascii=False)
print(f"{len(names):,} headings, {len(alias):,} aliases, {len(purge)} purged, {len(country)} country keywords -> {path} ({os.path.getsize(path) / 1e6:.0f} MB) in {time.time() - t0:.0f}s")
if not a.no_upload:
    subprocess.run(["modal", "volume", "put", "--force", "openalex-text-keywords", path, f"/lookup/{a.name}.json.gz"], check=True)
    print(f"uploaded to openalex-text-keywords:/lookup/{a.name}.json.gz")
