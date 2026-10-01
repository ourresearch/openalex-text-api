"""Keyword tagging for api.openalex.org/text*: the #1322 Qwen3-4B student (v3fix, the model that tagged every work) served by vLLM on one always-warm L4.

Deploy from desk:  modal deploy modal/keywords_model.py        (app: openalex-text-keywords; KEYWORDS_DEV=1 -> openalex-text-keywords-dev)
Call:              POST <url>  {"title": ..., "abstract": ...}   Authorization: Bearer $KEYWORDS_MODEL_TOKEN
Returns:           {"keywords": [{"id": "keywords/<kid>", "display_name": str, "score": float, "raw": str}], "model": str, "vocab": str, "ms": int}
                   GET <url>/health -> {"ok": true, ...}

Same prompt, 560-token truncation, greedy decoding and per-keyword confidence as the corpus run (oxjobs #1322 scratch/god/corpus_v3/modal_corpus_v3.py),
then the same normalisation the works went through (#1322 vocab/build_vocab.py, #1465): drop keywords under the confidence cut (mean token log-prob
< -1.2), raw string -> kid0 -> plural fold -> synonym map -> vocabulary heading; keywords outside the vocabulary are dropped; hard-purged kinds
(document types etc.), discipline labels not in the title and country keywords without evidence in the text are dropped; duplicates after mapping
keep their first position and highest confidence. Score = exp(mean token log-prob), the `score` works carry. The text endpoint only knows title +
abstract, so venue is sent as unknown.
Weights: Modal Volume openalex-text-keywords /models/student_qwen4b_v3fix (modal/copy_model.py).
Vocabulary lookup: /lookup/<VOCAB>.json.gz on the same Volume, built from the production tables by modal/build_lookup.py.
"""
import modal, os, time, threading

DEV = os.environ.get("KEYWORDS_DEV") == "1"   # KEYWORDS_DEV=1 modal deploy ... -> app openalex-text-keywords-dev, scales to zero (for testing)
app = modal.App("openalex-text-keywords-dev" if DEV else "openalex-text-keywords")
vol = modal.Volume.from_name("openalex-text-keywords")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("vllm>=0.10", "transformers>=4.51,<5.0", "fastapi[standard]", "regex")
         .env({"HF_HOME": "/vol/hf", "CUDA_HOME": "/usr/local/cuda", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_CACHE_ROOT": "/vol/cache/vllm", "KEYWORDS_DEV": "1" if DEV else "0"}))  # compile cache persists on the Volume: cold start skips ~55 s of torch.compile
MODEL_DIR = "/vol/models/student_qwen4b_v3fix"
MODEL_NAME = "student_qwen4b_v3fix"
VOCAB = os.environ.get("KEYWORDS_VOCAB", "keywords_v2")   # lookup file name; build_lookup.py writes /lookup/<name>.json.gz
MAX_PROMPT_TOKENS = 560   # as in training and the corpus run
MIN_LP = -1.2             # the corpus confidence cut (build_vocab.py --min-lp -1.2)
MAX_INPUT_CHARS = 6000


def user_text(title, abstract):   # = #1322 god/prompt_v3.user_text, the student's training prompt
    return "\n".join([
        f"Title: {(title or '(none)').strip()[:MAX_INPUT_CHARS]}",
        f"Abstract: {(abstract or '(none)').strip()[:MAX_INPUT_CHARS]}",
        "Venue: (unknown)",
    ])


def keyword_lps(tok, token_ids, logprobs, special=frozenset()):
    """= #1322 modal_corpus_v3.keyword_lps: split on ';', average the sampled-token log-probs per keyword; special tokens skipped."""
    kws, lps, cur, cur_lp = [], [], "", []
    for tid, d in zip(token_ids, logprobs):
        if tid in special: continue
        piece = tok.decode([tid]); lp = d[tid].logprob if d and tid in d else 0.0
        parts = piece.split(";")
        cur += parts[0]; cur_lp.append(lp)
        for p in parts[1:]:
            if cur.strip(): kws.append(cur.strip()); lps.append(sum(cur_lp) / len(cur_lp))
            cur, cur_lp = p, []
    if cur.strip() and cur_lp: kws.append(cur.strip()); lps.append(sum(cur_lp) / len(cur_lp))
    return kws, lps


# kid0 = #1322 vocab/kid.py (the Databricks UDF openalex.common.keywords_v2_kid0), verbatim
import re, unicodedata
GREEK = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu", "π": "pi", "σ": "sigma", "τ": "tau", "ω": "omega",
                       "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ω": "omega"})
def canon(s):
    s = unicodedata.normalize('NFKC', s).lower().strip(); s = re.sub(r'[‐-―_/\-]', ' ', s); s = re.sub(r'\s+', ' ', s)
    return s.strip(' .,;:"\'')
def kid0(s):
    if s is None: return None
    c = canon(s)
    if any(ch.isalpha() and not unicodedata.name(ch, '').startswith(('LATIN', 'GREEK')) for ch in c):
        return re.sub(r'[\W_]+', '-', c).strip('-') or None
    a = unicodedata.normalize('NFKD', c.translate(GREEK)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', a).strip('-') or None


STOP = {'as', 'in', 'the', 'a', 'an', 'of', 'on', 'and', 'for', 'to', 'with', 'by', 'at', 'from', 'or', 'is'}


def malformed(raw):   # the string-only part of build_vocab.py's 'mal' rule
    import regex
    return ('�' in raw or not regex.match(r'^[\p{L}\p{N}(\["#.\']', raw) or raw.lower() in STOP or regex.fullmatch(r'[0-9]{1,4}', raw)
            or not regex.search(r'\p{L}', raw) or raw.count('(') != raw.count(')'))


class Vocab:
    def __init__(self, path):
        import gzip, json, regex
        d = json.load(gzip.open(path, 'rt'))
        self.version = d['version']
        self.alias = d['alias']            # kid0 -> heading kid (plural fold + synonym map), only where they differ
        self.names = d['names']            # heading kid -> display_name (the whole vocabulary)
        self.purge = set(d['purge'])       # heading kids always dropped (document types and other non-subjects)
        self.disc = set(d['disc'])         # discipline labels: kept only when the phrase is in the title
        self.country = {k: regex.compile(rx) for k, rx in d['country'].items()}   # country keyword -> evidence regex over title + abstract

    def map(self, kws, lps, title, abstract):
        text = f"{title or ''} {abstract or ''}"; lt = (title or '').lower()
        out, seen = [], {}
        for raw, lp in zip(kws, lps):
            if lp < MIN_LP or malformed(raw): continue
            k0 = kid0(raw)
            if not k0: continue
            kid = self.alias.get(k0, k0)
            if kid not in self.names or kid in self.purge: continue
            if kid in self.disc and kid.replace('-', ' ') not in lt: continue
            if kid in self.country and not self.country[kid].search(text): continue
            if kid in seen:
                seen[kid]['lp'] = max(seen[kid]['lp'], lp); continue
            seen[kid] = {'id': f'keywords/{kid}', 'display_name': self.names[kid], 'lp': lp, 'raw': raw}
            out.append(seen[kid])
        import math
        return [{'id': x['id'], 'display_name': x['display_name'], 'score': round(math.exp(x['lp']), 3), 'raw': x['raw']} for x in out]


@app.cls(image=image, gpu="L4", volumes={"/vol": vol}, secrets=[modal.Secret.from_name("openalex-text-keywords")],
         min_containers=0 if DEV else 1, max_containers=2, scaledown_window=1200, timeout=120, startup_timeout=900, memory=16384, cpu=4)  # startup_timeout: load + compile ≈ 130 s on an L4; without it the 120 s request timeout kills startup
@modal.concurrent(max_inputs=8)   # requests queue inside one container (serialized by the lock) instead of booting a second one
class Tagger:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        t = time.time()
        self.vocab = Vocab(f"/vol/lookup/{VOCAB}.json.gz")
        self.llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.9, enable_prefix_caching=False)
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.special = frozenset(self.tok.all_special_ids)
        self.sp = SamplingParams(temperature=0.0, max_tokens=192, stop=["\n"], logprobs=1)
        self.lock = threading.Lock()
        self.tag_one("Warm-up", "A short abstract to warm the model.")
        vol.commit()  # persist the compile cache
        print(f"model + vocabulary {self.vocab.version} ready in {time.time() - t:.0f}s", flush=True)

    def tag_one(self, title, abstract):
        ids = self.tok(user_text(title, abstract) + "\nKeywords:", add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOKENS]
        with self.lock:
            o = self.llm.generate([{"prompt_token_ids": ids}], self.sp, use_tqdm=False)[0].outputs[0]
        kws, lps = keyword_lps(self.tok, o.token_ids, o.logprobs, self.special)
        return self.vocab.map(kws, lps, title, abstract)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        api = FastAPI()
        expected = f"Bearer {os.environ['KEYWORDS_MODEL_TOKEN']}"

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_NAME, "vocab": self.vocab.version}

        @api.post("/")
        async def tag(request: Request):
            if request.headers.get("authorization", "") != expected:
                raise HTTPException(status_code=401, detail="bad token")
            body = await request.json()
            title, abstract = body.get("title"), body.get("abstract")
            if not (title or abstract):
                raise HTTPException(status_code=400, detail="title or abstract required")
            t = time.time()
            kws = self.tag_one(title, abstract)
            return {"keywords": kws, "model": MODEL_NAME, "vocab": self.vocab.version, "ms": int((time.time() - t) * 1000)}

        return api
