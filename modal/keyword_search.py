"""Keyword search by meaning (oxjobs #1464): a short query ("antibacterial resistance", "heart attack") -> the vocabulary keywords that
name that concept, ranked. Called by openalex-elastic-api for GET /keywords?search.semantic=..., which then fetches the keyword docs from ES.

Index: every keyword's display name plus its synonym aliases (the synonym map's merged variant strings, e.g. "heart attack" for
myocardial-infarction), embedded with BAAI/bge-m3 (fp16, L2-normalised), all held on the GPU. A keyword's similarity = max over its name and
aliases; aliases farther than ALIAS_FLOOR from their keyword's name are dropped. Rank score = similarity + ALPHA * log10(works_count), so the
head keyword beats long-tail near-duplicates ("heart disease classification"). Eval (#1464, 289 paraphrase queries, Opus-judged blind): the
first result names the searched concept 58% of the time vs 30% for display-name text search, and never returns nothing (text search: 53%).

Build the index (re-run when the vocabulary or synonym map changes), after putting /vol/src/vocab.jsonl {kid, name, n} and
/vol/src/aliases.jsonl {alias, kid} on the volume:
    modal run modal/keyword_search.py::build
Serve:  modal deploy modal/keyword_search.py          (KEYWORD_SEARCH_DEV=1 for the dev app, which scales to zero)
    POST /search  {"query": "...", "k": 25}  Authorization: Bearer $KEYWORD_SEARCH_TOKEN
    -> {"results": [{"id", "display_name", "score", "similarity"}], "model", "vocab", "ms"}"""
import json, math, os, threading, time
import modal

DEV = os.environ.get("KEYWORD_SEARCH_DEV") == "1"
MODEL = "BAAI/bge-m3"
ALPHA = 0.06          # popularity weight, fit on half the #1464 eval, reported on the other half
ALIAS_FLOOR = 0.5     # cosine(alias, keyword name) below this -> alias dropped
CANDIDATES = 400      # rows taken from the matmul before folding aliases into their keyword and re-ranking
app = modal.App("openalex-keyword-search-dev" if DEV else "openalex-keyword-search")
vol = modal.Volume.from_name("openalex-keyword-search", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "sentence-transformers>=3.0", "numpy", "fastapi[standard]")
         .env({"HF_HOME": "/vol/hf"}))


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=3600, memory=65536, cpu=8)
def build_index():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    t = time.time()
    vocab = [json.loads(l) for l in open("/vol/src/vocab.jsonl")]
    pos = {r["kid"]: i for i, r in enumerate(vocab)}
    aliases = [a for a in (json.loads(l) for l in open("/vol/src/aliases.jsonl")) if a["kid"] in pos]
    m = SentenceTransformer(MODEL, device="cuda"); m.half()
    enc = lambda xs: m.encode(xs, batch_size=2048, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    N = enc([r["name"] for r in vocab])
    A = enc([a["alias"] for a in aliases])
    owner = np.array([pos[a["kid"]] for a in aliases], dtype=np.int32)
    keep = (A.astype(np.float32) * N[owner].astype(np.float32)).sum(1) >= ALIAS_FLOOR
    M = np.concatenate([N, A[keep]]); row_kw = np.concatenate([np.arange(len(vocab), dtype=np.int32), owner[keep]])
    os.makedirs("/vol/index", exist_ok=True)
    np.save("/vol/index/vectors.npy", M); np.save("/vol/index/row_keyword.npy", row_kw)
    with open("/vol/index/keywords.jsonl", "w") as f:
        for r in vocab: f.write(json.dumps({"kid": r["kid"], "name": r["name"], "n": r["n"]}) + "\n")
    meta = {"model": MODEL, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "keywords": len(vocab),
            "aliases": int(keep.sum()), "aliases_dropped": int((~keep).sum()), "alpha": ALPHA, "alias_floor": ALIAS_FLOOR}
    json.dump(meta, open("/vol/index/meta.json", "w"))
    vol.commit()
    return {**meta, "seconds": round(time.time() - t)}


@app.local_entrypoint()
def build():
    print(build_index.remote())


@app.cls(image=image, gpu=["T4", "L4", "A10G"], volumes={"/vol": vol}, secrets=[modal.Secret.from_name("openalex-keyword-search")],
         min_containers=0 if DEV else 1, max_containers=2, scaledown_window=1200, timeout=60, startup_timeout=600, memory=16384, cpu=4)
@modal.concurrent(max_inputs=16)
class Search:
    @modal.enter()
    def load(self):
        import numpy as np, torch
        from sentence_transformers import SentenceTransformer
        t = time.time()
        self.meta = json.load(open("/vol/index/meta.json"))
        self.kw = [json.loads(l) for l in open("/vol/index/keywords.jsonl")]
        self.M = torch.from_numpy(np.load("/vol/index/vectors.npy")).cuda()
        self.row_kw = torch.from_numpy(np.load("/vol/index/row_keyword.npy")).long().cuda()
        self.prior = torch.tensor([ALPHA * math.log10(max(r["n"], 1)) for r in self.kw], device="cuda")
        self.model = SentenceTransformer(MODEL, device="cuda"); self.model.half()
        self.lock = threading.Lock()
        self.search("warm up", 5)
        print(f"index {self.meta['keywords']:,} keywords + {self.meta['aliases']:,} aliases ready in {time.time() - t:.0f}s", flush=True)

    def search(self, query, k):
        import torch
        with self.lock:
            q = self.model.encode([query], normalize_embeddings=True, convert_to_tensor=True).half()
            sim, rows = torch.topk((q @ self.M.T)[0], CANDIDATES)
            kws = self.row_kw[rows]
        best = {}
        for s, i in zip(sim.float().tolist(), kws.tolist()):   # rows come sorted, so the first row seen per keyword is its max similarity
            best.setdefault(i, s)
        ids = torch.tensor(list(best), device="cuda"); sims = torch.tensor(list(best.values()), device="cuda")
        score = sims + self.prior[ids]
        order = torch.argsort(score, descending=True)[:k].tolist()
        ids, sims, score = ids.tolist(), sims.tolist(), score.tolist()
        return [{"id": f"https://openalex.org/keywords/{self.kw[ids[j]]['kid']}", "display_name": self.kw[ids[j]]["name"],
                 "score": round(score[j], 4), "similarity": round(sims[j], 4)} for j in order]

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        api = FastAPI()
        expected = f"Bearer {os.environ['KEYWORD_SEARCH_TOKEN']}"

        @api.get("/health")
        def health():
            return {"ok": True, **self.meta}

        @api.post("/search")
        async def search(request: Request):
            if request.headers.get("authorization", "") != expected:
                raise HTTPException(status_code=401, detail="bad token")
            body = await request.json()
            query = (body.get("query") or "").strip()[:500]
            if not query:
                raise HTTPException(status_code=400, detail="query required")
            k = max(1, min(int(body.get("k") or 25), 200))
            t = time.time()
            res = self.search(query, k)
            return {"results": res, "model": MODEL, "vocab": self.meta["built"], "ms": int((time.time() - t) * 1000)}

        return api
