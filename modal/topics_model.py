"""Topic tagging for api.openalex.org/text*: the OpenAlex topic classifier (q8b_2m, a Qwen3-8B classifier over the 4,516 topics plus
"not classifiable", distilled from Opus 5.5 labels on 2M works; oxjobs #1485, shipped by #1531) served by vLLM in FP8 on one always-warm L4.

Deploy from desk:  modal deploy modal/topics_model.py        (app: openalex-text-topics; TOPICS_DEV=1 -> openalex-text-topics-dev, scales to zero)
Call:              POST <url>  {"title": ..., "abstract": ...}   Authorization: Bearer $TOPICS_MODEL_TOKEN
Returns:           {"topics": [{"id", "display_name", "score", "subfield", "field", "domain"}], "not_classifiable": bool, "model": str, "ms": int}
                   GET <url>/health -> {"ok": true, ...}

Same input text, HF right truncation at 384 tokens, FP8 weights and topic rule as the corpus pass that assigned every work's topics
(#1531 scratch/corpus/modal_corpus.py): text = "Title: ...", "Abstract: ..." (only when there is one, first 4,000 characters),
"Venue: ..."; the text endpoint only knows title + abstract, so the venue is sent as unknown. The model's top class "not classifiable"
-> no topics; otherwise up to 3 topics in probability order (never the not-classifiable class), each with the model's calibrated
probability as `score`, the field works carry. The topics themselves (ids, names, hierarchy) are the unchanged vocabulary.
Weights: Modal Volume openalex-text-topics /models/q8b_2m (causal/ + head.pt; modal/copy_topics_model.py).
Topic lookup: /lookup/topics_q8b_2m.json on the same Volume (class order -> topic objects; modal/build_topics_lookup.py).
"""
import modal, os, time, threading

DEV = os.environ.get("TOPICS_DEV") == "1"
app = modal.App("openalex-text-topics-dev" if DEV else "openalex-text-topics")
vol = modal.Volume.from_name("openalex-text-topics")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm>=0.10,<0.11", "transformers>=4.53,<4.56", "numpy")   # = the corpus scorer's image
         .pip_install("fastapi[standard]")
         .env({"HF_HOME": "/vol/hf", "VLLM_CACHE_ROOT": "/vol/cache/vllm", "TOKENIZERS_PARALLELISM": "false", "TOPICS_DEV": "1" if DEV else "0"}))
MODEL_NAME = "q8b_2m"
MODEL_DIR = f"/vol/models/{MODEL_NAME}"
LOOKUP = f"/vol/lookup/topics_{MODEL_NAME}.json"
NOT_CLASSIFIABLE = 4516   # class index after the 4,516 topics
MAX_TOKENS = 384          # as in training and the corpus pass
MAX_ABSTRACT_CHARS = 4000
MAX_TITLE_CHARS = 4000
N_TOPICS = 3
MIN_SCORE = float(os.environ.get("TOPICS_MIN_SCORE", "0"))   # floor for topics after the first; must equal the production write's rule


def text(title, abstract, venue=None):   # = the corpus pass's text(): the student's training input
    p = [f"Title: {(title or '').strip()[:MAX_TITLE_CHARS] or '(none)'}"]
    if (abstract or "").strip(): p.append(f"Abstract: {abstract.strip()[:MAX_ABSTRACT_CHARS]}")
    p.append(f"Venue: {(venue or '').strip() or '(unknown)'}")
    return "\n".join(p)


@app.cls(image=image, gpu="L4", volumes={"/vol": vol}, secrets=[modal.Secret.from_name("openalex-text-topics")],
         min_containers=0 if DEV else 1, max_containers=2, scaledown_window=1200, timeout=120, startup_timeout=900, memory=32768, cpu=4)
@modal.concurrent(max_inputs=8)   # requests queue inside one container (serialized by the lock) instead of booting a second one
class Topics:
    @modal.enter()
    def load(self):
        import json, torch, torch.nn as nn
        from vllm import LLM
        from transformers import AutoTokenizer
        t = time.time()
        self.topics = json.load(open(LOOKUP))["topics"]
        assert len(self.topics) == NOT_CLASSIFIABLE
        self.tok = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/causal")
        kw = dict(model=f"{MODEL_DIR}/causal", dtype="bfloat16", max_model_len=512, gpu_memory_utilization=0.85, enable_prefix_caching=False, quantization="fp8")
        try: self.llm = LLM(runner="pooling", convert="embed", override_pooler_config={"pooling_type": "LAST", "normalize": False}, **kw)
        except TypeError: self.llm = LLM(task="embed", override_pooler_config={"pooling_type": "LAST", "normalize": False}, **kw)
        sd = torch.load(f"{MODEL_DIR}/head.pt")
        self.head = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); self.head.load_state_dict(sd); self.head.eval()
        self.lock = threading.Lock()
        self.classify(["Warm-up"], ["A short abstract to warm the model."])
        vol.commit()   # persist the compile cache
        print(f"model {MODEL_NAME} + {len(self.topics)} topics ready in {time.time() - t:.0f}s", flush=True)

    def classify(self, titles, abstracts, venues=None, k=10):
        """Top-k class indices and probabilities per input (the corpus pass's Scorer.run, unbatched)."""
        import torch, numpy as np
        from vllm.inputs import TokensPrompt
        venues = venues or [None] * len(titles)
        ids = self.tok([text(t, a, v) for t, a, v in zip(titles, abstracts, venues)], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        with self.lock:
            outs = self.llm.embed([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
            H = torch.tensor(np.array([o.outputs.embedding for o in outs]), dtype=torch.float32, device="cuda")
            with torch.no_grad(): v, i = torch.softmax(self.head(H), 1).topk(k, 1)
        return i.cpu().numpy(), v.cpu().numpy()

    def topics_for(self, top, prob):
        if int(top[0]) == NOT_CLASSIFIABLE:
            return [], True
        out = []
        for c, p in zip(top, prob):
            c = int(c)
            if c == NOT_CLASSIFIABLE: continue
            if out and p < MIN_SCORE: break
            out.append({**self.topics[c], "score": round(float(p), 4)})
            if len(out) == N_TOPICS: break
        return out, False

    @modal.method()
    def score_rows(self, rows: list):
        """Validation: rows [{ti, ab, ve}] -> top-10 class indices and probabilities (lists), to compare with the corpus pass."""
        top, prob = self.classify([r.get("ti") for r in rows], [r.get("ab") for r in rows], [r.get("ve") for r in rows])
        return top.tolist(), prob.tolist()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        api = FastAPI()
        expected = f"Bearer {os.environ['TOPICS_MODEL_TOKEN']}"

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_NAME, "topics": len(self.topics)}

        @api.post("/")
        async def tag(request: Request):
            if request.headers.get("authorization", "") != expected:
                raise HTTPException(status_code=401, detail="bad token")
            body = await request.json()
            title, abstract = body.get("title"), body.get("abstract")
            if not ((title or "").strip() or (abstract or "").strip()):
                raise HTTPException(status_code=400, detail="title or abstract required")
            t = time.time()
            top, prob = self.classify([title], [abstract])
            topics, nc = self.topics_for(top[0], prob[0])
            return {"topics": topics, "not_classifiable": nc, "model": MODEL_NAME, "ms": int((time.time() - t) * 1000)}

        return api
