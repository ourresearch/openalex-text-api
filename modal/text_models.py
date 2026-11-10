"""Topics and keywords for api.openalex.org/text* on ONE always-warm L4 (oxjobs #2136): the q8b_2m topic classifier (topics_model.py)
and the #1322 keyword student (keywords_model.py) as two vLLM engines in one container, so /text pays for one GPU instead of two.
Each model keeps the vLLM it ran alone: topics 0.10.2 in this process (= the #1531 corpus pass), keywords 0.22.1 in its own Python
(/opt/kw, keywords_worker.py). Freshly compiled on two machines, 0.10.2 gave the same keyword lists for 2,602 of 3,000 works and 0.22.1
for 3,000 of 3,000; accuracy against the #1322 gold is the same on both (F1 0.436). The compiled graphs are kept on the cache Volume, so a
restarted container loads them and answers as before.
Deploy from desk:  modal deploy modal/text_models.py        (app: openalex-text-models; TEXT_MODELS_DEV=1 -> openalex-text-models-dev, scales to zero)
Call:              POST <url>/topics    {"title": ..., "abstract": ...}   Authorization: Bearer $TOPICS_MODEL_TOKEN    (= topics_model.py's POST /)
                   POST <url>/keywords  {"title": ..., "abstract": ...}   Authorization: Bearer $KEYWORDS_MODEL_TOKEN  (= keywords_model.py's POST /)
                   GET <url>/health
Text, truncation, decoding and post-processing come from topics_model.py and keywords_model.py unchanged; only the serving moved.
GPU memory (L4, 22 GiB; the apps' own vLLM startup logs): topics 7.64 GiB FP8 weights + ~1.4 GiB activations and CUDA graphs,
keywords 7.56 GiB bf16 weights + about the same + its KV cache. Topics loads first: its FP8 conversion briefly needs the bf16 weights.
"""
import modal, os, time, threading
import topics_model as tm, keywords_model as km

DEV = os.environ.get("TEXT_MODELS_DEV") == "1"
app = modal.App("openalex-text-models-dev" if DEV else "openalex-text-models")
cache = modal.Volume.from_name("openalex-text-models", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")   # = keywords_model.py's base
         .pip_install("vllm==0.10.2", "transformers>=4.53,<4.56", "numpy", "regex", "psutil")
         .pip_install("fastapi[standard]")
         .run_commands("python -m venv /opt/kw", "/opt/kw/bin/pip install vllm==0.22.1 'transformers>=4.51,<5.0'")
         .env({"HF_HOME": "/cache/hf", "VLLM_CACHE_ROOT": "/cache/vllm", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0",
               "CUDA_HOME": "/usr/local/cuda", "TEXT_MODELS_DEV": "1" if DEV else "0"})
         .add_local_file(os.path.join(os.path.dirname(__file__), "keywords_worker.py"), "/root/keywords_worker.py")
         .add_local_python_source("topics_model", "keywords_model"))

TOPICS_DIR = f"/vt/models/{tm.MODEL_NAME}"
TOPICS_LOOKUP = f"/vt/lookup/topics_{tm.MODEL_NAME}.json"
KEYWORDS_DIR = f"/vk/models/{km.MODEL_NAME}"
KEYWORDS_VOCAB = f"/vk/lookup/{km.VOCAB}.json.gz"
TOPICS_GPU, KEYWORDS_GPU = 0.44, 0.46   # shares of the L4 for each engine's weights + activations + KV cache


@app.cls(image=image, gpu="L4", volumes={"/vt": tm.vol, "/vk": km.vol, "/cache": cache},
         secrets=[modal.Secret.from_name("openalex-text-topics"), modal.Secret.from_name("openalex-text-keywords")],
         min_containers=0 if DEV else 1, max_containers=2, scaledown_window=300 if DEV else 1200, timeout=120, startup_timeout=1200,
         memory=32768, cpu=4)
@modal.concurrent(max_inputs=16)   # each model has its own lock, so a topics call never waits behind a keywords call
class TextModels:
    @modal.enter()
    def load(self):
        import json, subprocess, torch, torch.nn as nn
        from vllm import LLM
        from transformers import AutoTokenizer
        t = time.time()
        self.topics = json.load(open(TOPICS_LOOKUP))["topics"]
        assert len(self.topics) == tm.NOT_CLASSIFIABLE
        self.ttok = AutoTokenizer.from_pretrained(f"{TOPICS_DIR}/causal")
        kw = dict(model=f"{TOPICS_DIR}/causal", dtype="bfloat16", max_model_len=512, gpu_memory_utilization=TOPICS_GPU, enable_prefix_caching=False, quantization="fp8")
        try: self.tllm = LLM(runner="pooling", convert="embed", override_pooler_config={"pooling_type": "LAST", "normalize": False}, **kw)
        except TypeError: self.tllm = LLM(task="embed", override_pooler_config={"pooling_type": "LAST", "normalize": False}, **kw)
        sd = torch.load(f"{TOPICS_DIR}/head.pt")
        self.head = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); self.head.load_state_dict(sd); self.head.eval()
        self.tlock = threading.Lock()
        self.vocab = km.Vocab(KEYWORDS_VOCAB)
        self.kproc = subprocess.Popen(["/opt/kw/bin/python", "/root/keywords_worker.py", KEYWORDS_DIR, str(KEYWORDS_GPU)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert json.loads(self.kproc.stdout.readline())["ready"], "keyword worker failed to start"
        self.ktok = AutoTokenizer.from_pretrained(KEYWORDS_DIR)
        self.special = frozenset(self.ktok.all_special_ids)
        self.klock = threading.Lock()
        self.classify(["Warm-up"], ["A short abstract to warm the model."])
        self.tag_one("Warm-up", "A short abstract to warm the model.")
        cache.commit()   # persist the compile cache
        print(f"topics {tm.MODEL_NAME} + keywords {km.MODEL_NAME} (vocabulary {self.vocab.version}) ready in {time.time() - t:.0f}s", flush=True)

    def classify(self, titles, abstracts, venues=None, k=10):
        """= topics_model.Topics.classify"""
        import torch, numpy as np
        from vllm.inputs import TokensPrompt
        venues = venues or [None] * len(titles)
        ids = self.ttok([tm.text(t, a, v) for t, a, v in zip(titles, abstracts, venues)], truncation=True, max_length=tm.MAX_TOKENS)["input_ids"]
        with self.tlock:
            outs = self.tllm.embed([TokensPrompt(prompt_token_ids=x) for x in ids], use_tqdm=False)
            H = torch.tensor(np.array([o.outputs.embedding for o in outs]), dtype=torch.float32, device="cuda")
            with torch.no_grad(): v, i = torch.softmax(self.head(H), 1).topk(k, 1)
        return i.cpu().numpy(), v.cpu().numpy()

    def topics_for(self, top, prob):
        """= topics_model.Topics.topics_for"""
        if int(top[0]) == tm.NOT_CLASSIFIABLE:
            return [], True
        out = []
        for c, p in zip(top, prob):
            c = int(c)
            if c == tm.NOT_CLASSIFIABLE: continue
            if out and p < tm.MIN_SCORE: break
            out.append({**self.topics[c], "score": round(float(p), 4)})
            if len(out) == tm.N_TOPICS: break
        return out, False

    def tag_one(self, title, abstract):
        """= keywords_model.Tagger.tag_one, generation in keywords_worker.py"""
        import json
        from types import SimpleNamespace
        ids = self.ktok(km.user_text(title, abstract) + "\nKeywords:", add_special_tokens=False)["input_ids"][:km.MAX_PROMPT_TOKENS]
        with self.klock:
            self.kproc.stdin.write(json.dumps({"ids": ids}) + "\n"); self.kproc.stdin.flush()
            line = self.kproc.stdout.readline()
        if not line:   # the worker died: end the container so Modal starts a fresh one
            print("keyword worker exited; stopping the container", flush=True); os._exit(1)
        o = json.loads(line)
        logprobs = [{t: SimpleNamespace(logprob=lp)} for t, lp in zip(o["tokens"], o["lps"])]
        kws, lps = km.keyword_lps(self.ktok, o["tokens"], logprobs, self.special)
        return self.vocab.map(kws, lps, title, abstract)

    @modal.method()
    def score_rows(self, rows: list):
        """= topics_model.Topics.score_rows: rows [{ti, ab, ve}] -> top-10 class indices and probabilities, to compare with the corpus pass."""
        top, prob = self.classify([r.get("ti") for r in rows], [r.get("ab") for r in rows], [r.get("ve") for r in rows])
        return top.tolist(), prob.tolist()

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Body, Header, HTTPException
        api = FastAPI()
        topics_auth = f"Bearer {os.environ['TOPICS_MODEL_TOKEN']}"
        keywords_auth = f"Bearer {os.environ['KEYWORDS_MODEL_TOKEN']}"

        @api.get("/health")
        def health():
            import psutil, torch
            p = psutil.Process()
            rss = sum(x.memory_info().rss for x in [p] + p.children(recursive=True))
            free, total = torch.cuda.mem_get_info()
            return {"ok": True, "topics_model": tm.MODEL_NAME, "topics": len(self.topics), "keywords_model": km.MODEL_NAME,
                    "vocab": self.vocab.version, "ram_gib": round(rss / 2**30, 2), "gpu_used_gib": round((total - free) / 2**30, 2)}

        # plain def handlers: FastAPI runs them on its thread pool, so the two models serve at the same time
        @api.post("/topics")
        def topics(body: dict = Body(...), authorization: str = Header("")):
            if authorization != topics_auth:
                raise HTTPException(status_code=401, detail="bad token")
            title, abstract = body.get("title"), body.get("abstract")
            if not ((title or "").strip() or (abstract or "").strip()):
                raise HTTPException(status_code=400, detail="title or abstract required")
            t = time.time()
            top, prob = self.classify([title], [abstract])
            topics, nc = self.topics_for(top[0], prob[0])
            return {"topics": topics, "not_classifiable": nc, "model": tm.MODEL_NAME, "ms": int((time.time() - t) * 1000)}

        @api.post("/keywords")
        def keywords(body: dict = Body(...), authorization: str = Header("")):
            if authorization != keywords_auth:
                raise HTTPException(status_code=401, detail="bad token")
            title, abstract = body.get("title"), body.get("abstract")
            if not (title or abstract):
                raise HTTPException(status_code=400, detail="title or abstract required")
            t = time.time()
            kws = self.tag_one(title, abstract)
            return {"keywords": kws, "model": km.MODEL_NAME, "vocab": self.vocab.version, "ms": int((time.time() - t) * 1000)}

        return api
