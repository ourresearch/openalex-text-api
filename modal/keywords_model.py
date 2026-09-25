"""Keyword tagging for api.openalex.org/text*: the #1322 Qwen3-4B student (distilled from Opus 5.5 indexer labels) served by vLLM on one always-warm L4.

Deploy from desk:  modal deploy modal/keywords_model.py        (app: openalex-text-keywords)
Call:              POST <url>  {"title": ..., "abstract": ...}   Authorization: Bearer $KEYWORDS_MODEL_TOKEN
Returns:           {"keywords": [{"keyword": str, "rank": int, "score": float}], "model": str, "ms": int}
                   GET <url>/health -> {"ok": true, ...}

Same prompt shape, 560-token truncation and greedy decoding as the corpus run (oxjobs #1322 scratch/god/modal_corpus.py), so text
submitted here gets the same keywords the works carry. The text endpoint only knows title + abstract; venue / year / type / language are
sent as unknown, which costs 0.9 F1 against the full prompt (#1359 EXPLORE § 4). Score is rank-derived (1 - (rank-1)/n), not a probability.
Weights: Modal Volume openalex-text-keywords /models/student_qwen4ball8 (copied from the research Volume oxjob1322 by modal/copy_model.py).
"""
import modal, os, time, threading

app = modal.App("openalex-text-keywords")
vol = modal.Volume.from_name("openalex-text-keywords")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install("vllm>=0.8", "transformers>=4.51,<5.0", "fastapi[standard]")
         .env({"HF_HOME": "/vol/hf", "CUDA_HOME": "/usr/local/cuda", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_CACHE_ROOT": "/vol/cache/vllm"}))  # compile cache persists on the Volume: cold start skips ~55 s of torch.compile
MODEL_DIR = "/vol/models/student_qwen4ball8"
MODEL_NAME = "student_qwen4ball8"
MAX_PROMPT_TOKENS = 560   # as in training and the corpus run
MAX_KEYWORDS = 10
MAX_INPUT_CHARS = 6000


def user_text(title, abstract):
    return "\n".join([
        f"Title: {(title or '(none)').strip()[:MAX_INPUT_CHARS]}",
        f"Abstract: {(abstract or '(none)').strip()[:MAX_INPUT_CHARS]}",
        "Venue: (unknown)",
        "Year: None; Type: article; Language: unknown",
    ])


@app.cls(image=image, gpu="L4", volumes={"/vol": vol}, secrets=[modal.Secret.from_name("openalex-text-keywords")],
         min_containers=1, max_containers=2, scaledown_window=1200, timeout=120, startup_timeout=900, memory=16384, cpu=4)  # startup_timeout: load + compile ≈ 130 s on an L4; without it the 120 s request timeout kills startup
@modal.concurrent(max_inputs=8)   # requests queue inside one container (serialized by the lock) instead of booting a second one
class Tagger:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        t = time.time()
        self.llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.9, enable_prefix_caching=False)
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.sp = SamplingParams(temperature=0.0, max_tokens=96, stop=["\n"])
        self.lock = threading.Lock()
        self.tag_one("Warm-up", "A short abstract to warm the model.")
        vol.commit()  # persist the compile cache
        print(f"model ready in {time.time() - t:.0f}s", flush=True)

    def tag_one(self, title, abstract):
        ids = self.tok(user_text(title, abstract) + "\nKeywords:", add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOKENS]
        with self.lock:
            text = self.llm.generate([{"prompt_token_ids": ids}], self.sp, use_tqdm=False)[0].outputs[0].text
        kws = [k.strip() for k in text.split(";") if k.strip()][:MAX_KEYWORDS]
        n = len(kws)
        return [{"keyword": k, "rank": i + 1, "score": round(1 - i / n, 3)} for i, k in enumerate(kws)]

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Request, HTTPException
        api = FastAPI()
        expected = f"Bearer {os.environ['KEYWORDS_MODEL_TOKEN']}"

        @api.get("/health")
        def health():
            return {"ok": True, "model": MODEL_NAME}

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
            return {"keywords": kws, "model": MODEL_NAME, "ms": int((time.time() - t) * 1000)}

        return api
