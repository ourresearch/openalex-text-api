"""Keyword generation for text_models.py, run by its own Python (/opt/kw: vLLM 0.22.1, the version openalex-text-keywords ran alone, which
gives the same keywords on every machine; vLLM 0.10.2, which topics needs, does not). JSON lines on stdin {"ids": [prompt token ids]} ->
{"tokens": [sampled token ids], "lps": [their log-probs]} on stdout. Decoding as keywords_model.Tagger: greedy, 192 tokens, stop at newline.
Usage (by text_models.py): /opt/kw/bin/python keywords_worker.py <model dir> <gpu share>
"""
import json, sys

out, sys.stdout = sys.stdout, sys.stderr   # vLLM's prints go to the log; stdout carries only answers
from vllm import LLM, SamplingParams

llm = LLM(model=sys.argv[1], dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=float(sys.argv[2]), enable_prefix_caching=False)
sp = SamplingParams(temperature=0.0, max_tokens=192, stop=["\n"], logprobs=1)
out.write('{"ready": true}\n'); out.flush()
for line in sys.stdin:
    o = llm.generate([{"prompt_token_ids": json.loads(line)["ids"]}], sp, use_tqdm=False)[0].outputs[0]
    lps = [d[t].logprob if d and t in d else 0.0 for t, d in zip(o.token_ids, o.logprobs)]
    out.write(json.dumps({"tokens": list(o.token_ids), "lps": lps}) + "\n"); out.flush()
