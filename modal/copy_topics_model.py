"""Copy the topic model (q8b_2m: the vLLM checkpoint causal/ + the classifier head head.pt) from the research Volume (oxjob1485)
into the production Volume openalex-text-topics.
modal run modal/copy_topics_model.py"""
import modal, shutil, os
app = modal.App("openalex-text-topics-copy")
src = modal.Volume.from_name("oxjob1485"); dst = modal.Volume.from_name("openalex-text-topics", create_if_missing=True)


@app.function(volumes={"/src": src, "/dst": dst}, timeout=3600, cpu=2, memory=4096)
def copy(name: str):
    os.makedirs(f"/dst/models/{name}", exist_ok=True)
    shutil.copytree(f"/src/models/{name}/causal", f"/dst/models/{name}/causal", dirs_exist_ok=True)
    for f in ("head.pt", "args.json"):
        shutil.copy2(f"/src/models/{name}/{f}", f"/dst/models/{name}/{f}")
    dst.commit()
    return sorted(f"{root[len('/dst/models/'):]}/{f} {os.path.getsize(os.path.join(root, f)) / 1e9:.2f} GB"
                  for root, _, fs in os.walk(f"/dst/models/{name}") for f in fs)


@app.local_entrypoint()
def main(name: str = "q8b_2m"):
    print("\n".join(copy.remote(name)))
