"""Copy a #1322 student's weights from the research Volume (oxjob1322) into the production Volume openalex-text-keywords.
modal run modal/copy_model.py [--name student_qwen4b_v3fix]"""
import modal, shutil, os
app = modal.App("openalex-text-keywords-copy")
src = modal.Volume.from_name("oxjob1322"); dst = modal.Volume.from_name("openalex-text-keywords", create_if_missing=True)
@app.function(volumes={"/src": src, "/dst": dst}, timeout=3600, cpu=2, memory=4096)
def copy(name: str):
    shutil.copytree(f"/src/models/{name}", f"/dst/models/{name}", dirs_exist_ok=True); dst.commit()
    return sorted(f"{f} {os.path.getsize(f'/dst/models/{name}/' + f) / 1e9:.2f} GB" for f in os.listdir(f"/dst/models/{name}"))
@app.local_entrypoint()
def main(name: str = "student_qwen4b_v3fix"): print("\n".join(copy.remote(name)))
