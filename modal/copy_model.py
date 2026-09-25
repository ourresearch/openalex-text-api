"""One-off: copy the #1322 student weights from the research Volume (oxjob1322) into the production Volume openalex-text-keywords.
modal run modal/copy_model.py"""
import modal, shutil, os
app = modal.App("openalex-text-keywords-copy")
src = modal.Volume.from_name("oxjob1322"); dst = modal.Volume.from_name("openalex-text-keywords", create_if_missing=True)
@app.function(volumes={"/src": src, "/dst": dst}, timeout=3600, cpu=2, memory=4096)
def copy():
    shutil.copytree("/src/models/student_qwen4ball8", "/dst/models/student_qwen4ball8", dirs_exist_ok=True); dst.commit()
    return sorted(f"{f} {os.path.getsize('/dst/models/student_qwen4ball8/'+f)/1e9:.2f} GB" for f in os.listdir("/dst/models/student_qwen4ball8"))
@app.local_entrypoint()
def main(): print("\n".join(copy.remote()))
