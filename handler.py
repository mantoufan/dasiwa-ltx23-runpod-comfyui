"""RunPod serverless handler wrapping the DaSiWa LTX 2.3 ComfyUI image.
Input: {"workflow": <ComfyUI API-format graph with timeline_data already set>,
        "images": [{"name":"scene_A.png","image":"<base64 or http url>"}, ...]}
Output: {"video_b64": "...", "filename": "...", "mime":"video/mp4"}
Reference images are written to the ComfyUI input dir; the workflow's
LTXDirector timeline_data references them by filename (multi-image keyframes)."""
import base64, json, os, subprocess, time, uuid, urllib.request
import requests
import runpod

COMFY = "http://127.0.0.1:8188"
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
INPUT_DIR = os.path.join(COMFYUI_DIR, "input")

def _wait_comfy(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{COMFY}/system_stats", timeout=5).ok:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False

# Cold start: launch the DaSiWa start script (syncs ComfyUI, downloads models on
# first run to the network volume, then runs ComfyUI). Background it, then wait.
print("[handler] launching DaSiWa ComfyUI ...", flush=True)
subprocess.Popen(["bash", "/start-comfy.sh"], stdout=subprocess.STDOUT if False else None)
_ready = _wait_comfy()
print(f"[handler] ComfyUI ready={_ready}", flush=True)

def _save_images(images):
    os.makedirs(INPUT_DIR, exist_ok=True)
    for im in images or []:
        name = os.path.basename(im["name"])
        data = im["image"]
        if isinstance(data, str) and data.startswith("http"):
            raw = requests.get(data, timeout=120).content
        else:
            if isinstance(data, str) and "," in data[:64]:
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
        with open(os.path.join(INPUT_DIR, name), "wb") as f:
            f.write(raw)

def _find_video(outputs):
    for _nid, o in outputs.items():
        for key in ("gifs", "images", "video"):
            for f in o.get(key, []):
                if f.get("filename", "").endswith((".mp4", ".webm")):
                    return f
    return None

def handler(job):
    inp = job.get("input", {})
    wf = inp.get("workflow")
    if not wf:
        return {"error": "missing 'workflow' (ComfyUI API-format graph)"}
    if not _wait_comfy(60):
        return {"error": "ComfyUI not ready"}
    try:
        _save_images(inp.get("images"))
    except Exception as e:
        return {"error": f"image save failed: {e}"}
    cid = str(uuid.uuid4())
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": cid}, timeout=60)
    if not r.ok:
        return {"error": f"/prompt rejected: {r.text[:500]}"}
    pid = r.json().get("prompt_id")
    # poll history
    deadline = time.time() + int(inp.get("timeout", 1200))
    while time.time() < deadline:
        h = requests.get(f"{COMFY}/history/{pid}", timeout=15).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {}).get("status_str")
            vf = _find_video(entry.get("outputs", {}))
            if vf:
                params = {"filename": vf["filename"], "subfolder": vf.get("subfolder", ""), "type": vf.get("type", "output")}
                vid = requests.get(f"{COMFY}/view", params=params, timeout=120).content
                return {"filename": vf["filename"], "mime": "video/mp4",
                        "video_b64": base64.b64encode(vid).decode(), "status": status}
            if status == "error":
                return {"error": "workflow execution error", "detail": str(entry.get("status", {}).get("messages", []))[:800]}
        time.sleep(3)
    return {"error": "generation timeout"}

runpod.serverless.start({"handler": handler})
