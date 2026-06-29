"""RunPod serverless handler for DaSiWa LTX 2.3 multi-image video generation.

Two input modes:
  A) High-level (recommended, used by clipboard):
     {
       "global_prompt": "a cinematic ...",
       "images": [{"name":"a.png","image":"<base64|http url>"}, ...],   # in scene order
       "duration_frames": 144,          # optional, default 144 (~6s @24fps)
       "frame_rate": 24,                # optional, default 24
       "guide_strength": "1.0",         # optional; scalar or comma list per image
       "seed": 123456,                  # optional; omit for random
       "segments": [...],               # optional explicit timeline; else auto-distribute
       "resolution_preset": "0.83 MP - HD",   # optional
       "aspect": "9:16 - Social"        # optional
     }
  B) Raw passthrough (testing): {"workflow": <API-format graph>, "images":[...]}

The handler loads a baked API-format template, injects timeline_data (multi-image
keyframes), prompt, duration and seed into the LTXDirector / RandomNoise nodes,
uploads the reference images to the ComfyUI input dir, runs the prompt and returns
the resulting mp4 as base64."""
import base64, json, os, re, subprocess, time, uuid
import requests
import runpod

COMFY = "http://127.0.0.1:8188"
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/runpod-volume/ComfyUI")
INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
TEMPLATE_PATH = os.environ.get("WF_TEMPLATE", "/dasiwa_api_template.json")


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


print("[handler] launching DaSiWa ComfyUI ...", flush=True)
subprocess.Popen(["bash", "/start-comfy.sh"])
print(f"[handler] ComfyUI ready={_wait_comfy()}", flush=True)


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


def _build_timeline(images, global_prompt, duration_frames, segments):
    """Auto-distribute images as keyframes: first @0, last @end, middle evenly."""
    if segments is None:
        names = [os.path.basename(im["name"]) for im in (images or [])]
        n = len(names)
        segments = []
        for i, nm in enumerate(names):
            if n == 1:
                start = 0
            elif i == n - 1:
                start = duration_frames - 1            # last = end frame
            else:
                start = round(i * (duration_frames - 1) / (n - 1)) if n > 1 else 0
            segments.append({"type": "image", "imageFile": nm, "start": int(start), "length": 1})
    return {
        "mainTrackEnabled": True,
        "global_prompt": global_prompt or "",
        "duration_frames": int(duration_frames),
        "start_frame": 0,
        "segments": segments,
    }


def _apply(wf, inp):
    """Inject high-level params into the API-format template (mutates a copy)."""
    images = inp.get("images") or []
    dur = int(inp.get("duration_frames", 144))
    fps = int(inp.get("frame_rate", 24))
    gp = inp.get("global_prompt", "")
    segments = inp.get("segments")  # explicit override or None -> auto
    tl = _build_timeline(images, gp, dur, segments)
    nseg = len(tl["segments"])

    gs = inp.get("guide_strength")
    if gs is None:
        gs = ",".join(["1.0"] * max(1, nseg))
    elif not isinstance(gs, str):
        gs = ",".join(str(x) for x in gs)

    for nid, node in wf.items():
        ct = node.get("class_type")
        ni = node.get("inputs", {})
        if ct == "LTXDirector":
            ni["timeline_data"] = json.dumps(tl)
            ni["guide_strength"] = gs
            ni["duration_frames"] = str(dur)
            ni["end_frame"] = str(dur)
            ni["start_frame"] = "0"
            ni["frame_rate"] = str(fps)
            ni["duration_seconds"] = str(round(dur / fps, 3))
            ni["end_second"] = str(round(dur / fps, 3))
        elif ct == "RandomNoise" and inp.get("seed") is not None:
            ni["noise_seed"] = int(inp["seed"])
        elif ct == "DaSiWa_ResolutionScaleCalculator":
            # Map a plain resolution label (480p/720p/1080p/4k) → the node's resolution_preset enum.
            res = str(inp.get("resolution") or "").lower()
            preset = {"480p": "480p", "720p": "720p", "1080p": "1080p",
                      "4k": "4K", "2160p": "4K", "2k": "2K"}.get(res)
            if inp.get("resolution_preset"):
                ni["resolution_preset"] = inp["resolution_preset"]
            elif preset:
                ni["resolution_preset"] = preset
            # Aspect: derive exact W:H from `ratio` (e.g. "9:16") via CUSTOM mode so any aspect works
            # (the preset list is portrait-only). Skip for adaptive/unknown.
            m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", str(inp.get("ratio") or inp.get("aspect") or ""))
            if m:
                ni["aspect_preset_when_not_image"] = "CUSTOM"
                ni["custom_aspect_width"] = int(m.group(1))
                ni["custom_aspect_height"] = int(m.group(2))

    # FAST MODE (default ON): base-model direct output. The DaSiWa workflow ends with a heavy
    # DaSiWa_RTX_UpscalerRefiner that pixel-upscales+refines EVERY decoded frame — by far the slowest
    # stage. For 480p/720p we don't need it: repoint the video output past the refiner to the decoded
    # frames it was reading, and ComfyUI prunes the orphaned refiner (+ its upstream-only nodes).
    if inp.get("fast", True):
        for node in wf.values():
            ni = node.get("inputs", {})
            img = ni.get("images")
            if isinstance(img, list) and len(img) == 2:
                src = wf.get(str(img[0]))
                if src and src.get("class_type") == "DaSiWa_RTX_UpscalerRefiner":
                    upstream = src.get("inputs", {}).get("images")
                    if isinstance(upstream, list) and len(upstream) == 2:
                        ni["images"] = upstream  # bypass the refiner → ComfyUI prunes it
    return wf


def _find_video(outputs):
    for _nid, o in outputs.items():
        for key in ("gifs", "images", "video"):
            for f in o.get(key, []):
                if f.get("filename", "").endswith((".mp4", ".webm")):
                    return f
    return None


def handler(job):
    inp = job.get("input", {})
    if not _wait_comfy(120):
        return {"error": "ComfyUI not ready"}
    try:
        _save_images(inp.get("images"))
    except Exception as e:
        return {"error": f"image save failed: {e}"}

    wf = inp.get("workflow")
    if wf is None:
        try:
            wf = json.load(open(TEMPLATE_PATH))
            wf = _apply(wf, inp)
        except Exception as e:
            return {"error": f"template build failed: {e}"}

    cid = str(uuid.uuid4())
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": cid}, timeout=60)
    if not r.ok:
        return {"error": f"/prompt rejected: {r.text[:600]}"}
    pid = r.json().get("prompt_id")

    deadline = time.time() + int(inp.get("timeout", 1100))
    while time.time() < deadline:
        h = requests.get(f"{COMFY}/history/{pid}", timeout=15).json()
        if pid in h:
            entry = h[pid]
            status = entry.get("status", {}).get("status_str")
            vf = _find_video(entry.get("outputs", {}))
            if vf:
                params = {"filename": vf["filename"], "subfolder": vf.get("subfolder", ""), "type": vf.get("type", "output")}
                vid = requests.get(f"{COMFY}/view", params=params, timeout=180).content
                return {"filename": vf["filename"], "mime": "video/mp4",
                        "video_b64": base64.b64encode(vid).decode(), "status": status}
            if status == "error":
                return {"error": "workflow execution error",
                        "detail": str(entry.get("status", {}).get("messages", []))[:1000]}
        time.sleep(3)
    return {"error": "generation timeout"}


runpod.serverless.start({"handler": handler})
