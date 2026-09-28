import os
import re
import json
import time
import shlex
import queue
import secrets
import threading
import subprocess
from pathlib import Path
from functools import wraps
from collections import deque

import requests
from flask import Flask, Response, jsonify, redirect, render_template_string, request
from huggingface_hub import HfApi, hf_hub_url

PANEL_PORT = int(os.getenv("PANEL_PORT", "8000"))
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "/runpod-volume/models"))
STATE_DIR = Path(os.getenv("STATE_DIR", "/runpod-volume/glm53-panel"))
CONFIG_PATH = STATE_DIR / "config.json"
HF_TOKEN = os.getenv("HF_TOKEN") or None
CLOUDFLARED_ENABLED = os.getenv("CLOUDFLARED_ENABLED", "true").lower() in ("1", "true", "yes", "on")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

PANEL_USER = "admin"
PANEL_PASSWORD = secrets.token_urlsafe(18)
API_KEY = "glm_" + secrets.token_urlsafe(24)

DEFAULT_CONFIG = {
    "repo": "huihui-ai/GLM-5.3-Flash-abliterated-GGUF",
    "quant": "UD-IQ1_S",
    "context_per_slot": 204800,
    "parallel": 4,
    "gpu_layers": 999,
    "kv_k": "q8_0",
    "kv_v": "q8_0",
    "flash_attn": True,
    "batch": 2048,
    "ubatch": 512,
    "threads": 0,
    "tensor_split": "",
    "extra_args": ""
}

def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            return {**DEFAULT_CONFIG, **data}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)

config = load_config()

def save_config():
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

app = Flask(__name__)

model_proc = None
model_thread = None
tunnel_proc = None
tunnel_url = None
model_started_at = None
model_state = "stopped"
model_error = ""
log_lines = deque(maxlen=1500)
speed_samples = deque(maxlen=5000)
download_lock = threading.Lock()
download_state = {
    "active": False,
    "status": "idle",
    "file": "",
    "downloaded": 0,
    "total": 0,
    "speed": 0.0,
    "eta": None,
    "error": ""
}

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)

def require_panel_auth(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != PANEL_USER or not secrets.compare_digest(auth.password or "", PANEL_PASSWORD):
            return Response("Authentication required", 401, {"WWW-Authenticate": 'Basic realm="GLM panel"'})
        return fn(*args, **kwargs)
    return wrapped

def require_api_key():
    header = request.headers.get("Authorization", "")
    return header == f"Bearer {API_KEY}"

def model_dir():
    safe_repo = config["repo"].replace("/", "--")
    return MODEL_ROOT / safe_repo / config["quant"]

def find_gguf():
    d = model_dir()
    if not d.exists():
        return None
    files = sorted(d.glob("*.gguf"))
    if not files:
        return None
    first_parts = [p for p in files if re.search(r"-00001-of-\d+\.gguf$", p.name)]
    return first_parts[0] if first_parts else files[0]

def list_quant_files(repo, quant):
    api = HfApi(token=HF_TOKEN)
    info = api.model_info(repo_id=repo, files_metadata=True, token=HF_TOKEN)
    matches = []
    q = quant.strip("/")
    for s in info.siblings or []:
        name = s.rfilename
        low = name.lower()
        if not low.endswith(".gguf"):
            continue
        if name.startswith(q + "/") or f"/{q}/" in name or q.lower() in Path(name).name.lower():
            size = getattr(s, "size", None) or 0
            matches.append((name, int(size)))
    if not matches:
        raise RuntimeError(f"No GGUF files found for quant '{quant}' in {repo}")
    return matches

def download_worker(repo, quant):
    global download_state
    with download_lock:
        try:
            files = list_quant_files(repo, quant)
            target_dir = MODEL_ROOT / repo.replace("/", "--") / quant
            target_dir.mkdir(parents=True, exist_ok=True)

            total = sum(size for _, size in files)
            already = 0
            for name, size in files:
                out = target_dir / Path(name).name
                if out.exists() and size and out.stat().st_size == size:
                    already += size

            download_state.update({
                "active": True, "status": "downloading", "file": "",
                "downloaded": already, "total": total, "speed": 0.0,
                "eta": None, "error": ""
            })
            start_t = time.time()
            start_bytes = already
            headers_base = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

            for remote_name, expected_size in files:
                out = target_dir / Path(remote_name).name
                if out.exists() and expected_size and out.stat().st_size == expected_size:
                    continue

                part = out.with_suffix(out.suffix + ".part")
                pos = part.stat().st_size if part.exists() else 0
                headers = dict(headers_base)
                if pos:
                    headers["Range"] = f"bytes={pos}-"

                url = hf_hub_url(repo_id=repo, filename=remote_name)
                download_state["file"] = Path(remote_name).name
                log(f"Downloading {remote_name} (resume={pos} bytes)")

                with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as r:
                    if r.status_code == 416 and expected_size and pos == expected_size:
                        part.rename(out)
                        continue
                    r.raise_for_status()
                    if pos and r.status_code != 206:
                        pos = 0
                        part.unlink(missing_ok=True)
                    mode = "ab" if pos else "wb"
                    with part.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            download_state["downloaded"] += len(chunk)
                            elapsed = max(time.time() - start_t, 0.001)
                            speed = max((download_state["downloaded"] - start_bytes) / elapsed, 0)
                            download_state["speed"] = speed
                            if total and speed > 0:
                                download_state["eta"] = max((total - download_state["downloaded"]) / speed, 0)

                part.rename(out)

            download_state.update({"active": False, "status": "complete", "speed": 0.0, "eta": 0})
            log(f"Download complete: {repo}:{quant}")
        except Exception as e:
            download_state.update({"active": False, "status": "error", "error": str(e)})
            log(f"DOWNLOAD ERROR: {e}")

def parse_llama_log(line):
    # llama.cpp output format changes over time; keep several permissive patterns.
    patterns = [
        r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second",
        r"([0-9]+(?:\.[0-9]+)?)\s+tokens/s",
        r"([0-9]+(?:\.[0-9]+)?)\s+t/s"
    ]
    for p in patterns:
        m = re.search(p, line, re.I)
        if m:
            try:
                speed_samples.append(float(m.group(1)))
            except ValueError:
                pass
            break

def pump_process(proc):
    global model_state, model_error
    try:
        for raw in iter(proc.stdout.readline, ""):
            if not raw:
                break
            line = raw.rstrip()
            log("[llama] " + line)
            parse_llama_log(line)
    finally:
        code = proc.wait()
        if model_proc is proc:
            model_state = "stopped" if code == 0 else "error"
            if code != 0:
                model_error = f"llama-server exited with code {code}"
            log(f"llama-server exited with code {code}")

def llama_ready():
    try:
        r = requests.get("http://127.0.0.1:8080/health", timeout=1.5)
        return r.status_code < 500
    except Exception:
        return False

def start_model():
    global model_proc, model_thread, model_started_at, model_state, model_error
    if model_proc and model_proc.poll() is None:
        return False, "Model is already running"

    gguf = find_gguf()
    if not gguf:
        return False, "Model is not downloaded"

    per_slot = int(config["context_per_slot"])
    parallel = int(config["parallel"])
    total_context = per_slot * parallel

    cmd = [
        "llama-server",
        "-m", str(gguf),
        "--host", "127.0.0.1",
        "--port", "8080",
        "-c", str(total_context),
        "-np", str(parallel),
        "-ngl", str(int(config["gpu_layers"])),
        "-ctk", config["kv_k"],
        "-ctv", config["kv_v"],
        "-b", str(int(config["batch"])),
        "-ub", str(int(config["ubatch"])),
        "--flash-attn", "on" if config["flash_attn"] else "off",
    ]
    if int(config.get("threads", 0)) > 0:
        cmd += ["-t", str(int(config["threads"]))]
    if config.get("tensor_split", "").strip():
        cmd += ["--tensor-split", config["tensor_split"].strip()]
    if config.get("extra_args", "").strip():
        cmd += shlex.split(config["extra_args"])

    log("Starting model:")
    log(" ".join(shlex.quote(x) for x in cmd))
    model_error = ""
    model_state = "loading"
    model_started_at = time.time()
    model_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    model_thread = threading.Thread(target=pump_process, args=(model_proc,), daemon=True)
    model_thread.start()
    return True, "Starting"

def stop_model():
    global model_proc, model_state
    if not model_proc or model_proc.poll() is not None:
        model_state = "stopped"
        return True, "Already stopped"
    log("Stopping llama-server...")
    model_proc.terminate()
    try:
        model_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        model_proc.kill()
    model_state = "stopped"
    return True, "Stopped"

def gpu_stats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ], text=True, timeout=3)
        rows = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 7:
                rows.append({
                    "index": int(p[0]), "name": p[1],
                    "vram_total_mb": float(p[2]), "vram_used_mb": float(p[3]),
                    "util": float(p[4]), "temp": float(p[5]), "power": float(p[6])
                })
        return rows
    except Exception:
        return []

def start_tunnel():
    global tunnel_proc, tunnel_url
    if not CLOUDFLARED_ENABLED:
        return
    cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PANEL_PORT}", "--no-autoupdate"]
    tunnel_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    rx = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for raw in iter(tunnel_proc.stdout.readline, ""):
        line = raw.rstrip()
        log("[cloudflared] " + line)
        m = rx.search(line)
        if m and not tunnel_url:
            tunnel_url = m.group(0)
            print("\n" + "=" * 72, flush=True)
            print(f"WEB PANEL: {tunnel_url}", flush=True)
            print(f"OPENAI BASE URL: {tunnel_url}/v1", flush=True)
            print(f"PANEL USER: {PANEL_USER}", flush=True)
            print(f"PANEL PASSWORD: {PANEL_PASSWORD}", flush=True)
            print(f"API KEY: {API_KEY}", flush=True)
            print("=" * 72 + "\n", flush=True)

def boot_banner():
    print("\n" + "=" * 72, flush=True)
    print("GLM-5.3 RUNPOD SERVERLESS PANEL", flush=True)
    print(f"LOCAL PANEL: http://0.0.0.0:{PANEL_PORT}", flush=True)
    print(f"PANEL USER: {PANEL_USER}", flush=True)
    print(f"PANEL PASSWORD: {PANEL_PASSWORD}", flush=True)
    print(f"API KEY: {API_KEY}", flush=True)
    print("=" * 72 + "\n", flush=True)

PANEL_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM-5.3 RunPod</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#0b0d10;color:#e8eaf0;margin:0}
.wrap{max-width:1180px;margin:28px auto;padding:0 18px}
h1{font-size:26px;margin:0 0 8px}.muted{color:#9298a6}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:#12151b;border:1px solid #252a35;border-radius:12px;padding:16px}
.c4{grid-column:span 4}.c6{grid-column:span 6}.c8{grid-column:span 8}.c12{grid-column:span 12}
@media(max-width:800px){.c4,.c6,.c8{grid-column:span 12}}
label{display:block;font-size:12px;color:#9aa1af;margin:10px 0 5px}
input,select{width:100%;box-sizing:border-box;background:#0d1015;color:#fff;border:1px solid #303746;border-radius:7px;padding:9px}
button{background:#6657ff;color:white;border:0;border-radius:7px;padding:9px 13px;cursor:pointer;margin:4px 5px 4px 0}
button.red{background:#b43b4a}button.gray{background:#333946}
.kv{font-size:12px;color:#9aa1af}.big{font-size:22px;font-weight:700}.ok{color:#50d890}.warn{color:#f2be54}.bad{color:#ff657a}
.progress{height:10px;background:#282d38;border-radius:8px;overflow:hidden}.bar{height:100%;background:#6657ff;width:0%}
pre{background:#090b0e;border:1px solid #242936;padding:12px;border-radius:8px;max-height:360px;overflow:auto;white-space:pre-wrap;font-size:12px}
.row{display:flex;gap:10px}.row>*{flex:1}.endpoint{word-break:break-all;font-family:monospace}
</style>
</head>
<body><div class="wrap">
<h1>GLM-5.3 Flash · RunPod</h1>
<div class="muted">Configure → download → start. The GGUF is kept on the Network Volume.</div>
<br>
<div class="grid">
<div class="card c4"><div class="kv">MODEL STATE</div><div id="state" class="big">...</div><div id="uptime" class="muted"></div></div>
<div class="card c4"><div class="kv">CURRENT SPEED</div><div id="speed" class="big">— tok/s</div><div id="avg" class="muted">avg —</div></div>
<div class="card c4"><div class="kv">GPU</div><div id="gpu" class="big">...</div><div id="vram" class="muted"></div></div>

<div class="card c8">
<form id="cfg">
<label>Hugging Face repository</label><input name="repo" value="{{c.repo}}">
<div class="row"><div><label>Quant / folder</label><input name="quant" value="{{c.quant}}"></div>
<div><label>Context per agent / slot</label><input name="context_per_slot" type="number" value="{{c.context_per_slot}}"></div>
<div><label>Parallel agents / slots</label><input name="parallel" type="number" value="{{c.parallel}}"></div></div>
<div class="row"><div><label>GPU layers</label><input name="gpu_layers" type="number" value="{{c.gpu_layers}}"></div>
<div><label>KV K</label><select name="kv_k"><option>{{c.kv_k}}</option><option>q8_0</option><option>f16</option><option>q4_0</option></select></div>
<div><label>KV V</label><select name="kv_v"><option>{{c.kv_v}}</option><option>q8_0</option><option>f16</option><option>q4_0</option></select></div></div>
<div class="row"><div><label>Batch</label><input name="batch" type="number" value="{{c.batch}}"></div>
<div><label>Ubatch</label><input name="ubatch" type="number" value="{{c.ubatch}}"></div>
<div><label>CPU threads (0=auto)</label><input name="threads" type="number" value="{{c.threads}}"></div></div>
<label>Tensor split (blank for one GPU)</label><input name="tensor_split" value="{{c.tensor_split}}">
<label>Extra llama-server args</label><input name="extra_args" value="{{c.extra_args}}">
<label><input style="width:auto" type="checkbox" name="flash_attn" {% if c.flash_attn %}checked{% endif %}> Flash Attention</label>
<button type="submit">Save settings</button>
</form>
<div class="muted">Total llama.cpp context = context per slot × parallel slots.</div>
</div>

<div class="card c4">
<div class="kv">OPENAI API</div>
<div class="endpoint" id="endpoint">{{endpoint}}</div><br>
<div class="kv">API KEY</div><div class="endpoint">{{api_key}}</div>
</div>

<div class="card c12">
<div class="row"><div><b>Model download</b><div id="dltext" class="muted">...</div></div>
<div style="text-align:right"><button onclick="act('/api/download')">Download</button>
<button onclick="act('/api/start')">Start model</button>
<button class="gray" onclick="act('/api/restart')">Restart</button>
<button class="red" onclick="act('/api/stop')">Stop</button></div></div>
<div class="progress"><div id="dlbar" class="bar"></div></div>
</div>

<div class="card c12"><b>Live logs</b><pre id="logs">Loading…</pre></div>
</div>
</div>
<script>
const fmtBytes=n=>{if(!n)return '0 B';let u=['B','KiB','MiB','GiB','TiB'],i=Math.min(Math.floor(Math.log(n)/Math.log(1024)),4);return (n/1024**i).toFixed(i>2?2:1)+' '+u[i]}
const fmtTime=s=>{if(s==null)return '—';s=Math.max(0,Math.round(s));let h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60;return (h?h+'h ':'')+(m?m+'m ':'')+x+'s'}
async function status(){
 let r=await fetch('/api/status'),d=await r.json();
 document.getElementById('state').textContent=d.model.state+(d.model.ready?' · READY':'');
 document.getElementById('uptime').textContent=d.model.uptime?'uptime '+fmtTime(d.model.uptime):'';
 document.getElementById('speed').textContent=(d.metrics.current_speed??'—')+' tok/s';
 document.getElementById('avg').textContent='avg '+(d.metrics.average_speed??'—')+' tok/s';
 if(d.gpus.length){let g=d.gpus[0];document.getElementById('gpu').textContent=g.name;document.getElementById('vram').textContent=(g.vram_used_mb/1024).toFixed(1)+' / '+(g.vram_total_mb/1024).toFixed(1)+' GB · '+g.util+'% · '+g.temp+'°C'}
 let dl=d.download, pct=dl.total?Math.min(100,dl.downloaded/dl.total*100):0;
 document.getElementById('dlbar').style.width=pct+'%';
 document.getElementById('dltext').textContent=dl.status+' · '+fmtBytes(dl.downloaded)+' / '+fmtBytes(dl.total)+(dl.speed?' · '+fmtBytes(dl.speed)+'/s · ETA '+fmtTime(dl.eta):'')+(dl.file?' · '+dl.file:'')+(dl.error?' · '+dl.error:'');
 document.getElementById('logs').textContent=d.logs.join('\n'); document.getElementById('logs').scrollTop=1e9;
 document.getElementById('endpoint').textContent=d.endpoint;
}
async function act(url){let r=await fetch(url,{method:'POST'});let d=await r.json(); if(!r.ok)alert(d.error||d.message||'Error');setTimeout(status,500)}
document.getElementById('cfg').addEventListener('submit',async e=>{e.preventDefault();let fd=new FormData(e.target),o=Object.fromEntries(fd.entries());o.flash_attn=fd.has('flash_attn');let r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});let d=await r.json();if(!r.ok)alert(d.error||'Error');else alert('Saved')})
status();setInterval(status,2000);
</script></body></html>
"""

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/")
@require_panel_auth
def panel():
    endpoint = (tunnel_url + "/v1") if tunnel_url else f"http://127.0.0.1:{PANEL_PORT}/v1"
    return render_template_string(PANEL_HTML, c=config, api_key=API_KEY, endpoint=endpoint)

@app.route("/api/config", methods=["POST"])
@require_panel_auth
def api_config():
    global config
    if model_proc and model_proc.poll() is None:
        return jsonify({"error": "Stop the model before changing configuration"}), 409
    d = request.get_json(force=True)
    try:
        new = dict(config)
        new["repo"] = str(d.get("repo", new["repo"])).strip()
        new["quant"] = str(d.get("quant", new["quant"])).strip()
        for k in ("context_per_slot", "parallel", "gpu_layers", "batch", "ubatch", "threads"):
            new[k] = int(d.get(k, new[k]))
        for k in ("kv_k", "kv_v", "tensor_split", "extra_args"):
            new[k] = str(d.get(k, new[k])).strip()
        new["flash_attn"] = bool(d.get("flash_attn", False))
        if new["context_per_slot"] < 1024 or new["parallel"] < 1:
            raise ValueError("Invalid context/parallel value")
        config = new
        save_config()
        return jsonify({"ok": True, "config": config})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/download", methods=["POST"])
@require_panel_auth
def api_download():
    if download_state["active"]:
        return jsonify({"error": "Download already active"}), 409
    t = threading.Thread(target=download_worker, args=(config["repo"], config["quant"]), daemon=True)
    t.start()
    return jsonify({"ok": True})

@app.route("/api/start", methods=["POST"])
@require_panel_auth
def api_start():
    ok, msg = start_model()
    return jsonify({"ok": ok, "message": msg, "error": None if ok else msg}), (200 if ok else 409)

@app.route("/api/stop", methods=["POST"])
@require_panel_auth
def api_stop():
    ok, msg = stop_model()
    return jsonify({"ok": ok, "message": msg})

@app.route("/api/restart", methods=["POST"])
@require_panel_auth
def api_restart():
    stop_model()
    ok, msg = start_model()
    return jsonify({"ok": ok, "message": msg, "error": None if ok else msg}), (200 if ok else 409)

@app.route("/api/status")
@require_panel_auth
def api_status():
    global model_state
    ready = llama_ready() if model_proc and model_proc.poll() is None else False
    if ready:
        model_state = "running"
    elif model_proc and model_proc.poll() is None:
        model_state = "loading"
    current = speed_samples[-1] if speed_samples else None
    avg = (sum(speed_samples) / len(speed_samples)) if speed_samples else None
    endpoint = (tunnel_url + "/v1") if tunnel_url else f"http://127.0.0.1:{PANEL_PORT}/v1"
    return jsonify({
        "model": {
            "state": model_state, "ready": ready,
            "error": model_error,
            "path": str(find_gguf() or ""),
            "uptime": (time.time() - model_started_at) if model_started_at and model_proc and model_proc.poll() is None else None,
            "context_per_slot": config["context_per_slot"],
            "parallel": config["parallel"],
            "total_context": config["context_per_slot"] * config["parallel"]
        },
        "download": dict(download_state),
        "metrics": {
            "current_speed": round(current, 2) if current is not None else None,
            "average_speed": round(avg, 2) if avg is not None else None
        },
        "gpus": gpu_stats(),
        "endpoint": endpoint,
        "logs": list(log_lines)[-300:]
    })

@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def proxy_v1(path):
    if not require_api_key():
        return jsonify({"error": {"message": "Invalid API key", "type": "authentication_error"}}), 401
    if not llama_ready():
        return jsonify({"error": {"message": "Model is not ready", "type": "server_error"}}), 503

    target = f"http://127.0.0.1:8080/v1/{path}"
    headers = {k: v for k, v in request.headers if k.lower() not in ("host", "content-length", "authorization")}
    try:
        upstream = requests.request(
            request.method, target, params=request.args,
            data=request.get_data(), headers=headers,
            stream=True, timeout=(10, 3600)
        )
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        out_headers = [(k, v) for k, v in upstream.headers.items() if k.lower() not in excluded]
        def generate():
            try:
                for chunk in upstream.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()
        return Response(generate(), status=upstream.status_code, headers=out_headers)
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "proxy_error"}}), 502

def main():
    boot_banner()
    save_config()
    if CLOUDFLARED_ENABLED:
        threading.Thread(target=start_tunnel, daemon=True).start()
    app.run(host="0.0.0.0", port=PANEL_PORT, threaded=True)

if __name__ == "__main__":
    main()
