"""Browser click demo: click on an image -> that point is the pointer -> R map + masks update.

    python demo.py --ckpt runs/C/last.pth [--images-dir datasets/coco/val2017] [--port 7860]

Then open http://localhost:7860. On a remote GPU server, forward the port first:

    ssh -p 4648 -L 7860:localhost:7860 delight@<server>

Standard library only (http.server); the model is wrapped by infer.Predictor.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from infer import Predictor, render_masks, render_r

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>LOVA pointer demo</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
 .row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
 .col{display:flex;flex-direction:column;gap:6px}
 canvas,img{max-width:100%;border:1px solid #444;background:#000}
 .pane{width:min(32vw,640px)}
 h4{margin:4px 0;font-weight:500;color:#aaa}
 table{border-collapse:collapse;font-size:13px}td,th{padding:2px 8px;border-bottom:1px solid #333;text-align:left}
 select,input,button{background:#222;color:#ddd;border:1px solid #555;padding:4px 6px}
 #stats{font-size:13px;color:#9c9}
 label{font-size:13px}
</style></head><body>
<div class="row" style="margin-bottom:10px">
 <input type="file" id="file" accept="image/*">
 <button id="sample" style="display:none">random sample image</button>
 <label>R mode <select id="rmode">
  <option value="pred">pred (R_θ)</option><option value="ones">ones (full compute)</option>
  <option value="zeros">zeros</option><option value="const:0.5">const 0.5</option><option value="const:0.25">const 0.25</option><option value="const:0.75">const 0.75</option>
 </select></label>
 <label>det thr <input type="number" id="thr" value="0.3" min="0" max="1" step="0.05" style="width:60px"></label>
 <span id="stats">load an image, then click on it</span>
</div>
<div class="row">
 <div class="col pane"><h4>image (click = pointer)</h4><canvas id="cv"></canvas></div>
 <div class="col pane"><h4>R field (red = high, blue = low)</h4><img id="rimg"></div>
 <div class="col pane"><h4>predicted masks</h4><img id="mimg"></div>
</div>
<div style="margin-top:10px"><table id="det"><thead><tr><th>#</th><th>class</th><th>score</th><th>mean R in mask</th><th>center</th><th>area px</th></tr></thead><tbody></tbody></table></div>
<script>
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let img=null,token=null,last=null,busy=false;
function draw(){ if(!img)return; cv.width=img.width;cv.height=img.height; ctx.drawImage(img,0,0);
  if(last){ctx.strokeStyle='#ff0';ctx.lineWidth=3;ctx.beginPath();ctx.arc(last.x,last.y,7,0,7);ctx.stroke();
  ctx.beginPath();ctx.moveTo(last.x-14,last.y);ctx.lineTo(last.x+14,last.y);ctx.moveTo(last.x,last.y-14);ctx.lineTo(last.x,last.y+14);ctx.stroke();}}
async function load(blob){ const r=await fetch('/upload',{method:'POST',body:blob}); const j=await r.json(); token=j.token;
  img=new Image(); img.onload=()=>{last=null;draw();document.getElementById('stats').textContent=`${j.w}x${j.h} loaded, click on it`;
  document.getElementById('rimg').src='';document.getElementById('mimg').src='';document.querySelector('#det tbody').innerHTML='';};
  img.src=URL.createObjectURL(blob); }
document.getElementById('file').onchange=e=>{ if(e.target.files[0]) load(e.target.files[0]); };
document.getElementById('sample').onclick=async()=>{ const r=await fetch('/sample'); load(await r.blob()); };
async function infer(){ if(!token||!last||busy)return; busy=true;
  const body={token,x:last.x,y:last.y,r_mode:document.getElementById('rmode').value,det_thr:parseFloat(document.getElementById('thr').value)};
  const r=await fetch('/infer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json(); busy=false;
  if(j.error){document.getElementById('stats').textContent='error: '+j.error;return;}
  document.getElementById('rimg').src='data:image/png;base64,'+j.r_png; document.getElementById('mimg').src='data:image/png;base64,'+j.mask_png;
  document.getElementById('stats').textContent=`pointer=(${last.x},${last.y})  R mean=${j.r_mean.toFixed(3)}  vcompute=${j.vcompute.toFixed(3)}  latency=${j.latency_ms.toFixed(1)} ms  ${j.detections.length} det`;
  document.querySelector('#det tbody').innerHTML=j.detections.map((d,i)=>`<tr><td>${i+1}</td><td>${d.label}</td><td>${d.score.toFixed(3)}</td><td>${d.mean_r.toFixed(2)}</td><td>(${d.center[0].toFixed(0)},${d.center[1].toFixed(0)})</td><td>${d.area}</td></tr>`).join(''); }
cv.onclick=e=>{ if(!img)return; const b=cv.getBoundingClientRect(); last={x:Math.round((e.clientX-b.left)*cv.width/b.width),y:Math.round((e.clientY-b.top)*cv.height/b.height)}; draw(); infer(); };
document.getElementById('rmode').onchange=infer; document.getElementById('thr').onchange=infer;
</script></body></html>"""


def b64png(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class State:
    def __init__(self, predictor: Predictor, images_dir: str | None):
        self.pred = predictor
        self.lock = threading.Lock()
        self.images: dict[str, Image.Image] = {}
        self.files = []
        if images_dir:
            self.files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
            self.images_dir = images_dir


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            if self.path == "/":
                page = HTML.replace('id="sample" style="display:none"', 'id="sample"') if state.files else HTML
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/sample") and state.files:
                name = random.choice(state.files)
                with open(os.path.join(state.images_dir, name), "rb") as f:
                    return self._send(200, f.read(), "image/jpeg")
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            if self.path == "/upload":
                img = Image.open(io.BytesIO(body)).convert("RGB")
                token = f"{random.getrandbits(48):012x}"
                if len(state.images) > 8:
                    state.images.clear()
                state.images[token] = img
                return self._json({"token": token, "w": img.width, "h": img.height})
            if self.path == "/infer":
                q = json.loads(body)
                img = state.images.get(q.get("token"))
                if img is None:
                    return self._json({"error": "image not loaded (upload again)"}, 400)
                xy = (q["x"], q["y"])
                det_thr = float(q.get("det_thr", 0.3))
                try:
                    with state.lock:
                        res = state.pred(img, xy, q.get("r_mode", "pred"))
                except Exception as e:  # surface model errors in the page
                    return self._json({"error": repr(e)}, 500)
                return self._json({
                    "r_png": b64png(render_r(img, res["r"], xy)),
                    "mask_png": b64png(render_masks(img, res, state.pred.class_names, det_thr, xy)),
                    "detections": state.pred.detections(res, det_thr),
                    "latency_ms": res["latency_ms"], "r_mean": res["r_mean"], "vcompute": res["vcompute"],
                })
            self._json({"error": "not found"}, 404)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images-dir", default=None, help="optional folder (e.g. datasets/coco/val2017) for the 'random sample' button")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    state = State(Predictor(a.ckpt, a.device), a.images_dir)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    print(f"model on {state.pred.device}, {len(state.files)} sample images. open http://localhost:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
