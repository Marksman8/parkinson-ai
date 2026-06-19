"""
Debugger FastAPI — runs on port 8080
Provides /run endpoint + live dashboard at /
"""
import json
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.debugger import PDAutoDebugger, Status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["debugger"] = PDAutoDebugger(project_root="..")
    yield
    state.clear()


app = FastAPI(
    title="PD System Auto-Debugger",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/run", summary="Run full system diagnostic")
async def run_debug():
    report = state["debugger"].run_and_save("debug_report.json")
    return JSONResponse(content={
        "timestamp": report.timestamp,
        "summary":   report.summary,
        "checks": [
            {
                "name":    c.name,
                "status":  c.status.value,
                "message": c.message,
                "detail":  c.detail,
                "fix":     c.fix,
                "ms":      int(c.duration * 1000),
            }
            for c in report.checks
        ],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "service": "pd-debugger"}


@app.get("/", response_class=HTMLResponse, summary="Live debug dashboard")
async def dashboard():
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PD System Debugger</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#06080f;--surf:#0d1117;--surf2:#161b27;--bdr:#1e2535;--txt:#e2e8f0;--dim:#94a3b8;--mut:#64748b;--ok:#10b981;--warn:#f59e0b;--err:#ef4444;--blue:#4f8ef7;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--txt);font-family:'JetBrains Mono',monospace;font-size:13px;min-height:100vh;padding:2rem;}
h1{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-0.02em;margin-bottom:0.3rem;}
.sub{color:var(--dim);font-size:11px;margin-bottom:2rem;}
.btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:0.6rem 1.4rem;font-family:inherit;font-size:12px;font-weight:500;cursor:pointer;transition:opacity 0.2s;}
.btn:hover{opacity:0.85;}
.btn:disabled{opacity:0.4;cursor:not-allowed;}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem;margin:1.5rem 0;}
.stat{background:var(--surf);border:1px solid var(--bdr);border-radius:8px;padding:1rem;text-align:center;}
.stat .n{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;margin-bottom:0.2rem;}
.stat .l{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:var(--mut);}
.n.ok{color:var(--ok)}.n.warn{color:var(--warn)}.n.err{color:var(--err)}.n.blue{color:var(--blue)}
.checks{display:flex;flex-direction:column;gap:0.6rem;margin-top:1.5rem;}
.check{background:var(--surf);border:1px solid var(--bdr);border-radius:8px;padding:0.8rem 1rem;display:grid;grid-template-columns:24px 1fr auto;gap:0.75rem;align-items:start;transition:border-color 0.2s;}
.check.ok{border-left:3px solid var(--ok);}
.check.warning{border-left:3px solid var(--warn);}
.check.error{border-left:3px solid var(--err);}
.check.skip{border-left:3px solid var(--mut);}
.icon{font-size:14px;margin-top:1px;}
.cname{font-weight:500;font-size:12px;color:var(--txt);margin-bottom:2px;}
.cmsg{font-size:11px;color:var(--dim);}
.cfix{font-size:10px;color:var(--warn);margin-top:4px;padding:4px 8px;background:rgba(245,158,11,0.08);border-radius:4px;border-left:2px solid var(--warn);}
.cerr{font-size:10px;color:var(--err);margin-top:4px;padding:4px 8px;background:rgba(239,68,68,0.08);border-radius:4px;border-left:2px solid var(--err);}
.cdur{font-size:10px;color:var(--mut);white-space:nowrap;}
.badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;}
.badge.ok{background:rgba(16,185,129,0.15);color:var(--ok);}
.badge.warning{background:rgba(245,158,11,0.15);color:var(--warn);}
.badge.error{background:rgba(239,68,68,0.15);color:var(--err);}
.badge.skip{background:rgba(100,116,139,0.15);color:var(--mut);}
#health-banner{border-radius:8px;padding:0.75rem 1.2rem;margin:1rem 0;font-size:12px;font-weight:500;display:none;}
#health-banner.healthy{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);color:var(--ok);}
#health-banner.degraded{background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);color:var(--warn);}
#health-banner.critical{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:var(--err);}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,0.2);border-top-color:#fff;border-radius:50%;animation:spin 0.6s linear infinite;vertical-align:middle;margin-right:6px;}
@keyframes spin{to{transform:rotate(360deg)}}
.ts{font-size:10px;color:var(--mut);margin-top:0.3rem;}
</style>
</head>
<body>
<h1>PD System Debugger</h1>
<p class="sub">Automated diagnostic for pd_multimodal_ai · Click Run to scan all components</p>
<div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;">
  <button class="btn" id="run-btn" onclick="runDebug()">▶ Run Diagnostics</button>
  <span id="ts" class="ts"></span>
</div>
<div id="health-banner"></div>
<div class="summary" id="summary" style="display:none"></div>
<div class="checks" id="checks"></div>

<script>
const ICONS = {ok:'✓', warning:'!', error:'✗', skip:'—'};

async function runDebug() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Scanning...';
  document.getElementById('checks').innerHTML = '';
  document.getElementById('summary').style.display = 'none';
  document.getElementById('health-banner').style.display = 'none';

  try {
    const r = await fetch('/run');
    const d = await r.json();

    document.getElementById('ts').textContent =
      'Last run: ' + new Date(d.timestamp).toLocaleTimeString();

    // Summary cards
    const s = d.summary;
    const sumEl = document.getElementById('summary');
    sumEl.style.display = 'grid';
    sumEl.innerHTML = `
      <div class="stat"><div class="n blue">${s.total}</div><div class="l">Total checks</div></div>
      <div class="stat"><div class="n ok">${s.ok}</div><div class="l">Passed</div></div>
      <div class="stat"><div class="n warn">${s.warnings}</div><div class="l">Warnings</div></div>
      <div class="stat"><div class="n err">${s.errors}</div><div class="l">Errors</div></div>
    `;

    // Health banner
    const banner = document.getElementById('health-banner');
    const hl = {
      healthy:  '✓ System healthy — all checks passed',
      degraded: '! System degraded — warnings need attention',
      critical: '✗ System critical — errors must be fixed before running',
    };
    banner.className = s.health;
    banner.textContent = hl[s.health] || s.health;
    banner.style.display = 'block';

    // Individual checks
    const checksEl = document.getElementById('checks');
    checksEl.innerHTML = '';
    for (const c of d.checks) {
      const div = document.createElement('div');
      div.className = 'check ' + c.status;
      let extra = '';
      if (c.status === 'error' && c.detail)
        extra += `<div class="cerr">${escHtml(c.detail.substring(0,300))}</div>`;
      if (c.fix)
        extra += `<div class="cfix">Fix: ${escHtml(c.fix)}</div>`;
      div.innerHTML = `
        <span class="icon" style="color:var(--${c.status==='ok'?'ok':c.status==='warning'?'warn':c.status==='error'?'err':'mut'})">${ICONS[c.status]}</span>
        <div>
          <div class="cname">${escHtml(c.name)}</div>
          <div class="cmsg">${escHtml(c.message)}</div>
          ${extra}
        </div>
        <div style="text-align:right">
          <span class="badge ${c.status}">${c.status}</span>
          <div class="cdur">${c.ms}ms</div>
        </div>`;
      checksEl.appendChild(div);
    }
  } catch(e) {
    document.getElementById('checks').innerHTML =
      `<div class="check error">
        <span class="icon" style="color:var(--err)">✗</span>
        <div><div class="cname">Debugger connection failed</div>
        <div class="cmsg">${escHtml(String(e))}</div>
        <div class="cfix">Fix: Make sure the debugger is running on port 8080</div></div>
        <span></span></div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Run Diagnostics';
  }
}
function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
window.onload = runDebug;
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
