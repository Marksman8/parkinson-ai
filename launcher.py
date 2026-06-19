"""
PD AI System — Unified Launcher

Spawns all backend services and serves a single sidebar-driven dashboard:
  - port 5000  → this launcher / dashboard
  - port 8000  → pd_multimodal  (chat, MRI, report, gene, drug, simulate)
  - port 8001  → module0_biomarker  (biomarker discovery, PPI, KEGG)
  - port 8080  → pd_debugger  (system diagnostics)
"""
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

app = FastAPI(title="PD AI Dashboard")


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PD AI · Clinical Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg-0:#0a0a0c;
  --bg-1:#111114;
  --bg-2:#17171c;
  --bg-3:#1e1e25;
  --bdr:#26262f;
  --bdr-2:#33333f;
  --txt:#e8e8ec;
  --txt-2:#a8a8b3;
  --txt-3:#6e6e78;
  --accent:#7c5cff;
  --accent-2:#5b8dff;
  --ok:#22c55e;
  --warn:#f59e0b;
  --err:#ef4444;
  --user-bubble:#1f1f27;
  --asst-bubble:transparent;
  --sb-w:260px;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:var(--bg-0);
  color:var(--txt);
  font-size:14.5px;
  line-height:1.55;
  overflow:hidden;
}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bdr-2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#444}

/* ── Layout ───────────────────────────────────────────────────── */
.app{display:flex;height:100vh;width:100vw}

/* ── Sidebar ──────────────────────────────────────────────────── */
.sidebar{
  width:var(--sb-w);
  background:var(--bg-1);
  border-right:1px solid var(--bdr);
  display:flex;
  flex-direction:column;
  flex-shrink:0;
}
.brand{
  padding:18px 18px 16px;
  display:flex;
  align-items:center;
  gap:10px;
  border-bottom:1px solid var(--bdr);
}
.brand-logo{
  width:30px;height:30px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:14px;color:#fff;
  box-shadow:0 4px 14px rgba(124,92,255,0.35);
}
.brand-name{font-weight:600;font-size:14.5px;letter-spacing:-0.01em}
.brand-sub{font-size:11px;color:var(--txt-3);margin-top:1px}

.nav{flex:1;overflow-y:auto;padding:10px 10px 14px}
.nav-group{margin-top:14px}
.nav-group-title{
  font-size:10.5px;
  text-transform:uppercase;
  letter-spacing:0.08em;
  color:var(--txt-3);
  padding:6px 10px;
  font-weight:600;
}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:8px 10px;
  border-radius:7px;
  color:var(--txt-2);
  cursor:pointer;
  font-size:13.5px;
  user-select:none;
  transition:background 0.12s ease,color 0.12s ease;
  margin-bottom:1px;
}
.nav-item:hover{background:var(--bg-2);color:var(--txt)}
.nav-item.active{background:var(--bg-3);color:var(--txt)}
.nav-item .ico{
  width:22px;height:22px;
  display:flex;align-items:center;justify-content:center;
  font-size:14px;
  flex-shrink:0;
}
.sidebar-footer{
  padding:10px 14px 14px;
  border-top:1px solid var(--bdr);
  font-size:11.5px;color:var(--txt-3);
}
.svc{display:flex;align-items:center;gap:8px;padding:3px 0}
.dot{width:7px;height:7px;border-radius:50%;background:var(--txt-3);flex-shrink:0}
.dot.on{background:var(--ok);box-shadow:0 0 0 3px rgba(34,197,94,0.15)}
.dot.off{background:var(--err)}
.dot.warm{background:var(--warn);box-shadow:0 0 0 3px rgba(245,158,11,0.18);animation:warm-pulse 1.4s ease-in-out infinite}
@keyframes warm-pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.svc .lbl-warm{color:var(--warn);font-size:10.5px;margin-left:6px;font-family:'JetBrains Mono',monospace}

/* ── Main ─────────────────────────────────────────────────────── */
.main{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--bg-0)}
.topbar{
  padding:14px 24px;
  border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;justify-content:space-between;
  flex-shrink:0;
}
.topbar h1{font-size:15px;font-weight:600;letter-spacing:-0.01em}
.topbar .sub{font-size:12px;color:var(--txt-3);margin-top:1px}
.topbar .right{font-size:12px;color:var(--txt-3);font-family:'JetBrains Mono',monospace}

.view{flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.view:not(.active){display:none}

/* ── Cards / forms (used by non-chat modules) ─────────────────── */
.scroll{flex:1;overflow-y:auto;padding:28px 32px}
.container{max-width:880px;margin:0 auto}
.card{
  background:var(--bg-1);
  border:1px solid var(--bdr);
  border-radius:12px;
  padding:22px;
  margin-bottom:18px;
}
.card h2{font-size:15px;font-weight:600;margin-bottom:6px}
.card p.desc{color:var(--txt-2);font-size:13px;margin-bottom:18px}
.field{margin-bottom:14px}
.field label{display:block;font-size:12px;color:var(--txt-2);margin-bottom:6px;font-weight:500}
.field input[type="text"],
.field input[type="number"],
.field textarea,
.field select{
  width:100%;
  background:var(--bg-2);
  border:1px solid var(--bdr);
  border-radius:8px;
  padding:10px 12px;
  color:var(--txt);
  font-family:inherit;
  font-size:13.5px;
  outline:none;
  transition:border-color 0.15s;
}
.field input:focus,.field textarea:focus,.field select:focus{border-color:var(--accent)}
.field textarea{min-height:120px;resize:vertical;line-height:1.5}
.field input[type="file"]{
  width:100%;
  background:var(--bg-2);
  border:1px dashed var(--bdr-2);
  border-radius:8px;
  padding:18px;
  color:var(--txt-2);
  font-size:13px;
  cursor:pointer;
}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--accent);color:#fff;
  border:none;border-radius:8px;
  padding:10px 18px;
  font-family:inherit;font-size:13.5px;font-weight:500;
  cursor:pointer;
  transition:opacity 0.15s,transform 0.05s;
}
.btn:hover{opacity:0.9}
.btn:active{transform:translateY(1px)}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.btn.ghost{background:transparent;border:1px solid var(--bdr-2);color:var(--txt-2)}
.btn.ghost:hover{color:var(--txt);border-color:var(--bdr-2)}
.result{
  margin-top:18px;
  background:var(--bg-2);
  border:1px solid var(--bdr);
  border-radius:10px;
  padding:16px 18px;
  white-space:pre-wrap;
  font-size:13.5px;
  line-height:1.65;
  color:var(--txt);
  max-height:520px;
  overflow-y:auto;
}
.result.empty{display:none}
.result img{max-width:100%;border-radius:8px;margin-top:10px;display:block}
.spinner{
  width:14px;height:14px;
  border:2px solid rgba(255,255,255,0.25);
  border-top-color:#fff;
  border-radius:50%;
  animation:spin 0.7s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.pill{
  display:inline-block;
  padding:3px 9px;
  border-radius:99px;
  background:var(--bg-3);
  color:var(--txt-2);
  font-size:11px;
  margin:2px 4px 2px 0;
  font-family:'JetBrains Mono',monospace;
}
.error{color:var(--err);font-size:13px}

/* ── Welcome view ─────────────────────────────────────────────── */
.welcome{
  flex:1;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:40px;
  text-align:center;
}
.welcome h2{
  font-size:30px;font-weight:600;letter-spacing:-0.02em;margin-bottom:10px;
  background:linear-gradient(135deg,#fff 0%,#a8a8b3 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.welcome p{color:var(--txt-2);max-width:540px;margin-bottom:32px;font-size:14.5px}
.welcome-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  gap:14px;
  width:100%;max-width:760px;
}
.welcome-card{
  background:var(--bg-1);
  border:1px solid var(--bdr);
  border-radius:12px;
  padding:18px;
  text-align:left;
  cursor:pointer;
  transition:border-color 0.15s,transform 0.1s;
}
.welcome-card:hover{border-color:var(--bdr-2);transform:translateY(-2px)}
.welcome-card .ic{font-size:20px;margin-bottom:10px}
.welcome-card h3{font-size:13.5px;font-weight:600;margin-bottom:4px}
.welcome-card p{font-size:12px;color:var(--txt-3);margin:0}

/* ── Chat view (ChatGPT style) ────────────────────────────────── */
.chat{flex:1;display:flex;flex-direction:column;min-height:0}
.chat-stream{
  flex:1;overflow-y:auto;
  padding:28px 0 24px;
}
.chat-empty{
  height:100%;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:var(--txt-3);
  padding:40px;text-align:center;
}
.chat-empty .big{font-size:24px;color:var(--txt);font-weight:500;margin-bottom:8px;letter-spacing:-0.01em}
.chat-empty .small{font-size:13.5px;max-width:420px}
.msg{
  max-width:760px;
  margin:0 auto 18px;
  padding:0 24px;
  display:flex;gap:14px;
  align-items:flex-start;
}
.msg .avatar{
  width:28px;height:28px;border-radius:6px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:600;
}
.msg.user .avatar{background:var(--bg-3);color:var(--txt)}
.msg.asst .avatar{background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#fff}
.msg .bubble{
  flex:1;
  font-size:14.5px;line-height:1.65;
  white-space:pre-wrap;word-wrap:break-word;
  padding-top:3px;
}
.msg.user .bubble{color:var(--txt)}
.msg.asst .bubble{color:var(--txt)}
.msg.asst .bubble.thinking{color:var(--txt-3);font-style:italic}

.chat-input-wrap{
  border-top:1px solid var(--bdr);
  padding:14px 24px 18px;
  background:var(--bg-0);
}
.chat-input-inner{
  max-width:760px;margin:0 auto;
  background:var(--bg-2);
  border:1px solid var(--bdr-2);
  border-radius:14px;
  padding:8px 8px 8px 16px;
  display:flex;align-items:flex-end;gap:8px;
  transition:border-color 0.15s;
}
.chat-input-inner:focus-within{border-color:var(--accent)}
#chat-text{
  flex:1;
  background:transparent;border:none;outline:none;resize:none;
  color:var(--txt);font-family:inherit;font-size:14.5px;
  line-height:1.55;
  padding:8px 0;
  max-height:200px;
  min-height:24px;
}
.send-btn{
  width:34px;height:34px;border-radius:8px;
  background:var(--accent);color:#fff;
  border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
  transition:opacity 0.15s;
}
.send-btn:hover{opacity:0.9}
.send-btn:disabled{opacity:0.4;cursor:not-allowed}
.chat-hint{
  max-width:760px;margin:8px auto 0;
  font-size:11px;color:var(--txt-3);text-align:center;
}

/* ── Debugger panel ───────────────────────────────────────────── */
.dbg-summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:18px}
.stat{background:var(--bg-1);border:1px solid var(--bdr);border-radius:10px;padding:14px;text-align:center}
.stat .n{font-size:24px;font-weight:600}
.stat .l{font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--txt-3);margin-top:2px}
.n.ok{color:var(--ok)} .n.warn{color:var(--warn)} .n.err{color:var(--err)} .n.blue{color:var(--accent-2)}
.dbg-checks{display:flex;flex-direction:column;gap:8px}
.dbg-check{
  background:var(--bg-1);border:1px solid var(--bdr);
  border-radius:10px;padding:10px 14px;
  display:grid;grid-template-columns:22px 1fr auto;gap:10px;align-items:start;
}
.dbg-check.ok{border-left:3px solid var(--ok)}
.dbg-check.warning{border-left:3px solid var(--warn)}
.dbg-check.error{border-left:3px solid var(--err)}
.dbg-check.skip{border-left:3px solid var(--txt-3)}
.dbg-check .cname{font-size:12.5px;font-weight:500}
.dbg-check .cmsg{font-size:11.5px;color:var(--txt-2);margin-top:2px}
.dbg-check .cfix{font-size:11px;color:var(--warn);margin-top:5px;padding:4px 7px;background:rgba(245,158,11,0.08);border-radius:4px}
.dbg-check .cdur{font-size:10.5px;color:var(--txt-3);font-family:'JetBrains Mono',monospace}
.badge{padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600;text-transform:uppercase}
.badge.ok{background:rgba(34,197,94,0.15);color:var(--ok)}
.badge.warning{background:rgba(245,158,11,0.15);color:var(--warn)}
.badge.error{background:rgba(239,68,68,0.15);color:var(--err)}
.badge.skip{background:rgba(110,110,120,0.15);color:var(--txt-3)}

/* ── MRI verdict card ────────────────────────────────────────── */
.verdict-card{
  background:var(--bg-1);border:1px solid var(--bdr);
  border-radius:12px;padding:18px;margin-top:16px;
  display:grid;grid-template-columns:minmax(0,1fr) 220px;gap:18px;
}
@media (max-width:760px){.verdict-card{grid-template-columns:1fr}}
.verdict-head{display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.v-badge{
  padding:5px 12px;border-radius:99px;font-size:12px;font-weight:600;
  border:1px solid var(--bdr-2);
}
.v-badge.pd-likely  {background:rgba(239,68,68,0.12); color:#fca5a5; border-color:rgba(239,68,68,0.35)}
.v-badge.pd-unlikely{background:rgba(34,197,94,0.12); color:#86efac; border-color:rgba(34,197,94,0.35)}
.v-badge.inconclusive,.v-badge.unknown{background:rgba(245,158,11,0.12); color:#fcd34d; border-color:rgba(245,158,11,0.35)}
.v-badge.normal     {background:rgba(91,141,255,0.12); color:#a5c1ff; border-color:rgba(91,141,255,0.35)}
.v-conf{font-size:11.5px;color:var(--txt-3);font-family:'JetBrains Mono',monospace}
.v-section{margin-top:12px}
.v-section h4{font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:var(--txt-3);margin-bottom:6px;font-weight:600}
.v-findings{list-style:none;padding:0;margin:0}
.v-findings li{padding:4px 0 4px 14px;position:relative;font-size:13px;color:var(--txt)}
.v-findings li::before{content:"›";position:absolute;left:0;color:var(--accent)}
.v-text{font-size:13px;line-height:1.6;color:var(--txt)}
.v-disclaimer{margin-top:14px;font-size:11.5px;color:var(--txt-3);padding-top:10px;border-top:1px solid var(--bdr)}
.v-image{width:100%;border-radius:10px;border:1px solid var(--bdr);object-fit:cover;background:var(--bg-2)}
.v-image-wrap{display:flex;flex-direction:column;gap:8px}
.v-image-wrap .cap{font-size:11px;color:var(--txt-3);text-align:center}
.v-raw{
  margin-top:14px;
  background:var(--bg-2);border:1px solid var(--bdr);border-radius:8px;
  padding:10px 12px;
  font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--txt-2);
  white-space:pre-wrap;max-height:240px;overflow:auto;
}
details.v-details{margin-top:14px}
details.v-details summary{cursor:pointer;font-size:11.5px;color:var(--txt-3);user-select:none}
details.v-details summary:hover{color:var(--txt-2)}

/* ── Tabs ────────────────────────────────────────────────────── */
.tabs{display:flex;gap:4px;background:var(--bg-2);padding:4px;border-radius:9px;margin-bottom:16px;width:fit-content}
.tab{
  padding:7px 14px;border-radius:6px;
  font-size:12.5px;font-weight:500;color:var(--txt-2);
  cursor:pointer;user-select:none;
  transition:background 0.12s,color 0.12s;
}
.tab:hover{color:var(--txt)}
.tab.active{background:var(--bg-3);color:var(--txt)}
.tab-panel{display:none}
.tab-panel.active{display:block}
</style>
</head>
<body>
<div class="app">

  <!-- ═════════ SIDEBAR ═════════ -->
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-logo">PD</div>
      <div>
        <div class="brand-name">PD AI</div>
        <div class="brand-sub">Clinical Intelligence</div>
      </div>
    </div>

    <nav class="nav">
      <div class="nav-group">
        <div class="nav-group-title">Overview</div>
        <div class="nav-item active" data-view="home"><span class="ico">⌂</span>Dashboard</div>
      </div>

      <div class="nav-group">
        <div class="nav-group-title">Clinical · Multimodal</div>
        <div class="nav-item" data-view="chat"><span class="ico">◈</span>Clinical Chat</div>
        <div class="nav-item" data-view="mri"><span class="ico">▤</span>MRI Analysis</div>
        <div class="nav-item" data-view="report"><span class="ico">▦</span>Report Summary</div>
        <div class="nav-item" data-view="gene"><span class="ico">⌬</span>Gene Analysis</div>
        <div class="nav-item" data-view="drug"><span class="ico">℞</span>Drug Recommendation</div>
        <div class="nav-item" data-view="simulate"><span class="ico">∿</span>Neuron Simulation</div>
      </div>

      <div class="nav-group">
        <div class="nav-group-title">Discovery</div>
        <div class="nav-item" data-view="biomarker"><span class="ico">◉</span>Biomarker Discovery</div>
      </div>

      <div class="nav-group">
        <div class="nav-group-title">Research</div>
        <div class="nav-item" data-view="evaluation"><span class="ico">▣</span>Evaluation Results</div>
      </div>

      <div class="nav-group">
        <div class="nav-group-title">System</div>
        <div class="nav-item" data-view="debugger"><span class="ico">⚙</span>Debugger</div>
      </div>
    </nav>

    <div class="sidebar-footer">
      <div class="svc"><span class="dot" id="dot-8000"></span><span>Multimodal · :8000</span><span id="lbl-8000"></span></div>
      <div class="svc"><span class="dot" id="dot-8001"></span><span>Biomarker · :8001</span><span id="lbl-8001"></span></div>
      <div class="svc"><span class="dot" id="dot-8080"></span><span>Debugger · :8080</span><span id="lbl-8080"></span></div>
    </div>
  </aside>

  <!-- ═════════ MAIN ═════════ -->
  <main class="main">

    <!-- ── Top bar ── -->
    <div class="topbar">
      <div>
        <h1 id="topbar-title">Dashboard</h1>
        <div class="sub" id="topbar-sub">Overview of all clinical AI modules</div>
      </div>
      <div class="right" id="clock"></div>
    </div>

    <!-- ── Home / Welcome ── -->
    <section class="view active" id="view-home">
      <div class="welcome">
        <h2>Parkinson's Clinical AI</h2>
        <p>A unified workspace for multimodal analysis, biomarker discovery, and live system diagnostics. Pick a module from the sidebar to begin.</p>
        <div class="welcome-grid">
          <div class="welcome-card" data-view="chat">
            <div class="ic">◈</div>
            <h3>Clinical Chat</h3>
            <p>Ask Parkinson's questions with full context.</p>
          </div>
          <div class="welcome-card" data-view="mri">
            <div class="ic">▤</div>
            <h3>MRI Analysis</h3>
            <p>Upload a brain scan for structural reasoning.</p>
          </div>
          <div class="welcome-card" data-view="biomarker">
            <div class="ic">◉</div>
            <h3>Biomarker Discovery</h3>
            <p>KEGG + STRING pipeline on clinical notes.</p>
          </div>
          <div class="welcome-card" data-view="drug">
            <div class="ic">℞</div>
            <h3>Drug Recommendation</h3>
            <p>Pharmacogenomics-aware suggestions.</p>
          </div>
          <div class="welcome-card" data-view="simulate">
            <div class="ic">∿</div>
            <h3>Neuron Simulation</h3>
            <p>Hodgkin-Huxley + dopamine kinetics.</p>
          </div>
          <div class="welcome-card" data-view="debugger">
            <div class="ic">⚙</div>
            <h3>Debugger</h3>
            <p>Run a full diagnostic across services.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ── Chat ── -->
    <section class="view" id="view-chat">
      <div class="chat">
        <div class="chat-stream" id="chat-stream">
          <div class="chat-empty" id="chat-empty">
            <div class="big">How can I help with your Parkinson's case?</div>
            <div class="small">Ask about symptoms, biomarkers, imaging, treatment options, or genetic risk — answers ground in the latest Module 0 discovery if one is active.</div>
          </div>
        </div>
        <div class="chat-input-wrap">
          <div class="chat-input-inner">
            <textarea id="chat-text" rows="1" placeholder="Message PD AI…"></textarea>
            <button class="send-btn" id="chat-send" title="Send">▲</button>
          </div>
          <div class="chat-hint">Enter to send · Shift+Enter for newline · Connected to pd_multimodal :8000</div>
        </div>
      </div>
    </section>

    <!-- ── MRI ── -->
    <section class="view" id="view-mri">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>MRI Analysis</h2>
          <p class="desc">Returns a structured verdict (PD-likely / unlikely / inconclusive / normal), confidence, key findings, and a recommended next step. Analyze a raw brain MRI image, or upload an existing radiology PDF report.</p>

          <div class="tabs" id="mri-tabs">
            <div class="tab active" data-tab="image">MRI Image</div>
            <div class="tab" data-tab="pdf">MRI Report (PDF)</div>
          </div>

          <div class="tab-panel active" id="mri-tab-image">
            <div class="field">
              <label>MRI image (PNG / JPG)</label>
              <input type="file" id="mri-file" accept="image/*">
            </div>
            <button class="btn" id="mri-go">Analyze MRI Image</button>
          </div>

          <div class="tab-panel" id="mri-tab-pdf">
            <div class="field">
              <label>Radiology MRI report (PDF)</label>
              <input type="file" id="mri-pdf-file" accept="application/pdf">
            </div>
            <button class="btn" id="mri-pdf-go">Analyze MRI Report</button>
            <div style="font-size:11.5px;color:var(--txt-3);margin-top:8px">
              Text-only reports parse fastest. Scanned/image-only PDFs may not extract — use the Image tab instead.
            </div>
          </div>

          <div class="result empty" id="mri-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Report ── -->
    <section class="view" id="view-report">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Clinical Report Summary</h2>
          <p class="desc">Upload a clinical PDF report — extracts patient summary, biomarker levels, diagnosis, treatment, and follow-up.</p>
          <div class="field">
            <label>Clinical PDF</label>
            <input type="file" id="report-file" accept="application/pdf">
          </div>
          <button class="btn" id="report-go">Summarize Report</button>
          <div class="result empty" id="report-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Gene ── -->
    <section class="view" id="view-gene">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Gene Mutation Analysis</h2>
          <p class="desc">Paste a genetic report or mutation list — returns per-mutation risk level, affected pathway, and counseling guidance.</p>
          <div class="field">
            <label>Genetic report text</label>
            <textarea id="gene-text" placeholder="e.g. LRRK2 G2019S heterozygous, GBA N370S heterozygous…"></textarea>
          </div>
          <button class="btn" id="gene-go">Interpret Mutations</button>
          <div class="result empty" id="gene-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Drug ── -->
    <section class="view" id="view-drug">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Drug Recommendation</h2>
          <p class="desc">Pharmacogenomics-aware suggestions based on patient profile.</p>
          <div class="row">
            <div class="field">
              <label>Mutations (comma-separated)</label>
              <input type="text" id="drug-mutations" placeholder="LRRK2 G2019S, GBA N370S">
            </div>
            <div class="field">
              <label>Disease stage</label>
              <select id="drug-stage">
                <option value="early">Early</option>
                <option value="moderate" selected>Moderate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label>Current symptoms (comma-separated)</label>
            <input type="text" id="drug-symptoms" placeholder="tremor, bradykinesia, rigidity">
          </div>
          <button class="btn" id="drug-go">Recommend Drugs</button>
          <div class="result empty" id="drug-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Simulate ── -->
    <section class="view" id="view-simulate">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Neuron Simulation</h2>
          <p class="desc">Run an ODE-based neuron model. Hodgkin-Huxley simulates action potentials; the dopamine model captures synaptic loss in PD.</p>
          <div class="row">
            <div class="field">
              <label>Model</label>
              <select id="sim-model">
                <option value="hh">Hodgkin-Huxley</option>
                <option value="dopamine">Dopamine kinetics</option>
              </select>
            </div>
            <div class="field">
              <label>I_ext (HH) — μA/cm²</label>
              <input type="number" id="sim-iext" value="10" step="0.5">
            </div>
          </div>
          <div class="field">
            <label>PD loss (dopamine model) — 0.0 healthy → 1.0 severe</label>
            <input type="number" id="sim-loss" value="0" min="0" max="1" step="0.1">
          </div>
          <button class="btn" id="sim-go">Run Simulation</button>
          <div class="result empty" id="sim-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Biomarker ── -->
    <section class="view" id="view-biomarker">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Biomarker Discovery</h2>
          <p class="desc">Module 0 — automated KEGG pathway + STRING PPI pipeline. Provide clinical text or upload a PDF; the system returns hub genes and a network graph.</p>
          <div class="field">
            <label>Clinical text</label>
            <textarea id="bio-text" placeholder="Patient presents with resting tremor, rigidity, and bradykinesia. LRRK2 G2019S mutation…"></textarea>
          </div>
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
            <button class="btn" id="bio-go-text">Discover from Text</button>
            <span style="color:var(--txt-3);font-size:12px">or upload PDF:</span>
            <input type="file" id="bio-file" accept="application/pdf" style="flex:1;max-width:280px;padding:8px;background:var(--bg-2);border:1px solid var(--bdr);border-radius:8px;color:var(--txt-2);font-size:12.5px">
            <button class="btn ghost" id="bio-go-pdf">Discover from PDF</button>
          </div>
          <div class="result empty" id="bio-result"></div>
        </div>
      </div></div>
    </section>

    <!-- ── Evaluation Results ── -->
    <section class="view" id="view-evaluation">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>Quantitative Evaluation</h2>
          <p class="desc">Reproducible benchmark results for the clinical and retrieval modules. Generated by the evaluation harnesses (<span class="pill">pd_clinical/eval</span><span class="pill">pd_rag/eval</span>).</p>
          <div class="tabs" id="eval-tabs">
            <div class="tab active" data-tab="clinical">Clinical · PPMI</div>
            <div class="tab" data-tab="rag">RAG Benchmark</div>
          </div>
          <div class="tab-panel active" id="eval-tab-clinical">
            <div id="eval-clinical"></div>
          </div>
          <div class="tab-panel" id="eval-tab-rag">
            <div id="eval-rag"></div>
          </div>
        </div>
      </div></div>
    </section>

    <!-- ── Debugger ── -->
    <section class="view" id="view-debugger">
      <div class="scroll"><div class="container">
        <div class="card">
          <h2>System Debugger</h2>
          <p class="desc">Live diagnostic across the multimodal stack — health, deps, model availability, ports, and integration.</p>
          <button class="btn" id="dbg-go">▶ Run Diagnostics</button>
          <span id="dbg-ts" style="margin-left:14px;color:var(--txt-3);font-size:11.5px;font-family:'JetBrains Mono',monospace"></span>
        </div>
        <div class="dbg-summary" id="dbg-summary" style="display:none"></div>
        <div class="dbg-checks" id="dbg-checks"></div>
      </div></div>
    </section>

  </main>
</div>

<script>
// ════════ Backend endpoints ════════
const API_MM = 'http://localhost:8000';
const API_BIO = 'http://localhost:8001';
const API_DBG = 'http://localhost:8080';

// ════════ Routing / view switching ════════
const VIEW_TITLES = {
  home:      ['Dashboard',           'Overview of all clinical AI modules'],
  chat:      ['Clinical Chat',       'Ask Parkinson\'s questions · pd_multimodal :8000'],
  mri:       ['MRI Analysis',        'Brain scan interpretation · /api/mri'],
  report:    ['Report Summary',      'Extract structured insights from PDF · /api/report'],
  gene:      ['Gene Analysis',       'Mutation risk interpretation · /api/gene'],
  drug:      ['Drug Recommendation', 'Pharmacogenomic suggestions · /api/drug'],
  simulate:  ['Neuron Simulation',   'Hodgkin-Huxley / dopamine kinetics · /api/simulate'],
  biomarker: ['Biomarker Discovery', 'KEGG + STRING pipeline · module0 :8001'],
  debugger:  ['Debugger',            'System diagnostics · pd_debugger :8080'],
  evaluation:['Evaluation Results',  'Quantitative benchmarks · PPMI classification + RAG'],
};

function setView(view){
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === view));
  const [t, s] = VIEW_TITLES[view] || ['', ''];
  document.getElementById('topbar-title').textContent = t;
  document.getElementById('topbar-sub').textContent   = s;
}
document.querySelectorAll('.nav-item').forEach(n =>
  n.addEventListener('click', () => setView(n.dataset.view)));
document.querySelectorAll('.welcome-card').forEach(c =>
  c.addEventListener('click', () => setView(c.dataset.view)));

// ════════ Clock ════════
function tick(){
  const d = new Date();
  document.getElementById('clock').textContent =
    d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
}
setInterval(tick, 1000); tick();

// ════════ Service health polling ════════
//   on   = /health responded OK
//   warm = service not yet listening (lifespan still running — models loading)
//   off  = stayed unreachable for more than 90s (probably crashed)
const svcState = {
  '8000': {firstSeenDown: Date.now(), ever: false},
  '8001': {firstSeenDown: Date.now(), ever: false},
  '8080': {firstSeenDown: Date.now(), ever: false},
};
async function pingService(url, port){
  const dot = document.getElementById('dot-' + port);
  const lbl = document.getElementById('lbl-' + port);
  const st  = svcState[port];
  try{
    const ctrl = new AbortController();
    const tid = setTimeout(()=>ctrl.abort(), 2500);
    const r = await fetch(url + '/health', {signal: ctrl.signal});
    clearTimeout(tid);
    if (r.ok) {
      st.ever = true;
      st.firstSeenDown = null;
      dot.classList.remove('off','warm'); dot.classList.add('on');
      if (lbl) lbl.textContent = '';
      return;
    }
    throw new Error('not ok');
  } catch {
    dot.classList.remove('on');
    // If we've never seen it up AND it's been <90s, it's still warming up
    if (!st.ever) {
      dot.classList.remove('off'); dot.classList.add('warm');
      if (lbl) lbl.innerHTML = '<span class="lbl-warm">· warming up</span>';
    } else {
      dot.classList.remove('warm'); dot.classList.add('off');
      if (lbl) lbl.textContent = '';
    }
  }
}
function isWarming(port){
  return document.getElementById('dot-' + port).classList.contains('warm');
}
async function pollHealth(){
  await Promise.all([
    pingService(API_MM,  '8000'),
    pingService(API_BIO, '8001'),
    pingService(API_DBG, '8080'),
  ]);
}
pollHealth();
setInterval(pollHealth, 4000);

// ════════ Helpers ════════
function showResult(id, text, isError){
  const el = document.getElementById(id);
  el.classList.remove('empty');
  el.innerHTML = '';
  if (isError) {
    const e = document.createElement('div');
    e.className = 'error';
    e.textContent = text;
    el.appendChild(e);
  } else if (typeof text === 'string') {
    el.textContent = text;
  } else {
    el.appendChild(text);
  }
}
function loadingBtn(btn, label){
  btn.disabled = true;
  btn._orig = btn.innerHTML;
  btn.innerHTML = `<span class="spinner"></span> ${label || 'Working…'}`;
}
function restoreBtn(btn){
  btn.disabled = false;
  if (btn._orig) btn.innerHTML = btn._orig;
}

// ════════ Chat module ════════
const chatStream = document.getElementById('chat-stream');
const chatEmpty  = document.getElementById('chat-empty');
const chatText   = document.getElementById('chat-text');
const chatSend   = document.getElementById('chat-send');

chatText.addEventListener('input', () => {
  chatText.style.height = 'auto';
  chatText.style.height = Math.min(chatText.scrollHeight, 200) + 'px';
});
chatText.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
chatSend.addEventListener('click', sendChat);

function addMsg(role, text){
  if (chatEmpty.parentNode) chatEmpty.remove();
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? 'You' : 'PD';
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.textContent = text;
  wrap.appendChild(av); wrap.appendChild(bub);
  chatStream.appendChild(wrap);
  chatStream.scrollTop = chatStream.scrollHeight;
  return bub;
}

async function sendChat(){
  const q = chatText.value.trim();
  if (!q) return;
  addMsg('user', q);
  chatText.value = '';
  chatText.style.height = 'auto';
  const bub = addMsg('asst', '…');
  bub.classList.add('thinking');
  chatSend.disabled = true;

  let acc = '';
  try {
    const fd = new FormData();
    fd.append('question', q);
    const r = await fetch(API_MM + '/api/chat/stream', {method:'POST', body:fd});
    if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);

    const reader  = r.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let started = false;

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      // SSE messages are separated by blank lines
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }

        if (obj.t) {
          if (!started) { bub.textContent = ''; bub.classList.remove('thinking'); started = true; }
          acc += obj.t;
          bub.textContent = acc;
          chatStream.scrollTop = chatStream.scrollHeight;
        } else if (obj.error) {
          bub.classList.remove('thinking');
          bub.textContent = (acc || '') + (acc ? '\n\n' : '') + '[Error] ' + obj.error;
        } else if (obj.done) {
          // stream complete
        }
      }
    }
    if (!acc && !bub.textContent.startsWith('[Error]')) {
      bub.classList.remove('thinking');
      bub.textContent = '(no response)';
    }
  } catch (e) {
    bub.classList.remove('thinking');
    if (isWarming('8000')) {
      bub.textContent =
        'The multimodal service on :8000 is still warming up — models are loading. ' +
        'First boot can take 30–120s. Watch the sidebar dot turn green, then resend.';
    } else {
      bub.textContent = 'Connection failed: ' + e.message + '\n\nIs pd_multimodal running on :8000?';
    }
  } finally {
    chatSend.disabled = false;
    chatStream.scrollTop = chatStream.scrollHeight;
  }
}

// ════════ MRI ════════
function verdictClass(v){
  const s = String(v||'').toLowerCase();
  if (s.includes('pd-likely') || s.includes('likely')) return 'pd-likely';
  if (s.includes('pd-unlikely') || s.includes('unlikely')) return 'pd-unlikely';
  if (s.includes('normal')) return 'normal';
  if (s.includes('inconclusive')) return 'inconclusive';
  return 'unknown';
}
function renderMRI(d, imgDataUrl, source){
  if (d.error) { showResult('mri-result', d.error, true); return; }
  const card = document.createElement('div');
  card.className = 'verdict-card';

  // Left column — verdict, findings, reasoning
  const left = document.createElement('div');
  const head = document.createElement('div');
  head.className = 'verdict-head';
  const vc = verdictClass(d.verdict);
  head.innerHTML =
    `<span class="v-badge ${vc}">${escHtml(d.verdict || 'Inconclusive')}</span>` +
    `<span class="v-conf">confidence · ${escHtml(d.confidence || 'Low')}</span>` +
    (source ? `<span class="v-conf">source · ${escHtml(source)}</span>` : '');
  left.appendChild(head);

  if (d.findings && d.findings.length) {
    const sec = document.createElement('div');
    sec.className = 'v-section';
    sec.innerHTML = '<h4>Key Findings</h4>';
    const ul = document.createElement('ul');
    ul.className = 'v-findings';
    d.findings.forEach(f => {
      const li = document.createElement('li');
      li.textContent = f;
      ul.appendChild(li);
    });
    sec.appendChild(ul);
    left.appendChild(sec);
  }
  if (d.reasoning) {
    const sec = document.createElement('div');
    sec.className = 'v-section';
    sec.innerHTML = '<h4>Reasoning</h4><div class="v-text"></div>';
    sec.querySelector('.v-text').textContent = d.reasoning;
    left.appendChild(sec);
  }
  if (d.recommendation) {
    const sec = document.createElement('div');
    sec.className = 'v-section';
    sec.innerHTML = '<h4>Recommendation</h4><div class="v-text"></div>';
    sec.querySelector('.v-text').textContent = d.recommendation;
    left.appendChild(sec);
  }

  const disc = document.createElement('div');
  disc.className = 'v-disclaimer';
  disc.textContent = 'AI assistive output — not a diagnosis. A neurologist must confirm.';
  left.appendChild(disc);

  // Raw output (collapsed)
  const det = document.createElement('details');
  det.className = 'v-details';
  det.innerHTML = '<summary>Raw model output</summary>';
  const raw = document.createElement('div');
  raw.className = 'v-raw';
  raw.textContent = d.analysis || '(empty)';
  det.appendChild(raw);
  left.appendChild(det);

  // Right column — uploaded image preview (or PDF icon when source is the report)
  const right = document.createElement('div');
  right.className = 'v-image-wrap';
  if (imgDataUrl) {
    const img = document.createElement('img');
    img.className = 'v-image';
    img.src = imgDataUrl;
    img.alt = 'MRI';
    right.appendChild(img);
    const cap = document.createElement('div');
    cap.className = 'cap';
    cap.textContent = 'Uploaded MRI';
    right.appendChild(cap);
  } else if (source === 'pdf-report') {
    const ph = document.createElement('div');
    ph.style.cssText =
      'width:100%;aspect-ratio:1/1;border-radius:10px;background:var(--bg-2);' +
      'border:1px solid var(--bdr);display:flex;align-items:center;justify-content:center;' +
      'font-size:42px;color:var(--txt-3)';
    ph.textContent = '▦';
    right.appendChild(ph);
    const cap = document.createElement('div');
    cap.className = 'cap';
    cap.textContent = 'Uploaded MRI report (PDF)';
    right.appendChild(cap);
  }

  card.appendChild(left);
  if (right.childElementCount) card.appendChild(right);
  showResult('mri-result', card, false);
}

// MRI tab switching
document.querySelectorAll('#mri-tabs .tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('#mri-tabs .tab').forEach(x => x.classList.toggle('active', x === t));
    const which = t.dataset.tab;
    document.getElementById('mri-tab-image').classList.toggle('active', which === 'image');
    document.getElementById('mri-tab-pdf').classList.toggle('active', which === 'pdf');
  });
});

document.getElementById('mri-go').addEventListener('click', async (e) => {
  const file = document.getElementById('mri-file').files[0];
  if (!file) return showResult('mri-result', 'Please choose an image first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Analyzing…');

  // Read the image to a data URL so we can preview it next to the verdict
  const imgDataUrl = await new Promise((res) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = () => res(null);
    fr.readAsDataURL(file);
  });

  try {
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(API_MM + '/api/mri', {method:'POST', body:fd});
    const d = await r.json();
    renderMRI(d, imgDataUrl, 'image');
  } catch (err) {
    showResult('mri-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

document.getElementById('mri-pdf-go').addEventListener('click', async (e) => {
  const file = document.getElementById('mri-pdf-file').files[0];
  if (!file) return showResult('mri-result', 'Please choose a PDF report first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Reading report…');
  try {
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(API_MM + '/api/mri/report', {method:'POST', body:fd});
    const d = await r.json();
    renderMRI(d, null, 'pdf-report');
  } catch (err) {
    showResult('mri-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Report ════════
document.getElementById('report-go').addEventListener('click', async (e) => {
  const file = document.getElementById('report-file').files[0];
  if (!file) return showResult('report-result', 'Please choose a PDF first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Summarizing…');
  try {
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(API_MM + '/api/report', {method:'POST', body:fd});
    const d = await r.json();
    if (d.error) return showResult('report-result', d.error, true);
    const wrap = document.createElement('div');
    if (d.biomarkers && d.biomarkers.length){
      const pills = document.createElement('div');
      pills.style.marginBottom = '10px';
      pills.innerHTML = '<strong style="font-size:12px;color:var(--txt-2)">Detected biomarkers:</strong><br>' +
        d.biomarkers.map(b => `<span class="pill">${b}</span>`).join('');
      wrap.appendChild(pills);
    }
    const sum = document.createElement('div');
    sum.style.whiteSpace = 'pre-wrap';
    sum.textContent = d.summary || '(no summary)';
    wrap.appendChild(sum);
    showResult('report-result', wrap, false);
  } catch (err) {
    showResult('report-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Gene ════════
document.getElementById('gene-go').addEventListener('click', async (e) => {
  const txt = document.getElementById('gene-text').value.trim();
  if (!txt) return showResult('gene-result', 'Paste the genetic report first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Interpreting…');
  try {
    const fd = new FormData(); fd.append('report_text', txt);
    const r = await fetch(API_MM + '/api/gene', {method:'POST', body:fd});
    const d = await r.json();
    showResult('gene-result', d.interpretation || d.error || JSON.stringify(d, null, 2), !!d.error);
  } catch (err) {
    showResult('gene-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Drug ════════
document.getElementById('drug-go').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  loadingBtn(btn, 'Generating…');
  try {
    const fd = new FormData();
    fd.append('mutations', document.getElementById('drug-mutations').value);
    fd.append('stage',     document.getElementById('drug-stage').value);
    fd.append('symptoms',  document.getElementById('drug-symptoms').value);
    const r = await fetch(API_MM + '/api/drug', {method:'POST', body:fd});
    const d = await r.json();
    showResult('drug-result', d.recommendations || d.error || JSON.stringify(d, null, 2), !!d.error);
  } catch (err) {
    showResult('drug-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Simulate ════════
document.getElementById('sim-go').addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  loadingBtn(btn, 'Simulating…');
  try {
    const fd = new FormData();
    fd.append('model',   document.getElementById('sim-model').value);
    fd.append('pd_loss', document.getElementById('sim-loss').value);
    fd.append('I_ext',   document.getElementById('sim-iext').value);
    const r = await fetch(API_MM + '/api/simulate', {method:'POST', body:fd});
    const d = await r.json();
    if (d.error) return showResult('sim-result', d.error, true);
    const wrap = document.createElement('div');
    const meta = document.createElement('div');
    meta.innerHTML = `<strong>${d.model_name}</strong><br>` +
      `<span style="font-size:12px;color:var(--txt-3);font-family:'JetBrains Mono',monospace">` +
      JSON.stringify(d.params || {}, null, 2) + '</span>';
    wrap.appendChild(meta);
    if (d.figure_path) {
      const img = document.createElement('img');
      const fname = d.figure_path.split(/[\\/]/).pop();
      img.src = API_MM + '/api/plot/' + encodeURIComponent(fname);
      img.alt = 'simulation';
      wrap.appendChild(img);
    }
    showResult('sim-result', wrap, false);
  } catch (err) {
    showResult('sim-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Biomarker (Module 0) ════════
function renderBio(d){
  if (d.error) { showResult('bio-result', d.error, true); return; }
  const wrap = document.createElement('div');

  const head = document.createElement('div');
  head.style.marginBottom = '10px';
  head.innerHTML = `<strong style="font-size:13px">Session: </strong>
    <span style="font-family:'JetBrains Mono',monospace;color:var(--txt-2);font-size:12px">${d.session_id || '—'}</span>`;
  wrap.appendChild(head);

  if (d.top_genes && d.top_genes.length){
    const g = document.createElement('div');
    g.style.marginBottom = '10px';
    g.innerHTML = '<strong style="font-size:12px;color:var(--txt-2)">Top hub genes:</strong><br>' +
      d.top_genes.map(x => {
        const name = typeof x === 'string' ? x : (x.gene || x.name || JSON.stringify(x));
        return `<span class="pill">${name}</span>`;
      }).join('');
    wrap.appendChild(g);
  }
  if (d.explanation || d.summary){
    const s = document.createElement('div');
    s.style.whiteSpace = 'pre-wrap';
    s.style.marginTop = '8px';
    s.textContent = d.explanation || d.summary;
    wrap.appendChild(s);
  }
  if (d.session_id){
    const links = document.createElement('div');
    links.style.marginTop = '12px';
    links.style.fontSize = '12.5px';
    links.innerHTML =
      `<a href="${API_BIO}/report/${d.session_id}" target="_blank" style="color:var(--accent-2);margin-right:14px">↓ PDF Report</a>` +
      `<a href="${API_BIO}/graph/${d.session_id}"  target="_blank" style="color:var(--accent-2)">↓ PPI Graph</a>`;
    wrap.appendChild(links);

    const img = document.createElement('img');
    img.src = API_BIO + '/graph/' + d.session_id;
    img.alt = 'PPI network';
    img.onerror = () => img.remove();
    wrap.appendChild(img);
  } else {
    const raw = document.createElement('pre');
    raw.style.fontSize = '11.5px';
    raw.style.color = 'var(--txt-3)';
    raw.style.marginTop = '10px';
    raw.textContent = JSON.stringify(d, null, 2);
    wrap.appendChild(raw);
  }
  showResult('bio-result', wrap, false);
}

document.getElementById('bio-go-text').addEventListener('click', async (e) => {
  const txt = document.getElementById('bio-text').value.trim();
  if (!txt) return showResult('bio-result', 'Enter clinical text first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Discovering…');
  try {
    const fd = new FormData(); fd.append('clinical_text', txt);
    const r = await fetch(API_BIO + '/discover', {method:'POST', body:fd});
    const d = await r.json();
    renderBio(d);
  } catch (err) {
    showResult('bio-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

document.getElementById('bio-go-pdf').addEventListener('click', async (e) => {
  const file = document.getElementById('bio-file').files[0];
  if (!file) return showResult('bio-result', 'Choose a PDF first.', true);
  const btn = e.currentTarget;
  loadingBtn(btn, 'Discovering…');
  try {
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch(API_BIO + '/discover/pdf', {method:'POST', body:fd});
    const d = await r.json();
    renderBio(d);
  } catch (err) {
    showResult('bio-result', 'Request failed: ' + err.message, true);
  } finally { restoreBtn(btn); }
});

// ════════ Debugger ════════
const ICONS = {ok:'✓', warning:'!', error:'✗', skip:'—'};

async function runDebug(){
  const btn = document.getElementById('dbg-go');
  loadingBtn(btn, 'Scanning…');
  const checksEl = document.getElementById('dbg-checks');
  const sumEl    = document.getElementById('dbg-summary');
  checksEl.innerHTML = '';
  sumEl.style.display = 'none';
  try {
    const r = await fetch(API_DBG + '/run');
    const d = await r.json();
    document.getElementById('dbg-ts').textContent =
      'Last run · ' + new Date(d.timestamp).toLocaleTimeString();

    const s = d.summary || {};
    sumEl.style.display = 'grid';
    sumEl.innerHTML = `
      <div class="stat"><div class="n blue">${s.total||0}</div><div class="l">Total</div></div>
      <div class="stat"><div class="n ok">${s.ok||0}</div><div class="l">Passed</div></div>
      <div class="stat"><div class="n warn">${s.warnings||0}</div><div class="l">Warnings</div></div>
      <div class="stat"><div class="n err">${s.errors||0}</div><div class="l">Errors</div></div>`;

    for (const c of (d.checks || [])) {
      const row = document.createElement('div');
      row.className = 'dbg-check ' + c.status;
      let extra = '';
      if (c.status === 'error' && c.detail)
        extra += `<div class="cfix" style="color:var(--err);background:rgba(239,68,68,0.08)">${escHtml(String(c.detail).substring(0,400))}</div>`;
      if (c.fix)
        extra += `<div class="cfix">Fix: ${escHtml(c.fix)}</div>`;
      row.innerHTML = `
        <span style="color:var(--${c.status==='ok'?'ok':c.status==='warning'?'warn':c.status==='error'?'err':'txt-3'});font-size:13px;margin-top:1px">${ICONS[c.status]||'·'}</span>
        <div>
          <div class="cname">${escHtml(c.name)}</div>
          <div class="cmsg">${escHtml(c.message||'')}</div>
          ${extra}
        </div>
        <div style="text-align:right">
          <span class="badge ${c.status}">${c.status}</span>
          <div class="cdur">${c.ms||0}ms</div>
        </div>`;
      checksEl.appendChild(row);
    }
  } catch (err) {
    checksEl.innerHTML =
      `<div class="dbg-check error">
        <span style="color:var(--err);font-size:13px">✗</span>
        <div><div class="cname">Debugger connection failed</div>
        <div class="cmsg">${escHtml(String(err))}</div>
        <div class="cfix">Fix: make sure pd_debugger is running on :8080</div></div>
        <span></span>
      </div>`;
  } finally { restoreBtn(btn); }
}
function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
document.getElementById('dbg-go').addEventListener('click', runDebug);

// ════════ Evaluation Results ════════
function renderEval(arm, containerId){
  const el = document.getElementById(containerId);
  el.innerHTML = '<div style="color:var(--txt-3);font-size:13px;padding:8px 0">Loading…</div>';
  fetch('/api/eval/' + arm).then(r => r.json()).then(d => {
    if (!d.available){
      el.innerHTML =
        '<div class="error">No results yet for this module.</div>' +
        '<div style="color:var(--txt-3);font-size:12.5px;margin-top:8px">Run the harness, then reload:<br>' +
        (arm === 'clinical'
          ? '<span class="pill">python -m pd_clinical.eval.prepare_ppmi --raw-dir &lt;PPMI_csvs&gt;</span><br><span class="pill">python -m pd_clinical.eval.train_eval</span>'
          : '<span class="pill">python -m pd_rag.eval.build_benchmark</span><br><span class="pill">python -m pd_rag.eval.run_eval</span>') +
        '</div>';
      return;
    }
    el.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'v-raw';
    pre.style.maxHeight = '460px';
    pre.style.fontSize = '12px';
    pre.textContent = d.markdown;
    el.appendChild(pre);
    (d.figures || []).forEach(name => {
      const img = document.createElement('img');
      img.src = '/api/eval/' + arm + '/figures/' + encodeURIComponent(name);
      img.alt = name;
      img.style.cssText = 'max-width:100%;border-radius:8px;margin-top:12px;border:1px solid var(--bdr);background:#fff';
      el.appendChild(img);
    });
  }).catch(e => { el.innerHTML = '<div class="error">Failed to load: ' + e.message + '</div>'; });
}

document.querySelectorAll('#eval-tabs .tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('#eval-tabs .tab').forEach(x => x.classList.toggle('active', x === t));
    const which = t.dataset.tab;
    document.getElementById('eval-tab-clinical').classList.toggle('active', which === 'clinical');
    document.getElementById('eval-tab-rag').classList.toggle('active', which === 'rag');
    renderEval(which, 'eval-' + which);
  });
});

// load clinical results the first time the Evaluation view is opened
let _evalLoaded = false;
document.querySelector('.nav-item[data-view="evaluation"]').addEventListener('click', () => {
  if (!_evalLoaded){ renderEval('clinical', 'eval-clinical'); _evalLoaded = true; }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
async def health():
    return {"status": "ok", "service": "pd-launcher"}


# ── Evaluation results (read harness outputs) ─────────────────────────
_EVAL_ROOT = Path(__file__).resolve().parent
_EVAL_DIRS = {
    "clinical": _EVAL_ROOT / "pd_clinical" / "eval" / "results",
    "rag":      _EVAL_ROOT / "pd_rag" / "eval" / "results",
}


@app.get("/api/eval/{arm}")
async def eval_results(arm: str):
    d = _EVAL_DIRS.get(arm)
    if d is None:
        return JSONResponse({"available": False, "error": "unknown arm"}, status_code=404)
    md = d / "results.md"
    figs_dir = d / "figures"
    figures = sorted(p.name for p in figs_dir.glob("*.png")) if figs_dir.exists() else []
    return JSONResponse({
        "available": md.exists(),
        "markdown": md.read_text(encoding="utf-8") if md.exists() else "",
        "figures": figures,
    })


@app.get("/api/eval/{arm}/figures/{name}")
async def eval_figure(arm: str, name: str):
    d = _EVAL_DIRS.get(arm)
    if d is None:
        return JSONResponse({"error": "unknown arm"}, status_code=404)
    figs_dir = (d / "figures").resolve()
    f = (figs_dir / name).resolve()
    # path-traversal guard: resolved file must stay inside the figures dir
    if not str(f).startswith(str(figs_dir)) or not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(f), media_type="image/png")


def main():
    root_dir = Path(__file__).resolve().parent

    # Find the venv interpreter. Check a few likely locations:
    #   <root>/.venv, <root>/pd_env, <root>/venv,
    #   <root>/../.venv, <root>/../pd_env  (project root one level up)
    candidates = [
        root_dir / ".venv"   / "Scripts" / "python.exe",
        root_dir / "pd_env"  / "Scripts" / "python.exe",
        root_dir / "venv"    / "Scripts" / "python.exe",
        root_dir.parent / ".venv"  / "Scripts" / "python.exe",
        root_dir.parent / "pd_env" / "Scripts" / "python.exe",
        root_dir.parent / "venv"   / "Scripts" / "python.exe",
    ]
    env_python = next((p for p in candidates if p.exists()), None)
    python_exe = str(env_python if env_python else Path(sys.executable))

    if env_python:
        print(f"Using venv interpreter: {python_exe}")
    else:
        print(f"!  No venv found, using current Python: {python_exe}")
        print("   If services fail to start, packages may be missing from this Python.")

    print("\n" + "=" * 60)
    print("Spawning child services...")

    p1 = subprocess.Popen(
        [python_exe, "api/main.py"],
        cwd=str(root_dir / "pd_multimodal"),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    p2 = subprocess.Popen(
        [python_exe, "api/main.py"],
        cwd=str(root_dir / "module0_biomarker"),
        stdout=sys.stdout, stderr=sys.stderr,
    )
    p3 = subprocess.Popen(
        [python_exe, "api/main.py"],
        cwd=str(root_dir / "pd_debugger"),
        stdout=sys.stdout, stderr=sys.stderr,
    )

    print("Services spawned on :8000, :8001, :8080")
    print("Dashboard → http://localhost:5000")
    print("=" * 60 + "\n")

    time.sleep(2)
    webbrowser.open("http://localhost:5000")

    children = (p1, p2, p3)

    def _shutdown(*_):
        print("\nShutting down launcher and child services...")
        for p in children:
            try: p.terminate()
            except Exception: pass
        # Wait briefly for graceful exit, then force-kill anything still alive
        for p in children:
            try: p.wait(timeout=4)
            except Exception:
                try: p.kill()
                except Exception: pass
        # Belt-and-suspenders: if the children spawned grandchildren (uvicorn reloader),
        # use taskkill to wipe the whole tree on Windows.
        if sys.platform == "win32":
            for p in children:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass
        print("Backend services terminated.")

    import atexit, signal
    atexit.register(_shutdown)
    try: signal.signal(signal.SIGINT,  lambda *_: (_shutdown(), sys.exit(0)))
    except Exception: pass
    try: signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(0)))
    except Exception: pass

    try:
        uvicorn.run("launcher:app", host="0.0.0.0", port=5000, log_level="info")
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
