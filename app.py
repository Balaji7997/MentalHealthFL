import logging
import os
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Maximum request size: 8 MB
app.config["JSON_SORT_KEYS"] = False


@app.after_request
def add_security_headers(response):
    """Add baseline browser security and privacy headers without external dependencies."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store" if request.path == "/predict" else "no-cache")
    return response

# Image analysis stays off until the project has a suitable validated model/dataset.
ENABLE_IMAGE_ANALYSIS = os.getenv("ENABLE_IMAGE_ANALYSIS", "false").strip().lower() in {"1", "true", "yes"}
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def get_severity_predictor():
    """Load the project predictor only when a prediction is requested."""
    from models.predictor import predict_severity
    return predict_severity

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MentalHealthFL | Mental Wellbeing Check-in</title>
<meta name="description" content="An experimental AI research tool for text-based mental wellbeing check-ins, with voice input and support resources.">
<meta name="theme-color" content="#1769aa">
<style>
* { box-sizing: border-box; }
:root { color-scheme: dark; }
body {
    margin: 0; font-family: Arial, sans-serif;
    background: #0f172a; color: #e2e8f0;
}
main { max-width: 900px; margin: 28px auto; padding: 18px; }
header { margin-bottom: 24px; }
h1 { margin-bottom: 8px; color: #93c5fd; }
h2 { margin-top: 0; }
p { line-height: 1.6; color: #cbd5e1; }
.card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 16px; padding: 22px; margin-bottom: 18px;
}
textarea {
    width: 100%; min-height: 120px; resize: vertical;
    background: #0f172a; color: white; border: 1px solid #475569;
    border-radius: 10px; padding: 13px; font: inherit;
}
button {
    border: 0; border-radius: 10px; padding: 12px 16px;
    margin: 7px 5px 0 0; font-weight: bold; cursor: pointer;
}
.primary { background: #2563eb; color: white; }
.secondary { background: #334155; color: white; }
.danger { background: #b91c1c; color: white; }
button:disabled { opacity: .5; cursor: not-allowed; }
.status { color: #93c5fd; min-height: 24px; }
.big { font-size: 25px; font-weight: bold; color: #93c5fd; }
.small { font-size: 13px; color: #94a3b8; }
.prob-row { margin: 14px 0; }
.prob-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
.track { background: #0f172a; height: 9px; border-radius: 9px; overflow: hidden; }
.bar { height: 100%; background: #60a5fa; width: 0; }
.image-controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0; }
input[type="file"] { display: block; width: 100%; margin-top: 12px; color: #cbd5e1; }
#faceVideo, #facePreview img {
    display: block; width: 100%; max-width: 480px; max-height: 360px;
    object-fit: contain; border-radius: 12px; margin-top: 12px;
    background: #0f172a;
}
#faceVideo { display: none; }
#imageResults { margin-top: 16px; }
.expression-row { margin: 12px 0; }
.note { font-size: 13px; color: #94a3b8; }
footer { font-size: 12px; color: #94a3b8; margin-top: 24px; line-height: 1.6; }
@media (max-width: 600px) {
    main { margin: 8px auto; padding: 12px; }
    .card { padding: 16px; }
    button { padding: 11px 13px; }
}

/* MentalHealthFL UI refresh: presentation-only styles; API and model logic stay unchanged. */
:root {
    color-scheme: dark;
    --bg: #080d1a;
    --panel: rgba(17, 25, 45, .88);
    --panel-strong: #111a2e;
    --line: rgba(148, 163, 184, .17);
    --muted: #9aa9c2;
    --text: #eef4ff;
    --mint: #5eead4;
    --blue: #818cf8;
}
body {
    min-height: 100vh;
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
      radial-gradient(ellipse at 10% 0%, rgba(79, 70, 229, .20), transparent 38%),
      radial-gradient(ellipse at 100% 18%, rgba(20, 184, 166, .12), transparent 30%),
      var(--bg);
}
main { max-width: 1060px; margin: 0 auto; padding: 34px 24px 50px; }
header {
    position: relative; overflow: hidden; margin-bottom: 24px; padding: 28px 30px;
    border: 1px solid var(--line); border-radius: 24px;
    background: linear-gradient(120deg, rgba(79,70,229,.20), rgba(20,184,166,.08) 70%), var(--panel);
    box-shadow: 0 22px 70px rgba(0,0,0,.20);
}
header:after {
    content: ""; position: absolute; width: 190px; height: 190px; right: -55px; top: -75px;
    border: 1px solid rgba(94,234,212,.22); border-radius: 50%;
    box-shadow: 0 0 0 24px rgba(94,234,212,.035), 0 0 0 50px rgba(129,140,248,.035);
    pointer-events: none;
}
header h1 { color: var(--text); font-size: clamp(28px, 4vw, 42px); letter-spacing: -1.6px; margin: 10px 0 8px; }
header p { max-width: 680px; margin: 0; color: #b6c3d9; font-size: 15px; }
.eyebrow {
    display: inline-flex; align-items: center; gap: 8px; padding: 7px 11px;
    border: 1px solid rgba(94,234,212,.22); border-radius: 999px;
    color: #9ff8e8; background: rgba(20,184,166,.08); font-size: 11px;
    font-weight: 800; letter-spacing: 1.3px; text-transform: uppercase;
}
.eyebrow:before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 12px rgba(94,234,212,.7); }
.card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 22px;
    padding: 26px; margin-bottom: 20px; box-shadow: 0 14px 40px rgba(0,0,0,.13);
    backdrop-filter: blur(14px);
}
.card h2 { font-size: 20px; letter-spacing: -.4px; margin: 0 0 8px; }
.card > p { color: #aab8cf; margin-top: 8px; }
label { display: block; margin: 20px 0 9px; color: #dce7f7; font-size: 13px; font-weight: 700; }
textarea {
    min-height: 145px; padding: 16px; line-height: 1.65; color: var(--text);
    background: rgba(5, 10, 23, .72); border: 1px solid #2b3853; border-radius: 15px;
    outline: none; transition: border-color .2s, box-shadow .2s;
}
textarea:focus { border-color: rgba(94,234,212,.65); box-shadow: 0 0 0 4px rgba(20,184,166,.09); }
textarea::placeholder { color: #65748e; }
button {
    border-radius: 12px; padding: 12px 17px; margin: 8px 7px 0 0;
    font: inherit; font-size: 13px; font-weight: 750; letter-spacing: .1px;
    transition: transform .18s, filter .18s, border-color .18s; cursor: pointer;
}
button:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.08); }
.primary { color: #06121d; background: linear-gradient(135deg, #5eead4, #93c5fd); box-shadow: 0 8px 22px rgba(45,212,191,.13); }
.secondary { color: #dce8fb; background: #1a2740; border: 1px solid #34435f; }
.danger { color: #fecaca; background: rgba(153,27,27,.22); border: 1px solid rgba(248,113,113,.35); }
button:disabled { opacity: .48; cursor: not-allowed; transform: none; }
.status { display: flex; align-items: center; gap: 9px; min-height: 38px; padding: 9px 12px; border: 1px solid rgba(129,140,248,.18); border-radius: 11px; background: rgba(99,102,241,.07); color: #b9c6ff; font-size: 13px; }
.big { font-size: 30px; font-weight: 800; letter-spacing: -.7px; color: #8ff4df; }
.small, .note { font-size: 12px; color: var(--muted); line-height: 1.65; }
#textResult { border-color: rgba(94,234,212,.22); background: linear-gradient(135deg, rgba(20,184,166,.07), transparent 55%), var(--panel); }
#textResult h3 { margin-top: 25px; font-size: 14px; color: #dce7f7; }
.prob-row { margin: 15px 0; }
.prob-head { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 7px; font-size: 13px; color: #d4deef; }
.track { background: #080f20; height: 8px; border-radius: 999px; overflow: hidden; border: 1px solid rgba(148,163,184,.08); }
.bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #5eead4); transition: width .5s ease; }
footer { border-top: 1px solid var(--line); padding: 18px 4px 0; color: #8291aa; font-size: 12px; }
/* Image analysis is intentionally hidden until the correct image-based model is available. DOM stays present so existing JS remains safe. */
.image-module { display: none !important; }
/* UX polish */
html { scroll-behavior: smooth; }
::selection { background: rgba(94,234,212,.28); color: #fff; }
header { padding: clamp(24px, 4vw, 42px); }
header h1 { max-width: 760px; line-height: 1.08; }
header p { max-width: 620px; line-height: 1.8; }
main { max-width: 1120px; }
.workspace { display:grid; grid-template-columns:minmax(0,1.12fr) minmax(280px,.88fr); gap:20px; align-items:start; }
.workspace > .card { margin-bottom:20px; }
.input-card { position:relative; }
.section-top { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; }
.step-tag { flex:none; padding:7px 10px; border-radius:10px; background:rgba(129,140,248,.11); border:1px solid rgba(129,140,248,.22); color:#c7d2fe; font-size:11px; font-weight:800; letter-spacing:.7px; }
.quick-prompts { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0 4px; }
.prompt-chip { width:auto; margin:0; padding:8px 11px; border-radius:999px; color:#b9c8df; background:rgba(26,39,64,.7); border:1px solid #34435f; font-size:12px; font-weight:650; }
.prompt-chip:hover { border-color:rgba(94,234,212,.55); }
.input-meta { display:flex; justify-content:space-between; gap:10px; margin-top:8px; font-size:11px; color:var(--muted); }
.action-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.action-row button { margin:0; }
#textResult { position:sticky; top:16px; }
.result-top { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
.result-icon { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; background:rgba(94,234,212,.10); border:1px solid rgba(94,234,212,.2); font-size:21px; }
.confidence-pill { display:inline-flex; padding:7px 10px; border-radius:999px; background:rgba(94,234,212,.09); border:1px solid rgba(94,234,212,.2); color:#9ff8e8; font-size:12px; font-weight:800; }
.privacy-note { display:flex; gap:10px; align-items:flex-start; margin-top:16px; padding:12px 13px; border-radius:13px; border:1px solid var(--line); background:rgba(5,10,23,.32); color:#9aa9c2; font-size:12px; line-height:1.6; }
.privacy-note span:first-child { font-size:16px; }
.status:before { content:''; display:block; width:7px; height:7px; border-radius:50%; background:#5eead4; flex:none; }
button:focus-visible { outline:3px solid rgba(94,234,212,.55); outline-offset:3px; }
textarea { min-height:170px; }
#analyze { min-height:45px; }
footer { line-height:1.8; }
@media (max-width: 820px) { .workspace { grid-template-columns:1fr; } #textResult { position:static; } }
@media (max-width: 640px) { .section-top { gap:8px; } .step-tag { padding:6px 8px; } .action-row { display:grid; grid-template-columns:1fr; } .action-row button { width:100%; } .quick-prompts { gap:6px; } .prompt-chip { width:auto; } }
@media (max-width: 640px) {
    main { padding: 18px 13px 30px; }
    header { padding: 22px 20px; border-radius: 19px; }
    .card { padding: 20px 17px; border-radius: 18px; }
    button { width: 100%; margin: 7px 0 0; }
    .big { font-size: 26px; }
}

/* Research-informed redesign: calm clinical palette, accessible spacing, clear support pathways. */
:root { color-scheme: light; --bg:#f5f7f2; --panel:#ffffff; --panel-strong:#ffffff; --line:#dfe7df; --muted:#63766d; --text:#1d352d; --mint:#1e7964; --blue:#547d72; }
body { color:var(--text); background:radial-gradient(ellipse at 0% 0%,#e2f1e9 0,transparent 34%),radial-gradient(ellipse at 100% 10%,#e9eee0 0,transparent 30%),var(--bg); }
main { max-width:1160px; padding-top:18px; }
.site-nav { display:flex; justify-content:space-between; align-items:center; gap:22px; padding:10px 2px 22px; }
.brand { display:flex; align-items:center; gap:11px; color:#1d352d; text-decoration:none; font-weight:850; letter-spacing:-.4px; }
.brand-mark { display:grid; place-items:center; width:43px; height:43px; border-radius:15px; color:#fff; background:#1e7964; font-size:25px; }
.brand-accent { color:#1e7964; }
.brand small { display:block; color:#73847b; font-size:10px; font-weight:600; letter-spacing:.2px; margin-top:3px; }
.nav-links { display:flex; align-items:center; gap:23px; }
.nav-links a { color:#52685e; text-decoration:none; font-size:13px; font-weight:700; }
.nav-links a:hover { color:#14644f; }
.nav-links .nav-cta { padding:11px 15px; border-radius:12px; background:#1e7964; color:white; }
header { color:var(--text); border:1px solid #dbe8df; background:linear-gradient(115deg,rgba(255,255,255,.97),rgba(234,246,238,.93)); box-shadow:0 20px 55px rgba(31,71,52,.07); }
header:after { border-color:rgba(30,121,100,.18); box-shadow:0 0 0 24px rgba(30,121,100,.04),0 0 0 50px rgba(84,125,114,.04); }
header h1 { color:#183b30; letter-spacing:-1.8px; }
header p { color:#5d7268; }
.eyebrow { color:#226c59; border-color:#cde4d5; background:#edf7f0; }
.eyebrow:before { background:#319875; box-shadow:0 0 0 4px rgba(49,152,117,.12); }
.card { background:rgba(255,255,255,.95); border-color:#e0e8e0; box-shadow:0 12px 36px rgba(36,67,50,.055); backdrop-filter:blur(10px); }
.card h2 { color:#1d3c30; }
.card > p { color:#66786e; }
label { color:#355548; }
textarea { color:#233e33; background:#fbfdfb; border-color:#d6e3d9; }
textarea:focus { border-color:#64a58c; box-shadow:0 0 0 4px rgba(55,139,104,.10); }
textarea::placeholder { color:#91a097; }
.primary { color:white; background:#1e7964; box-shadow:0 7px 18px rgba(30,121,100,.14); }
.secondary { color:#355548; background:#f2f7f3; border-color:#dce7de; }
.danger { color:#a23d3d; background:#fff4f2; border-color:#f1d5d1; }
.status { color:#286e58; border-color:#d9e9de; background:#f2f8f3; }
.status:before { background:#329875; }
.big { color:#1e7964; }
.small,.note { color:#728278; }
#textResult { border-color:#d4e8db; background:linear-gradient(135deg,#f0f8f2,transparent 60%),#fff; }
#textResult h3 { color:#34574a; }
.track { background:#edf2ed; border-color:#e4ebe4; }
.bar { background:linear-gradient(90deg,#478d75,#8fc7a6); }
.confidence-pill { color:#226c59; background:#edf7f0; border-color:#cfe5d6; }
.step-tag { color:#3d7060; background:#f0f7f2; border-color:#d8e8dc; }
.prompt-chip { color:#4f6a5d; background:#f5f9f5; border-color:#dce8de; }
.prompt-chip:hover { border-color:#8fc5a7; }
.privacy-note { color:#6c7d72; background:#f8fbf8; border-color:#e2eae2; }
.intro-strip { display:flex; align-items:center; gap:14px; margin:0 0 22px; padding:17px 20px; border:1px solid #dce9de; border-radius:18px; background:rgba(255,255,255,.74); color:#496456; line-height:1.55; }
.intro-strip strong { color:#254b3b; }
.intro-strip small { color:#718278; }
.intro-icon { display:grid; place-items:center; flex:none; width:42px; height:42px; border-radius:14px; background:#e6f3e9; color:#2b8064; font-size:22px; }
.below-grid,.resource-section { margin:54px 0; }
.section-heading { max-width:650px; margin-bottom:20px; }
.section-heading h2 { margin:12px 0 7px; font-size:clamp(24px,3vw,32px); letter-spacing:-.9px; color:#1d3c30; }
.section-heading p { margin:0; color:#6c7e73; line-height:1.7; }
.info-grid,.resource-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.info-card,.resource-card { position:relative; padding:23px; border:1px solid #e0e8df; border-radius:20px; background:rgba(255,255,255,.9); box-shadow:0 10px 28px rgba(36,67,50,.04); }
.info-number { position:absolute; right:19px; top:17px; color:#b4c8ba; font-size:12px; font-weight:850; letter-spacing:1px; }
.info-icon { display:grid; place-items:center; width:42px; height:42px; margin-bottom:17px; border-radius:14px; color:#28795f; background:#eaf5ed; font-size:22px; }
.info-card h3,.resource-card h3 { margin:0 0 9px; color:#294b3c; font-size:17px; letter-spacing:-.3px; }
.info-card p,.resource-card p { margin:0; color:#6c7e73; font-size:13px; line-height:1.75; }
.resource-card { display:block; color:inherit; text-decoration:none; transition:transform .2s,border-color .2s,box-shadow .2s; }
.resource-card:hover { transform:translateY(-3px); border-color:#b9d7c2; box-shadow:0 16px 32px rgba(36,67,50,.08); }
.resource-label,.support-kicker { color:#388064; font-size:10px; font-weight:850; letter-spacing:1.2px; }
.resource-card h3 { display:flex; justify-content:space-between; margin-top:13px; }
.resource-card h3 span { color:#43876d; }
.resource-link { display:inline-block; margin-top:17px; color:#28795f; font-size:12px; font-weight:800; }
.support-banner { display:flex; justify-content:space-between; align-items:center; gap:22px; margin:48px 0 30px; padding:27px 30px; border-radius:23px; color:#edf8f0; background:linear-gradient(115deg,#1d5d4b,#267a60); box-shadow:0 16px 36px rgba(30,95,73,.16); }
.support-banner h2 { margin:8px 0; color:white; font-size:clamp(22px,3vw,30px); letter-spacing:-.7px; }
.support-banner p { margin:0; color:#d3e9dc; line-height:1.7; }
.support-kicker { color:#bde7cf; }
.support-action { display:flex; flex-direction:column; align-items:flex-start; gap:9px; min-width:180px; }
.support-action strong { font-size:15px; }
.support-action strong a { color:#fff; font-size:26px; text-decoration:none; }
.support-button { display:inline-block; padding:10px 13px; border:1px solid rgba(255,255,255,.3); border-radius:11px; color:#fff; text-decoration:none; font-size:12px; font-weight:750; }
.site-footer { display:grid; grid-template-columns:.8fr 1.2fr; gap:20px; border-top:1px solid #dce6dc; padding-top:22px; color:#718176; font-size:12px; line-height:1.8; }
.site-footer strong { color:#315443; font-size:14px; }
.site-footer p { margin:5px 0; color:#718176; font-size:12px; }
.footer-bottom { grid-column:1/-1; display:flex; justify-content:space-between; gap:15px; padding-top:12px; border-top:1px solid #e1e9e0; }
.footer-bottom a { color:#28795f; font-weight:800; text-decoration:none; }
button:focus-visible,a:focus-visible { outline:3px solid #4a9a7b; outline-offset:3px; }
@media(max-width:800px) { .nav-links { gap:12px; } .nav-links a:not(.nav-cta) { display:none; } .info-grid,.resource-grid { grid-template-columns:1fr; } .support-banner { align-items:flex-start; flex-direction:column; padding:24px; } .site-footer { grid-template-columns:1fr; } .footer-bottom { grid-column:auto; } }
@media(max-width:640px) { .site-nav { align-items:flex-start; } .brand small { max-width:160px; } .intro-strip { align-items:flex-start; padding:14px; } .support-action { min-width:0; } .below-grid,.resource-section { margin:38px 0; } }
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto !important; transition:none !important; animation:none !important; } }



/* Clinical-inspired doctor theme: preserves the existing page structure and functionality. */
:root {
  color-scheme: light;
  --bg: #f3f8fc;
  --panel: rgba(255,255,255,.96);
  --panel-strong: #ffffff;
  --line: #dbe7f0;
  --muted: #60758a;
  --text: #18334b;
  --mint: #0f9f9a;
  --blue: #2476c7;
}
html { background: var(--bg); }
body {
  min-height: 100vh;
  color: var(--text);
  background:
    radial-gradient(ellipse at 8% 0%, rgba(51,153,218,.11), transparent 34%),
    radial-gradient(ellipse at 100% 12%, rgba(22,163,157,.09), transparent 30%),
    linear-gradient(180deg, #f8fbfe 0%, #eef5fa 100%);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
}
main { max-width: 1060px; margin: 0 auto; padding: 30px 24px 48px; }
header {
  position: relative; overflow: hidden; margin-bottom: 24px; padding: 30px;
  border: 1px solid #d7e7f2; border-radius: 24px;
  background: linear-gradient(120deg, rgba(225,242,255,.98), rgba(232,250,247,.96) 78%), #fff;
  box-shadow: 0 14px 40px rgba(33,83,120,.08);
}
header::after {
  content: "+"; position: absolute; right: 28px; top: 8px;
  color: rgba(36,118,199,.10); font-size: 118px; font-weight: 300; line-height: 1;
  pointer-events: none;
}
h1 { color: #1768a8; letter-spacing: -.035em; }
h2 { color: #214b69; }
p { color: #526b80; }
.card {
  background: rgba(255,255,255,.94); color: var(--text);
  border: 1px solid #dce8f1; border-radius: 18px; padding: 24px; margin-bottom: 18px;
  box-shadow: 0 9px 28px rgba(35,76,105,.055);
}
textarea {
  background: #fbfdff; color: #18334b; border: 1px solid #c9dce9;
  border-radius: 12px; padding: 14px; font: inherit; line-height: 1.6;
  transition: border-color .18s ease, box-shadow .18s ease;
}
textarea:focus, input:focus-visible, button:focus-visible {
  outline: 3px solid rgba(36,118,199,.18); outline-offset: 2px;
}
textarea:focus { border-color: #4b9bd7; box-shadow: 0 0 0 4px rgba(36,118,199,.08); }
button { border-radius: 11px; padding: 12px 17px; transition: transform .15s ease, box-shadow .15s ease, background .15s ease; }
button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 14px rgba(26,91,145,.12); }
.primary { background: #2476c7; color: #fff; }
.primary:hover:not(:disabled) { background: #1c65ad; }
.secondary { background: #e8f2f8; color: #285575; border: 1px solid #d2e3ef; }
.danger { background: #fff0f0; color: #a52c36; border: 1px solid #f1d0d3; }
.status { color: #137f80; min-height: 24px; }
.big { color: #1768a8; }
.small, .note, footer { color: #6b8092; }
.track { background: #e7eff5; height: 10px; border-radius: 10px; }
.bar { background: linear-gradient(90deg, #2789ce, #20aaa0); border-radius: inherit; transition: width .5s ease; }
input[type="file"] { color: #526b80; }
#faceVideo, #facePreview img { background: #edf4f8; border: 1px solid #d8e6ef; }
footer { border-top: 1px solid #dce8f1; padding-top: 16px; }
::selection { background: #cdeafa; color: #173c59; }
@media (max-width: 600px) {
  main { margin: 0 auto; padding: 16px 12px 30px; }
  header { padding: 23px 19px; border-radius: 18px; }
  header::after { right: 12px; top: 8px; font-size: 84px; }
  .card { padding: 18px; border-radius: 15px; }
}


/* Clinical redesign: clean medical palette and distinct new logo/brand. */
:root { color-scheme: light; --bg:#f3f7fb; --panel:#ffffff; --panel-strong:#ffffff; --line:#dbe5ef; --muted:#61758b; --text:#18344d; --mint:#0d9488; --blue:#1769aa; }
html { scroll-behavior:smooth; }
body { background:#f3f7fb; color:#18344d; font-family:Inter,"Segoe UI",Arial,sans-serif; }
main { max-width:1160px; padding:24px 28px 52px; }
.site-nav { background:#fff; border:1px solid #dfe8f0; box-shadow:0 5px 20px rgba(22,54,82,.045); border-radius:16px; padding:12px 18px; }
.brand { display:flex; align-items:center; gap:11px; color:#163b5c; text-decoration:none; font-weight:800; }
.brand-mark { display:grid; place-items:center; width:45px; height:45px; border-radius:13px; background:#e7f4fb; color:#1769aa; }
.brand-mark svg { width:34px; height:34px; }
.brand-copy { display:flex; flex-direction:column; gap:3px; line-height:1.1; }
.brand-copy strong { font-size:19px; letter-spacing:-.55px; }
.brand-copy strong span { color:#0d9488; }
.brand-copy small { color:#6c8296; font-size:9px; letter-spacing:1.15px; font-weight:800; }
.nav-links a { color:#48647c; font-weight:650; }
.nav-links .nav-cta { background:#1769aa; color:#fff; border-radius:10px; padding:11px 15px; }
header { margin-top:22px; padding:44px 46px; border:1px solid #dce8f1; border-radius:22px; background:linear-gradient(110deg,#fff 0%,#f5fbff 68%,#e9f6f6 100%); box-shadow:0 12px 32px rgba(22,54,82,.06); }
header:after { width:230px; height:230px; right:-80px; top:-100px; border-color:rgba(23,105,170,.13); box-shadow:0 0 0 24px rgba(23,105,170,.035),0 0 0 52px rgba(13,148,136,.03); }
.hero-brandline { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
.hero-logo { display:grid; place-items:center; width:44px; height:44px; color:#1769aa; background:#e6f3fb; border:1px solid #d2e8f5; border-radius:13px; }
.hero-logo svg { width:32px; height:32px; }
header .eyebrow { color:#1769aa; background:#eaf5fc; border:1px solid #d5eaf7; }
header .eyebrow:before { background:#0d9488; box-shadow:none; }
header h1 { color:#173b59; font-size:clamp(32px,4.8vw,51px); line-height:1.12; letter-spacing:-1.8px; margin:0 0 16px; }
header h1 span { color:#1769aa; }
header p { max-width:720px; color:#536b80; font-size:16px; line-height:1.75; }
.hero-trust { display:flex; flex-wrap:wrap; gap:10px 22px; margin-top:25px; color:#48647b; font-size:12px; font-weight:650; }
.hero-trust span { display:flex; align-items:center; gap:7px; }
.trust-dot { width:7px; height:7px; display:inline-block; border-radius:50%; background:#0d9488; }
.intro-strip { background:#eaf6f6; border:1px solid #d3ece9; color:#234e5d; border-radius:14px; }
.intro-strip small { color:#567581; }
.card { background:#fff; border:1px solid #dfe8f0; border-radius:18px; box-shadow:0 7px 24px rgba(22,54,82,.045); backdrop-filter:none; }
.card h2, .card h3 { color:#1b405f; }
.card > p { color:#61758b; }
label { color:#294b67; }
textarea { color:#18344d; background:#fbfdff; border:1px solid #cbd9e6; border-radius:12px; }
textarea:focus { border-color:#4a9bc9; box-shadow:0 0 0 4px rgba(23,105,170,.10); }
textarea::placeholder { color:#8b9eaf; }
button { border-radius:10px; }
.primary { color:#fff; background:#1769aa; box-shadow:0 5px 12px rgba(23,105,170,.14); }
.primary:hover:not(:disabled) { background:#125b95; filter:none; }
.secondary { color:#245574; background:#edf5fa; border:1px solid #d5e5f0; }
.danger { color:#a33b43; background:#fff1f1; border:1px solid #f0d0d2; }
.status { color:#315e7c; background:#f0f7fb; border:1px solid #d9eaf4; }
.big { color:#0d8078; }
.small,.note { color:#6d8295; }
#textResult { border-color:#cfe8e5; background:#faffff; }
#textResult h3 { color:#294b67; }
.prob-head { color:#3c5870; }
.track { background:#e8eff5; border:0; }
.bar { background:linear-gradient(90deg,#1769aa,#19a79b); }
.below-grid > *, .resource-card { background:#fff; border-color:#dfe8f0; }
.resource-section h2, .below-grid h2 { color:#1b405f; }
.support-banner { background:linear-gradient(110deg,#e9f5fb,#eaf8f5); border:1px solid #d5e9ee; color:#1d455f; border-radius:18px; }
.support-banner p { color:#526f82; }
.support-button { background:#1769aa; color:white; }
footer,.site-footer { color:#6a8093; border-color:#dce6ee; }
footer p { color:#6a8093; }
@media(max-width:700px) { main { padding:12px 13px 32px; } .site-nav { align-items:flex-start; gap:12px; } .nav-links { gap:8px; flex-wrap:wrap; } header { padding:28px 22px; } .hero-trust { flex-direction:column; gap:9px; } .brand-copy strong { font-size:17px; } }


/* Added wellbeing tools, improved accessibility and clearer result guidance. */
:root { --clinical-blue:#1769aa; --clinical-teal:#0d9488; }
.site-nav { position:sticky; top:10px; z-index:20; backdrop-filter:blur(12px); }
.brand-mark { box-shadow:inset 0 0 0 1px #d4e8f5; }
.wellbeing-section { margin:48px 0; }
.wellbeing-grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.wellbeing-card { background:#fff; border:1px solid #dce8f1; border-radius:20px; padding:24px; box-shadow:0 9px 28px rgba(35,76,105,.045); }
.wellbeing-card h3 { margin:0 0 8px; color:#214b69; }
.wellbeing-card p { line-height:1.7; }
.mood-options { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }
.mood-choice { margin:0; width:auto; border:1px solid #d5e3ed; color:#35566f; background:#f6fafc; }
.mood-choice[aria-pressed="true"] { color:#fff; background:#1769aa; border-color:#1769aa; }
.mood-feedback { min-height:44px; padding:12px; border-radius:12px; background:#f0f7fb; color:#31566e; line-height:1.6; }
.breath-orb-wrap { display:grid; place-items:center; padding:18px 0; }
.breath-orb { display:grid; place-items:center; width:124px; height:124px; border-radius:50%; color:#1769aa; background:radial-gradient(circle,#fff 35%,#e0f1fc 36%,#d0e9f8 70%,#c1e5e4 100%); border:1px solid #c8e0ee; box-shadow:0 0 0 12px #f4f9fc; font-weight:800; text-align:center; transition:transform 1s ease; }
.breath-orb.expand { transform:scale(1.14); }
.result-explanation { margin-top:14px; padding:14px; border:1px solid #dbe8f1; background:#f5f9fc; border-radius:13px; color:#48657a; line-height:1.7; font-size:13px; }
#loadingLabel { display:none; }
button:disabled { opacity:.58; }
@media(max-width:760px) { .wellbeing-grid { grid-template-columns:1fr; } .site-nav { position:relative; top:auto; } }
@media(prefers-reduced-motion:reduce) { .breath-orb { transition:none; } }

</style>
</head>
<body>
<main>
<div id="top"></div>
<nav class="site-nav" aria-label="Main navigation">
  <a class="brand" href="#top" aria-label="MentalHealthFL home">
    <span class="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 48 48" role="img" aria-label="Medical cross logo">
        <path d="M24 3.5 42 10v12.2c0 10.2-7.4 17.7-18 22.3C13.4 39.9 6 32.4 6 22.2V10L24 3.5Z" fill="currentColor" opacity=".14"/>
        <path d="M24 3.5 42 10v12.2c0 10.2-7.4 17.7-18 22.3C13.4 39.9 6 32.4 6 22.2V10L24 3.5Z" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"/>
        <path d="M21 13h6v8h8v6h-8v8h-6v-8h-8v-6h8z" fill="currentColor"/>
      </svg>
    </span>
    <span class="brand-copy"><strong>MentalHealth<span>FL</span></strong><small>MENTAL WELLBEING RESEARCH</small></span>
  </a>
  <div class="nav-links"><a href="#checkin">Check-in</a><a href="#how-it-works">How it works</a><a href="#resources">Resources</a><a href="#wellbeing">Wellbeing</a><a class="nav-cta" href="#support">Find support</a></div>
</nav>

<header>
    <div class="hero-brandline"><span class="hero-logo" aria-hidden="true"><svg viewBox="0 0 48 48"><path d="M24 3.5 42 10v12.2c0 10.2-7.4 17.7-18 22.3C13.4 39.9 6 32.4 6 22.2V10L24 3.5Z" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M21 13h6v8h8v6h-8v8h-6v-8h-8v-6h8z" fill="currentColor"/></svg></span><span class="eyebrow">AI-assisted mental wellbeing</span></div>
    <h1>A thoughtful check-in.<br><span>Support for your wellbeing.</span></h1>
    <p>Share what has been on your mind using text or your microphone. Your check-in provides an experimental AI model result—not a diagnosis—and can help you reflect on possible next steps.</p>
    <div class="hero-trust"><span><b class="trust-dot"></b> Private-by-design experience</span><span><b class="trust-dot"></b> Text and voice check-in</span><span><b class="trust-dot"></b> Support resources available</span></div>
</header>

<div class="intro-strip"><span class="intro-icon">☀</span><span><strong>A small check-in can be a good first step.</strong><br><small>Take a moment, share only what feels comfortable, and explore your options.</small></span></div>
<div class="workspace" id="checkin">
<section class="card input-card">
    <div class="section-top"><div><span class="eyebrow">Text &amp; voice check-in</span>
    <h2 style="margin-top:14px">What has been on your mind?</h2></div><span class="step-tag">STEP 01</span></div>
    <p>There is no need to write perfectly. A sentence or two is enough to try the model.</p>

    <div class="action-row">
      <button id="start" class="primary" type="button" onclick="startVoice()">🎙️ Start speaking</button>
      <button id="stop" class="danger" type="button" onclick="stopVoice()" disabled>■ Stop listening</button>
    </div>

    <p id="status" class="status" role="status" aria-live="polite">Ready when you are.</p>

    <label for="inputText">Your check-in</label>
    <textarea id="inputText" maxlength="10000" aria-describedby="inputHint" placeholder="For example: I have a lot on my mind about exams and I am finding it difficult to relax..."></textarea>
    <div class="input-meta"><span id="inputHint">Only share what you feel comfortable sharing.</span><span id="charCount">0 / 10,000</span></div>
    <div class="quick-prompts" aria-label="Example prompts">
      <button class="prompt-chip" type="button" onclick="usePrompt('I feel calm today and I am managing my daily tasks well.')">Feeling calm</button>
      <button class="prompt-chip" type="button" onclick="usePrompt('I feel stressed about my exams and deadlines.')">Exam stress</button>
      <button class="prompt-chip" type="button" onclick="usePrompt('I have been feeling overwhelmed and finding it hard to focus.')">Feeling overwhelmed</button>
    </div>
    <div class="action-row">
      <button id="analyze" class="primary" type="button" onclick="analyzeText()">Analyze check-in →</button>
      <button class="secondary" type="button" onclick="clearText()">Clear</button>
    </div>
    <div class="privacy-note"><span>🔒</span><span>This is an AI research demo, not a diagnosis or a substitute for professional support. Avoid entering identifying or highly sensitive personal details.</span></div>
</section>

<section id="textResult" class="card" aria-live="polite" style="display:none">
    <div class="section-top"><div><span class="eyebrow">Your model output</span><h2 style="margin-top:14px">Check-in result</h2></div><span class="step-tag">STEP 02</span></div>
    <div class="result-top" style="margin-top:20px"><div><p class="small" style="margin:0 0 7px">Predicted category</p><div id="severity" class="big"></div></div><div class="result-icon" aria-hidden="true">✦</div></div>
    <p style="margin:16px 0 8px">Model confidence</p><span id="confidence" class="confidence-pill"></span>
    <p class="small" style="margin-top:14px">Confidence reflects the model's estimate, not a guarantee of correctness or a clinical assessment.</p>
    <div id="resultExplanation" class="result-explanation" role="status"></div>
    <h3>Probability by category</h3>
    <div id="probabilities"></div>
    <div class="action-row"><button class="primary" type="button" onclick="speakTextResult()">🔊 Read result aloud</button><button class="secondary" type="button" onclick="stopSpeech()">Stop audio</button></div>
</section>
</div>

<section class="below-grid" id="how-it-works" aria-label="How MentalHealthFL works">
  <div class="section-heading"><span class="eyebrow">A simple process</span><h2>How your check-in works</h2><p>Designed to be clear and easy to use, whether you type or speak.</p></div>
  <div class="info-grid">
    <article class="info-card"><span class="info-number">01</span><div class="info-icon">✎</div><h3>Share in your words</h3><p>Write a short check-in or use your microphone. You choose what to share.</p></article>
    <article class="info-card"><span class="info-number">02</span><div class="info-icon">⌁</div><h3>Review the model output</h3><p>The research model estimates a severity category and displays its class probabilities.</p></article>
    <article class="info-card"><span class="info-number">03</span><div class="info-icon">♡</div><h3>Choose a next step</h3><p>Use the result as a conversation starter—not a diagnosis—and seek support when needed.</p></article>
  </div>
</section>

<section class="resource-section" id="resources">
  <div class="section-heading"><span class="eyebrow">Trusted information</span><h2>Explore wellbeing resources</h2><p>External resources can offer practical guidance and professional information.</p></div>
  <div class="resource-grid">
    <a class="resource-card" href="https://www.nhs.uk/every-mind-matters/" target="_blank" rel="noopener noreferrer"><span class="resource-label">PRACTICAL TIPS</span><h3>Every Mind Matters <span>↗</span></h3><p>Ideas for managing stress, anxiety, low mood, and sleep.</p><span class="resource-link">Visit NHS resource</span></a>
    <a class="resource-card" href="https://www.mind.org.uk/" target="_blank" rel="noopener noreferrer"><span class="resource-label">SUPPORT & INFORMATION</span><h3>Mind <span>↗</span></h3><p>Mental-health information and guidance on finding support.</p><span class="resource-link">Visit Mind</span></a>
    <a class="resource-card" href="https://www.nimh.nih.gov/health" target="_blank" rel="noopener noreferrer"><span class="resource-label">RESEARCH-BASED INFO</span><h3>NIMH <span>↗</span></h3><p>Evidence-informed information about mental health conditions and research.</p><span class="resource-link">Explore health topics</span></a>
  </div>
</section>


<section class="wellbeing-section" id="wellbeing">
  <div class="section-heading"><span class="eyebrow">Small steps, at your pace</span><h2>Wellbeing tools</h2><p>These optional activities are separate from the AI model and do not send or save your responses to the server.</p></div>
  <div class="wellbeing-grid">
    <article class="wellbeing-card" aria-labelledby="moodTitle">
      <h3 id="moodTitle">How are you feeling right now?</h3>
      <p>Choose an option for a private moment of reflection. This is not a clinical screening.</p>
      <div class="mood-options" role="group" aria-label="Choose your current mood">
        <button type="button" class="mood-choice" data-mood="Calm" aria-pressed="false">☀ Calm</button>
        <button type="button" class="mood-choice" data-mood="Okay" aria-pressed="false">🙂 Okay</button>
        <button type="button" class="mood-choice" data-mood="Stressed" aria-pressed="false">〰 Stressed</button>
        <button type="button" class="mood-choice" data-mood="Low" aria-pressed="false">☁ Low</button>
        <button type="button" class="mood-choice" data-mood="Overwhelmed" aria-pressed="false">🌧 Overwhelmed</button>
      </div>
      <div class="mood-feedback" id="moodFeedback" role="status" aria-live="polite">Your selection stays in this page only and clears when you refresh.</div>
    </article>
    <article class="wellbeing-card" aria-labelledby="breathTitle">
      <h3 id="breathTitle">A short breathing pause</h3>
      <p>Try a comfortable pace. Stop if you feel dizzy or uncomfortable; breathing exercises are optional.</p>
      <div class="breath-orb-wrap"><div id="breathOrb" class="breath-orb" aria-live="polite"><span id="breathCue">Ready<br>when you are</span></div></div>
      <p class="small" id="breathStatus" role="status" aria-live="polite">Suggested cycle: breathe in, breathe out gently. No need to hold your breath.</p>
      <div class="action-row"><button type="button" class="primary" id="breathStart">Start 1-minute exercise</button><button type="button" class="secondary" id="breathStop" disabled>Stop</button></div>
    </article>
  </div>
</section>

<section class="support-banner" id="support"><div><span class="support-kicker">YOU DO NOT HAVE TO HANDLE IT ALONE</span><h2>Need to talk to someone?</h2><p>In India, Tele-MANAS offers 24/7 tele-mental-health support.</p></div><div class="support-action"><strong>Call <a href="tel:14416">14416</a></strong><a class="support-button" href="https://telemanas.mohfw.gov.in/" target="_blank" rel="noopener noreferrer">Official Tele-MANAS site ↗</a></div></section>

<section class="card image-module" aria-hidden="true">
    <h2>🖼️ Facial-expression image analysis</h2>
    <p>Analyze a face image for visible expression and connect it with your existing MentalHealthFL severity model. Because your trained RoBERTa + LoRA model accepts text—not pixels—you must also provide a short description of how you feel. The severity result comes from that text, not from the face image.</p>

    <label for="imageMentalText">How have you been feeling? (required for severity prediction)</label>
    <textarea id="imageMentalText" placeholder="For example: I have been feeling stressed about exams and struggling to sleep."></textarea>

    <label for="faceImage">Upload an image (maximum 8 MB)</label>
    <input id="faceImage" type="file" accept="image/jpeg,image/png,image/webp">

    <label for="cameraSelect">Camera device (choose the laptop webcam)</label>
    <select id="cameraSelect" style="display:block;width:100%;max-width:480px;margin-top:8px;padding:10px;border-radius:8px;background:#0f172a;color:#e2e8f0;border:1px solid #475569;">
        <option value="">Default camera</option>
    </select>
    <p class="small">Open this dashboard in Chrome on your laptop. The list shows cameras available to that browser.</p>

    <div class="image-controls">
        <button type="button" id="startCamera" class="primary">Start laptop webcam</button>
        <button type="button" id="captureImage" class="primary">Capture &amp; analyze</button>
        <button type="button" id="stopCamera" class="danger">Stop webcam</button>
        <button type="button" id="analyzeImage" class="secondary">Analyze uploaded image</button>
    </div>

    <video id="faceVideo" autoplay playsinline></video>
    <div id="facePreview"></div>
    <p id="imageStatus" class="status" role="status">Choose an image or start the webcam.</p>
    <div id="imageResults" aria-live="polite"></div>
    <p class="note">Facial-expression scores do not reliably measure stress, mental-health severity, or diagnose a condition.</p>
</section>

<footer class="site-footer">
  <div><strong>MentalHealthFL</strong><p>Student research project exploring AI-assisted text-based wellbeing check-ins.</p></div>
  <p>This tool is experimental and does not diagnose a condition or replace a qualified professional. Model confidence is not clinical certainty. Avoid entering identifying or highly sensitive details. Browser speech recognition may use browser or external services.</p>
  <div class="footer-bottom"><span>Built for clarity, care, and responsible research.</span><a href="#top">Back to top ↑</a></div>
</footer>
</main>

<script>
/* ---------- Existing voice/text feature ---------- */
let recognition = null;
let finalTranscript = "";
let isRecording = false;
let lastTextResult = null;
let textRequestPending = false;

const statusEl = document.getElementById("status");
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const analyzeBtn = document.getElementById("analyze");
const inputEl = document.getElementById("inputText");

function setStatus(message) {
    statusEl.textContent = message;
}

function updateCharCount() {
    document.getElementById("charCount").textContent = inputEl.value.length.toLocaleString() + " / 10,000";
}

function usePrompt(text) {
    inputEl.value = text;
    updateCharCount();
    inputEl.focus();
    setStatus("Example added. Edit it to reflect your own words, or analyze it as a demo.");
}

inputEl.addEventListener("input", updateCharCount);

function startVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        setStatus("Speech recognition is unavailable in this browser. Try a supported version of Chrome.");
        return;
    }
    if (isRecording) return;

    window.speechSynthesis?.cancel();
    finalTranscript = "";
    inputEl.value = "";
    recognition = new SpeechRecognition();
    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = true;

    isRecording = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Listening... Speak now.");

    recognition.onresult = (event) => {
        let interim = "";
        let finalPart = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) finalPart += transcript + " ";
            else interim += transcript;
        }
        if (finalPart) finalTranscript += finalPart;
        inputEl.value = (finalTranscript + interim).trim();
        updateCharCount();
    };

    recognition.onerror = (event) => {
        setStatus("Speech recognition error: " + event.error);
    };

    recognition.onend = () => {
        isRecording = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        const text = inputEl.value.trim();
        if (text) {
            setStatus("Speech captured. Analyzing text...");
            analyzeText();
        } else {
            setStatus("No speech was captured. Please try again or type text.");
        }
    };

    try {
        recognition.start();
    } catch (error) {
        isRecording = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        setStatus("Could not start recording. Please try again.");
    }
}

function stopVoice() {
    if (recognition && isRecording) {
        setStatus("Stopping recording...");
        recognition.stop();
    }
}

async function analyzeText() {
    const text = inputEl.value.trim();
    if (!text) {
        setStatus("Please speak or enter some text first.");
        return;
    }
    if (textRequestPending) return;

    textRequestPending = true;
    analyzeBtn.disabled = true;
    setStatus("Running model prediction...");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text})
        });
        // Read the body as text first so empty/non-JSON server responses
        // produce a useful error instead of "Unexpected end of JSON input".
        const responseText = await response.text();
        let data = {};

        if (responseText.trim()) {
            try {
                data = JSON.parse(responseText);
            } catch (_parseError) {
                throw new Error(
                    `Server returned a non-JSON response (HTTP ${response.status}). Check Render logs.`
                );
            }
        }

        if (!response.ok) {
            throw new Error(
                data.error || `Prediction failed (HTTP ${response.status}). Check Render logs.`
            );
        }

        if (!data || typeof data.severity !== "string" ||
            typeof data.confidence !== "number" || !data.probabilities) {
            throw new Error("The server response is missing prediction fields. Check Render logs.");
        }

        lastTextResult = data;
        document.getElementById("textResult").style.display = "block";
        document.getElementById("severity").textContent = data.severity;
        document.getElementById("confidence").textContent = (data.confidence * 100).toFixed(2) + "%";
        const severityText = String(data.severity || "").toLowerCase();
        const explanations = {
          minimal: "The model placed this text in its Minimal category. This is only a text-model estimate and does not rule out distress or a need for support.",
          mild: "The model placed this text in its Mild category. Consider what support or self-care feels useful to you; the category is not a diagnosis.",
          moderate: "The model placed this text in its Moderate category. If these feelings are affecting daily life, consider speaking with a qualified health professional or someone you trust.",
          severe: "The model placed this text in its Severe category. This is not a diagnosis. Consider contacting a qualified mental-health professional promptly; if you are in immediate danger, contact local emergency services."
        };
        document.getElementById("resultExplanation").textContent = explanations[severityText] || "This is an experimental model estimate, not a diagnosis. Consider your own experience and seek professional guidance when needed.";
        document.getElementById("textResult").scrollIntoView({behavior: "smooth", block: "nearest"});

        const container = document.getElementById("probabilities");
        container.replaceChildren();

        for (const [label, probability] of Object.entries(data.probabilities || {})) {
            const row = document.createElement("div");
            row.className = "prob-row";
            const head = document.createElement("div");
            head.className = "prob-head";
            const name = document.createElement("span");
            name.textContent = label;
            const value = document.createElement("span");
            value.textContent = (probability * 100).toFixed(2) + "%";
            head.append(name, value);

            const track = document.createElement("div");
            track.className = "track";
            const bar = document.createElement("div");
            bar.className = "bar";
            bar.style.width = Math.max(0, Math.min(100, probability * 100)) + "%";
            track.appendChild(bar);
            row.append(head, track);
            container.appendChild(row);
        }

        setStatus("Text prediction complete.");
        speakTextResult();
    } catch (error) {
        setStatus("Text prediction error: " + error.message);
    } finally {
        textRequestPending = false;
        analyzeBtn.disabled = false;
    }
}

function speakTextResult() {
    if (!lastTextResult || !("speechSynthesis" in window)) {
        setStatus("No text prediction to speak, or speech output is unsupported.");
        return;
    }
    window.speechSynthesis.cancel();
    const message = "The predicted category is " + lastTextResult.severity +
        ". The model confidence is " + (lastTextResult.confidence * 100).toFixed(1) + " percent.";
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = "en-IN";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}

function stopSpeech() {
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

function clearText() {
    if (isRecording && recognition) recognition.stop();
    stopSpeech();
    inputEl.value = "";
    updateCharCount();
    document.getElementById("textResult").style.display = "none";
    lastTextResult = null;
    finalTranscript = "";
    setStatus("Ready.");
}


/* ---------- Optional private wellbeing tools: no server calls or storage ---------- */
const moodFeedback = document.getElementById("moodFeedback");
const moodAdvice = {
  Calm: "Notice what is helping you feel steady today. You may want to keep making space for it.",
  Okay: "It is fine to feel somewhere in the middle. Take the next step at a pace that works for you.",
  Stressed: "If it helps, choose one small task, take a short break, or talk with someone you trust.",
  Low: "Be gentle with yourself. A basic need, a small routine, or reaching out to a trusted person may be a starting point.",
  Overwhelmed: "Pause if you can, focus on one immediate need, and consider reaching out to someone you trust or a professional."
};
document.querySelectorAll(".mood-choice").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mood-choice").forEach(item => item.setAttribute("aria-pressed", "false"));
    button.setAttribute("aria-pressed", "true");
    const mood = button.dataset.mood;
    moodFeedback.textContent = mood + ": " + moodAdvice[mood] + " This choice is not saved.";
  });
});
let breathTimer = null;
let breathStartedAt = 0;
let breathPhaseTimer = null;
const breathOrb = document.getElementById("breathOrb");
const breathCue = document.getElementById("breathCue");
const breathStatus = document.getElementById("breathStatus");
const breathStart = document.getElementById("breathStart");
const breathStop = document.getElementById("breathStop");
function stopBreathing(message = "Exercise stopped. Take your time.") {
  if (breathTimer) clearInterval(breathTimer);
  if (breathPhaseTimer) clearTimeout(breathPhaseTimer);
  breathTimer = null; breathPhaseTimer = null;
  breathOrb.classList.remove("expand");
  breathCue.innerHTML = "Ready<br>when you are";
  breathStatus.textContent = message;
  breathStart.disabled = false; breathStop.disabled = true;
}
function breathingPhase() {
  const elapsed = Math.floor((Date.now() - breathStartedAt) / 1000);
  if (elapsed >= 60) { stopBreathing("One minute complete. Notice how you feel; you can repeat or stop here."); return; }
  const phase = elapsed % 10;
  if (phase < 4) {
    breathCue.textContent = "Breathe in"; breathOrb.classList.add("expand");
    breathStatus.textContent = "Breathe in gently for a comfortable count of four.";
  } else {
    breathCue.textContent = "Breathe out"; breathOrb.classList.remove("expand");
    breathStatus.textContent = "Breathe out slowly and comfortably. Do not force or hold your breath.";
  }
}
breathStart.addEventListener("click", () => {
  if (breathTimer) return;
  breathStartedAt = Date.now(); breathStart.disabled = true; breathStop.disabled = false;
  breathingPhase(); breathTimer = setInterval(breathingPhase, 1000);
});
breathStop.addEventListener("click", () => stopBreathing());
window.addEventListener("beforeunload", () => { if (breathTimer) clearInterval(breathTimer); });

/* ---------- Image upload and webcam feature ---------- */
const imageInput = document.getElementById("faceImage");
const video = document.getElementById("faceVideo");
const imagePreview = document.getElementById("facePreview");
const imageStatus = document.getElementById("imageStatus");
const imageResults = document.getElementById("imageResults");
let cameraStream = null;
let selectedImage = null;
let previewObjectUrl = null;
let imageRequestPending = false;

function setImageStatus(message) {
    imageStatus.textContent = message;
}

function showImagePreview(blob) {
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = URL.createObjectURL(blob);
    imagePreview.replaceChildren();
    const img = document.createElement("img");
    img.src = previewObjectUrl;
    img.alt = "Selected image preview";
    imagePreview.appendChild(img);
}

async function analyzeImageBlob(blob, filename) {
    if (!blob) {
        setImageStatus("Select an image or capture a webcam frame first.");
        return;
    }
    if (blob.size > 8 * 1024 * 1024) {
        setImageStatus("Please use an image smaller than 8 MB.");
        return;
    }
    const mentalText = document.getElementById("imageMentalText").value.trim();
    if (!mentalText) {
        setImageStatus("Enter a short description of how you feel so your trained severity model can analyze it.");
        return;
    }
    if (imageRequestPending) return;

    imageRequestPending = true;
    document.getElementById("analyzeImage").disabled = true;
    document.getElementById("captureImage").disabled = true;
    setImageStatus("Running your severity model and analyzing facial expression...");
    imageResults.replaceChildren();

    const formData = new FormData();
    formData.append("image", blob, filename || "image.jpg");
    formData.append("mental_text", mentalText);

    try {
        const response = await fetch("/analyze-image", {method: "POST", body: formData});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Image analysis failed.");

        const severityTitle = document.createElement("h3");
        severityTitle.textContent = "Your MentalHealthFL severity model result";
        imageResults.appendChild(severityTitle);

        const severity = document.createElement("p");
        severity.className = "big";
        severity.textContent = data.severity_prediction.severity;
        imageResults.appendChild(severity);

        const severityConfidence = document.createElement("p");
        severityConfidence.textContent = "Text-model confidence: " + (Number(data.severity_prediction.confidence) * 100).toFixed(2) + "%";
        imageResults.appendChild(severityConfidence);

        const severityNote = document.createElement("p");
        severityNote.className = "note";
        severityNote.textContent = "This severity prediction was generated from your written description, not inferred from your face.";
        imageResults.appendChild(severityNote);

        const severityClasses = document.createElement("h4");
        severityClasses.textContent = "Severity class probabilities";
        imageResults.appendChild(severityClasses);
        for (const [label, probability] of Object.entries(data.severity_prediction.probabilities || {})) {
            const row = document.createElement("div");
            row.className = "prob-row";
            const head = document.createElement("div");
            head.className = "prob-head";
            const name = document.createElement("span"); name.textContent = label;
            const value = document.createElement("span"); value.textContent = (Number(probability) * 100).toFixed(2) + "%";
            head.append(name, value);
            const track = document.createElement("div"); track.className = "track";
            const bar = document.createElement("div"); bar.className = "bar";
            bar.style.width = Math.max(0, Math.min(100, Number(probability) * 100)) + "%";
            track.appendChild(bar); row.append(head, track); imageResults.appendChild(row);
        }

        const divider = document.createElement("hr");
        divider.style.borderColor = "#334155";
        imageResults.appendChild(divider);

        const heading = document.createElement("h3");
        heading.textContent = "Facial expression estimate: " + data.expression;
        heading.className = "big";
        imageResults.appendChild(heading);

        const confidence = document.createElement("p");
        confidence.textContent = "Top expression score: " + Number(data.confidence).toFixed(2) + "%";
        imageResults.appendChild(confidence);

        const subheading = document.createElement("h3");
        subheading.textContent = "Expression scores";
        imageResults.appendChild(subheading);

        for (const [label, score] of Object.entries(data.probabilities || {})) {
            const row = document.createElement("div");
            row.className = "expression-row";
            const head = document.createElement("div");
            head.className = "prob-head";
            const name = document.createElement("span");
            name.textContent = label;
            const value = document.createElement("span");
            value.textContent = Number(score).toFixed(2) + "%";
            head.append(name, value);

            const track = document.createElement("div");
            track.className = "track";
            const bar = document.createElement("div");
            bar.className = "bar";
            bar.style.width = Math.max(0, Math.min(100, Number(score) || 0)) + "%";
            track.appendChild(bar);
            row.append(head, track);
            imageResults.appendChild(row);
        }

        const note = document.createElement("p");
        note.className = "note";
        note.textContent = data.note || "";
        imageResults.appendChild(note);
        setImageStatus("Image analysis completed.");
    } catch (error) {
        setImageStatus("Image analysis error: " + error.message);
    } finally {
        imageRequestPending = false;
        document.getElementById("analyzeImage").disabled = false;
        document.getElementById("captureImage").disabled = false;
    }
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;
    selectedImage = file;
    showImagePreview(file);
    setImageStatus("Image selected. Click Analyze uploaded image.");
});

document.getElementById("analyzeImage").addEventListener("click", () => {
    analyzeImageBlob(selectedImage, "uploaded-image");
});

const cameraSelect = document.getElementById("cameraSelect");

async function refreshCameraList(preferredDeviceId = "") {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === "videoinput");
    const previous = preferredDeviceId || cameraSelect.value;
    cameraSelect.replaceChildren();

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "System default camera";
    cameraSelect.appendChild(defaultOption);

    cameras.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
    });

    // Prefer a camera whose browser-provided label looks like a laptop webcam.
    if (previous && cameras.some(device => device.deviceId === previous)) {
        cameraSelect.value = previous;
    } else {
        const likelyLaptop = cameras.find(device =>
            /integrated|built.?in|laptop|hd webcam|webcam/i.test(device.label || "")
        );
        if (likelyLaptop) cameraSelect.value = likelyLaptop.deviceId;
    }
}

async function startCamera() {
    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Camera access is unavailable. Open http://127.0.0.1:5000 in Chrome on the laptop.");
        }

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        const deviceId = cameraSelect.value;
        const videoConstraints = deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } };

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });
        video.srcObject = cameraStream;
        video.style.display = "block";
        video.muted = true;
        await new Promise((resolve, reject) => {
            if (video.readyState >= 2 && video.videoWidth > 0) return resolve();
            const timer = setTimeout(() => reject(new Error("Camera preview did not become ready. Try another camera.")), 10000);
            video.onloadedmetadata = () => {
                video.play().then(() => {
                    clearTimeout(timer);
                    resolve();
                }).catch(error => {
                    clearTimeout(timer);
                    reject(error);
                });
            };
        });
        await video.play();
        const activeDeviceId = cameraStream.getVideoTracks()[0]?.getSettings()?.deviceId || deviceId;
        await refreshCameraList(activeDeviceId);

        // If the device list reveals a likely built-in laptop webcam, switch to it.
        const selectedDeviceId = cameraSelect.value;
        if (!deviceId && selectedDeviceId && selectedDeviceId !== activeDeviceId) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            video.srcObject = cameraStream;
            await video.play();
        }
        setImageStatus("Camera started. Confirm the preview shows your face, then click Capture & analyze.");
    } catch (error) {
        setImageStatus("Could not start webcam: " + error.message + ". Check Chrome camera permissions and select the laptop webcam from the list.");
    }
}

document.getElementById("startCamera").addEventListener("click", startCamera);

cameraSelect.addEventListener("change", async () => {
    if (cameraStream) await startCamera();
});

// Ask permission once on demand, then enumerate available cameras and labels.
document.getElementById("startCamera").addEventListener("click", async () => {
    try { await refreshCameraList(cameraSelect.value); } catch (_) { /* handled by startCamera */ }
});

document.getElementById("captureImage").addEventListener("click", async () => {
    if (!cameraStream || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
        setImageStatus("Start the laptop webcam first and wait until the live preview appears.");
        return;
    }

    // Give the browser a moment to render a live frame before drawing the canvas.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let min = 255, max = 0;
    for (let i = 0; i < pixels.length; i += 4) {
        min = Math.min(min, pixels[i], pixels[i + 1], pixels[i + 2]);
        max = Math.max(max, pixels[i], pixels[i + 1], pixels[i + 2]);
    }
    if (max === 0 || max - min < 3) {
        setImageStatus("The camera frame is black or blank. Check the live preview, choose another camera, and try again.");
        return;
    }

    canvas.toBlob(blob => {
        if (!blob) {
            setImageStatus("Could not capture the webcam frame. Please try again.");
            return;
        }
        selectedImage = blob;
        showImagePreview(blob);
        analyzeImageBlob(blob, "laptop-webcam-capture.jpg");
    }, "image/jpeg", 0.92);
});

document.getElementById("stopCamera").addEventListener("click", () => {
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    video.srcObject = null;
    video.style.display = "none";
    setImageStatus("Webcam stopped.");
});

window.addEventListener("beforeunload", () => {
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
});
</script>
</body>
</html>
"""


@app.route("/health", methods=["GET"])
def health():
    """Lightweight health check; does not load the large ML model."""
    return jsonify({"status": "ok", "service": "MentalHealthFL", "image_analysis_enabled": ENABLE_IMAGE_ANALYSIS, "model_loaded": False}), 200


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide non-empty text."}), 400

    if len(text) > 10000:
        return jsonify({"error": "Please keep the text under 10,000 characters."}), 400

    try:
        predict_severity = get_severity_predictor()
        result = predict_severity(text.strip(), prefer_federated=False)
        return jsonify({
            "severity": result["severity"],
            "confidence": float(result["confidence"]),
            "probabilities": {
                label: float(probability)
                for label, probability in result["probabilities"].items()
            },
        })
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the Python terminal for details."
        }), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if not ENABLE_IMAGE_ANALYSIS:
        return jsonify({"error": "Image analysis is disabled. Text and voice check-in are available."}), 503
    try:
        from image_processing import analyze_facial_expression
    except ImportError:
        app.logger.exception("Image analysis dependencies are unavailable")
        return jsonify({"error": "Image analysis dependencies are not installed."}), 503
    if "image" not in request.files:
        return jsonify({"error": "Please select or capture an image."}), 400

    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "No image was selected."}), 400

    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify({"error": "The image is empty."}), 400

    if len(image_bytes) > 8 * 1024 * 1024:
        return jsonify({"error": "Image must be smaller than 8 MB."}), 413

    mental_text = request.form.get("mental_text", "").strip()
    if not mental_text:
        return jsonify({"error": "Please describe how you feel so the trained severity model can analyze your text."}), 400
    if len(mental_text) > 10000:
        return jsonify({"error": "Please keep the description under 10,000 characters."}), 400

    try:
        predict_severity = get_severity_predictor()
        result = analyze_facial_expression(image_bytes)
        severity_result = predict_severity(mental_text, prefer_federated=False)
        result["severity_prediction"] = {
            "severity": severity_result["severity"],
            "confidence": float(severity_result["confidence"]),
            "probabilities": {
                label: float(probability)
                for label, probability in severity_result["probabilities"].items()
            },
        }
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Facial-expression analysis failed")
        return jsonify({
            "error": str(exc) or "Image analysis failed. Try a clear, front-facing photo."
        }), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large. Use an image smaller than 8 MB."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
