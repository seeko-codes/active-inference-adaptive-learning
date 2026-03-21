"""
Problem browser — browse all question types without playing through sessions.

Local:  python3 browse_server.py       → http://localhost:8001
Vercel: deployed as /browse via api/browse.py
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BROWSE_PASSWORD = os.environ.get("BROWSE_PASSWORD", "Money2k10")

app = FastAPI(title="Problem Browser")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = None

def _load():
    global _cache
    if _cache is not None:
        return _cache

    from domain.render import render_problems, render_supplementary_problems
    from domain.knowledge_space import TIERS
    from domain.ast_nodes import Leaf, Op

    DATA_DIR = Path(__file__).parent / "data"
    ks_path = DATA_DIR / "knowledge_space.json"

    ast_registry = {}
    for tier_def in TIERS:
        for name, node in tier_def["seeds"].items():
            ast_registry[name] = node

    problems = []
    if ks_path.exists():
        with open(ks_path) as f:
            dataset = json.load(f)
        problems = render_problems(dataset, ast_registry, num_numeric_variants=3)

    problems.extend(render_supplementary_problems())
    _cache = problems
    return _cache


def _check_auth(x_password: str = Header(None)):
    if x_password != BROWSE_PASSWORD:
        return False
    return True


@app.get("/browse/api/problems")
def get_problems(question_type: str = None, skill: str = None, tier: int = None,
                 limit: int = 50, offset: int = 0, x_password: str = Header(None)):
    if not _check_auth(x_password):
        return JSONResponse({"error": "unauthorized"}, 401)

    problems = _load()
    filtered = problems

    if question_type:
        filtered = [p for p in filtered if p.get("question_type") == question_type]
    if skill:
        filtered = [p for p in filtered if skill in p.get("skills_tested", []) or skill in p.get("derivation_path", [])]
    if tier is not None:
        filtered = [p for p in filtered if p.get("tier") == tier]

    total = len(filtered)
    page = filtered[offset:offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "problems": page}


@app.get("/browse/api/stats")
def get_stats(x_password: str = Header(None)):
    if not _check_auth(x_password):
        return JSONResponse({"error": "unauthorized"}, 401)

    problems = _load()
    types = {}
    skills = {}
    tiers = {}
    for p in problems:
        qt = p.get("question_type", "?")
        types[qt] = types.get(qt, 0) + 1
        for s in p.get("skills_tested", []):
            skills[s] = skills.get(s, 0) + 1
        t = p.get("tier", -1)
        tiers[t] = tiers.get(t, 0) + 1

    return {
        "total": len(problems),
        "by_type": dict(sorted(types.items())),
        "by_skill": dict(sorted(skills.items())),
        "by_tier": dict(sorted(tiers.items(), key=lambda x: x[0])),
    }


# Also serve at root paths for local dev
@app.get("/api/problems")
def get_problems_local(question_type: str = None, skill: str = None, tier: int = None,
                       limit: int = 50, offset: int = 0, x_password: str = Header(None)):
    if not _check_auth(x_password):
        return JSONResponse({"error": "unauthorized"}, 401)
    return get_problems(question_type, skill, tier, limit, offset, x_password)


@app.get("/api/stats")
def get_stats_local(x_password: str = Header(None)):
    if not _check_auth(x_password):
        return JSONResponse({"error": "unauthorized"}, 401)
    return get_stats(x_password)


@app.get("/browse", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Problem Browser</title>
<style>
  :root { --bg: #0a0a0a; --card: #141414; --border: #2a2a2a; --text: #e0e0e0;
          --muted: #888; --primary: #7c6ef0; --primary-dim: #7c6ef020; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, system-ui, sans-serif; background: var(--bg);
         color: var(--text); line-height: 1.5; padding: 1.5rem; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin-bottom: 1rem; }
  .stats-bar { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 0.5rem 1rem; font-size: 0.85rem; }
  .stat b { color: var(--primary); }
  .filters { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1.5rem; align-items: center; }
  select, button, input[type=password] { background: var(--card); color: var(--text); border: 1px solid var(--border);
                   border-radius: 6px; padding: 0.4rem 0.75rem; font-size: 0.85rem; cursor: pointer; }
  select:focus, button:focus, input:focus { outline: 2px solid var(--primary); }
  button { background: var(--primary); color: #fff; border: none; font-weight: 600; }
  button:hover { opacity: 0.9; }
  button.secondary { background: var(--card); border: 1px solid var(--border); color: var(--text); font-weight: 400; }
  .count { font-size: 0.8rem; color: var(--muted); }
  .cards { display: flex; flex-direction: column; gap: 1rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
  .badge { display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px;
           background: var(--primary-dim); color: var(--primary); border: 1px solid #7c6ef030; margin-right: 0.4rem; }
  .badge.tier { background: #1a1a2e; color: var(--muted); border-color: var(--border); }
  .expression { text-align: center; font-size: 1.3rem; padding: 0.75rem 0; font-family: 'Times New Roman', serif; }
  .vs-display { display: flex; align-items: center; justify-content: center; gap: 1.5rem; padding: 0.75rem 0; }
  .vs-display span.vs { color: var(--muted); font-size: 0.8rem; font-family: sans-serif; }
  .vs-display .expr { font-size: 1.2rem; font-family: 'Times New Roman', serif; }
  .arrow-display { display: flex; align-items: center; justify-content: center; gap: 1rem; padding: 0.75rem 0; }
  .arrow-display .expr { font-size: 1.2rem; font-family: 'Times New Roman', serif; }
  .arrow-display .arrow { color: var(--primary); font-weight: bold; font-size: 1.2rem; }
  .prompt { font-size: 0.9rem; color: var(--text); margin: 0.5rem 0; white-space: pre-wrap; }
  .hint { background: var(--primary-dim); border: 1px solid #7c6ef015; border-radius: 6px;
          padding: 0.5rem 0.75rem; font-size: 0.8rem; color: #a99cf0; margin-top: 0.5rem; }
  .answer { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
  .answer b { color: #6ee7b7; }
  .choices { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
  .choice { font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px;
            background: #1a1a2e; border: 1px solid var(--border); }
  .choice.correct { border-color: #6ee7b7; color: #6ee7b7; }
  .pager { display: flex; gap: 0.5rem; justify-content: center; margin-top: 1.5rem; align-items: center; }
  .meta { font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem; display: flex; gap: 1rem; flex-wrap: wrap; }
  .worked-example { background: var(--primary-dim); border: 1px solid #7c6ef015; border-radius: 8px;
                    padding: 1rem; margin-bottom: 0.5rem; }
  .worked-example .rule { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
                          color: var(--primary); margin-bottom: 0.4rem; }
  .error-display { background: #2a1515; border: 1px solid #3a2020; border-radius: 8px;
                   padding: 0.75rem; margin-bottom: 0.5rem; }
  .error-display .wrong { color: #f87171; font-family: 'Times New Roman', serif; font-size: 1.1rem; }
  #login-screen { display: flex; flex-direction: column; align-items: center; justify-content: center;
                  min-height: 60vh; gap: 1rem; }
  #login-screen h2 { font-size: 1.1rem; color: var(--muted); }
  #login-screen .login-box { display: flex; gap: 0.5rem; }
  #login-screen input { width: 220px; }
  #login-error { color: #f87171; font-size: 0.8rem; min-height: 1.2rem; }
  #main-content { display: none; }
</style>
</head>
<body>

<div id="login-screen">
  <h1>Problem Browser</h1>
  <h2>Enter password to continue</h2>
  <div class="login-box">
    <input type="password" id="pwd" placeholder="Password" onkeydown="if(event.key==='Enter')doLogin()"/>
    <button onclick="doLogin()">Enter</button>
  </div>
  <div id="login-error"></div>
</div>

<div id="main-content">
<h1>Problem Browser</h1>
<div class="stats-bar" id="stats"></div>

<div class="filters">
  <select id="typeFilter"><option value="">All Types</option></select>
  <select id="skillFilter"><option value="">All Skills</option></select>
  <select id="tierFilter"><option value="">All Tiers</option></select>
  <button onclick="loadProblems(0)">Filter</button>
  <button class="secondary" onclick="resetFilters()">Reset</button>
  <span class="count" id="resultCount"></span>
</div>

<div class="cards" id="cards"></div>
<div class="pager" id="pager"></div>
</div>

<script>
const LIMIT = 20;
let currentOffset = 0;
let password = sessionStorage.getItem('browse_pwd') || '';

// Detect base path: /browse on Vercel, / locally
const isVercel = location.pathname.startsWith('/browse');
const apiBase = isVercel ? '/browse/api' : '/api';

function apiFetch(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const url = `${apiBase}${path}${qs ? '?' + qs : ''}`;
  return fetch(url, { headers: { 'X-Password': password } });
}

async function doLogin() {
  password = document.getElementById('pwd').value;
  const r = await apiFetch('/stats');
  if (r.ok) {
    sessionStorage.setItem('browse_pwd', password);
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';
    loadStats();
    loadProblems(0);
  } else {
    document.getElementById('login-error').textContent = 'Wrong password';
  }
}

// Auto-login if password is cached
(async () => {
  if (password) {
    const r = await apiFetch('/stats');
    if (r.ok) {
      document.getElementById('login-screen').style.display = 'none';
      document.getElementById('main-content').style.display = 'block';
      loadStats();
      loadProblems(0);
      return;
    }
    sessionStorage.removeItem('browse_pwd');
    password = '';
  }
  document.getElementById('pwd').focus();
})();

async function loadStats() {
  const r = await apiFetch('/stats');
  const s = await r.json();
  document.getElementById('stats').innerHTML =
    `<div class="stat"><b>${s.total}</b> total problems</div>` +
    `<div class="stat"><b>${Object.keys(s.by_type).length}</b> question types</div>` +
    `<div class="stat"><b>${Object.keys(s.by_skill).length}</b> skills</div>` +
    `<div class="stat"><b>${Object.keys(s.by_tier).length}</b> tiers</div>`;

  const typeSelect = document.getElementById('typeFilter');
  for (const [t, c] of Object.entries(s.by_type)) {
    typeSelect.innerHTML += `<option value="${t}">${t} (${c})</option>`;
  }
  const skillSelect = document.getElementById('skillFilter');
  for (const [sk, c] of Object.entries(s.by_skill)) {
    skillSelect.innerHTML += `<option value="${sk}">${sk} (${c})</option>`;
  }
  const tierSelect = document.getElementById('tierFilter');
  for (const [ti, c] of Object.entries(s.by_tier)) {
    tierSelect.innerHTML += `<option value="${ti}">Tier ${ti} (${c})</option>`;
  }
}

async function loadProblems(offset) {
  currentOffset = offset;
  const type = document.getElementById('typeFilter').value;
  const skill = document.getElementById('skillFilter').value;
  const tier = document.getElementById('tierFilter').value;

  const params = { limit: LIMIT, offset };
  if (type) params.question_type = type;
  if (skill) params.skill = skill;
  if (tier) params.tier = tier;

  const r = await apiFetch('/problems', params);
  const data = await r.json();

  document.getElementById('resultCount').textContent = `${data.total} results`;
  renderCards(data.problems);
  renderPager(data.total, offset);
}

function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function renderCards(problems) {
  const container = document.getElementById('cards');
  container.innerHTML = problems.map((p, i) => {
    const idx = currentOffset + i + 1;
    let body = '';

    if (p.question_type === 'worked_example' && p.example_before) {
      body += `<div class="worked-example">
        <div class="rule">${esc(p.rule_demonstrated)}</div>
        <div class="arrow-display">
          <span class="expr">${esc(p.example_before)}</span>
          <span class="arrow">=</span>
          <span class="expr">${esc(p.example_after)}</span>
        </div>
        ${p.rule_description ? `<div style="font-size:0.8rem;color:var(--muted)">${esc(p.rule_description)}</div>` : ''}
      </div>`;
    }

    if (p.question_type === 'identify_property' && p.expression_before) {
      body += `<div class="arrow-display">
        <span class="expr">${esc(p.expression_before)}</span>
        <span class="arrow">&rarr;</span>
        <span class="expr">${esc(p.expression_after)}</span>
      </div>`;
    }

    if (['equivalent','non_equivalent','boundary_test','custom_operation'].includes(p.question_type) && p.expression_a) {
      body += `<div class="vs-display">
        <span class="expr">${esc(p.expression_a)}</span>
        <span class="vs">vs</span>
        <span class="expr">${esc(p.expression_b)}</span>
      </div>`;
    }

    if (p.question_type === 'find_error' && p.original_expression) {
      body += `<div class="error-display">
        <div style="font-size:0.75rem;color:var(--muted)">Student claims:</div>
        <div class="wrong">${esc(p.original_expression)} = ${esc(p.claimed_answer)}</div>
      </div>`;
    }

    const skipExpr = ['identify_property','equivalent','non_equivalent','boundary_test',
                      'custom_operation','worked_example','find_error'];
    if (p.student_sees && !skipExpr.includes(p.question_type)) {
      body += `<div class="expression">${esc(p.student_sees)}</div>`;
    }

    body += `<div class="prompt">${esc(p.prompt)}</div>`;

    if (p.scaffolding?.hint) {
      body += `<div class="hint">${esc(p.scaffolding.hint)}</div>`;
    }

    if (p.choices) {
      body += `<div class="choices">${p.choices.map(c =>
        `<span class="choice${c === String(p.expected_answer) ? ' correct' : ''}">${esc(c)}</span>`
      ).join('')}</div>`;
    }

    const ans = p.expected_answer;
    body += `<div class="answer">Expected: <b>${esc(ans)}</b></div>`;

    const skills = (p.skills_tested || []).join(', ');
    body += `<div class="meta">
      <span>ID: ${esc(p.problem_id)}</span>
      ${skills ? `<span>Skills: ${esc(skills)}</span>` : ''}
      ${p.seed_name ? `<span>Seed: ${esc(p.seed_name)}</span>` : ''}
    </div>`;

    return `<div class="card">
      <div class="card-header">
        <div>
          <span class="badge">${esc(p.question_type)}</span>
          <span class="badge tier">Tier ${p.tier}</span>
          <span style="font-size:0.75rem;color:var(--muted)">#${idx}</span>
        </div>
      </div>
      ${body}
    </div>`;
  }).join('');
}

function renderPager(total, offset) {
  const pages = Math.ceil(total / LIMIT);
  const current = Math.floor(offset / LIMIT);
  if (pages <= 1) { document.getElementById('pager').innerHTML = ''; return; }

  let html = '';
  if (current > 0)
    html += `<button class="secondary" onclick="loadProblems(${(current-1)*LIMIT})">&laquo; Prev</button>`;

  const start = Math.max(0, current - 3);
  const end = Math.min(pages, current + 4);
  for (let i = start; i < end; i++) {
    if (i === current)
      html += `<button style="opacity:1">${i+1}</button>`;
    else
      html += `<button class="secondary" onclick="loadProblems(${i*LIMIT})">${i+1}</button>`;
  }

  if (current < pages - 1)
    html += `<button class="secondary" onclick="loadProblems(${(current+1)*LIMIT})">Next &raquo;</button>`;

  document.getElementById('pager').innerHTML = html;
}

function resetFilters() {
  document.getElementById('typeFilter').value = '';
  document.getElementById('skillFilter').value = '';
  document.getElementById('tierFilter').value = '';
  loadProblems(0);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    print("Problem Browser: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
