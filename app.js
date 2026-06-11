/* Simple_RAG web UI — vanilla JS, talks to server.py's JSON API. */

const state = {
  project: null,
  session: null,
  asking: false,
};

const $ = (id) => document.getElementById(id);

// ------------------------------- api ---------------------------------- //
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function pollJob(jobId, onDone, onProgress = null, intervalMs = 2000) {
  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (err) {
      onDone({ status: "error", error: err.message });
      return;
    }
    if (job.status === "running") {
      if (onProgress) onProgress(job);
      setTimeout(tick, intervalMs);
    } else {
      onDone(job);
    }
  };
  setTimeout(tick, intervalMs);
}

// ----------------------------- projects -------------------------------- //
async function loadProjects() {
  const projects = await api("/api/projects");
  const list = $("projectList");
  list.innerHTML = "";
  for (const p of projects) {
    const li = document.createElement("li");
    li.classList.toggle("active", p.name === state.project);
    li.innerHTML = `<span class="name"></span><span class="count"></span>`;
    li.querySelector(".name").textContent = p.name;
    li.querySelector(".count").textContent = `${p.chunks} chunks`;
    li.onclick = () => selectProject(p.name);
    list.appendChild(li);
  }
}

async function selectProject(name) {
  state.project = name;
  state.session = null;
  $("docsSection").hidden = false;
  $("chatsSection").hidden = false;
  await Promise.all([loadProjects(), loadFiles(), loadSessions()]);
  const sessions = await api(`/api/projects/${name}/sessions`);
  if (sessions.length) {
    selectSession(sessions[0].name);
  } else {
    newChat();
  }
}

$("newProjectBtn").onclick = async () => {
  const name = prompt("New project name:");
  if (!name) return;
  try {
    const p = await api("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadProjects();
    selectProject(p.name);
  } catch (err) {
    alert(err.message);
  }
};

// ------------------------------ files ---------------------------------- //
async function loadFiles() {
  const files = await api(`/api/projects/${state.project}/files`);
  const list = $("fileList");
  list.innerHTML = "";
  for (const f of files) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="count"></span>`;
    li.querySelector(".name").textContent = f.name;
    if (f.status === "pending") {
      li.querySelector(".count").textContent = "new";
      li.querySelector(".count").classList.add("pending-badge");
      li.title = "Will be indexed when you ask your next question";
    } else {
      li.querySelector(".count").textContent = `${f.chunks}`;
      li.title = `${f.chunks} chunks · ingested ${f.ingested_at || "?"}`;
    }
    list.appendChild(li);
  }
}

$("attachBtn").onclick = () => {
  if (!state.project) { alert("Select a project first."); return; }
  $("fileInput").click();
};
$("fileInput").onchange = () => {
  if ($("fileInput").files.length) uploadFiles($("fileInput").files);
};

async function uploadFiles(fileList) {
  if (!state.project) { alert("Select a project first."); return; }
  const form = new FormData();
  for (const f of fileList) form.append("files", f);
  const status = $("uploadStatus");
  status.hidden = false;
  status.textContent = "Adding…";
  try {
    const { saved } = await api(`/api/projects/${state.project}/upload`, {
      method: "POST", body: form,
    });
    status.textContent =
      `Added ${saved.length} file(s) — indexed with your next question.`;
    await loadFiles();
    setTimeout(() => { status.hidden = true; }, 6000);
  } catch (err) {
    status.textContent = `Upload failed: ${err.message}`;
  }
  $("fileInput").value = "";
}

// ----------------------------- sessions -------------------------------- //
async function loadSessions() {
  const sessions = await api(`/api/projects/${state.project}/sessions`);
  const list = $("sessionList");
  list.innerHTML = "";
  for (const s of sessions) {
    const li = document.createElement("li");
    li.classList.toggle("active", s.name === state.session);
    li.innerHTML = `<span class="name"></span><span class="count"></span>`;
    li.querySelector(".name").textContent = s.preview || s.name;
    li.querySelector(".count").textContent = `${s.turns}`;
    li.title = s.name;
    li.onclick = () => selectSession(s.name);
    list.appendChild(li);
  }
}

function newChat() {
  const stamp = new Date().toISOString().slice(0, 16)
    .replace("T", "_").replace(":", "");
  selectSession(`chat_${stamp}`);
}
$("newChatBtn").onclick = newChat;

async function selectSession(name) {
  state.session = name;
  $("emptyState").hidden = true;
  $("chatView").hidden = false;
  $("chatTitle").textContent = state.project;
  $("chatMeta").textContent = name;
  await loadSessions();
  const history = await api(
    `/api/projects/${state.project}/sessions/${name}`);
  const box = $("messages");
  box.innerHTML = "";
  for (const turn of history.turns) renderTurn(turn);
  box.scrollTop = box.scrollHeight;
}

// ----------------------------- rendering ------------------------------- //
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}

function renderUser(text) {
  const row = el(`<div class="msg-row msg-user"><div class="bubble"></div></div>`);
  row.querySelector(".bubble").textContent = text;
  $("messages").appendChild(row);
  return row;
}

function renderTurn(turn) {
  renderUser(turn.query);
  const row = el(`<div class="msg-row msg-assistant">
      <div class="answer"></div>
      <div class="msg-meta">
        <span class="conf-badge"></span>
        <span class="timing"></span>
        <span class="model"></span>
        <button class="frag-toggle"></button>
      </div>
      <div class="fragments" hidden></div>
    </div>`);
  row.querySelector(".answer").textContent = turn.answer;
  row.querySelector(".conf-badge").textContent =
    `confidence ${Number(turn.confidence).toFixed(2)}`;
  const secs = turn.timing_seconds ? turn.timing_seconds.total : null;
  row.querySelector(".timing").textContent =
    secs != null ? fmtSecs(secs) : "";
  row.querySelector(".model").textContent = turn.model || "";

  const frags = turn.fragments || [];
  const toggle = row.querySelector(".frag-toggle");
  const fragBox = row.querySelector(".fragments");
  toggle.textContent = `${frags.length} sources`;
  for (const [i, f] of frags.entries()) {
    const meta = f.metadata || {};
    const item = el(`<div class="fragment"><div class="frag-head"></div><div class="frag-text"></div></div>`);
    let pages = "";
    if (meta.page_start != null) {
      pages = meta.page_start === meta.page_end
        ? ` · p. ${meta.page_start}`
        : ` · pp. ${meta.page_start}–${meta.page_end}`;
    }
    item.querySelector(".frag-head").textContent =
      `${i + 1} · ${meta.source_label || "?"}${pages} · chunk ${meta.chunk_index ?? "?"} · ` +
      `confidence ${(f.confidence ?? 0).toFixed(3)}`;
    item.querySelector(".frag-text").textContent = f.text;
    fragBox.appendChild(item);
  }
  toggle.onclick = () => { fragBox.hidden = !fragBox.hidden; };

  $("messages").appendChild(row);
}

function fmtSecs(s) {
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

// ------------------------------- ask ----------------------------------- //
const input = $("questionInput");
const sendBtn = $("sendBtn");

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
  sendBtn.disabled = !input.value.trim() || state.asking;
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendQuestion();
  }
});
sendBtn.onclick = sendQuestion;

async function sendQuestion() {
  const question = input.value.trim();
  if (!question || state.asking || !state.project) return;
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  state.asking = true;

  renderUser(question);
  const thinkRow = el(`<div class="msg-row msg-assistant">
      <div class="thinking"><span class="pulse"></span>
      <span class="phase">Working locally…</span>
      <span class="timer">0s</span></div></div>`);
  $("messages").appendChild(thinkRow);
  $("messages").scrollTop = $("messages").scrollHeight;

  const start = Date.now();
  const timer = setInterval(() => {
    thinkRow.querySelector(".timer").textContent =
      fmtSecs((Date.now() - start) / 1000);
  }, 1000);

  try {
    const { job_id } = await api(
      `/api/projects/${state.project}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session: state.session }),
      });
    pollJob(job_id, async (job) => {
      clearInterval(timer);
      thinkRow.remove();
      // renderTurn re-renders the user bubble; remove the optimistic one.
      const rows = $("messages").querySelectorAll(".msg-row.msg-user");
      if (rows.length) rows[rows.length - 1].remove();
      if (job.status === "done") {
        renderTurn(job.result);
        loadSessions();
        if (job.result.indexed_now) {
          loadFiles();
          loadProjects();
        }
      } else {
        renderUser(question);
        $("messages").appendChild(
          el(`<div class="msg-row msg-assistant"><div class="error-note"></div></div>`));
        $("messages").lastChild.querySelector(".error-note").textContent =
          job.error || "Unknown error.";
      }
      $("messages").scrollTop = $("messages").scrollHeight;
      state.asking = false;
      sendBtn.disabled = !input.value.trim();
    }, (job) => {
      const labels = {
        indexing: "Indexing new documents…",
        answering: "Retrieving and generating locally…",
      };
      thinkRow.querySelector(".phase").textContent =
        labels[job.phase] || "Working locally…";
    });
  } catch (err) {
    clearInterval(timer);
    thinkRow.querySelector(".thinking").outerHTML =
      `<div class="error-note"></div>`;
    thinkRow.querySelector(".error-note").textContent = err.message;
    state.asking = false;
  }
}

// ------------------------------- boot ----------------------------------- //
loadProjects();
