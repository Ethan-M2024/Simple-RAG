"""FastAPI web UI for local_rag_pipeline.py.

Wraps the existing pipeline (no logic duplicated) behind a small JSON API
and serves a Claude-style chat frontend from ./static.

Run:
    python3 webapp/server.py
    # then open http://127.0.0.1:8400
"""

import os
import sys
import json
import time
import uuid
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_DIR = os.path.join(REPO_ROOT, "projects")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

sys.path.insert(0, REPO_ROOT)
from local_rag_pipeline import (  # noqa: E402
    DeterministicRAGEngine,
    LocalVectorStorage,
    ProjectWorkspace,
    ingest_project,
    resolve_model,
)

app = FastAPI(title="Simple_RAG UI")

# ----------------------------------------------------------------------- #
# Per-project store cache + locks. Opening a store loads the FastEmbed
# model, so each project's store is created once and reused. All chroma
# work for a project is serialized through its lock.
# ----------------------------------------------------------------------- #
_stores: Dict[str, LocalVectorStorage] = {}
_project_locks: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _safe_name(name: str) -> str:
    cleaned = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ")
    cleaned = cleaned.strip().replace(" ", "_")
    if not cleaned:
        raise HTTPException(400, "Invalid project name.")
    return cleaned


def _project_lock(name: str) -> threading.Lock:
    with _registry_lock:
        if name not in _project_locks:
            _project_locks[name] = threading.Lock()
        return _project_locks[name]


def _get_store(workspace: ProjectWorkspace) -> LocalVectorStorage:
    with _registry_lock:
        store = _stores.get(workspace.name)
        if store is None:
            store = LocalVectorStorage(
                persist_path=workspace.chroma_path,
                collection_name="project_chunks",
            )
            _stores[workspace.name] = store
        return store


def _workspace(name: str) -> ProjectWorkspace:
    name = _safe_name(name)
    if not os.path.isdir(os.path.join(PROJECTS_DIR, name)):
        raise HTTPException(404, f"Project '{name}' not found.")
    return ProjectWorkspace(name, base_dir=PROJECTS_DIR)


def _uploads_dir(ws: ProjectWorkspace) -> str:
    path = os.path.join(ws.root, "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def _pending_files(ws: ProjectWorkspace) -> List[str]:
    """Uploaded files not yet indexed at their current contents."""
    manifest = ws.load_manifest()
    pending = []
    uploads = _uploads_dir(ws)
    for name in sorted(os.listdir(uploads)):
        path = os.path.join(uploads, name)
        if os.path.isfile(path) and not ws.is_indexed(path, manifest):
            pending.append(path)
    return pending


def _new_job(kind: str, project: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "project": project,
            "status": "running",
            "phase": None,
            "started": time.time(),
            "result": None,
            "error": None,
        }
    return job_id


def _set_phase(job_id: str, phase: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["phase"] = phase


def _finish_job(job_id: str, result: Any = None, error: Optional[str] = None) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = "error" if error else "done"
        job["result"] = result
        job["error"] = error
        job["elapsed"] = round(time.time() - job["started"], 1)


# ------------------------------- models -------------------------------- #
class ProjectCreate(BaseModel):
    name: str


class AskRequest(BaseModel):
    question: str
    session: str = "default"
    n_results: int = 6
    model: str = "7b"


# ------------------------------ endpoints ------------------------------ #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/projects")
def list_projects() -> List[Dict[str, Any]]:
    out = []
    if os.path.isdir(PROJECTS_DIR):
        for name in sorted(os.listdir(PROJECTS_DIR)):
            root = os.path.join(PROJECTS_DIR, name)
            if not os.path.isdir(root) or name.startswith("."):
                continue
            ws = ProjectWorkspace(name, base_dir=PROJECTS_DIR)
            manifest = ws.load_manifest()
            files = manifest.get("files", {})
            out.append({
                "name": name,
                "files": len(files),
                "chunks": sum(f.get("chunks", 0) for f in files.values()),
            })
    return out


@app.post("/api/projects")
def create_project(body: ProjectCreate) -> Dict[str, Any]:
    name = _safe_name(body.name)
    root = os.path.join(PROJECTS_DIR, name)
    if os.path.isdir(root):
        raise HTTPException(409, f"Project '{name}' already exists.")
    ProjectWorkspace(name, base_dir=PROJECTS_DIR)  # creates directories
    return {"name": name, "files": 0, "chunks": 0}


@app.get("/api/projects/{project}/files")
def project_files(project: str) -> List[Dict[str, Any]]:
    ws = _workspace(project)
    manifest = ws.load_manifest()
    out = []
    for path, rec in manifest.get("files", {}).items():
        out.append({
            "name": os.path.basename(path),
            "status": "indexed",
            "chunks": rec.get("chunks", 0),
            "ingested_at": rec.get("ingested_at"),
        })
    indexed_names = {f["name"] for f in out}
    for path in _pending_files(ws):
        name = os.path.basename(path)
        if name not in indexed_names:
            out.append({"name": name, "status": "pending", "chunks": 0,
                        "ingested_at": None})
    return sorted(out, key=lambda f: f["name"])


@app.get("/api/projects/{project}/sessions")
def project_sessions(project: str) -> List[Dict[str, Any]]:
    ws = _workspace(project)
    out = []
    for fname in sorted(os.listdir(ws.history_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ws.history_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            turns = data.get("turns", [])
            out.append({
                "name": fname[:-5],
                "turns": len(turns),
                "updated": turns[-1].get("timestamp") if turns else None,
                "preview": turns[0].get("query", "")[:80] if turns else "",
            })
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda s: s["updated"] or "", reverse=True)
    return out


@app.get("/api/projects/{project}/sessions/{session}")
def session_history(project: str, session: str) -> Dict[str, Any]:
    ws = _workspace(project)
    path = os.path.join(ws.history_dir, f"{_safe_name(session)}.json")
    if not os.path.exists(path):
        return {"project": project, "session": session, "turns": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.post("/api/projects/{project}/upload")
async def upload_documents(
    project: str, files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """Save uploads immediately; indexing is deferred to the next /ask."""
    ws = _workspace(project)
    uploads_dir = _uploads_dir(ws)

    saved: List[str] = []
    for up in files:
        fname = os.path.basename(up.filename or "")
        if not fname:
            continue
        dest = os.path.join(uploads_dir, fname)
        with open(dest, "wb") as fh:
            fh.write(await up.read())
        saved.append(fname)
    if not saved:
        raise HTTPException(400, "No files received.")
    return {"saved": saved}


@app.post("/api/projects/{project}/ask")
def ask(project: str, body: AskRequest) -> Dict[str, Any]:
    ws = _workspace(project)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Empty question.")
    session = _safe_name(body.session)
    model = resolve_model(body.model)
    job_id = _new_job("ask", ws.name)

    def work() -> None:
        try:
            with _project_lock(ws.name):
                store = _get_store(ws)

                # Deferred ingest: index any uploads added since the last ask.
                pending = _pending_files(ws)
                indexed_now = 0
                if pending:
                    _set_phase(job_id, "indexing")
                    for path in pending:
                        indexed_now += ingest_project(store, ws, path, 1000, 200)

                if store.collection.count() == 0:
                    _finish_job(job_id, error="Project has no documents yet. "
                                "Add a document first.")
                    return

                _set_phase(job_id, "answering")
                t0 = time.perf_counter()
                retrieved, confidence = store.retrieve_context(
                    question, n_results=body.n_results)
                retrieve_secs = time.perf_counter() - t0

                engine = DeterministicRAGEngine(model_name=model)
                t0 = time.perf_counter()
                answer = engine.generate_answer(question, retrieved)
                generate_secs = time.perf_counter() - t0

                turn = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "query": question,
                    "answer": answer,
                    "confidence": round(confidence, 4),
                    "model": model,
                    "n_results": body.n_results,
                    "timing_seconds": {
                        "retrieve": round(retrieve_secs, 3),
                        "generate": round(generate_secs, 3),
                        "total": round(retrieve_secs + generate_secs, 3),
                    },
                    "fragments": retrieved,
                }
                ws.append_turn(session, turn)
            turn["indexed_now"] = indexed_now
            _finish_job(job_id, turn)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI.
            _finish_job(job_id, error=str(exc))

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job.")
        return dict(job)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import webbrowser
    import uvicorn
    threading.Timer(1.5, webbrowser.open, args=("http://127.0.0.1:8400",)).start()
    uvicorn.run(app, host="127.0.0.1", port=8400)
