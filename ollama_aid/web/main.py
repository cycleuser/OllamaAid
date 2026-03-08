"""
OllamaAid - Web interface (Flask)
REST API + simple HTML dashboard for all features.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request

from ollama_aid.__version__ import __version__
from ollama_aid.core.i18n import I18n
from ollama_aid.core.models import RunnerBackend, RunnerConfig
from ollama_aid.core.runner import ExternalRunner

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
)
i18n = I18n("en")
_runner = ExternalRunner()


# ======================================================================
# API routes
# ======================================================================

@app.route("/")
def index():
    return render_template("index.html", version=__version__)


@app.route("/api/language", methods=["POST"])
def set_language():
    lang = request.json.get("language", "en") if request.is_json else "en"
    i18n.set_language(lang)
    return jsonify({"success": True, "language": i18n.language})


@app.route("/api/models")
def api_list_models():
    from ollama_aid.core.manager import OllamaManager
    result = OllamaManager().list_models()
    if not result.success:
        return jsonify({"success": False, "error": result.error}), 500
    models = [
        {
            "name": m.name, "tag": m.tag, "full_name": m.full_name,
            "id": m.model_id, "size": m.size, "size_bytes": m.size_bytes,
            "modified": m.modified_date,
        }
        for m in (result.data or [])
    ]
    return jsonify({"success": True, "models": models})


@app.route("/api/models/<path:model_name>/info")
def api_model_info(model_name: str):
    from ollama_aid.core.manager import OllamaManager
    result = OllamaManager().show_model_info(model_name)
    return jsonify(result.to_dict())


@app.route("/api/models/<path:model_name>/delete", methods=["POST"])
def api_delete_model(model_name: str):
    from ollama_aid.core.manager import OllamaManager
    result = OllamaManager().delete_model(model_name)
    return jsonify(result.to_dict())


@app.route("/api/models/<path:model_name>/update", methods=["POST"])
def api_update_model(model_name: str):
    from ollama_aid.core.manager import OllamaManager
    result = OllamaManager().update_model(model_name)
    return jsonify(result.to_dict())


@app.route("/api/models/<path:model_name>/resolve")
def api_resolve_model(model_name: str):
    result = ExternalRunner.resolve_model(model_name)
    return jsonify(result.to_dict())


@app.route("/api/trends")
def api_trends():
    from ollama_aid.core.trends import fetch_trends
    result = fetch_trends()
    if not result.success:
        return jsonify({"success": False, "error": result.error}), 500
    data = [t.to_dict() for t in (result.data or [])]
    return jsonify({"success": True, "models": data})


@app.route("/api/trends/download", methods=["POST"])
def api_trends_download():
    body = request.get_json(force=True) if request.is_json else {}
    model_name = body.get("model_name", "").strip()
    tag = body.get("tag", "").strip()
    if not model_name or not tag:
        return jsonify({"success": False, "error": "model_name and tag are required"}), 400
    full_name = f"{model_name}:{tag}"
    from ollama_aid.core.manager import OllamaManager
    result = OllamaManager().update_model(full_name)
    return jsonify(result.to_dict())


@app.route("/api/test", methods=["POST"])
def api_test():
    body = request.get_json(force=True) if request.is_json else {}
    models = body.get("models", [])
    if not models:
        return jsonify({"success": False, "error": "No models specified"}), 400
    from ollama_aid.core.tester import run_tests
    result = run_tests(models)
    if not result.success:
        return jsonify({"success": False, "error": result.error}), 500
    data = [r.to_dict() for r in (result.data or [])]
    return jsonify({"success": True, "results": data})


@app.route("/api/runner/start", methods=["POST"])
def api_runner_start():
    body = request.get_json(force=True) if request.is_json else {}
    model_name = body.get("model_name", "")
    if not model_name:
        return jsonify({"success": False, "error": "model_name required"}), 400
    backend_str = body.get("backend", "llama.cpp")
    backend = RunnerBackend.LLAMA_CPP if "llama" in backend_str.lower() else RunnerBackend.VLLM
    cfg = RunnerConfig(
        backend=backend,
        model_name=model_name,
        host=body.get("host", "127.0.0.1"),
        port=int(body.get("port", 8080)),
        gpu_layers=int(body.get("gpu_layers", -1)),
        context_size=int(body.get("context_size", 4096)),
        threads=int(body.get("threads", 0)),
        batch_size=int(body.get("batch_size", 512)),
        tensor_parallel_size=int(body.get("tensor_parallel_size", 1)),
        dtype=body.get("dtype", "auto"),
        max_model_len=int(body.get("max_model_len", 0)),
        extra_args=body.get("extra_args", []),
    )
    result = _runner.start(cfg)
    return jsonify(result.to_dict())


@app.route("/api/runner/stop", methods=["POST"])
def api_runner_stop():
    result = _runner.stop()
    return jsonify(result.to_dict())


@app.route("/api/runner/status")
def api_runner_status():
    result = _runner.status()
    return jsonify(result.to_dict())


# ======================================================================
# Entry point
# ======================================================================

def main(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main(debug=True)
