"""
OllamaAid - Public API
All public functions return ``ToolResult`` objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core.models import (
    DEFAULT_TEST_SCENARIOS,
    RunnerBackend,
    RunnerConfig,
    TestScenario,
    ToolResult,
)


# Re-export ToolResult so callers can ``from ollama_aid import ToolResult``
__all__ = [
    "ToolResult",
    "list_models",
    "export_model",
    "import_model",
    "delete_model",
    "update_model",
    "show_model_info",
    "fetch_trends",
    "test_model",
    "export_test_csv",
    "run_with_backend",
    "stop_backend",
    "resolve_model_path",
]


def list_models(*, ollama_path: Optional[str] = None) -> ToolResult:
    """List all locally downloaded Ollama models."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).list_models()


def export_model(
    model_name: str,
    export_dir: str,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Export a model to a directory (GGUF + Modelfile)."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).export_model(model_name, export_dir)


def import_model(
    gguf_path: str,
    new_name: str,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Import a GGUF file as a new Ollama model."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).import_model(gguf_path, new_name)


def delete_model(
    model_name: str,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Delete a model from Ollama."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).delete_model(model_name)


def update_model(
    model_name: str,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Pull the latest version of a model."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).update_model(model_name)


def show_model_info(
    model_name: str,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Return detailed information about a model."""
    from .core.manager import OllamaManager
    return OllamaManager(ollama_path).show_model_info(model_name)


def fetch_trends() -> ToolResult:
    """Scrape Ollama model trend data from ollama.com/search."""
    from .core.trends import fetch_trends as _fetch
    return _fetch()


def test_model(
    models: List[str],
    scenarios: Optional[List[TestScenario]] = None,
    *,
    ollama_path: Optional[str] = None,
) -> ToolResult:
    """Run benchmark tests on one or more models."""
    from .core.tester import run_tests
    return run_tests(models, scenarios, ollama_path=ollama_path)


def export_test_csv(results: list, path: str) -> ToolResult:
    """Export test results to CSV."""
    from .core.tester import export_results_csv
    return export_results_csv(results, path)


def resolve_model_path(model_name: str) -> ToolResult:
    """Resolve the on-disk path for an Ollama-managed model."""
    from .core.runner import ExternalRunner
    return ExternalRunner.resolve_model(model_name)


# Singleton runner instance
_runner = None


def _get_runner():
    global _runner
    if _runner is None:
        from .core.runner import ExternalRunner
        _runner = ExternalRunner()
    return _runner


def run_with_backend(
    model_name: str,
    backend: str = "llama.cpp",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    gpu_layers: int = -1,
    context_size: int = 4096,
    threads: int = 0,
    batch_size: int = 512,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: int = 0,
    extra_args: Optional[List[str]] = None,
) -> ToolResult:
    """Start an external inference server for an Ollama-managed model."""
    cfg = RunnerConfig(
        backend=RunnerBackend(backend),
        model_name=model_name,
        host=host,
        port=port,
        gpu_layers=gpu_layers,
        context_size=context_size,
        threads=threads,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        extra_args=extra_args or [],
    )
    return _get_runner().start(cfg)


def stop_backend() -> ToolResult:
    """Stop the running external inference server."""
    return _get_runner().stop()


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "List all locally downloaded Ollama models",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_model",
            "description": "Export an Ollama model to GGUF + Modelfile",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model name"},
                    "export_dir": {"type": "string", "description": "Export directory"},
                },
                "required": ["model_name", "export_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "import_model",
            "description": "Import a GGUF file as a new Ollama model",
            "parameters": {
                "type": "object",
                "properties": {
                    "gguf_path": {"type": "string", "description": "Path to GGUF file"},
                    "new_name": {"type": "string", "description": "New model name"},
                },
                "required": ["gguf_path", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_model",
            "description": "Delete an Ollama model",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model name to delete"},
                },
                "required": ["model_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_trends",
            "description": "Fetch Ollama model trends from ollama.com",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_with_backend",
            "description": "Start external inference server (vLLM/llama.cpp) for an Ollama model",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "backend": {"type": "string", "enum": ["vllm", "llama.cpp"]},
                    "port": {"type": "integer"},
                },
                "required": ["model_name"],
            },
        },
    },
]


def dispatch(name: str, kwargs: dict) -> dict:
    """Dispatch an OpenAI function-call by name. Returns a dict."""
    funcs = {
        "list_models": lambda **kw: list_models(**kw),
        "export_model": lambda **kw: export_model(**kw),
        "import_model": lambda **kw: import_model(**kw),
        "delete_model": lambda **kw: delete_model(**kw),
        "fetch_trends": lambda **kw: fetch_trends(),
        "run_with_backend": lambda **kw: run_with_backend(**kw),
        "stop_backend": lambda **kw: stop_backend(),
    }
    fn = funcs.get(name)
    if fn is None:
        return {"success": False, "error": f"Unknown function: {name}"}
    return fn(**kwargs).to_dict()
