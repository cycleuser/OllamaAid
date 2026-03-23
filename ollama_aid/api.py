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
    "benchmark_embedding",
    "benchmark_embeddings",
    "benchmark_reranker",
    "benchmark_rerankers",
    "benchmark_translation",
    "benchmark_translations",
    "benchmark_tool_call",
    "benchmark_tool_calls",
    "benchmark_code_generation",
    "benchmark_code_generations",
    "benchmark_chat",
    "benchmark_chats",
    "detect_model_type",
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


def fetch_trends(*, limit: int = 100) -> ToolResult:
    """Scrape Ollama model trend data from ollama.com/search.
    
    Args:
        limit: Maximum number of models to return (default: 100, use -1 for all)
    """
    from .core.trends import fetch_trends as _fetch
    return _fetch(limit=limit)


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
    log_cb=None,
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
    return _get_runner().start(cfg, log_cb=log_cb)


def stop_backend() -> ToolResult:
    """Stop the running external inference server."""
    return _get_runner().stop()


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def detect_model_type(model_name: str) -> str:
    """Detect model type from its name.
    
    Returns one of: embedding, reranker, code, chat, vision, thinking, unknown
    """
    from .core.benchmark_data import detect_model_type as _detect
    return _detect(model_name)


# ---------------------------------------------------------------------------
# Embedding model benchmark
# ---------------------------------------------------------------------------

def benchmark_embedding(
    model: str,
    language: str = "both",
) -> ToolResult:
    """Benchmark a single embedding model.
    
    Parameters
    ----------
    model: Model name (e.g., "bge-large:335m")
    language: "en", "zh", or "both"
    
    Returns
    -------
    ToolResult with EmbeddingBenchResult
    
    References
    ----------
    - MTEB: Muennighoff et al. (2023). https://arxiv.org/abs/2210.07316
    - STS Benchmark: Cer et al. (2017). https://arxiv.org/abs/1708.00055
    - BEIR: Thakur et al. (2021). https://arxiv.org/abs/2104.08663
    """
    from .core.embedder import benchmark_embedding as _bench
    result = _bench(model, language=language)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_embeddings(
    models: List[str],
    language: str = "both",
    progress_cb=None,
) -> ToolResult:
    """Benchmark multiple embedding models.
    
    Parameters
    ----------
    models: List of model names
    language: "en", "zh", or "both"
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of EmbeddingBenchResult
    """
    from .core.embedder import benchmark_embeddings as _bench
    return _bench(models, language=language, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Reranker model benchmark
# ---------------------------------------------------------------------------

def benchmark_reranker(model: str) -> ToolResult:
    """Benchmark a single reranker model.
    
    Parameters
    ----------
    model: Model name (e.g., "dengcao/Qwen3-Reranker-4B:Q8_0")
    
    Returns
    -------
    ToolResult with RerankerBenchResult
    
    References
    ----------
    - BEIR: Thakur et al. (2021). https://arxiv.org/abs/2104.08663
    - MS MARCO: Nguyen et al. (2016). https://arxiv.org/abs/1611.09268
    """
    from .core.reranker import benchmark_reranker as _bench
    result = _bench(model)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_rerankers(models: List[str], progress_cb=None) -> ToolResult:
    """Benchmark multiple reranker models.
    
    Parameters
    ----------
    models: List of model names
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of RerankerBenchResult
    """
    from .core.reranker import benchmark_rerankers as _bench
    return _bench(models, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Translation model benchmark
# ---------------------------------------------------------------------------

def benchmark_translation(model: str) -> ToolResult:
    """Benchmark a single translation model.
    
    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    
    Returns
    -------
    ToolResult with TranslationBenchResult
    """
    from .core.translator import benchmark_translation as _bench
    result = _bench(model)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_translations(models: List[str], progress_cb=None) -> ToolResult:
    """Benchmark multiple translation models.
    
    Parameters
    ----------
    models: List of model names
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of TranslationBenchResult
    """
    from .core.translator import benchmark_translations as _bench
    return _bench(models, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Tool calling benchmark
# ---------------------------------------------------------------------------

def benchmark_tool_call(model: str) -> ToolResult:
    """Benchmark tool calling capability of a model.
    
    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    
    Returns
    -------
    ToolResult with ToolCallBenchResult
    """
    from .core.tool_caller import benchmark_tool_call as _bench
    result = _bench(model)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_tool_calls(models: List[str], progress_cb=None) -> ToolResult:
    """Benchmark multiple models on tool calling.
    
    Parameters
    ----------
    models: List of model names
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of ToolCallBenchResult
    """
    from .core.tool_caller import benchmark_tool_calls as _bench
    return _bench(models, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Code generation benchmark
# ---------------------------------------------------------------------------

def benchmark_code_generation(model: str) -> ToolResult:
    """Benchmark code generation capability of a model.
    
    Parameters
    ----------
    model: Model name (e.g., "codellama:7b")
    
    Returns
    -------
    ToolResult with CodeGenBenchResult
    """
    from .core.code_eval import benchmark_code_generation as _bench
    result = _bench(model)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_code_generations(models: List[str], progress_cb=None) -> ToolResult:
    """Benchmark multiple models on code generation.
    
    Parameters
    ----------
    models: List of model names
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of CodeGenBenchResult
    """
    from .core.code_eval import benchmark_code_generations as _bench
    return _bench(models, progress_cb=progress_cb)


# ---------------------------------------------------------------------------
# Chat benchmark
# ---------------------------------------------------------------------------

def benchmark_chat(model: str) -> ToolResult:
    """Benchmark chat/completion capability of a model.
    
    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    
    Returns
    -------
    ToolResult with ChatBenchResult
    """
    from .core.chat_eval import benchmark_chat as _bench
    result = _bench(model)
    return ToolResult(
        success=result.success,
        data=result.to_dict(),
        error=result.error,
    )


def benchmark_chats(models: List[str], progress_cb=None) -> ToolResult:
    """Benchmark multiple models on chat/completion.
    
    Parameters
    ----------
    models: List of model names
    progress_cb: Optional progress callback
    
    Returns
    -------
    ToolResult with list of ChatBenchResult
    """
    from .core.chat_eval import benchmark_chats as _bench
    return _bench(models, progress_cb=progress_cb)


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
