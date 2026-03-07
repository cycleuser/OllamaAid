"""
OllamaAid - Unified Ollama Model Management, Trends & Testing Tool
"""

from .__version__ import __version__, __app_name__, __app_name_cn__
from .api import (
    ToolResult,
    list_models,
    export_model,
    import_model,
    delete_model,
    update_model,
    fetch_trends,
    test_model,
    run_with_backend,
)

__all__ = [
    "__version__",
    "__app_name__",
    "__app_name_cn__",
    "ToolResult",
    "list_models",
    "export_model",
    "import_model",
    "delete_model",
    "update_model",
    "fetch_trends",
    "test_model",
    "run_with_backend",
]
