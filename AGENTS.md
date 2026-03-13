# AGENTS.md

Guidelines for AI coding agents working on the OllamaAid codebase.

## Project Overview

OllamaAid is a unified Ollama model management, trends viewing, and testing tool. It provides CLI, GUI (PySide6), and Web (Flask) interfaces for managing Ollama models, viewing model trends from ollama.com, benchmarking models, and running models via external backends (vLLM/llama.cpp).

## Build / Install / Test Commands

```bash
# Install in development mode with all dependencies
pip install -e ".[all,test]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_core.py -v

# Run a single test class
python -m pytest tests/test_core.py::TestToolResult -v

# Run a single test case
python -m pytest tests/test_core.py::TestToolResult::test_success -v

# Run tests with coverage
python -m pytest tests/ -v --cov=ollama_aid

# Install the package
pip install -e .

# Build for PyPI
python -m build
```

## Code Style Guidelines

### Imports

```python
# Standard library imports first (alphabetically)
from __future__ import annotations
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports second (alphabetically)
import requests

# Local imports last (use relative imports for internal modules)
from .config import find_ollama
from .models import ModelInfo, ToolResult
```

### Type Hints

- Use `from __future__ import annotations` for modern type hint syntax
- Always annotate function parameters and return types
- Use `Optional[T]` for nullable types, not `T | None`
- Use `List[T]`, `Dict[K, V]` instead of `list[T]`, `dict[K, V]` for compatibility

```python
def list_models(*, ollama_path: Optional[str] = None) -> ToolResult:
    ...

def _run(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
    ...
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Module-level logger**: `log = logging.getLogger(__name__)`
- **Enum classes**: `PascalCase`, values are `lower_case`

```python
# Constants
OLLAMA_API_BASE = "http://127.0.0.1:11434"
DEFAULT_TEST_SCENARIOS: List[TestScenario] = [...]

# Enum
class RunnerBackend(str, Enum):
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"

# Private method
def _parse_size_to_bytes(size_str: str) -> int:
    ...
```

### Data Classes

Use `@dataclass` for data models with `to_dict()` method:

```python
@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"success": self.success}
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d
```

### Error Handling

Use the `ToolResult` pattern for all public API functions:

```python
def list_models(*, ollama_path: Optional[str] = None) -> ToolResult:
    if not self.ollama_path:
        return ToolResult(success=False, error="Ollama executable not found")
    try:
        # ... operation ...
        return ToolResult(success=True, data=models, metadata={"count": len(models)})
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="Ollama list timed out")
    except Exception as exc:
        return ToolResult(success=False, error=str(exc))
```

### Logging

```python
import logging
log = logging.getLogger(__name__)

log.info("pull_model('%s') called", model_name)
log.warning("Ollama service not reachable at %s", OLLAMA_API_BASE)
log.debug("run_with_backend(model=%s, backend=%s)", args.model, args.backend)
```

### Docstrings

Use triple-quoted docstrings at module and function level:

```python
"""
OllamaAid - Ollama model management
List, export, import, delete, update models via the Ollama CLI and HTTP API.
"""

def list_models(self) -> ToolResult:
    """List all locally downloaded Ollama models."""
    ...
```

### CLI Command Pattern

Each CLI command is a separate function handling argparse args:

```python
def cmd_list(args):
    from ollama_aid.api import list_models
    result = list_models()
    if args.json_output:
        _json_out({"success": result.success, "models": data})
        return
    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)
    # ... display results ...
```

## Project Structure

```
ollama_aid/
├── __init__.py          # Public exports + __all__
├── __main__.py          # Entry point for python -m ollama_aid
├── __version__.py       # Version string
├── api.py               # Public API + TOOLS + dispatch()
├── core/
│   ├── config.py        # Find Ollama/vLLM/llama.cpp paths
│   ├── i18n.py          # Bilingual EN/ZH translation
│   ├── manager.py       # OllamaManager class
│   ├── models.py        # Data classes, ToolResult
│   ├── runner.py        # ExternalRunner for vLLM/llama.cpp
│   ├── tester.py        # Benchmark runner
│   └── trends.py        # Web scraping ollama.com
├── cli/main.py          # argparse CLI (10 subcommands)
├── gui/main.py          # PySide6 tabbed GUI
└── web/
    ├── main.py          # Flask REST API
    └── templates/       # HTML templates
tests/
└── test_core.py         # Test suite
```

## Testing Guidelines

- Tests use `pytest` framework
- Test classes follow `Test<ClassName>` naming
- Test methods follow `test_<description>` naming
- Group related tests in classes:

```python
class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, data={"key": "val"})
        assert r.success is True
        assert r.data == {"key": "val"}

    def test_failure(self):
        r = ToolResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"
```

## Key Patterns

1. **ToolResult everywhere**: All public functions return `ToolResult` objects
2. **Lazy imports**: Import dependencies inside functions to avoid import errors
3. **Keyword-only arguments**: Use `*` to enforce keyword-only args in public APIs
4. **Backward-compatible aliases**: `TestScenario = BenchScenario` for renaming
5. **Environment isolation**: GUI launches in subprocess to avoid library conflicts

## Dependencies

- **Core**: `requests`, `beautifulsoup4`, `lxml`
- **GUI (optional)**: `PySide6`
- **Web (optional)**: `flask`
- **Tester (optional)**: `pandas`, `matplotlib`, `numpy`
- **Test**: `pytest`