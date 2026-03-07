"""
OllamaAid - Configuration helpers
Locate Ollama, detect platform, resolve model storage paths.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_ollama() -> Optional[str]:
    """Locate the Ollama executable on the system."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates: list[str] = []
    system = platform.system()
    if system == "Windows":
        candidates = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Ollama\ollama.exe"),
            r"C:\Program Files\Ollama\ollama.exe",
            r"C:\Program Files (x86)\Ollama\ollama.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/usr/local/bin/ollama",
            str(Path.home() / ".ollama" / "ollama"),
            "/opt/homebrew/bin/ollama",
        ]
    else:
        candidates = [
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            str(Path.home() / ".ollama" / "ollama"),
            "/snap/bin/ollama",
        ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def get_ollama_models_dir() -> Path:
    """Return the directory where Ollama stores downloaded models (blobs)."""
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        return Path(env_dir)
    system = platform.system()
    if system == "Windows":
        return Path(os.path.expandvars(r"%USERPROFILE%\.ollama\models"))
    elif system == "Darwin":
        return Path.home() / ".ollama" / "models"
    else:
        return Path.home() / ".ollama" / "models"


def find_backend(backend: str) -> Optional[str]:
    """Locate vLLM or llama.cpp server executable."""
    if backend == "vllm":
        exe = shutil.which("vllm")
        if exe:
            return exe
        # vllm is a Python package, check importability
        try:
            result = subprocess.run(
                ["python", "-m", "vllm.entrypoints.openai.api_server", "--help"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return "python -m vllm.entrypoints.openai.api_server"
        except Exception:
            pass
        return None

    if backend in ("llama.cpp", "llama-cpp", "llamacpp"):
        for name in ("llama-server", "llama-cpp-server", "server"):
            exe = shutil.which(name)
            if exe:
                return exe
        # Common build paths
        candidates = []
        system = platform.system()
        if system == "Windows":
            candidates = [
                r"C:\llama.cpp\build\bin\Release\llama-server.exe",
                r"C:\llama.cpp\build\bin\llama-server.exe",
                os.path.expandvars(r"%USERPROFILE%\llama.cpp\build\bin\Release\llama-server.exe"),
            ]
        else:
            candidates = [
                "/usr/local/bin/llama-server",
                str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"),
                str(Path.home() / "llama.cpp" / "llama-server"),
            ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    return None


def resolve_model_path(model_name: str) -> Optional[Path]:
    """Resolve the GGUF / blob path for an Ollama-managed model.

    Ollama stores models as blobs under ``<models_dir>/blobs/`` with
    sha256 digest names.  We run ``ollama show --modelfile <model>`` to
    extract the ``FROM`` path which points to the actual file.
    """
    ollama = find_ollama()
    if not ollama:
        return None
    try:
        result = subprocess.run(
            [ollama, "show", "--modelfile", model_name],
            capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.upper().startswith("FROM "):
                path_str = line[5:].strip()
                p = Path(path_str)
                if p.exists():
                    return p
    except Exception:
        pass
    return None


def is_ollama_running() -> bool:
    """Check whether the Ollama service is reachable."""
    try:
        import requests
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
