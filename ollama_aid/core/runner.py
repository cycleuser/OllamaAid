"""
OllamaAid - External runner integration
Launch Ollama-managed models via vLLM or llama.cpp for better performance.
"""

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from .config import find_backend, find_ollama, get_ollama_models_dir, resolve_model_path
from .models import RunnerBackend, RunnerConfig, ToolResult

log = logging.getLogger(__name__)


class ExternalRunner:
    """Manage an external inference server (vLLM or llama.cpp)."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    # ------------------------------------------------------------------
    # Model path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_model(model_name: str) -> ToolResult:
        """Resolve an Ollama model name to its on-disk path and metadata.

        Returns a ``ToolResult`` whose ``data`` dict contains:
        * ``model_path`` - absolute path to the weights file
        * ``model_name`` - the original model name
        * ``info``       - extra metadata from ``ollama show``
        """
        path = resolve_model_path(model_name)
        if path is None:
            return ToolResult(
                success=False,
                error=f"Cannot resolve model path for '{model_name}'. "
                      "Is Ollama installed and has the model been pulled?",
            )
        # Gather extra info
        info: dict = {"model_name": model_name, "model_path": str(path)}
        ollama = find_ollama()
        if ollama:
            try:
                r = subprocess.run(
                    [ollama, "show", model_name],
                    capture_output=True, text=True, encoding="utf-8", timeout=10,
                )
                if r.returncode == 0:
                    for line in r.stdout.splitlines():
                        line = line.strip()
                        if ":" in line:
                            k, v = line.split(":", 1)
                            info[k.strip().lower().replace(" ", "_")] = v.strip()
            except Exception:
                pass
        return ToolResult(success=True, data=info)

    # ------------------------------------------------------------------
    # Build command line
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llamacpp_cmd(cfg: RunnerConfig, backend_exe: str) -> List[str]:
        cmd = [backend_exe]
        cmd += ["-m", cfg.model_path]
        cmd += ["--host", cfg.host]
        cmd += ["--port", str(cfg.port)]
        if cfg.gpu_layers != 0:
            cmd += ["-ngl", str(cfg.gpu_layers)]
        if cfg.context_size > 0:
            cmd += ["-c", str(cfg.context_size)]
        if cfg.threads > 0:
            cmd += ["-t", str(cfg.threads)]
        if cfg.batch_size > 0:
            cmd += ["-b", str(cfg.batch_size)]
        cmd += cfg.extra_args
        return cmd

    @staticmethod
    def _build_vllm_cmd(cfg: RunnerConfig, backend_exe: str) -> List[str]:
        # vllm can be invoked as ``python -m vllm.entrypoints.openai.api_server``
        if backend_exe.startswith("python"):
            cmd = backend_exe.split()
        else:
            cmd = [backend_exe, "serve"]
        cmd += [cfg.model_path]
        cmd += ["--host", cfg.host]
        cmd += ["--port", str(cfg.port)]
        if cfg.tensor_parallel_size > 1:
            cmd += ["--tensor-parallel-size", str(cfg.tensor_parallel_size)]
        if cfg.dtype and cfg.dtype != "auto":
            cmd += ["--dtype", cfg.dtype]
        if cfg.max_model_len > 0:
            cmd += ["--max-model-len", str(cfg.max_model_len)]
        if cfg.gpu_layers >= 0:
            cmd += ["--gpu-memory-utilization", "0.9"]
        cmd += cfg.extra_args
        return cmd

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(
        self,
        config: RunnerConfig,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> ToolResult:
        """Start the external inference server.

        Parameters
        ----------
        config:
            ``RunnerConfig`` with backend, model_path, host, port, etc.
        log_cb:
            Optional callback ``f(line)`` for real-time log output.
        """
        if self.is_running:
            return ToolResult(success=False, error="Server is already running")

        log.info("Runner start requested: backend=%s model=%s",
                 config.backend.value, config.model_name or config.model_path)

        # Resolve model path from Ollama if not absolute
        if not config.model_path or not Path(config.model_path).exists():
            if config.model_name:
                log.info("Resolving model '%s' to disk path...", config.model_name)
                res = self.resolve_model(config.model_name)
                if not res.success:
                    log.error("Model resolve failed: %s", res.error)
                    return res
                config.model_path = res.data["model_path"]
                log.info("Resolved model path: %s", config.model_path)
            else:
                return ToolResult(success=False, error="No model_path or model_name specified")

        backend_name = config.backend.value
        log.info("Looking for backend '%s'...", backend_name)
        backend_exe = find_backend(backend_name)
        if not backend_exe:
            if backend_name in ("llama.cpp", "llama-cpp", "llamacpp"):
                searched = "llama-server, llama-cli, llama-cpp-server, server"
            elif backend_name == "vllm":
                searched = "vllm, python -m vllm.entrypoints.openai.api_server"
            else:
                searched = backend_name
            log.error("Backend not found. Searched: %s", searched)
            return ToolResult(
                success=False,
                error=f"Backend '{backend_name}' not found. "
                      f"Searched for: {searched}. "
                      f"Please install it and ensure the binary is on your PATH.",
            )
        log.info("Using backend executable: %s", backend_exe)

        if config.backend == RunnerBackend.LLAMA_CPP:
            cmd = self._build_llamacpp_cmd(config, backend_exe)
        else:
            cmd = self._build_vllm_cmd(config, backend_exe)

        if log_cb:
            log_cb(f"[runner] Starting: {' '.join(cmd)}")

        try:
            creation_flags = 0
            if sys.platform == "win32":
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                creationflags=creation_flags,
            )
            self._running = True

            # Background thread to relay logs
            collected_lines: list[str] = []

            def _reader():
                assert self._process is not None and self._process.stdout is not None
                for line in self._process.stdout:
                    line = line.rstrip()
                    collected_lines.append(line)
                    if log_cb:
                        log_cb(line)
                    if not self._running:
                        break

            self._log_thread = threading.Thread(target=_reader, daemon=True)
            self._log_thread.start()

            # Brief grace period to detect immediate crashes (e.g. model
            # load failures).  If the process exits within a few seconds it
            # almost certainly failed to start.  We poll several times to
            # catch both fast failures and slower GPU-init failures.
            for _ in range(10):
                time.sleep(0.5)
                if self._process.poll() is not None:
                    break
            if self._process.poll() is not None:
                rc = self._process.returncode
                self._log_thread.join(timeout=2)
                # Collect last few log lines for error context
                tail = "\n".join(collected_lines[-10:]) if collected_lines else "(no output)"
                self._process = None
                self._running = False
                return ToolResult(
                    success=False,
                    error=f"Backend process exited immediately (code {rc}).\n{tail}",
                )

            return ToolResult(
                success=True,
                data={
                    "pid": self._process.pid,
                    "cmd": " ".join(cmd),
                    "host": config.host,
                    "port": config.port,
                },
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def stop(self) -> ToolResult:
        """Stop the external inference server."""
        if not self.is_running:
            return ToolResult(success=True, data="Server is not running")
        self._running = False
        try:
            if sys.platform == "win32":
                self._process.terminate()  # type: ignore[union-attr]
            else:
                os.kill(self._process.pid, signal.SIGTERM)  # type: ignore[union-attr]
            self._process.wait(timeout=10)  # type: ignore[union-attr]
        except Exception:
            try:
                self._process.kill()  # type: ignore[union-attr]
            except Exception:
                pass
        self._process = None
        return ToolResult(success=True, data="Server stopped")

    def status(self) -> ToolResult:
        """Return current server status."""
        if self.is_running:
            return ToolResult(
                success=True,
                data={"running": True, "pid": self._process.pid},  # type: ignore[union-attr]
            )
        return ToolResult(success=True, data={"running": False})
