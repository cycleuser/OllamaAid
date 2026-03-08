"""
OllamaAid - Ollama model management
List, export, import, delete, update models via the Ollama CLI.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from .config import find_ollama, resolve_model_path
from .models import MODELFILE_TEMPLATES, ModelInfo, ToolResult

log = logging.getLogger(__name__)


class OllamaManager:
    """High-level wrapper around the Ollama CLI for model management."""

    def __init__(self, ollama_path: Optional[str] = None):
        self.ollama_path = ollama_path or find_ollama()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run(self, args: list, timeout: int = 30) -> subprocess.CompletedProcess:
        if not self.ollama_path:
            raise RuntimeError("Ollama executable not found")
        cmd = [self.ollama_path] + args
        return subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", timeout=timeout,
        )

    @staticmethod
    def _parse_size_to_bytes(size_str: str) -> int:
        """Convert human-readable size like '3.2 GB' to bytes."""
        size_str = size_str.strip().upper()
        m = re.match(r"([\d.]+)\s*(GB|MB|KB|TB|B)", size_str)
        if not m:
            return 0
        val = float(m.group(1))
        unit = m.group(2)
        multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        return int(val * multiplier.get(unit, 1))

    # ------------------------------------------------------------------
    # Model CRUD
    # ------------------------------------------------------------------

    def list_models(self) -> ToolResult:
        """List all locally downloaded Ollama models."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        try:
            result = self._run(["list"], timeout=15)
            if result.returncode != 0:
                return ToolResult(success=False, error=result.stderr.strip())

            models: List[ModelInfo] = []
            lines = result.stdout.strip().splitlines()
            if not lines:
                return ToolResult(success=True, data=[])

            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) < 4:
                    continue
                full_name = parts[0]
                name, tag = (full_name.split(":", 1) + ["latest"])[:2]
                model_id = parts[1] if len(parts) > 1 else ""
                # Size may be two tokens like "3.2 GB"
                size_str = ""
                date_str = ""
                idx = 2
                if idx < len(parts):
                    size_str = parts[idx]
                    if idx + 1 < len(parts) and parts[idx + 1] in ("B", "KB", "MB", "GB", "TB"):
                        size_str += " " + parts[idx + 1]
                        idx += 2
                    else:
                        idx += 1
                date_str = " ".join(parts[idx:])

                models.append(ModelInfo(
                    name=name,
                    tag=tag,
                    model_id=model_id,
                    full_name=full_name,
                    size=size_str,
                    size_bytes=self._parse_size_to_bytes(size_str),
                    modified_date=date_str,
                ))
            return ToolResult(success=True, data=models, metadata={"count": len(models)})
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Ollama list timed out")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def export_model(self, model_name: str, export_dir: str) -> ToolResult:
        """Export a model to a directory as GGUF + Modelfile."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        try:
            # Get modelfile to find the FROM path
            result = self._run(["show", "--modelfile", model_name], timeout=15)
            if result.returncode != 0:
                return ToolResult(success=False, error=f"Cannot get modelfile: {result.stderr.strip()}")

            modelfile_content = result.stdout
            model_path: Optional[str] = None
            for line in modelfile_content.splitlines():
                line = line.strip()
                if line.upper().startswith("FROM "):
                    model_path = line[5:].strip()
                    break

            if not model_path or not Path(model_path).exists():
                return ToolResult(success=False, error="Cannot locate model file on disk")

            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            safe_name = model_name.replace(":", "_").replace("/", "_")

            # Copy model file
            src = Path(model_path)
            dst = export_path / f"{safe_name}.gguf"
            shutil.copy2(str(src), str(dst))

            # Save modelfile
            mf_dst = export_path / f"{safe_name}.Modelfile"
            mf_dst.write_text(modelfile_content, encoding="utf-8")

            return ToolResult(
                success=True,
                data={"model_file": str(dst), "modelfile": str(mf_dst)},
                metadata={"model": model_name},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def import_model(self, gguf_path: str, new_name: str) -> ToolResult:
        """Import a GGUF file as a new Ollama model."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        if not Path(gguf_path).is_file():
            return ToolResult(success=False, error=f"File not found: {gguf_path}")
        if not re.match(r"^[a-zA-Z0-9_-]+(?::[a-zA-Z0-9_.-]+)?$", new_name):
            return ToolResult(success=False, error=f"Invalid model name: {new_name}")
        try:
            # Determine template
            template_key = "default"
            name_lower = new_name.lower()
            for key in MODELFILE_TEMPLATES:
                if key in name_lower:
                    template_key = key
                    break
            content = MODELFILE_TEMPLATES[template_key].format(model_path=gguf_path)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                result = self._run(["create", new_name, "-f", tmp_path], timeout=300)
            finally:
                os.unlink(tmp_path)

            if result.returncode != 0:
                return ToolResult(success=False, error=result.stderr.strip())
            return ToolResult(success=True, data={"model": new_name})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def delete_model(self, model_name: str) -> ToolResult:
        """Delete a model from Ollama."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        try:
            result = self._run(["rm", model_name], timeout=30)
            if result.returncode != 0:
                return ToolResult(success=False, error=result.stderr.strip())
            return ToolResult(success=True, data={"deleted": model_name})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def update_model(self, model_name: str) -> ToolResult:
        """Pull the latest version of a model."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        try:
            result = self._run(["pull", model_name], timeout=600)
            if result.returncode != 0:
                return ToolResult(success=False, error=result.stderr.strip())
            return ToolResult(success=True, data={"updated": model_name})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def show_model_info(self, model_name: str) -> ToolResult:
        """Return detailed information about a model."""
        if not self.ollama_path:
            return ToolResult(success=False, error="Ollama executable not found")
        try:
            result = self._run(["show", model_name], timeout=15)
            if result.returncode != 0:
                return ToolResult(success=False, error=result.stderr.strip())
            info: dict = {"raw": result.stdout}
            section = ""
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Section headers: 2-space indent, single word/phrase, no value
                # e.g. "  Model", "  Capabilities", "  Parameters", "  License"
                if line.startswith("  ") and not line.startswith("    "):
                    section = stripped.lower()
                    continue
                # Key-value pairs: 4+ space indent, key and value separated by 2+ spaces
                # e.g. "    architecture        qwen3"
                if line.startswith("    "):
                    parts = re.split(r"\s{2,}", stripped, maxsplit=1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(" ", "_")
                        val = parts[1].strip()
                        # Accumulate duplicate keys (e.g. multiple "stop" values)
                        if key in info:
                            existing = info[key]
                            if isinstance(existing, list):
                                existing.append(val)
                            else:
                                info[key] = [existing, val]
                        else:
                            info[key] = val
                    elif len(parts) == 1 and section == "capabilities":
                        # Capability entries have no value, just the name
                        caps = info.get("capabilities", [])
                        if isinstance(caps, str):
                            caps = [caps]
                        caps.append(stripped)
                        info["capabilities"] = caps
            # Convert lists to comma-separated strings for display
            for key, val in info.items():
                if isinstance(val, list):
                    info[key] = ", ".join(val)
            return ToolResult(success=True, data=info, metadata={"model": model_name})
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
