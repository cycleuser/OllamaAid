"""
OllamaAid - Model tester
Run benchmarks against one or more Ollama models.
"""

import csv
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional

from .config import find_ollama
from .models import (
    DEFAULT_TEST_SCENARIOS,
    TestMetrics,
    TestResult,
    TestScenario,
    ToolResult,
)

log = logging.getLogger(__name__)


def _parse_duration(text: str) -> float:
    """Convert duration strings like ``2.5s``, ``1m30s``, ``500ms`` to seconds."""
    if not text:
        return 0.0
    text = text.strip().lower()
    total = 0.0
    for m in re.finditer(r"([\d.]+)\s*(h|m(?!s)|s|ms|µs|us)", text):
        val = float(m.group(1))
        unit = m.group(2)
        if unit == "h":
            total += val * 3600
        elif unit == "m":
            total += val * 60
        elif unit == "s":
            total += val
        elif unit == "ms":
            total += val / 1000
        elif unit in ("µs", "us"):
            total += val / 1_000_000
    if total == 0.0:
        try:
            total = float(text.rstrip("s"))
        except ValueError:
            pass
    return total


def _run_ollama_verbose(
    ollama_path: str, model: str, prompt: str, timeout: int = 300,
) -> tuple[str, TestMetrics]:
    """Run ``ollama run --verbose`` and parse metrics from stderr."""
    metrics = TestMetrics()
    try:
        result = subprocess.run(
            [ollama_path, "run", "--verbose", model, prompt],
            capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )
        response = result.stdout.strip()
        stderr = result.stderr

        # Parse metrics from verbose output
        patterns = {
            "total_duration_sec": r"total\s+duration[:\s]+([\d.]+\S*)",
            "load_duration_sec": r"load\s+duration[:\s]+([\d.]+\S*)",
            "prompt_eval_duration_sec": r"prompt\s+eval\s+duration[:\s]+([\d.]+\S*)",
            "eval_duration_sec": r"eval\s+duration[:\s]+([\d.]+\S*)",
        }
        for attr, pat in patterns.items():
            m = re.search(pat, stderr, re.IGNORECASE)
            if m:
                setattr(metrics, attr, _parse_duration(m.group(1)))

        count_patterns = {
            "prompt_tokens": r"prompt\s+eval\s+count[:\s]+(\d+)",
            "completion_tokens": r"eval\s+count[:\s]+(\d+)",
        }
        for attr, pat in count_patterns.items():
            m = re.search(pat, stderr, re.IGNORECASE)
            if m:
                setattr(metrics, attr, int(m.group(1)))

        rate_patterns = {
            "prompt_eval_rate_tps": r"prompt\s+eval\s+rate[:\s]+([\d.]+)",
            "eval_rate_tps": r"eval\s+rate[:\s]+([\d.]+)",
        }
        for attr, pat in rate_patterns.items():
            m = re.search(pat, stderr, re.IGNORECASE)
            if m:
                setattr(metrics, attr, float(m.group(1)))

        return response, metrics
    except subprocess.TimeoutExpired:
        return "", metrics
    except Exception as exc:
        log.warning("Error running ollama verbose: %s", exc)
        return "", metrics


def _get_self_evaluation(
    ollama_path: str, model: str, scenario: str, response: str,
) -> float:
    """Ask the model to self-score its response on a 1-10 scale."""
    prompt = (
        f"Rate the quality of the following response on a scale of 1 to 10. "
        f"Only output the numeric score.\n\n"
        f"Task: {scenario}\n\nResponse:\n{response[:500]}"
    )
    try:
        result = subprocess.run(
            [ollama_path, "run", model, prompt],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", result.stdout)
        if m:
            score = float(m.group(1))
            return min(max(score, 1.0), 10.0)
    except Exception:
        pass
    return 5.0


def run_tests(
    models: List[str],
    scenarios: Optional[List[TestScenario]] = None,
    *,
    ollama_path: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> ToolResult:
    """Run benchmark tests on the given models and scenarios.

    Parameters
    ----------
    models:
        List of model names (e.g. ``["llama3.2:3b", "qwen2:7b"]``).
    scenarios:
        Test scenarios; defaults to ``DEFAULT_TEST_SCENARIOS``.
    ollama_path:
        Explicit path to the Ollama executable.
    progress_cb:
        Optional callback ``f(message)`` to report progress.
    stop_flag:
        Optional callable returning ``True`` to abort testing.

    Returns a ``ToolResult`` with ``data`` being a list of ``TestResult``.
    """
    exe = ollama_path or find_ollama()
    if not exe:
        return ToolResult(success=False, error="Ollama executable not found")
    if not models:
        return ToolResult(success=False, error="No models specified")

    scenarios = scenarios or DEFAULT_TEST_SCENARIOS
    results: List[TestResult] = []

    for model in models:
        for sc in scenarios:
            if stop_flag and stop_flag():
                return ToolResult(
                    success=True, data=results,
                    metadata={"count": len(results), "stopped": True},
                )
            prompt = sc.prompt_template.format(input=sc.user_input) if sc.user_input else sc.prompt_template
            if progress_cb:
                progress_cb(f"Testing {model} on '{sc.name}'...")

            response, metrics = _run_ollama_verbose(exe, model, prompt)
            if response:
                metrics.self_score = _get_self_evaluation(exe, model, sc.name, response)

            results.append(TestResult(
                model=model, scenario=sc.name,
                response=response, metrics=metrics,
            ))

    return ToolResult(success=True, data=results, metadata={"count": len(results)})


def export_results_csv(results: List[TestResult], path: str) -> ToolResult:
    """Export test results to a CSV file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "model", "scenario", "self_score", "eval_rate_tps",
            "total_duration_sec", "load_duration_sec",
            "prompt_eval_duration_sec", "eval_duration_sec",
            "prompt_tokens", "completion_tokens",
            "prompt_eval_rate_tps", "ttft_ms", "response",
        ]
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        return ToolResult(success=True, data=str(p))
    except Exception as exc:
        return ToolResult(success=False, error=str(exc))
