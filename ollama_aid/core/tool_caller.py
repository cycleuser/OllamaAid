"""
OllamaAid - Tool calling benchmark
Evaluate tool calling / function calling capabilities.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import TOOL_CALL_BENCHMARK, ToolCallSample
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class ToolCallMetrics:
    """Metrics from tool calling evaluation."""
    avg_latency_ms: float = 0.0
    function_accuracy: float = 0.0
    parameter_accuracy: float = 0.0
    total_accuracy: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "function_accuracy": round(self.function_accuracy, 4),
            "parameter_accuracy": round(self.parameter_accuracy, 4),
            "total_accuracy": round(self.total_accuracy, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class ToolCallBenchResult:
    """Result of tool calling benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: ToolCallMetrics = field(default_factory=ToolCallMetrics)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "model": self.model,
            "success": self.success,
        }
        if self.error:
            d["error"] = self.error
        d["metrics"] = self.metrics.to_dict()
        d["details"] = self.details
        return d


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from model response."""
    try:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        if text.strip().startswith('{'):
            return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_function_call(text: str) -> tuple:
    """Extract function name and parameters from model response."""
    text_lower = text.lower()

    json_obj = _extract_json(text)
    if json_obj:
        if "name" in json_obj:
            return json_obj.get("name", ""), json_obj.get("parameters", {})
        if "function" in json_obj:
            func = json_obj["function"]
            return func.get("name", ""), func.get("parameters", {})

    name_patterns = [
        r'(?:call|invoke|use)\s+(\w+)',
        r'"name"\s*:\s*"(\w+)"',
        r'function\s+(\w+)',
        r'(\w+)\s*\(',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text_lower)
        if match:
            func_name = match.group(1)
            if func_name not in ['if', 'else', 'for', 'while', 'def', 'return']:
                return func_name, {}

    return "", {}


def get_tool_call(
    model: str,
    query: str,
    tools: List[dict],
    timeout: int = 60,
) -> Optional[tuple]:
    """Get tool call from Ollama API.

    Returns tuple of (function_name, parameters) or None.
    """
    try:
        import requests

        tools_json = json.dumps(tools, indent=2)
        prompt = f"""You have access to the following tools:

{tools_json}

User query: {query}

Based on the user query, determine which function to call and what parameters to use. Output your response in JSON format:
{{"name": "function_name", "parameters": {{"param1": "value1", ...}}}}"""

        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Tool call API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        response_text = data.get("response", "").strip()
        func_name, params = _extract_function_call(response_text)
        return (func_name, params, response_text)
    except Exception as e:
        log.debug(f"Failed to get tool call from {model}: {e}")
        return None


def evaluate_tool_calling(
    model: str,
    samples: List[ToolCallSample],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate tool calling performance."""
    if progress_cb:
        progress_cb(f"  Evaluating tool calling for {model}...")

    function_correct = 0
    param_correct = 0
    total = 0
    latencies: List[float] = []

    for sample in samples:
        start = time.time()
        result = get_tool_call(model, sample.query, sample.tools)
        latencies.append((time.time() - start) * 1000)

        if result is None:
            continue

        func_name, params, _ = result
        total += 1

        if func_name == sample.expected_function or func_name.lower() == sample.expected_function.lower():
            function_correct += 1

        param_match = True
        for key, expected_val in sample.expected_params.items():
            if key not in params:
                param_match = False
                break
            if str(params[key]).lower() != str(expected_val).lower():
                param_match = False
                break
        if param_match and sample.expected_params:
            param_correct += 1
        elif not sample.expected_params and not params:
            param_correct += 1

    return {
        "function_accuracy": function_correct / total if total > 0 else 0,
        "parameter_accuracy": param_correct / total if total > 0 else 0,
        "samples_evaluated": total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_tool_call(
    model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolCallBenchResult:
    """Run comprehensive benchmark on tool calling capability.

    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    progress_cb: Optional callback for progress updates

    Returns
    -------
    ToolCallBenchResult with metrics and details
    """
    result = ToolCallBenchResult(model=model)
    latencies: List[float] = []
    tests_passed = 0
    tests_total = 1

    test_result = get_tool_call(
        model,
        "Hello",
        [{"type": "function", "function": {"name": "test", "parameters": {}}}],
    )
    if test_result is None:
        result.success = False
        result.error = f"Failed to get tool call from model '{model}'. Check if model is loaded."
        return result

    eval_result = evaluate_tool_calling(model, TOOL_CALL_BENCHMARK, progress_cb)

    result.details["tool_calling"] = {
        "function_accuracy": round(eval_result["function_accuracy"], 4),
        "parameter_accuracy": round(eval_result["parameter_accuracy"], 4),
        "samples": eval_result["samples_evaluated"],
    }
    result.metrics.function_accuracy = eval_result["function_accuracy"]
    result.metrics.parameter_accuracy = eval_result["parameter_accuracy"]
    result.metrics.total_accuracy = (eval_result["function_accuracy"] + eval_result["parameter_accuracy"]) / 2

    if eval_result.get("avg_latency_ms"):
        latencies.append(eval_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if result.metrics.function_accuracy > 0.6:
        tests_passed += 1
    if result.metrics.parameter_accuracy > 0.4:
        tests_passed += 1
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total + 2

    total = 0.0
    if result.metrics.function_accuracy > 0:
        total += result.metrics.function_accuracy * 50
    if result.metrics.parameter_accuracy > 0:
        total += result.metrics.parameter_accuracy * 50
    result.metrics.total_score = total

    return result


def benchmark_tool_calls(
    models: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple models on tool calling.

    Parameters
    ----------
    models: List of model names
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of ToolCallBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[ToolCallBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking tool calling for {model}...")
        result = benchmark_tool_call(model, progress_cb=progress_cb)
        results.append(result)

    successful = [r for r in results if r.success]
    return ToolResult(
        success=len(successful) > 0,
        data=results,
        metadata={
            "total": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
        },
    )
