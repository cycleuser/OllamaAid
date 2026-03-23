"""
OllamaAid - Chat/Completion benchmark
Evaluate chat model responses for quality, relevance, and helpfulness.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import CHAT_BENCHMARK, ChatSample
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class ChatMetrics:
    """Metrics from chat/completion evaluation."""
    avg_latency_ms: float = 0.0
    response_length: float = 0.0
    relevance_score: float = 0.0
    helpfulness_score: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "response_length": round(self.response_length, 2),
            "relevance_score": round(self.relevance_score, 4),
            "helpfulness_score": round(self.helpfulness_score, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class ChatBenchResult:
    """Result of chat/completion benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: ChatMetrics = field(default_factory=ChatMetrics)
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


def _compute_keyword_overlap(text1: str, text2: str) -> float:
    """Compute keyword overlap between two texts."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    if not words2:
        return 0.0

    overlap = len(words1 & words2)
    return overlap / len(words2)


def _compute_length_ratio(response: str, reference: str) -> float:
    """Compute length similarity ratio."""
    len_response = len(response)
    len_reference = len(reference)

    if len_reference == 0:
        return 0.0

    ratio = len_response / len_reference
    if ratio > 1.5:
        return 1.5 / ratio
    return ratio


def get_chat_response(
    model: str,
    system: str,
    user_message: str,
    timeout: int = 60,
) -> Optional[str]:
    """Get chat response from Ollama API."""
    try:
        import requests

        if "qwen" in model.lower():
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
        elif "llama" in model.lower():
            prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user_message} [/INST]"
        elif "gemma" in model.lower():
            prompt = f"<start_of_turn>model\n{system}\n{user_message}<end_of_turn>\n<start_of_turn>model"
        else:
            prompt = f"System: {system}\n\nUser: {user_message}\n\nAssistant:"

        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Chat API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        log.debug(f"Failed to get chat response from {model}: {e}")
        return None


def evaluate_chat(
    model: str,
    samples: List[ChatSample],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate chat/completion performance."""
    if progress_cb:
        progress_cb(f"  Evaluating chat for {model}...")

    relevance_scores: List[float] = []
    length_ratios: List[float] = []
    latencies: List[float] = []
    lengths: List[int] = []

    for sample in samples:
        start = time.time()
        response = get_chat_response(model, sample.system, sample.user_message)
        latencies.append((time.time() - start) * 1000)

        if not response:
            continue

        lengths.append(len(response))

        relevance = _compute_keyword_overlap(response, sample.reference_response)
        relevance_scores.append(relevance)

        length_ratio = _compute_length_ratio(response, sample.reference_response)
        length_ratios.append(length_ratio)

    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    helpfulness = (avg_relevance + sum(length_ratios) / len(length_ratios)) / 2 if length_ratios else 0

    return {
        "relevance": avg_relevance,
        "helpfulness": helpfulness,
        "avg_length": avg_length,
        "samples_evaluated": len(relevance_scores),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_chat(
    model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ChatBenchResult:
    """Run comprehensive benchmark on chat/completion capability.

    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    progress_cb: Optional callback for progress updates

    Returns
    -------
    ChatBenchResult with metrics and details
    """
    result = ChatBenchResult(model=model)
    latencies: List[float] = []
    tests_passed = 0
    tests_total = 1

    test_response = get_chat_response(
        model,
        "You are a helpful assistant.",
        "Hello, how are you?",
    )
    if not test_response:
        result.success = False
        result.error = f"Failed to get chat response from model '{model}'. Check if model is loaded."
        return result

    eval_result = evaluate_chat(model, CHAT_BENCHMARK, progress_cb)

    result.details["chat"] = {
        "relevance": round(eval_result["relevance"], 4),
        "helpfulness": round(eval_result["helpfulness"], 4),
        "avg_length": round(eval_result["avg_length"], 2),
        "samples": eval_result["samples_evaluated"],
    }
    result.metrics.relevance_score = eval_result["relevance"]
    result.metrics.helpfulness_score = eval_result["helpfulness"]
    result.metrics.response_length = eval_result["avg_length"]

    if eval_result.get("avg_latency_ms"):
        latencies.append(eval_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if result.metrics.relevance_score > 0.2:
        tests_passed += 1
    if result.metrics.helpfulness_score > 0.3:
        tests_passed += 1
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total + 2

    total = 0.0
    if result.metrics.relevance_score > 0:
        total += result.metrics.relevance_score * 50
    if result.metrics.helpfulness_score > 0:
        total += result.metrics.helpfulness_score * 50
    result.metrics.total_score = total

    return result


def benchmark_chats(
    models: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple models on chat/completion.

    Parameters
    ----------
    models: List of model names
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of ChatBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[ChatBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking chat for {model}...")
        result = benchmark_chat(model, progress_cb=progress_cb)
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
