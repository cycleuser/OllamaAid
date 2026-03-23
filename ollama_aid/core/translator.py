"""
OllamaAid - Translation model benchmark
Evaluate translation models using BLEU and semantic similarity metrics.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import TRANSLATION_BENCHMARK, TranslationSample
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class TranslationMetrics:
    """Metrics from translation model evaluation."""
    avg_latency_ms: float = 0.0
    bleu_score: float = 0.0
    semantic_similarity: float = 0.0
    accuracy: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "bleu_score": round(self.bleu_score, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "accuracy": round(self.accuracy, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class TranslationBenchResult:
    """Result of translation model benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: TranslationMetrics = field(default_factory=TranslationMetrics)
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


def _simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    return re.findall(r'\w+', text.lower())


def _bleu_score(candidate: str, reference: str) -> float:
    """Calculate simplified BLEU score."""
    cand_tokens = _simple_tokenize(candidate)
    ref_tokens = _simple_tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    matches = sum(1 for t in cand_tokens if t in ref_tokens)
    precision = matches / len(cand_tokens) if cand_tokens else 0

    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)
    overlap = len(cand_set & ref_set)
    recall = overlap / len(ref_set) if ref_set else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _word_overlap_score(candidate: str, reference: str) -> float:
    """Calculate word overlap score."""
    cand_tokens = set(_simple_tokenize(candidate))
    ref_tokens = set(_simple_tokenize(reference))

    if not ref_tokens:
        return 0.0

    overlap = len(cand_tokens & ref_tokens)
    return overlap / len(ref_tokens)


def get_translation(
    model: str,
    text: str,
    source_lang: str,
    target_lang: str,
    timeout: int = 60,
) -> Optional[str]:
    """Get translation from Ollama API."""
    try:
        import requests

        if "qwen" in model.lower():
            prompt = f"Translate the following from {source_lang} to {target_lang}. Only output the translation, nothing else.\n\n{text}"
        elif "llama" in model.lower():
            prompt = f"Translate this text to {target_lang}: {text}"
        elif "mistral" in model.lower() or "gemma" in model.lower():
            prompt = f"Translate to {target_lang}: {text}"
        else:
            prompt = f"Translate from {source_lang} to {target_lang}: {text}"

        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Translation API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        log.debug(f"Failed to get translation from {model}: {e}")
        return None


def evaluate_translation(
    model: str,
    samples: List[TranslationSample],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate translation performance using BLEU and overlap metrics."""
    if progress_cb:
        progress_cb(f"  Evaluating translation for {model}...")

    bleu_scores: List[float] = []
    overlap_scores: List[float] = []
    latencies: List[float] = []

    for sample in samples:
        start = time.time()
        translation = get_translation(
            model,
            sample.source_text,
            sample.source_lang,
            sample.target_lang,
        )
        latencies.append((time.time() - start) * 1000)

        if translation:
            bleu = _bleu_score(translation, sample.reference)
            bleu_scores.append(bleu)

            overlap = _word_overlap_score(translation, sample.reference)
            overlap_scores.append(overlap)

    return {
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
        "word_overlap": sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0,
        "samples_evaluated": len(bleu_scores),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_translation(
    model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> TranslationBenchResult:
    """Run comprehensive benchmark on a translation model.

    Parameters
    ----------
    model: Model name (e.g., "qwen2.5:7b")
    progress_cb: Optional callback for progress updates

    Returns
    -------
    TranslationBenchResult with metrics and details
    """
    result = TranslationBenchResult(model=model)
    latencies: List[float] = []
    tests_passed = 0
    tests_total = 1

    test_translation = get_translation(
        model,
        "Hello world",
        "en",
        "zh",
    )
    if not test_translation:
        result.success = False
        result.error = f"Failed to get translation from model '{model}'. Check if model is loaded."
        return result

    trans_result = evaluate_translation(model, TRANSLATION_BENCHMARK, progress_cb)

    result.details["translation"] = {
        "bleu": round(trans_result["bleu"], 4),
        "word_overlap": round(trans_result["word_overlap"], 4),
        "samples": trans_result["samples_evaluated"],
    }
    result.metrics.bleu_score = trans_result["bleu"]
    result.metrics.semantic_similarity = trans_result["word_overlap"]
    result.metrics.accuracy = trans_result["word_overlap"]

    if trans_result.get("avg_latency_ms"):
        latencies.append(trans_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if result.metrics.bleu_score > 0.3:
        tests_passed += 1
    if result.metrics.semantic_similarity > 0.4:
        tests_passed += 1
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total + 2

    total = 0.0
    if result.metrics.bleu_score > 0:
        total += min(result.metrics.bleu_score * 50, 40)
    if result.metrics.semantic_similarity > 0:
        total += min(result.metrics.semantic_similarity * 60, 60)
    result.metrics.total_score = total

    return result


def benchmark_translations(
    models: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple translation models.

    Parameters
    ----------
    models: List of model names
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of TranslationBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[TranslationBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking translation for {model}...")
        result = benchmark_translation(model, progress_cb=progress_cb)
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
