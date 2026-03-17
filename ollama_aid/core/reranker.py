"""
OllamaAid - Reranker model benchmark
Evaluate reranker models using standard metrics.

References
----------
- BEIR: Thakur et al. (2021). "BEIR: A Heterogenous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models"
  https://arxiv.org/abs/2104.08663
- MS MARCO: Nguyen et al. (2016). "MS MARCO: A Human Generated MAchine Reading
  COmprehension Dataset"
  https://arxiv.org/abs/1611.09268
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import RERANK_BENCHMARK
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class RerankerMetrics:
    """Metrics from reranker model evaluation."""
    avg_latency_ms: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "mrr": round(self.mrr, 4),
            "map_score": round(self.map_score, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class RerankerBenchResult:
    """Result of reranker model benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: RerankerMetrics = field(default_factory=RerankerMetrics)
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


def get_rerank_score(model: str, query: str, document: str, timeout: int = 60) -> Optional[float]:
    """Get relevance score from reranker model via Ollama API.

    Reranker models typically output a relevance score between 0 and 1.
    We use the generate API and parse the numeric output.
    """
    try:
        import requests
        prompt = f"Rate the relevance of the following document to the query on a scale of 0 to 1. Output only the number.\n\nQuery: {query}\n\nDocument: {document}"
        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Rerank API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        response_text = data.get("response", "").strip()
        import re
        m = re.search(r"[\d.]+", response_text)
        if m:
            score = float(m.group())
            return min(max(score, 0.0), 1.0)
        return None
    except Exception as e:
        log.debug(f"Failed to get rerank score from {model}: {e}")
        return None


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute DCG@K (Discounted Cumulative Gain at K)."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    dcg = relevances[0]
    for i, rel in enumerate(relevances[1:], 2):
        import math
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(predicted_relevances: List[int], ideal_relevances: List[int], k: int) -> float:
    """Compute NDCG@K (Normalized DCG at K)."""
    dcg = dcg_at_k(predicted_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(predicted_relevances: List[int]) -> float:
    """Compute Average Precision for a single query."""
    if not predicted_relevances or sum(predicted_relevances) == 0:
        return 0.0
    relevant_count = 0
    precision_sum = 0.0
    for i, rel in enumerate(predicted_relevances):
        if rel > 0:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    return precision_sum / sum(predicted_relevances)


def reciprocal_rank(predicted_relevances: List[int]) -> float:
    """Compute Reciprocal Rank."""
    for i, rel in enumerate(predicted_relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_reranking(
    model: str,
    samples: List,
    top_k: int = 5,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate reranking performance.

    Based on BEIR reranking task format.
    Returns NDCG@K, MRR, and MAP scores.
    """
    if progress_cb:
        progress_cb(f"  Evaluating reranking for {model}...")

    ndcg_scores: List[float] = []
    mrr_scores: List[float] = []
    map_scores: List[float] = []
    latencies: List[float] = []

    for sample in samples:
        start = time.time()
        query = sample.query
        documents = sample.documents
        gold_relevances = sample.relevance_scores

        doc_scores: List[tuple] = []
        for i, doc in enumerate(documents):
            score = get_rerank_score(model, query, doc)
            if score is not None:
                doc_scores.append((i, score))

        latencies.append((time.time() - start) * 1000)

        if not doc_scores:
            continue

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_order = [idx for idx, _ in doc_scores]
        predicted_relevances = [gold_relevances[idx] for idx in predicted_order]

        ndcg = ndcg_at_k(predicted_relevances, gold_relevances, top_k)
        ndcg_scores.append(ndcg)

        rr = reciprocal_rank(predicted_relevances)
        mrr_scores.append(rr)

        ap = average_precision(predicted_relevances)
        map_scores.append(ap)

    return {
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        "map": sum(map_scores) / len(map_scores) if map_scores else 0,
        "top_k": top_k,
        "samples_evaluated": len(ndcg_scores),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_reranker(
    model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> RerankerBenchResult:
    """Run comprehensive benchmark on a single reranker model.

    Parameters
    ----------
    model: Model name (e.g., "dengcao/Qwen3-Reranker-4B:Q8_0")
    progress_cb: Optional callback for progress updates

    Returns
    -------
    RerankerBenchResult with metrics and details
    """
    result = RerankerBenchResult(model=model)
    latencies: List[float] = []
    tests_passed = 0
    tests_total = 1

    test_score = get_rerank_score(model, "test query", "test document")
    if test_score is None:
        result.success = False
        result.error = f"Failed to get rerank score from model '{model}'. Check if model is loaded and supports reranking."
        return result

    rerank_result = evaluate_reranking(model, RERANK_BENCHMARK, progress_cb=progress_cb)
    result.details["reranking"] = {
        "ndcg_at_5": round(rerank_result["ndcg_at_k"], 4),
        "mrr": round(rerank_result["mrr"], 4),
        "map": round(rerank_result["map"], 4),
        "samples": rerank_result["samples_evaluated"],
    }
    result.metrics.ndcg_at_k = rerank_result["ndcg_at_k"]
    result.metrics.mrr = rerank_result["mrr"]
    result.metrics.map_score = rerank_result["map"]
    if rerank_result.get("avg_latency_ms"):
        latencies.append(rerank_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if result.metrics.ndcg_at_k > 0.5:
        tests_passed += 1
    if result.metrics.mrr > 0.5:
        tests_passed += 1
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total + 1

    total = 0.0
    if result.metrics.ndcg_at_k > 0:
        total += result.metrics.ndcg_at_k * 40
    if result.metrics.mrr > 0:
        total += result.metrics.mrr * 35
    if result.metrics.map_score > 0:
        total += result.metrics.map_score * 25
    result.metrics.total_score = total

    return result


def benchmark_rerankers(
    models: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple reranker models.

    Parameters
    ----------
    models: List of model names
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of RerankerBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[RerankerBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking {model}...")
        result = benchmark_reranker(model, progress_cb=progress_cb)
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