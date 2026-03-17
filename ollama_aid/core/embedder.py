"""
OllamaAid - Embedding model benchmark
Evaluate embedding models using MTEB-style metrics.

References
----------
- MTEB: Muennighoff et al. (2023). "MTEB: Massive Text Embedding Benchmark"
  https://arxiv.org/abs/2210.07316
- STS Benchmark: Cer et al. (2017). "SemEval-2017 Task 1: Semantic Textual Similarity"
  https://arxiv.org/abs/1708.00055
- BEIR: Thakur et al. (2021). "BEIR: A Heterogenous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models"
  https://arxiv.org/abs/2104.08663
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import (
    RETRIEVAL_BENCHMARK_EN,
    RETRIEVAL_BENCHMARK_ZH,
    STS_BENCHMARK_EN,
    STS_BENCHMARK_ZH,
    CROSS_LINGUAL_PAIRS,
)
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class EmbeddingMetrics:
    """Metrics from embedding model evaluation."""
    embedding_dim: int = 0
    avg_latency_ms: float = 0.0
    sts_spearman: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_recall_at_k: float = 0.0
    cross_lingual_score: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "sts_spearman": round(self.sts_spearman, 4),
            "retrieval_mrr": round(self.retrieval_mrr, 4),
            "retrieval_recall_at_k": round(self.retrieval_recall_at_k, 4),
            "cross_lingual_score": round(self.cross_lingual_score, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class EmbeddingBenchResult:
    """Result of embedding model benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: EmbeddingMetrics = field(default_factory=EmbeddingMetrics)
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


def get_embedding(model: str, text: str, timeout: int = 60) -> Optional[List[float]]:
    """Get embedding vector from Ollama API."""
    try:
        import requests
        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Embedding API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        return data.get("embedding")
    except Exception as e:
        log.debug(f"Failed to get embedding from {model}: {e}")
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    n = len(x)

    def rank(arr: List[float]) -> List[float]:
        sorted_idx = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and arr[sorted_idx[j]] == arr[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)
    d2 = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1 - (6 * d2) / (n * (n ** 2 - 1))


def evaluate_sts(
    model: str,
    samples: List,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate semantic textual similarity.

    Based on STS Benchmark (Cer et al., 2017).
    Returns Spearman correlation between predicted and gold similarity scores.
    """
    if progress_cb:
        progress_cb(f"  Evaluating STS for {model}...")

    predictions: List[float] = []
    gold_scores: List[float] = []
    latencies: List[float] = []
    dim = 0

    for sample in samples:
        start = time.time()
        e1 = get_embedding(model, sample.sentence1)
        e2 = get_embedding(model, sample.sentence2)
        latencies.append((time.time() - start) * 1000)

        if e1 and e2:
            dim = len(e1)
            sim = cosine_similarity(e1, e2)
            predictions.append(sim)
            gold_scores.append(sample.score)

    if len(predictions) < 2:
        return {"spearman": 0.0, "samples_evaluated": 0, "dim": dim}

    spearman = spearman_correlation(predictions, gold_scores)

    return {
        "spearman": spearman,
        "samples_evaluated": len(predictions),
        "dim": dim,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def evaluate_retrieval(
    model: str,
    samples: List,
    top_k: int = 3,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate retrieval performance.

    Based on BEIR benchmark (Thakur et al., 2021).
    Returns MRR (Mean Reciprocal Rank) and Recall@K.
    """
    if progress_cb:
        progress_cb(f"  Evaluating retrieval for {model}...")

    mrr_scores: List[float] = []
    recall_scores: List[float] = []
    latencies: List[float] = []
    dim = 0

    for sample in samples:
        start = time.time()
        query_emb = get_embedding(model, sample.query)
        if not query_emb:
            continue
        dim = len(query_emb)

        doc_scores: List[tuple] = []
        for i, doc in enumerate(sample.documents):
            doc_emb = get_embedding(model, doc)
            if doc_emb:
                sim = cosine_similarity(query_emb, doc_emb)
                doc_scores.append((i, sim))

        latencies.append((time.time() - start) * 1000)

        if not doc_scores:
            continue

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        relevant = set(sample.relevant_doc_ids)

        rr = 0.0
        for rank, (idx, _) in enumerate(doc_scores, 1):
            if idx in relevant:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        retrieved = [idx for idx, _ in doc_scores[:top_k]]
        hit = len(set(retrieved) & relevant)
        recall = hit / len(relevant) if relevant else 0
        recall_scores.append(recall)

    return {
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        "recall_at_k": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
        "top_k": top_k,
        "samples_evaluated": len(mrr_scores),
        "dim": dim,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def evaluate_cross_lingual(
    model: str,
    pairs: List[tuple],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate cross-lingual semantic similarity."""
    if progress_cb:
        progress_cb(f"  Evaluating cross-lingual for {model}...")

    similarities: List[float] = []
    latencies: List[float] = []
    dim = 0

    for text1, text2, expected in pairs:
        start = time.time()
        e1 = get_embedding(model, text1)
        e2 = get_embedding(model, text2)
        latencies.append((time.time() - start) * 1000)

        if e1 and e2:
            dim = len(e1)
            sim = cosine_similarity(e1, e2)
            similarities.append(sim)

    if not similarities:
        return {"avg_similarity": 0.0, "samples_evaluated": 0, "dim": dim}

    return {
        "avg_similarity": sum(similarities) / len(similarities),
        "samples_evaluated": len(similarities),
        "dim": dim,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_embedding(
    model: str,
    language: str = "en",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> EmbeddingBenchResult:
    """Run comprehensive benchmark on a single embedding model.

    Parameters
    ----------
    model: Model name (e.g., "bge-large:335m")
    language: "en", "zh", or "both"
    progress_cb: Optional callback for progress updates

    Returns
    -------
    EmbeddingBenchResult with metrics and details
    """
    result = EmbeddingBenchResult(model=model)
    all_latencies: List[float] = []
    dim = 0
    tests_passed = 0
    tests_total = 0

    test_embedding = get_embedding(model, "test")
    if not test_embedding:
        result.success = False
        result.error = f"Failed to get embedding from model '{model}'. Check if model is loaded."
        return result

    dim = len(test_embedding)
    result.metrics.embedding_dim = dim

    if language in ("en", "both"):
        tests_total += 1
        sts_result = evaluate_sts(model, STS_BENCHMARK_EN, progress_cb)
        result.details["sts_en"] = {
            "spearman": round(sts_result["spearman"], 4),
            "samples": sts_result["samples_evaluated"],
        }
        result.metrics.sts_spearman = sts_result["spearman"]
        if sts_result["spearman"] > 0.5:
            tests_passed += 1
        if sts_result.get("avg_latency_ms"):
            all_latencies.append(sts_result["avg_latency_ms"])

        tests_total += 1
        ret_result = evaluate_retrieval(model, RETRIEVAL_BENCHMARK_EN, progress_cb=progress_cb)
        result.details["retrieval_en"] = {
            "mrr": round(ret_result["mrr"], 4),
            "recall_at_3": round(ret_result["recall_at_k"], 4),
            "samples": ret_result["samples_evaluated"],
        }
        result.metrics.retrieval_mrr = ret_result["mrr"]
        result.metrics.retrieval_recall_at_k = ret_result["recall_at_k"]
        if ret_result["mrr"] > 0.5:
            tests_passed += 1
        if ret_result.get("avg_latency_ms"):
            all_latencies.append(ret_result["avg_latency_ms"])

    if language in ("zh", "both"):
        tests_total += 1
        sts_zh_result = evaluate_sts(model, STS_BENCHMARK_ZH, progress_cb)
        result.details["sts_zh"] = {
            "spearman": round(sts_zh_result["spearman"], 4),
            "samples": sts_zh_result["samples_evaluated"],
        }
        if sts_zh_result["spearman"] > result.metrics.sts_spearman:
            result.metrics.sts_spearman = sts_zh_result["spearman"]
        if sts_zh_result["spearman"] > 0.5:
            tests_passed += 1
        if sts_zh_result.get("avg_latency_ms"):
            all_latencies.append(sts_zh_result["avg_latency_ms"])

        tests_total += 1
        ret_zh_result = evaluate_retrieval(model, RETRIEVAL_BENCHMARK_ZH, progress_cb=progress_cb)
        result.details["retrieval_zh"] = {
            "mrr": round(ret_zh_result["mrr"], 4),
            "recall_at_3": round(ret_zh_result["recall_at_k"], 4),
            "samples": ret_zh_result["samples_evaluated"],
        }
        if ret_zh_result["mrr"] > result.metrics.retrieval_mrr:
            result.metrics.retrieval_mrr = ret_zh_result["mrr"]
        if ret_zh_result["mrr"] > 0.5:
            tests_passed += 1
        if ret_zh_result.get("avg_latency_ms"):
            all_latencies.append(ret_zh_result["avg_latency_ms"])

    tests_total += 1
    cl_result = evaluate_cross_lingual(model, CROSS_LINGUAL_PAIRS, progress_cb)
    result.details["cross_lingual"] = {
        "avg_similarity": round(cl_result["avg_similarity"], 4),
        "samples": cl_result["samples_evaluated"],
    }
    result.metrics.cross_lingual_score = cl_result["avg_similarity"]
    if cl_result["avg_similarity"] > 0.6:
        tests_passed += 1
    if cl_result.get("avg_latency_ms"):
        all_latencies.append(cl_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total

    total = 0.0
    if result.metrics.sts_spearman > 0:
        total += result.metrics.sts_spearman * 25
    if result.metrics.retrieval_mrr > 0:
        total += result.metrics.retrieval_mrr * 30
    if result.metrics.retrieval_recall_at_k > 0:
        total += result.metrics.retrieval_recall_at_k * 20
    if result.metrics.cross_lingual_score > 0:
        total += min(result.metrics.cross_lingual_score * 100, 25)
    result.metrics.total_score = total

    return result


def benchmark_embeddings(
    models: List[str],
    language: str = "both",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple embedding models.

    Parameters
    ----------
    models: List of model names
    language: "en", "zh", or "both"
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of EmbeddingBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[EmbeddingBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking {model}...")
        result = benchmark_embedding(model, language=language, progress_cb=progress_cb)
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