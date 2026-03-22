#!/usr/bin/env python
"""
OllamaAid - Embedding model benchmark CLI

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

Usage
-----
    # Benchmark specific models
    python scripts/benchmark_embeddings.py bge-large:335m bge-m3:567m

    # Benchmark all known embedding models
    python scripts/benchmark_embeddings.py --all

    # JSON output
    python scripts/benchmark_embeddings.py --all --json

    # Chinese only evaluation
    python scripts/benchmark_embeddings.py bge-large:335m --language zh
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

EMBEDDING_MODELS: List[str] = [
    "all-minilm:22m",
    "all-minilm:33m",
    "bge-large:335m",
    "bge-m3:567m",
    "embeddinggemma:latest",
    "granite-embedding:278m",
    "granite-embedding:30m",
    "mxbai-embed-large:335m",
    "nomic-embed-text:latest",
    "nomic-embed-text-v2-moe:latest",
    "paraphrase-multilingual:278m",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
    "qwen3-embedding:8b",
    "snowflake-arctic-embed:22m",
    "snowflake-arctic-embed:33m",
    "snowflake-arctic-embed:110m",
    "snowflake-arctic-embed:137m",
    "snowflake-arctic-embed:335m",
    "snowflake-arctic-embed2:568m",
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/benchmark_embeddings.py bge-large:335m
    python scripts/benchmark_embeddings.py --all
    python scripts/benchmark_embeddings.py --all --json
    python scripts/benchmark_embeddings.py bge-large:335m --language zh

References:
    - MTEB: https://arxiv.org/abs/2210.07316
    - STS Benchmark: https://arxiv.org/abs/1708.00055
    - BEIR: https://arxiv.org/abs/2104.08663
        """,
    )
    parser.add_argument("models", nargs="*", help="Model names to test")
    parser.add_argument("--all", action="store_true", help="Test all known embedding models")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--language", choices=["en", "zh", "both"], default="both",
                        help="Language for evaluation (default: both)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    models = args.models if args.models else (EMBEDDING_MODELS if args.all else [])

    if not models:
        parser.print_help()
        print("\nError: No models specified. Use --all or provide model names.")
        sys.exit(1)

    try:
        from ollama_aid.api import benchmark_embeddings
    except ImportError:
        print("Error: ollama_aid not installed. Run: pip install -e .")
        sys.exit(1)

    if not args.quiet:
        print(f"Benchmarking {len(models)} embedding model(s)...\n")

    def progress_cb(msg: str) -> None:
        if not args.quiet:
            print(f"  {msg}")

    result = benchmark_embeddings(models, language=args.language, progress_cb=progress_cb)

    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)

    results = result.data or []

    if args.json:
        output = {
            "success": result.success,
            "metadata": result.metadata,
            "results": [r.to_dict() if hasattr(r, 'to_dict') else r for r in results],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print("\n" + "=" * 90)
    print(f"{'Model':<40} {'Dim':>6} {'STS':>8} {'MRR':>8} {'Recall':>8} {'Score':>8}")
    print("=" * 90)

    for r in results:
        if hasattr(r, 'success'):
            if r.success:
                m = r.metrics
                print(f"{r.model:<40} {m.embedding_dim:>6} "
                      f"{m.sts_spearman:>8.4f} {m.retrieval_mrr:>8.4f} "
                      f"{m.retrieval_recall_at_k:>8.4f} {m.total_score:>8.2f}")
            else:
                print(f"{r.model:<40} ERROR: {r.error}")
        else:
            if r.get("success"):
                m = r.get("metrics", {})
                print(f"{r['model']:<40} {m.get('embedding_dim', 0):>6} "
                      f"{m.get('sts_spearman', 0):>8.4f} {m.get('retrieval_mrr', 0):>8.4f} "
                      f"{m.get('retrieval_recall_at_k', 0):>8.4f} {m.get('total_score', 0):>8.2f}")
            else:
                print(f"{r['model']:<40} ERROR: {r.get('error', 'unknown')}")

    print("=" * 90)

    successful = [r for r in results if (r.success if hasattr(r, 'success') else r.get("success"))]
    print(f"\nCompleted: {len(successful)}/{len(results)} models")

    if successful:
        best = max(successful, key=lambda x: (x.metrics.total_score if hasattr(x, 'metrics') 
                                               else x.get("metrics", {}).get("total_score", 0)))
        best_name = best.model if hasattr(best, 'model') else best['model']
        best_score = best.metrics.total_score if hasattr(best, 'metrics') else best.get("metrics", {}).get("total_score", 0)
        print(f"Best model: {best_name} (score: {best_score:.2f})")


if __name__ == "__main__":
    main()