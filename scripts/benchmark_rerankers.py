#!/usr/bin/env python
"""
OllamaAid - Reranker model benchmark CLI

Evaluate reranker models using standard retrieval metrics.

References
----------
- BEIR: Thakur et al. (2021). "BEIR: A Heterogenous Benchmark for Zero-shot
  Evaluation of Information Retrieval Models"
  https://arxiv.org/abs/2104.08663
- MS MARCO: Nguyen et al. (2016). "MS MARCO: A Human Generated MAchine Reading
  COmprehension Dataset"
  https://arxiv.org/abs/1611.09268

Usage
-----
    # Benchmark specific models
    python scripts/benchmark_rerankers.py "dengcao/Qwen3-Reranker-0.6B:Q8_0"

    # JSON output
    python scripts/benchmark_rerankers.py "dengcao/Qwen3-Reranker-0.6B:Q8_0" --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

RERANKER_MODELS: List[str] = [
    "dengcao/Qwen3-Reranker-0.6B:Q8_0",
    "dengcao/Qwen3-Reranker-4B:Q8_0",
]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama reranker models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/benchmark_rerankers.py "dengcao/Qwen3-Reranker-0.6B:Q8_0"
    python scripts/benchmark_rerankers.py --all
    python scripts/benchmark_rerankers.py --all --json

References:
    - BEIR: https://arxiv.org/abs/2104.08663
    - MS MARCO: https://arxiv.org/abs/1611.09268
        """,
    )
    parser.add_argument("models", nargs="*", help="Model names to test")
    parser.add_argument("--all", action="store_true", help="Test all known reranker models")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    models = args.models if args.models else (RERANKER_MODELS if args.all else [])

    if not models:
        parser.print_help()
        print("\nError: No models specified. Use --all or provide model names.")
        sys.exit(1)

    try:
        from ollama_aid.api import benchmark_rerankers
    except ImportError:
        print("Error: ollama_aid not installed. Run: pip install -e .")
        sys.exit(1)

    if not args.quiet:
        print(f"Benchmarking {len(models)} reranker model(s)...\n")

    def progress_cb(msg: str) -> None:
        if not args.quiet:
            print(f"  {msg}")

    result = benchmark_rerankers(models, progress_cb=progress_cb)

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

    print("\n" + "=" * 80)
    print(f"{'Model':<40} {'NDCG@5':>8} {'MRR':>8} {'MAP':>8} {'Score':>8}")
    print("=" * 80)

    for r in results:
        if hasattr(r, 'success'):
            if r.success:
                m = r.metrics
                print(f"{r.model:<40} {m.ndcg_at_k:>8.4f} "
                      f"{m.mrr:>8.4f} {m.map_score:>8.4f} {m.total_score:>8.2f}")
            else:
                print(f"{r.model:<40} ERROR: {r.error}")
        else:
            if r.get("success"):
                m = r.get("metrics", {})
                print(f"{r['model']:<40} {m.get('ndcg_at_k', 0):>8.4f} "
                      f"{m.get('mrr', 0):>8.4f} {m.get('map_score', 0):>8.4f} "
                      f"{m.get('total_score', 0):>8.2f}")
            else:
                print(f"{r['model']:<40} ERROR: {r.get('error', 'unknown')}")

    print("=" * 80)

    successful = [r for r in results if (r.success if hasattr(r, 'success') else r.get("success"))]
    print(f"\nCompleted: {len(successful)}/{len(results)} models")


if __name__ == "__main__":
    main()