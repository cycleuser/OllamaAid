#!/usr/bin/env python
"""
OllamaAid - Unified Benchmark Script

Comprehensive benchmark for all Ollama model types:
- Embedding: Semantic textual similarity, retrieval, cross-lingual
- Reranker: NDCG, MRR, MAP for document re-ranking
- Translation: BLEU score, word overlap for translation tasks
- Tool Calling: Function name and parameter accuracy
- Code Generation: Syntax validity, test pass rate
- Chat: Response relevance and helpfulness

Usage:
    # Benchmark all models (auto-detect type)
    python scripts/benchmark_all.py

    # Benchmark specific type
    python scripts/benchmark_all.py --type embedding
    python scripts/benchmark_all.py --type translation
    python scripts/benchmark_all.py --type tool
    python scripts/benchmark_all.py --type code
    python scripts/benchmark_all.py --type chat

    # Benchmark specific models
    python scripts/benchmark_all.py --models qwen2.5:7b,llama3.1:8b

    # JSON output
    python scripts/benchmark_all.py --type all --json

Examples:
    python scripts/benchmark_all.py -t embedding -q
    python scripts/benchmark_all.py -t translation --models qwen2.5:7b
    python scripts/benchmark_all.py -t all --json -o results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Callable, Optional

OLLAMA_API_BASE = "http://127.0.0.1:11434"


BENCHMARK_TYPES = {
    "embedding": "Semantic textual similarity, retrieval, and cross-lingual benchmarks",
    "reranker": "Document re-ranking with NDCG, MRR, and MAP metrics",
    "translation": "Translation quality with BLEU and word overlap metrics",
    "tool": "Tool/function calling accuracy",
    "code": "Code generation with syntax and test pass rate",
    "chat": "Chat response quality and helpfulness",
    "all": "Run all benchmark types",
}


def list_available_models() -> List[str]:
    """List all available Ollama models."""
    try:
        import requests
        resp = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


def detect_model_type(model_name: str) -> str:
    """Detect model type from name."""
    name_lower = model_name.lower()
    
    embedding_keywords = ["embed", "embedding", "minilm", "bge", "nomic", "arctic", "granite-embedding", "paraphrase", "e5", "mxbai"]
    reranker_keywords = ["rerank", "reranker"]
    code_keywords = ["code", "coder", "codellama", "codestral", "deepseek-coder", "qwen2.5-coder"]
    chat_keywords = ["chat", "instruct", "qwen", "llama", "mistral", "gemma", "phi", "yi", "glm", "-command"]
    
    if any(kw in name_lower for kw in embedding_keywords):
        return "embedding"
    if any(kw in name_lower for kw in reranker_keywords):
        return "reranker"
    if any(kw in name_lower for kw in code_keywords):
        return "code"
    if any(kw in name_lower for kw in chat_keywords):
        return "chat"
    
    return "chat"


def benchmark_embedding_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark embedding models."""
    try:
        from ollama_aid.core.embedder import benchmark_embedding
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Embedding: {model}")
            result = benchmark_embedding(model, language="both")
            results.append(result.to_dict())
        return {"type": "embedding", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "embedding", "error": str(e), "count": 0}


def benchmark_reranker_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark reranker models."""
    try:
        from ollama_aid.core.reranker import benchmark_reranker
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Reranker: {model}")
            result = benchmark_reranker(model)
            results.append(result.to_dict())
        return {"type": "reranker", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "reranker", "error": str(e), "count": 0}


def benchmark_translation_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark translation capabilities."""
    try:
        from ollama_aid.core.translator import benchmark_translation
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Translation: {model}")
            result = benchmark_translation(model)
            results.append(result.to_dict())
        return {"type": "translation", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "translation", "error": str(e), "count": 0}


def benchmark_tool_calling_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark tool calling capabilities."""
    try:
        from ollama_aid.core.tool_caller import benchmark_tool_call
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Tool Calling: {model}")
            result = benchmark_tool_call(model)
            results.append(result.to_dict())
        return {"type": "tool", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "tool", "error": str(e), "count": 0}


def benchmark_code_generation_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark code generation capabilities."""
    try:
        from ollama_aid.core.code_eval import benchmark_code_generation
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Code Generation: {model}")
            result = benchmark_code_generation(model)
            results.append(result.to_dict())
        return {"type": "code", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "code", "error": str(e), "count": 0}


def benchmark_chat_models(models: List[str], progress_cb: Optional[Callable] = None) -> dict:
    """Benchmark chat capabilities."""
    try:
        from ollama_aid.core.chat_eval import benchmark_chat
        results = []
        for model in models:
            if progress_cb:
                progress_cb(f"  Chat: {model}")
            result = benchmark_chat(model)
            results.append(result.to_dict())
        return {"type": "chat", "results": results, "count": len(results)}
    except ImportError as e:
        return {"type": "chat", "error": str(e), "count": 0}


def run_all_benchmarks(
    models: List[str],
    benchmark_type: str = "all",
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Run benchmarks based on type."""
    results = {
        "total_models": len(models),
        "benchmarks": {},
        "summary": {},
    }
    
    if benchmark_type in ("embedding", "all"):
        embedding_models = [m for m in models if detect_model_type(m) == "embedding"]
        if embedding_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(embedding_models)} embedding models...]")
            bench_result = benchmark_embedding_models(embedding_models, progress_cb)
            results["benchmarks"]["embedding"] = bench_result
            results["summary"]["embedding"] = f"{bench_result.get('count', 0)} models tested"
    
    if benchmark_type in ("reranker", "all"):
        reranker_models = [m for m in models if detect_model_type(m) == "reranker"]
        if reranker_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(reranker_models)} reranker models...]")
            bench_result = benchmark_reranker_models(reranker_models, progress_cb)
            results["benchmarks"]["reranker"] = bench_result
            results["summary"]["reranker"] = f"{bench_result.get('count', 0)} models tested"
    
    if benchmark_type in ("translation", "all"):
        translation_models = [m for m in models if detect_model_type(m) in ("chat", "translation")]
        translation_models = [m for m in translation_models if detect_model_type(m) != "embedding"]
        if translation_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(translation_models)} translation-capable models...]")
            bench_result = benchmark_translation_models(translation_models, progress_cb)
            results["benchmarks"]["translation"] = bench_result
            results["summary"]["translation"] = f"{bench_result.get('count', 0)} models tested"
    
    if benchmark_type in ("tool", "all"):
        tool_models = [m for m in models if detect_model_type(m) in ("chat", "tool")]
        tool_models = [m for m in tool_models if detect_model_type(m) not in ("embedding", "reranker")]
        if tool_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(tool_models)} tool-calling models...]")
            bench_result = benchmark_tool_calling_models(tool_models, progress_cb)
            results["benchmarks"]["tool"] = bench_result
            results["summary"]["tool"] = f"{bench_result.get('count', 0)} models tested"
    
    if benchmark_type in ("code", "all"):
        code_models = [m for m in models if detect_model_type(m) == "code"]
        if code_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(code_models)} code generation models...]")
            bench_result = benchmark_code_generation_models(code_models, progress_cb)
            results["benchmarks"]["code"] = bench_result
            results["summary"]["code"] = f"{bench_result.get('count', 0)} models tested"
    
    if benchmark_type in ("chat", "all"):
        chat_models = [m for m in models if detect_model_type(m) == "chat"]
        chat_models = [m for m in chat_models if detect_model_type(m) not in ("embedding", "reranker", "code")]
        if chat_models:
            if progress_cb:
                progress_cb(f"\n[Benchmarking {len(chat_models)} chat models...]")
            bench_result = benchmark_chat_models(chat_models, progress_cb)
            results["benchmarks"]["chat"] = bench_result
            results["summary"]["chat"] = f"{bench_result.get('count', 0)} models tested"
    
    return results


def print_summary(results: dict):
    """Print benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total models tested: {results.get('total_models', 0)}")
    print("-" * 70)
    
    for bench_type, bench_result in results.get("benchmarks", {}).items():
        count = bench_result.get("count", 0)
        error = bench_result.get("error")
        if error:
            print(f"  {bench_type.capitalize():<15} Error: {error}")
        else:
            print(f"  {bench_type.capitalize():<15} {count} models tested")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-t", "--type",
        choices=list(BENCHMARK_TYPES.keys()),
        default="all",
        help="Type of benchmark to run",
    )
    parser.add_argument(
        "-m", "--models",
        default="",
        help="Comma-separated model names (empty for all available)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-o", "--output",
        help="Save results to file",
    )
    
    args = parser.parse_args()
    
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        if not args.quiet:
            print("Discovering available models...")
        model_names = list_available_models()
        if not model_names:
            print("Error: No models found. Is Ollama running?", file=sys.stderr)
            sys.exit(1)
    
    if not args.quiet:
        print(f"Found {len(model_names)} models to benchmark")
        print(f"Benchmark type: {BENCHMARK_TYPES.get(args.type, 'all')}")
    
    def progress(msg):
        if not args.quiet:
            print(msg)
    
    results = run_all_benchmarks(model_names, args.type, progress)
    
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_summary(results)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
