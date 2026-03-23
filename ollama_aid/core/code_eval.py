"""
OllamaAid - Code generation benchmark
Evaluate code generation capabilities.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .benchmark_data import CODE_GEN_BENCHMARK, CodeGenSample
from .models import ToolResult

log = logging.getLogger(__name__)

OLLAMA_API_BASE = "http://127.0.0.1:11434"


@dataclass
class CodeGenMetrics:
    """Metrics from code generation evaluation."""
    avg_latency_ms: float = 0.0
    syntax_validity: float = 0.0
    function_correctness: float = 0.0
    test_pass_rate: float = 0.0
    total_score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0

    def to_dict(self) -> dict:
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "syntax_validity": round(self.syntax_validity, 4),
            "function_correctness": round(self.function_correctness, 4),
            "test_pass_rate": round(self.test_pass_rate, 4),
            "total_score": round(self.total_score, 2),
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


@dataclass
class CodeGenBenchResult:
    """Result of code generation benchmark."""
    model: str
    success: bool = True
    error: Optional[str] = None
    metrics: CodeGenMetrics = field(default_factory=CodeGenMetrics)
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


LANG_PATTERNS = {
    "python": {
        "def": r"def\s+\w+\s*\(",
        "class": r"class\s+\w+",
        "import": r"^import\s|^from\s+import",
        "comment": r"#",
    },
    "javascript": {
        "function": r"function\s+\w+\s*\(",
        "const": r"const\s+\w+\s*=",
        "let": r"let\s+\w+\s*=",
        "arrow": r"=>",
    },
    "go": {
        "func": r"func\s+\w+",
        "package": r"package\s+\w+",
        "import": r"import\s+\(",
    },
    "java": {
        "class": r"public\s+class\s+\w+",
        "void": r"void\s+\w+\s*\(",
        "import": r"^import\s",
    },
}


def _extract_code(text: str, language: str) -> str:
    """Extract code block from response."""
    text = text.strip()

    code_block_match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    if '```' in text:
        lines = text.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)
        if code_lines:
            return '\n'.join(code_lines).strip()

    return text


def _check_syntax(code: str, language: str) -> bool:
    """Check if generated code has valid syntax structure."""
    if not code:
        return False

    patterns = LANG_PATTERNS.get(language.lower(), LANG_PATTERNS["python"])

    has_structure = any(re.search(p, code) for p in patterns.values())
    has_matching_parens = code.count('(') == code.count(')')
    has_matching_braces = code.count('{') == code.count('}')
    has_matching_brackets = code.count('[') == code.count(']')

    return has_structure and has_matching_parens and has_matching_braces and has_matching_brackets


def _test_python_code(code: str, test_cases: List[str]) -> tuple:
    """Test Python code against test cases."""
    if not code or 'def ' not in code:
        return False, "No function definition found"

    func_match = re.search(r'def\s+(\w+)\s*\(', code)
    if not func_match:
        return False, "Cannot find function name"

    func_name = func_match.group(1)
    test_code = f"\n{code}\n"

    for test in test_cases:
        try:
            exec(test_code)
            result = eval(test)
            if not result:
                return False, f"Test failed: {test}"
        except Exception as e:
            return False, f"Error executing test: {e}"

    return True, "All tests passed"


def get_code_generation(
    model: str,
    prompt: str,
    language: str,
    timeout: int = 60,
) -> Optional[str]:
    """Get code generation from Ollama API."""
    try:
        import requests

        if "codellama" in model.lower() or "qwen2.5-coder" in model.lower():
            full_prompt = f"<s>[INST] Write a {language} function for the following:\n{prompt} [/INST]"
        else:
            full_prompt = f"Write a {language} function:\n{prompt}\n\nOnly output the code, no explanations."

        resp = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.debug(f"Code gen API returned {resp.status_code} for model {model}")
            return None
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        log.debug(f"Failed to get code generation from {model}: {e}")
        return None


def evaluate_code_generation(
    model: str,
    samples: List[CodeGenSample],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate code generation performance."""
    if progress_cb:
        progress_cb(f"  Evaluating code generation for {model}...")

    syntax_valid = 0
    test_passed = 0
    total = 0
    latencies: List[float] = []

    for sample in samples:
        start = time.time()
        code_response = get_code_generation(model, sample.prompt, sample.language)
        latencies.append((time.time() - start) * 1000)

        if not code_response:
            continue

        total += 1
        code = _extract_code(code_response, sample.language)

        if _check_syntax(code, sample.language):
            syntax_valid += 1

        if sample.language.lower() == "python":
            passed, _ = _test_python_code(code, sample.test_cases)
            if passed:
                test_passed += 1

    return {
        "syntax_validity": syntax_valid / total if total > 0 else 0,
        "test_pass_rate": test_passed / total if total > 0 else 0,
        "samples_evaluated": total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def benchmark_code_generation(
    model: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> CodeGenBenchResult:
    """Run comprehensive benchmark on code generation capability.

    Parameters
    ----------
    model: Model name (e.g., "codellama:7b")
    progress_cb: Optional callback for progress updates

    Returns
    -------
    CodeGenBenchResult with metrics and details
    """
    result = CodeGenBenchResult(model=model)
    latencies: List[float] = []
    tests_passed = 0
    tests_total = 1

    test_response = get_code_generation(
        model,
        "return the sum of two numbers",
        "python",
    )
    if not test_response:
        result.success = False
        result.error = f"Failed to get code generation from model '{model}'. Check if model is loaded."
        return result

    eval_result = evaluate_code_generation(model, CODE_GEN_BENCHMARK, progress_cb)

    result.details["code_generation"] = {
        "syntax_validity": round(eval_result["syntax_validity"], 4),
        "test_pass_rate": round(eval_result["test_pass_rate"], 4),
        "samples": eval_result["samples_evaluated"],
    }
    result.metrics.syntax_validity = eval_result["syntax_validity"]
    result.metrics.test_pass_rate = eval_result["test_pass_rate"]
    result.metrics.function_correctness = eval_result["syntax_validity"]

    if eval_result.get("avg_latency_ms"):
        latencies.append(eval_result["avg_latency_ms"])

    result.metrics.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    if result.metrics.syntax_validity > 0.6:
        tests_passed += 1
    if result.metrics.test_pass_rate > 0.3:
        tests_passed += 1
    result.metrics.tests_passed = tests_passed
    result.metrics.tests_total = tests_total + 2

    total = 0.0
    if result.metrics.syntax_validity > 0:
        total += result.metrics.syntax_validity * 40
    if result.metrics.test_pass_rate > 0:
        total += result.metrics.test_pass_rate * 60
    result.metrics.total_score = total

    return result


def benchmark_code_generations(
    models: List[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> ToolResult:
    """Benchmark multiple models on code generation.

    Parameters
    ----------
    models: List of model names
    progress_cb: Progress callback

    Returns
    -------
    ToolResult with list of CodeGenBenchResult
    """
    if not models:
        return ToolResult(success=False, error="No models specified")

    results: List[CodeGenBenchResult] = []

    for model in models:
        if progress_cb:
            progress_cb(f"Benchmarking code generation for {model}...")
        result = benchmark_code_generation(model, progress_cb=progress_cb)
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
