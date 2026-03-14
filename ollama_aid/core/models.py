"""
OllamaAid - Data models
All data classes used across the application.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ToolResult (standard API return type)
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standard return type for all public API functions."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {"success": self.success}
        if self.data is not None:
            d["data"] = self.data
        if self.error is not None:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Ollama model information."""
    name: str
    tag: str = ""
    model_id: str = ""
    full_name: str = ""
    size: str = ""
    size_bytes: int = 0
    modified_date: str = ""
    quantization: str = ""
    family: str = ""
    parameter_size: str = ""
    digest: str = ""

    def __post_init__(self):
        if not self.full_name and self.name:
            self.full_name = f"{self.name}:{self.tag}" if self.tag else self.name


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

@dataclass
class TrendData:
    """Ollama model trend data scraped from ollama.com."""
    name: str
    pulls: float = 0.0
    min_params: float = 0.0
    max_params: float = 0.0
    param_details: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    updated: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pulls": self.pulls,
            "min_params": self.min_params,
            "max_params": self.max_params,
            "param_details": self.param_details,
            "tags": self.tags,
            "description": self.description,
            "updated": self.updated,
            "url": self.url,
        }


def parse_param_tags(param_details: str) -> list[str]:
    """Parse param_details string into lowercase Ollama-compatible tags.

    >>> parse_param_tags("0.8B, 2B, 4B, 9B")
    ['0.8b', '2b', '4b', '9b']
    >>> parse_param_tags("")
    []
    """
    if not param_details:
        return []
    return [t.strip().lower() for t in param_details.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

@dataclass
class BenchScenario:
    """A test scenario for model evaluation."""
    name: str
    description: str = ""
    prompt_template: str = ""
    evaluation_criteria: str = ""
    user_input: str = ""


# Backward-compatible alias
TestScenario = BenchScenario


@dataclass
class BenchMetrics:
    """Performance metrics from a single test run."""
    total_duration_sec: float = 0.0
    load_duration_sec: float = 0.0
    prompt_eval_duration_sec: float = 0.0
    eval_duration_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_rate_tps: float = 0.0
    eval_rate_tps: float = 0.0
    ttft_ms: float = 0.0
    self_score: float = 0.0


# Backward-compatible alias
TestMetrics = BenchMetrics


@dataclass
class BenchResult:
    """Result of a model test."""
    model: str
    scenario: str
    response: str = ""
    metrics: BenchMetrics = field(default_factory=BenchMetrics)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "scenario": self.scenario,
            "response": self.response,
            "total_duration_sec": self.metrics.total_duration_sec,
            "load_duration_sec": self.metrics.load_duration_sec,
            "prompt_eval_duration_sec": self.metrics.prompt_eval_duration_sec,
            "eval_duration_sec": self.metrics.eval_duration_sec,
            "prompt_tokens": self.metrics.prompt_tokens,
            "completion_tokens": self.metrics.completion_tokens,
            "prompt_eval_rate_tps": self.metrics.prompt_eval_rate_tps,
            "eval_rate_tps": self.metrics.eval_rate_tps,
            "ttft_ms": self.metrics.ttft_ms,
            "self_score": self.metrics.self_score,
        }
        if self.error:
            d["error"] = self.error
        return d


# Backward-compatible alias
TestResult = BenchResult


# ---------------------------------------------------------------------------
# External runner (vLLM / llama.cpp)
# ---------------------------------------------------------------------------

class RunnerBackend(str, Enum):
    """Supported external runner backends."""
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"


@dataclass
class RunnerConfig:
    """Configuration for running a model through an external backend."""
    backend: RunnerBackend = RunnerBackend.LLAMA_CPP
    model_path: str = ""
    model_name: str = ""
    host: str = "127.0.0.1"
    port: int = 8080
    gpu_layers: int = -1
    context_size: int = 4096
    threads: int = 0
    batch_size: int = 512
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: int = 0
    extra_args: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "host": self.host,
            "port": self.port,
            "gpu_layers": self.gpu_layers,
            "context_size": self.context_size,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "extra_args": self.extra_args,
        }


# ---------------------------------------------------------------------------
# Modelfile templates
# ---------------------------------------------------------------------------

MODELFILE_TEMPLATES: Dict[str, str] = {
    "qwen": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.8\n'
        'PARAMETER top_k 20\n'
        'PARAMETER repeat_penalty 1.05\n'
        'TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n"""\n'
        'SYSTEM "You are a helpful assistant."\n'
    ),
    "llama": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.8\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.1\n'
    ),
    "mistral": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.05\n'
    ),
    "gemma": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.95\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.0\n'
    ),
    "phi": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.1\n'
    ),
    "yi": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.05\n'
    ),
    "deepseek": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.05\n'
    ),
    "codellama": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.2\n'
        'PARAMETER top_p 0.95\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.05\n'
    ),
    "default": (
        'FROM {model_path}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER top_p 0.9\n'
        'PARAMETER top_k 40\n'
        'PARAMETER repeat_penalty 1.1\n'
    ),
}


# ---------------------------------------------------------------------------
# Default test scenarios
# ---------------------------------------------------------------------------

DEFAULT_TEST_SCENARIOS: List[TestScenario] = [
    TestScenario(
        name="Natural Language to Code",
        description="Convert natural language descriptions to code",
        prompt_template="Please write a Python function: {input}",
        evaluation_criteria="Code correctness, completeness, style",
        user_input="Write a function to find all prime numbers up to N using the Sieve of Eratosthenes",
    ),
    TestScenario(
        name="Translation",
        description="Chinese-English bidirectional translation",
        prompt_template="Please translate the following text:\n{input}",
        evaluation_criteria="Translation accuracy, fluency, naturalness",
        user_input="The quick brown fox jumps over the lazy dog.",
    ),
    TestScenario(
        name="Code Explanation",
        description="Explain code functionality in plain language",
        prompt_template="Please explain the following code:\n{input}",
        evaluation_criteria="Accuracy, clarity, completeness",
        user_input="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    ),
    TestScenario(
        name="Question Answering",
        description="Answer technical questions",
        prompt_template="{input}",
        evaluation_criteria="Accuracy, depth, clarity",
        user_input="What is the difference between a process and a thread?",
    ),
]
