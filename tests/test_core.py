"""
OllamaAid - Test suite
"""

import pytest
from ollama_aid.core.models import (
    ModelInfo, TestMetrics, TestResult, TestScenario,
    TrendData, RunnerConfig, RunnerBackend,
    MODELFILE_TEMPLATES, DEFAULT_TEST_SCENARIOS, ToolResult,
)


class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, data={"key": "val"})
        assert r.success is True
        assert r.data == {"key": "val"}
        assert r.error is None

    def test_failure(self):
        r = ToolResult(success=False, error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_to_dict(self):
        r = ToolResult(success=True, data=[1, 2], metadata={"k": "v"})
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"] == [1, 2]
        assert d["metadata"] == {"k": "v"}

    def test_to_dict_omits_none(self):
        r = ToolResult(success=True)
        d = r.to_dict()
        assert "data" not in d
        assert "error" not in d


class TestModelInfo:
    def test_full_name_auto(self):
        m = ModelInfo(name="llama3", tag="7b")
        assert m.full_name == "llama3:7b"

    def test_full_name_explicit(self):
        m = ModelInfo(name="llama3", tag="7b", full_name="custom:name")
        assert m.full_name == "custom:name"


class TestTrendData:
    def test_to_dict(self):
        t = TrendData(name="test", pulls=1000, tags=["chat", "code"])
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["pulls"] == 1000
        assert d["tags"] == ["chat", "code"]


class TestTestModels:
    def test_metrics_defaults(self):
        m = TestMetrics()
        assert m.total_duration_sec == 0.0
        assert m.eval_rate_tps == 0.0

    def test_result_to_dict(self):
        r = TestResult(model="m1", scenario="s1")
        d = r.to_dict()
        assert d["model"] == "m1"
        assert d["scenario"] == "s1"
        assert d["self_score"] == 0.0

    def test_default_scenarios_exist(self):
        assert len(DEFAULT_TEST_SCENARIOS) >= 4
        for s in DEFAULT_TEST_SCENARIOS:
            assert s.name
            assert s.prompt_template


class TestRunnerConfig:
    def test_defaults(self):
        cfg = RunnerConfig()
        assert cfg.backend == RunnerBackend.LLAMA_CPP
        assert cfg.port == 8080
        assert cfg.gpu_layers == -1

    def test_to_dict(self):
        cfg = RunnerConfig(backend=RunnerBackend.VLLM, port=9000)
        d = cfg.to_dict()
        assert d["backend"] == "vllm"
        assert d["port"] == 9000


class TestModelfileTemplates:
    def test_all_keys(self):
        expected = {"qwen", "llama", "mistral", "gemma", "phi", "yi", "deepseek", "codellama", "default"}
        assert expected.issubset(set(MODELFILE_TEMPLATES.keys()))

    def test_format_placeholder(self):
        for key, tmpl in MODELFILE_TEMPLATES.items():
            result = tmpl.format(model_path="/test/path.gguf")
            assert "FROM /test/path.gguf" in result


class TestI18n:
    def test_english(self):
        from ollama_aid.core.i18n import I18n
        i = I18n("en")
        assert "OllamaAid" in i.t("app_title")

    def test_chinese(self):
        from ollama_aid.core.i18n import I18n
        i = I18n("zh")
        assert "Ollama" in i.t("app_title")

    def test_format_args(self):
        from ollama_aid.core.i18n import I18n
        i = I18n("en")
        assert "5" in i.t("models_loaded", 5)

    def test_switch_language(self):
        from ollama_aid.core.i18n import I18n
        i = I18n("en")
        assert i.language == "en"
        i.set_language("zh")
        assert i.language == "zh"


class TestAPI:
    def test_imports(self):
        from ollama_aid import (
            ToolResult, list_models, export_model, import_model,
            delete_model, update_model, fetch_trends, test_model,
            run_with_backend,
        )
        assert callable(list_models)
        assert callable(fetch_trends)

    def test_tools_schema(self):
        from ollama_aid.api import TOOLS
        assert isinstance(TOOLS, list)
        assert len(TOOLS) >= 5
        for tool in TOOLS:
            assert tool["type"] == "function"
            assert "name" in tool["function"]

    def test_dispatch_unknown(self):
        from ollama_aid.api import dispatch
        result = dispatch("nonexistent_func", {})
        assert result["success"] is False
        assert "Unknown" in result["error"]


class TestTesterParsing:
    def test_parse_duration(self):
        from ollama_aid.core.tester import _parse_duration
        assert _parse_duration("2.5s") == pytest.approx(2.5)
        assert _parse_duration("500ms") == pytest.approx(0.5)
        assert _parse_duration("1m30s") == pytest.approx(90.0)
        assert _parse_duration("1h2m3s") == pytest.approx(3723.0)
        assert _parse_duration("") == 0.0


class TestConfig:
    def test_parse_size(self):
        from ollama_aid.core.manager import OllamaManager
        assert OllamaManager._parse_size_to_bytes("3.2 GB") == int(3.2 * 1024**3)
        assert OllamaManager._parse_size_to_bytes("500 MB") == int(500 * 1024**2)
        assert OllamaManager._parse_size_to_bytes("invalid") == 0

    def test_get_ollama_models_dir(self):
        from ollama_aid.core.config import get_ollama_models_dir
        d = get_ollama_models_dir()
        assert "models" in str(d).lower() or "ollama" in str(d).lower()
