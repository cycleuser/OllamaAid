"""
OllamaAid - PySide6 GUI
Tabbed interface combining model management, trends, testing and runner.
"""

from __future__ import annotations

import sys
import threading
from typing import List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QComboBox, QLineEdit, QMessageBox,
    QFileDialog, QTextEdit, QCheckBox, QGroupBox,
    QSpinBox, QSplitter, QInputDialog, QFormLayout,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont

from ollama_aid.__version__ import __version__, __app_name_cn__
from ollama_aid.core.i18n import I18n
from ollama_aid.core.models import (
    DEFAULT_TEST_SCENARIOS, ModelInfo, RunnerBackend,
    RunnerConfig, TestResult, TrendData,
)


# ======================================================================
# Worker threads
# ======================================================================

class WorkerThread(QThread):
    """Generic worker thread emitting (success, message, data)."""
    finished = Signal(bool, str, object)
    progress = Signal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            if hasattr(result, "success"):
                self.finished.emit(result.success, result.error or "", result.data)
            else:
                self.finished.emit(True, "", result)
        except Exception as exc:
            self.finished.emit(False, str(exc), None)


class TrendsWorkerThread(QThread):
    data_ready = Signal(list)
    error_occurred = Signal(str)
    progress_updated = Signal(int)

    def run(self):
        self.progress_updated.emit(10)
        from ollama_aid.core.trends import fetch_trends
        self.progress_updated.emit(50)
        result = fetch_trends()
        self.progress_updated.emit(90)
        if result.success:
            self.progress_updated.emit(100)
            self.data_ready.emit(result.data)
        else:
            self.error_occurred.emit(result.error or "Unknown error")


class TestWorkerThread(QThread):
    result_ready = Signal(object)
    log_message = Signal(str)
    test_finished = Signal()

    def __init__(self, models: list, scenarios: list):
        super().__init__()
        self.models = models
        self.scenarios = scenarios
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from ollama_aid.core.tester import run_tests
        result = run_tests(
            self.models, self.scenarios,
            progress_cb=lambda msg: self.log_message.emit(msg),
            stop_flag=lambda: self._stop,
        )
        if result.success and result.data:
            for r in result.data:
                self.result_ready.emit(r)
        self.test_finished.emit()


# ======================================================================
# Custom table items
# ======================================================================

class NumericItem(QTableWidgetItem):
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._val = value

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self._val < other._val
        return super().__lt__(other)


class TimeItem(QTableWidgetItem):
    def __init__(self, text: str):
        super().__init__(text)
        from ollama_aid.core.trends import _parse_time_to_days
        self._days = _parse_time_to_days(text)

    def __lt__(self, other):
        if isinstance(other, TimeItem):
            return self._days < other._days
        return super().__lt__(other)


# ======================================================================
# Tab panels
# ======================================================================

class ManagerTab(QWidget):
    """Model management tab."""

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self._models: list[ModelInfo] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_refresh = QPushButton(self.i18n.t("btn_refresh"))
        self.btn_refresh.clicked.connect(self.load_models)
        ctrl.addWidget(self.btn_refresh)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(self.i18n.t("search_placeholder"))
        self.search_box.textChanged.connect(self._filter)
        ctrl.addWidget(self.search_box)

        self.sort_combo = QComboBox()
        for key in ("sort_name_asc", "sort_name_desc", "sort_size_asc",
                     "sort_size_desc", "sort_date_asc", "sort_date_desc"):
            self.sort_combo.addItem(self.i18n.t(key), key)
        self.sort_combo.currentIndexChanged.connect(self._sort)
        ctrl.addWidget(self.sort_combo)
        layout.addLayout(ctrl)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            self.i18n.t("model_name"), self.i18n.t("model_tag"),
            self.i18n.t("model_id"), self.i18n.t("model_size"),
            self.i18n.t("model_modified"),
        ])
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Action buttons
        btn_layout = QHBoxLayout()
        for label_key, handler in [
            ("btn_export", self._export), ("btn_import", self._import),
            ("btn_delete", self._delete), ("btn_update", self._update),
        ]:
            btn = QPushButton(self.i18n.t(label_key))
            btn.clicked.connect(handler)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.status = QLabel(self.i18n.t("click_refresh"))
        layout.addWidget(self.status)

    def _selected_model(self) -> Optional[ModelInfo]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._models):
            QMessageBox.warning(self, self.i18n.t("warning"), self.i18n.t("no_model_selected"))
            return None
        return self._models[row]

    def load_models(self):
        self.btn_refresh.setEnabled(False)
        self.status.setText(self.i18n.t("loading"))
        from ollama_aid.core.manager import OllamaManager
        self._worker = WorkerThread(OllamaManager().list_models)
        self._worker.finished.connect(self._on_models_loaded)
        self._worker.start()

    def _on_models_loaded(self, success, error, data):
        self.btn_refresh.setEnabled(True)
        if not success:
            self.status.setText(self.i18n.t("operation_failed", error))
            return
        self._models = data or []
        self._populate()
        self.status.setText(self.i18n.t("models_loaded", len(self._models)))

    def _populate(self):
        self.table.setRowCount(len(self._models))
        for row, m in enumerate(self._models):
            self.table.setItem(row, 0, QTableWidgetItem(m.name))
            self.table.setItem(row, 1, QTableWidgetItem(m.tag))
            self.table.setItem(row, 2, QTableWidgetItem(m.model_id))
            self.table.setItem(row, 3, QTableWidgetItem(m.size))
            self.table.setItem(row, 4, QTableWidgetItem(m.modified_date))

    def _filter(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            match = any(
                text in (self.table.item(row, c).text().lower() if self.table.item(row, c) else "")
                for c in range(self.table.columnCount())
            )
            self.table.setRowHidden(row, not match)

    def _sort(self):
        key = self.sort_combo.currentData()
        if not key or not self._models:
            return
        if key == "sort_name_asc":
            self._models.sort(key=lambda m: m.full_name.lower())
        elif key == "sort_name_desc":
            self._models.sort(key=lambda m: m.full_name.lower(), reverse=True)
        elif key == "sort_size_asc":
            self._models.sort(key=lambda m: m.size_bytes)
        elif key == "sort_size_desc":
            self._models.sort(key=lambda m: m.size_bytes, reverse=True)
        elif key == "sort_date_asc":
            self._models.sort(key=lambda m: m.modified_date)
        elif key == "sort_date_desc":
            self._models.sort(key=lambda m: m.modified_date, reverse=True)
        self._populate()

    def _export(self):
        m = self._selected_model()
        if not m:
            return
        d = QFileDialog.getExistingDirectory(self, "Export Directory")
        if not d:
            return
        from ollama_aid.core.manager import OllamaManager
        self._worker = WorkerThread(OllamaManager().export_model, m.full_name, d)
        self._worker.finished.connect(
            lambda ok, err, _: self.status.setText(
                self.i18n.t("export_success", m.full_name) if ok
                else self.i18n.t("operation_failed", err)
            )
        )
        self._worker.start()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF File", "", "GGUF Files (*.gguf);;All (*)")
        if not path:
            return
        name, ok = QInputDialog.getText(self, "Model Name", "Enter name for the imported model:")
        if not ok or not name.strip():
            return
        from ollama_aid.core.manager import OllamaManager
        self._worker = WorkerThread(OllamaManager().import_model, path, name.strip())
        self._worker.finished.connect(
            lambda ok, err, _: (
                self.status.setText(self.i18n.t("import_success", name) if ok
                                    else self.i18n.t("operation_failed", err)),
                self.load_models() if ok else None,
            )
        )
        self._worker.start()

    def _delete(self):
        m = self._selected_model()
        if not m:
            return
        reply = QMessageBox.question(
            self, self.i18n.t("confirm"),
            self.i18n.t("delete_confirm", m.full_name),
        )
        if reply != QMessageBox.Yes:
            return
        from ollama_aid.core.manager import OllamaManager
        self._worker = WorkerThread(OllamaManager().delete_model, m.full_name)
        self._worker.finished.connect(
            lambda ok, err, _: (
                self.status.setText(self.i18n.t("delete_success", m.full_name) if ok
                                    else self.i18n.t("operation_failed", err)),
                self.load_models() if ok else None,
            )
        )
        self._worker.start()

    def _update(self):
        m = self._selected_model()
        if not m:
            return
        self.status.setText(f"Pulling {m.full_name}...")
        from ollama_aid.core.manager import OllamaManager
        self._worker = WorkerThread(OllamaManager().update_model, m.full_name)
        self._worker.finished.connect(
            lambda ok, err, _: (
                self.status.setText(self.i18n.t("update_success", m.full_name) if ok
                                    else self.i18n.t("operation_failed", err)),
                self.load_models() if ok else None,
            )
        )
        self._worker.start()


class TrendsTab(QWidget):
    """Trends viewer tab."""

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self._data: list[TrendData] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_refresh = QPushButton(self.i18n.t("refresh_data"))
        self.btn_refresh.clicked.connect(self.refresh)
        ctrl.addWidget(self.btn_refresh)
        self.status = QLabel(self.i18n.t("click_refresh"))
        ctrl.addWidget(self.status)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            self.i18n.t("model_name"), self.i18n.t("pulls"),
            self.i18n.t("min_params"), self.i18n.t("max_params"),
            self.i18n.t("param_details"), self.i18n.t("function_tags"),
            self.i18n.t("update_time"), self.i18n.t("access_link"),
        ])
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.table.horizontalHeader()
        for i in range(8):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(7, 200)
        layout.addWidget(self.table)

    def refresh(self):
        self.btn_refresh.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText(self.i18n.t("loading"))
        self._worker = TrendsWorkerThread()
        self._worker.data_ready.connect(self._on_data)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.progress_updated.connect(self.progress.setValue)
        self._worker.start()

    def _fmt_num(self, n: float) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(int(n)) if n else "0"

    def _fmt_params(self, p: float) -> str:
        if p >= 1000:
            return f"{p / 1000:.1f}B"
        if p >= 1:
            return f"{p:.1f}B"
        if p > 0:
            return f"{p * 1000:.0f}M"
        return self.i18n.t("unknown")

    def _on_data(self, data: list):
        self.btn_refresh.setEnabled(True)
        self.progress.setVisible(False)
        self._data = data
        self.table.setRowCount(len(data))
        for row, t in enumerate(data):
            self.table.setItem(row, 0, QTableWidgetItem(t.name))
            self.table.setItem(row, 1, NumericItem(self._fmt_num(t.pulls), t.pulls))
            self.table.setItem(row, 2, NumericItem(self._fmt_params(t.min_params), t.min_params))
            self.table.setItem(row, 3, NumericItem(self._fmt_params(t.max_params), t.max_params))
            self.table.setItem(row, 4, QTableWidgetItem(t.param_details or self.i18n.t("unknown")))
            self.table.setItem(row, 5, QTableWidgetItem(", ".join(t.tags) if t.tags else self.i18n.t("none")))
            self.table.setItem(row, 6, TimeItem(t.updated or self.i18n.t("unknown")))
            self.table.setItem(row, 7, QTableWidgetItem(t.url))
        self.status.setText(self.i18n.t("loaded_trends", len(data)))

    def _on_error(self, msg: str):
        self.btn_refresh.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText(self.i18n.t("error"))
        QMessageBox.warning(self, self.i18n.t("error"), msg)


class TesterTab(QWidget):
    """Model tester tab."""

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self._results: list[TestResult] = []
        self._worker: Optional[TestWorkerThread] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Model checkboxes
        self.model_group = QGroupBox(self.i18n.t("select_models"))
        self.model_layout = QVBoxLayout(self.model_group)
        self.btn_load_models = QPushButton(self.i18n.t("btn_refresh"))
        self.btn_load_models.clicked.connect(self._load_models)
        self.model_layout.addWidget(self.btn_load_models)
        left_layout.addWidget(self.model_group)

        # Scenario checkboxes
        self.scenario_group = QGroupBox(self.i18n.t("select_scenarios"))
        scenario_layout = QVBoxLayout(self.scenario_group)
        self._scenario_checks: list[QCheckBox] = []
        for sc in DEFAULT_TEST_SCENARIOS:
            cb = QCheckBox(sc.name)
            cb.setChecked(True)
            self._scenario_checks.append(cb)
            scenario_layout.addWidget(cb)
        left_layout.addWidget(self.scenario_group)

        # Buttons
        self.btn_start = QPushButton(self.i18n.t("btn_start_test"))
        self.btn_start.clicked.connect(self._start)
        left_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton(self.i18n.t("btn_stop_test"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        left_layout.addWidget(self.btn_stop)
        self.btn_csv = QPushButton(self.i18n.t("btn_export_csv"))
        self.btn_csv.clicked.connect(self._export_csv)
        left_layout.addWidget(self.btn_csv)
        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QTabWidget()
        # Results table
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels([
            self.i18n.t("col_model"), self.i18n.t("col_scenario"),
            self.i18n.t("col_score"), self.i18n.t("col_eval_rate"),
            self.i18n.t("col_total_dur"),
        ])
        self.result_table.setSortingEnabled(True)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right.addTab(self.result_table, self.i18n.t("tab_results"))

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right.addTab(self.log_text, self.i18n.t("tab_logs"))

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        self.status = QLabel("")
        layout.addWidget(self.status)

    def _load_models(self):
        from ollama_aid.core.manager import OllamaManager
        result = OllamaManager().list_models()
        # Clear old checkboxes (keep button)
        for cb in list(self._model_checks()):
            cb.setParent(None)
        if result.success and result.data:
            for m in result.data:
                cb = QCheckBox(m.full_name)
                self.model_layout.addWidget(cb)

    def _model_checks(self) -> list[QCheckBox]:
        checks = []
        for i in range(self.model_layout.count()):
            w = self.model_layout.itemAt(i).widget()
            if isinstance(w, QCheckBox):
                checks.append(w)
        return checks

    def _start(self):
        models = [cb.text() for cb in self._model_checks() if cb.isChecked()]
        if not models:
            QMessageBox.warning(self, self.i18n.t("warning"), self.i18n.t("no_model_selected"))
            return
        scenarios = [
            DEFAULT_TEST_SCENARIOS[i]
            for i, cb in enumerate(self._scenario_checks) if cb.isChecked()
        ]
        if not scenarios:
            return
        self._results.clear()
        self.result_table.setRowCount(0)
        self.log_text.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker = TestWorkerThread(models, scenarios)
        self._worker.result_ready.connect(self._on_result)
        self._worker.log_message.connect(lambda msg: self.log_text.append(msg))
        self._worker.test_finished.connect(self._on_finished)
        self._worker.start()

    def _stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_result(self, r: TestResult):
        self._results.append(r)
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_table.setItem(row, 0, QTableWidgetItem(r.model))
        self.result_table.setItem(row, 1, QTableWidgetItem(r.scenario))
        self.result_table.setItem(row, 2, NumericItem(f"{r.metrics.self_score:.1f}", r.metrics.self_score))
        self.result_table.setItem(row, 3, NumericItem(f"{r.metrics.eval_rate_tps:.1f}", r.metrics.eval_rate_tps))
        self.result_table.setItem(row, 4, NumericItem(f"{r.metrics.total_duration_sec:.2f}", r.metrics.total_duration_sec))
        self.status.setText(self.i18n.t("test_progress", r.model, r.scenario))

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText(self.i18n.t("test_complete", len(self._results)))

    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "results.csv", "CSV (*.csv)")
        if not path:
            return
        from ollama_aid.core.tester import export_results_csv
        export_results_csv(self._results, path)
        self.status.setText(f"Exported to {path}")


class RunnerTab(QWidget):
    """External runner (vLLM / llama.cpp) tab."""

    def __init__(self, i18n: I18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self._runner = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["llama.cpp", "vllm"])
        form.addRow(self.i18n.t("runner_backend"), self.backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        refresh_btn = QPushButton(self.i18n.t("btn_refresh"))
        refresh_btn.clicked.connect(self._load_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo)
        model_row.addWidget(refresh_btn)
        form.addRow(self.i18n.t("runner_model"), model_row)

        self.host_edit = QLineEdit("127.0.0.1")
        form.addRow(self.i18n.t("runner_host"), self.host_edit)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8080)
        form.addRow(self.i18n.t("runner_port"), self.port_spin)
        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(-1, 999)
        self.gpu_spin.setValue(-1)
        form.addRow(self.i18n.t("runner_gpu_layers"), self.gpu_spin)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(128, 131072)
        self.ctx_spin.setValue(4096)
        form.addRow(self.i18n.t("runner_ctx_size"), self.ctx_spin)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 256)
        self.threads_spin.setValue(0)
        form.addRow(self.i18n.t("runner_threads"), self.threads_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 8192)
        self.batch_spin.setValue(512)
        form.addRow(self.i18n.t("runner_batch_size"), self.batch_spin)
        self.tp_spin = QSpinBox()
        self.tp_spin.setRange(1, 16)
        self.tp_spin.setValue(1)
        form.addRow(self.i18n.t("runner_tp_size"), self.tp_spin)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["auto", "float16", "bfloat16", "float32"])
        form.addRow(self.i18n.t("runner_dtype"), self.dtype_combo)
        self.maxlen_spin = QSpinBox()
        self.maxlen_spin.setRange(0, 131072)
        self.maxlen_spin.setValue(0)
        form.addRow(self.i18n.t("runner_max_model_len"), self.maxlen_spin)
        self.extra_edit = QLineEdit()
        form.addRow(self.i18n.t("runner_extra_args"), self.extra_edit)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton(self.i18n.t("btn_start_server"))
        self.btn_start.clicked.connect(self._start)
        btn_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton(self.i18n.t("btn_stop_server"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.status = QLabel("")
        layout.addWidget(self.status)

    def _load_models(self):
        from ollama_aid.core.manager import OllamaManager
        result = OllamaManager().list_models()
        self.model_combo.clear()
        if result.success and result.data:
            for m in result.data:
                self.model_combo.addItem(m.full_name)

    def _start(self):
        from ollama_aid.core.runner import ExternalRunner
        from ollama_aid.core.models import RunnerBackend, RunnerConfig
        self._runner = ExternalRunner()
        backend_str = self.backend_combo.currentText()
        backend = RunnerBackend.LLAMA_CPP if "llama" in backend_str.lower() else RunnerBackend.VLLM
        extra = self.extra_edit.text().split() if self.extra_edit.text().strip() else []
        cfg = RunnerConfig(
            backend=backend,
            model_name=self.model_combo.currentText(),
            host=self.host_edit.text(),
            port=self.port_spin.value(),
            gpu_layers=self.gpu_spin.value(),
            context_size=self.ctx_spin.value(),
            threads=self.threads_spin.value(),
            batch_size=self.batch_spin.value(),
            tensor_parallel_size=self.tp_spin.value(),
            dtype=self.dtype_combo.currentText(),
            max_model_len=self.maxlen_spin.value(),
            extra_args=extra,
        )
        self.log_text.clear()
        result = self._runner.start(cfg, log_cb=lambda line: self.log_text.append(line))
        if result.success:
            d = result.data
            self.status.setText(self.i18n.t("server_running", d["host"], d["port"]))
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.status.setText(self.i18n.t("server_failed", result.error))

    def _stop(self):
        if self._runner:
            self._runner.stop()
        self.status.setText(self.i18n.t("server_stopped"))
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)


# ======================================================================
# Main window
# ======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.i18n = I18n("en")
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle(self.i18n.t("app_title"))
        self.setGeometry(100, 100, 1300, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Language selector
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        lang_label = QLabel(self.i18n.t("language") + ":")
        top_bar.addWidget(lang_label)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "中文"])
        self.lang_combo.currentTextChanged.connect(self._change_lang)
        top_bar.addWidget(self.lang_combo)
        layout.addLayout(top_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.manager_tab = ManagerTab(self.i18n)
        self.trends_tab = TrendsTab(self.i18n)
        self.tester_tab = TesterTab(self.i18n)
        self.runner_tab = RunnerTab(self.i18n)
        self.tabs.addTab(self.manager_tab, self.i18n.t("tab_manager"))
        self.tabs.addTab(self.trends_tab, self.i18n.t("tab_trends"))
        self.tabs.addTab(self.tester_tab, self.i18n.t("tab_tester"))
        self.tabs.addTab(self.runner_tab, self.i18n.t("tab_runner"))
        layout.addWidget(self.tabs)

        font = QFont()
        font.setPointSize(10)
        self.setFont(font)

    def _change_lang(self, text: str):
        lang = "zh" if text == "中文" else "en"
        self.i18n.set_language(lang)
        # Rebuild is simplest for full i18n refresh
        self.setWindowTitle(self.i18n.t("app_title"))
        for i, key in enumerate(("tab_manager", "tab_trends", "tab_tester", "tab_runner")):
            self.tabs.setTabText(i, self.i18n.t(key))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OllamaAid")
    app.setApplicationVersion(__version__)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
