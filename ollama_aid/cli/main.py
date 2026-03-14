"""
OllamaAid - Command-line interface
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional


def _json_out(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _table_print(headers: list[str], rows: list[list[str]]) -> None:
    """Simple aligned table printer."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ======================================================================
# Sub-command handlers
# ======================================================================

def cmd_list(args):
    from ollama_aid.api import list_models
    result = list_models()
    if args.json_output:
        data = [
            {"name": m.full_name, "size": m.size, "modified": m.modified_date, "id": m.model_id}
            for m in (result.data or [])
        ]
        _json_out({"success": result.success, "models": data, "count": len(data)})
        return
    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)
    models = result.data or []
    if not models:
        print("No models found.")
        return
    rows = [[m.full_name, m.size, m.modified_date, m.model_id] for m in models]
    _table_print(["NAME", "SIZE", "MODIFIED", "ID"], rows)
    if not args.quiet:
        print(f"\nTotal: {len(models)} models")


def cmd_export(args):
    from ollama_aid.api import export_model
    result = export_model(args.model, args.output)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        print(f"Exported to: {result.data['model_file']}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_import(args):
    from ollama_aid.api import import_model
    result = import_model(args.file, args.name)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        print(f"Imported model: {result.data['model']}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_delete(args):
    from ollama_aid.api import delete_model
    if not args.yes:
        answer = input(f"Delete model '{args.model}'? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled.")
            return
    result = delete_model(args.model)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        print(f"Deleted: {args.model}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_update(args):
    from ollama_aid.api import update_model
    print(f"Pulling latest version of {args.model}...")
    result = update_model(args.model)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        print(f"Updated: {args.model}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_info(args):
    from ollama_aid.api import show_model_info
    result = show_model_info(args.model)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        for k, v in result.data.items():
            if k == "raw":
                continue
            print(f"  {k}: {v}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_trends(args):
    from ollama_aid.api import fetch_trends
    if not args.quiet:
        print("Fetching trends from ollama.com...")
    limit = getattr(args, "limit", 100)
    result = fetch_trends(limit=limit)
    if args.json_output:
        data = [t.to_dict() for t in (result.data or [])]
        _json_out({"success": result.success, "models": data})
        return
    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)
    trends = result.data or []
    if not trends:
        print("No trends data.")
        return

    def _fmt_pulls(p):
        if p >= 1e9:
            return f"{p / 1e9:.1f}B"
        if p >= 1e6:
            return f"{p / 1e6:.1f}M"
        if p >= 1e3:
            return f"{p / 1e3:.1f}K"
        return str(int(p))

    rows = [
        [t.name, _fmt_pulls(t.pulls), t.param_details or "-", ", ".join(t.tags) or "-", t.updated or "-"]
        for t in trends
    ]
    _table_print(["MODEL", "PULLS", "PARAMS", "TAGS", "UPDATED"], rows)
    if not args.quiet:
        print(f"\nTotal: {len(trends)} models")


def cmd_test(args):
    from ollama_aid.api import test_model, export_test_csv
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models specified.", file=sys.stderr)
        sys.exit(1)

    def _progress(msg):
        if not args.quiet:
            print(f"  {msg}")

    from ollama_aid.core.tester import run_tests
    result = run_tests(models, progress_cb=_progress)
    if args.json_output:
        data = [r.to_dict() for r in (result.data or [])]
        _json_out({"success": result.success, "results": data})
        return
    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)
    results = result.data or []
    rows = [
        [r.model, r.scenario, f"{r.metrics.self_score:.1f}",
         f"{r.metrics.eval_rate_tps:.1f}", f"{r.metrics.total_duration_sec:.2f}"]
        for r in results
    ]
    _table_print(["MODEL", "SCENARIO", "SCORE", "EVAL RATE (t/s)", "DURATION (s)"], rows)

    if args.output:
        export_test_csv(results, args.output)
        if not args.quiet:
            print(f"\nExported to: {args.output}")


def cmd_run(args):
    from ollama_aid.api import run_with_backend, stop_backend
    log = logging.getLogger("ollama_aid.cli")
    if args.stop:
        result = stop_backend()
        if result.success:
            print("Server stopped.")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        return

    extra = args.extra_args.split() if args.extra_args else []
    verbose = getattr(args, "verbose", False)
    print(f"Starting {args.backend} server for {args.model}...")
    log.debug("run_with_backend(model=%s, backend=%s, host=%s, port=%s)",
              args.model, args.backend, args.host, args.port)

    def _log_line(line):
        print(f"  [server] {line}")

    result = run_with_backend(
        model_name=args.model,
        backend=args.backend,
        host=args.host,
        port=args.port,
        gpu_layers=args.gpu_layers,
        context_size=args.ctx_size,
        threads=args.threads,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        extra_args=extra,
        log_cb=_log_line if verbose else None,
    )
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        d = result.data
        print(f"Server started (PID {d['pid']})")
        print(f"  Endpoint: http://{d['host']}:{d['port']}")
        print("Press Ctrl+C to stop...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_backend()
            print("\nServer stopped.")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


def cmd_resolve(args):
    from ollama_aid.api import resolve_model_path
    result = resolve_model_path(args.model)
    if args.json_output:
        _json_out(result.to_dict())
        return
    if result.success:
        print(f"Model: {result.data['model_name']}")
        print(f"Path:  {result.data['model_path']}")
        for k, v in result.data.items():
            if k not in ("model_name", "model_path"):
                print(f"  {k}: {v}")
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


# ======================================================================
# GUI launcher with subprocess isolation
# ======================================================================


def _make_gui_env(verbose: bool = False, force_xcb: bool = False) -> dict:
    """Build a sanitised environment dict for the GUI subprocess.

    Removes LD_LIBRARY_PATH / LD_PRELOAD entries that point to CUDA /
    NVIDIA / PyTorch / vLLM libraries so they cannot interfere with Qt.
    """
    import os
    import re

    env = os.environ.copy()

    # Force software OpenGL to avoid GPU driver conflicts
    env["QT_OPENGL"] = "software"

    if force_xcb:
        env["QT_QPA_PLATFORM"] = "xcb"

    if verbose:
        env["QT_DEBUG_PLUGINS"] = "1"
        env["QT_LOGGING_RULES"] = "*.debug=true"

    # Strip library-path entries that drag in CUDA / NVIDIA / torch libs
    _POISON = re.compile(
        r"(cuda|nvidia|nccl|cudnn|cupti|cublas|cufft|curand|cusolver|"
        r"cusparse|torch|pytorch|vllm)",
        re.IGNORECASE,
    )
    for var in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        val = env.get(var)
        if not val:
            continue
        cleaned = os.pathsep.join(
            p for p in val.split(os.pathsep) if not _POISON.search(p)
        )
        if cleaned:
            env[var] = cleaned
        else:
            env.pop(var, None)

    return env


def _run_gui_subprocess(env: dict, verbose: bool = False):
    """Spawn the GUI subprocess and wait for it.  Returns the exit code."""
    import signal
    import subprocess

    cmd = [sys.executable, "-u", "-m", "ollama_aid.gui.main"]

    venv = env.get("VIRTUAL_ENV") or env.get("CONDA_PREFIX")
    print("[ollama-aid] Launching GUI  (PID will follow)", file=sys.stderr)
    if venv:
        print(f"[ollama-aid] Active environment: {venv}", file=sys.stderr)
    platform = env.get("QT_QPA_PLATFORM", "(auto)")
    print(f"[ollama-aid] QT_QPA_PLATFORM={platform}", file=sys.stderr)
    print(f"[ollama-aid] Command: {' '.join(cmd)}", file=sys.stderr)
    sys.stderr.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
        )
    except FileNotFoundError:
        print(
            "Error: Python executable not found.\n"
            "Install PySide6 with:  pip install ollama-aid[gui]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[ollama-aid] GUI process started (PID {proc.pid})", file=sys.stderr)
    sys.stderr.flush()

    def _forward_signal(signum, _frame):
        try:
            proc.send_signal(signum)
        except OSError:
            pass

    prev_int = signal.signal(signal.SIGINT, _forward_signal)
    prev_term = signal.signal(signal.SIGTERM, _forward_signal)

    try:
        rc = proc.wait()
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    return rc


def _launch_gui(verbose: bool = False) -> None:
    """Launch the PySide6 GUI in a **subprocess**.

    Running the GUI in a separate process avoids C-level segfaults caused
    by shared-library conflicts when the CLI process has already loaded
    packages such as PyTorch/vLLM whose native libraries clash with Qt.

    If the first attempt crashes (SIGSEGV / SIGABRT), the launcher
    automatically retries once with ``QT_QPA_PLATFORM=xcb`` which avoids
    Wayland-specific driver issues on NVIDIA systems.
    """
    import signal

    env = _make_gui_env(verbose=verbose)
    rc = _run_gui_subprocess(env, verbose=verbose)

    # Normal exit
    if rc == 0:
        return

    # SIGTERM / SIGINT are normal shutdown (Ctrl-C, window manager close)
    if rc < 0 and -rc in (signal.SIGTERM, signal.SIGINT):
        sys.exit(0)

    # Crashed with SIGSEGV/SIGABRT – auto-retry with xcb if we haven't already
    is_crash = rc < 0 and -rc in (signal.SIGSEGV, signal.SIGABRT)
    already_xcb = env.get("QT_QPA_PLATFORM") == "xcb"

    if is_crash and not already_xcb:
        signame = _sig_name(-rc)
        print(
            f"\n[ollama-aid] GUI crashed ({signame}) on platform "
            f"'{env.get('QT_QPA_PLATFORM', 'auto')}'.",
            file=sys.stderr,
        )
        print(
            "[ollama-aid] Retrying with QT_QPA_PLATFORM=xcb ...\n",
            file=sys.stderr,
        )
        env2 = _make_gui_env(verbose=verbose, force_xcb=True)
        rc = _run_gui_subprocess(env2, verbose=verbose)
        if rc == 0:
            return
        if rc < 0 and -rc in (signal.SIGTERM, signal.SIGINT):
            sys.exit(0)

    # Still failing – print diagnostics
    if rc < 0:
        signame = _sig_name(-rc)
        print(
            f"\n{'=' * 60}\n"
            f"ERROR: GUI process killed by signal {-rc} ({signame}).\n"
            f"{'=' * 60}",
            file=sys.stderr,
        )
        if "SEGV" in signame or "ABRT" in signame:
            print(
                "\nThis is typically caused by native library conflicts\n"
                "(e.g. PyTorch/vLLM CUDA libs vs PySide6/Qt).\n"
                "\n--- Suggestions ---\n"
                "  1. Run outside the current virtualenv:\n"
                "       deactivate && ollama-aid --gui\n"
                "  2. Use the web dashboard instead:\n"
                "       ollama-aid --web\n"
                "  3. Install PySide6 in a clean venv:\n"
                "       python -m venv ~/gui-env && ~/gui-env/bin/pip install ollama-aid[gui]\n"
                "       ~/gui-env/bin/ollama-aid --gui",
                file=sys.stderr,
            )
        sys.exit(1)
    else:
        print(f"[ollama-aid] GUI exited with code {rc}", file=sys.stderr)
        sys.exit(rc)


def _sig_name(signum: int) -> str:
    import signal
    try:
        return signal.Signals(signum).name
    except (ValueError, AttributeError):
        return "unknown"


# ======================================================================
# Main entry
# ======================================================================

def main(argv: Optional[list[str]] = None) -> None:
    from ollama_aid.__version__ import __version__

    # Shared flags available on every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    common.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")
    common.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose / debug logging",
    )

    parser = argparse.ArgumentParser(
        prog="ollama-aid",
        description="OllamaAid - Ollama model management, trends & testing assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
        epilog=(
            "examples:\n"
            "  ollama-aid list                     List local models\n"
            "  ollama-aid trends                   Fetch model trends\n"
            "  ollama-aid test m1,m2 -o out.csv    Benchmark models\n"
            "  ollama-aid run qwen3:0.6b           Run via llama.cpp\n"
            "  ollama-aid --gui                    Launch PySide6 GUI\n"
            "  ollama-aid --web                    Launch Flask web dashboard\n"
            "  ollama-aid --web --port 8000        Web on custom port\n"
        ),
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--gui", action="store_true", help="Launch PySide6 GUI")
    parser.add_argument("--web", action="store_true", help="Launch Flask web dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host for --web (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port for --web (default: 5000)")

    sub = parser.add_subparsers(dest="command", title="commands")

    # --- list ---
    p_list = sub.add_parser("list", parents=[common], help="List local Ollama models")
    p_list.set_defaults(func=cmd_list)

    # --- export ---
    p_export = sub.add_parser("export", parents=[common], help="Export a model to GGUF + Modelfile")
    p_export.add_argument("model", help="Model name (e.g. llama3.2:3b)")
    p_export.add_argument("-o", "--output", required=True, help="Export directory")
    p_export.set_defaults(func=cmd_export)

    # --- import ---
    p_import = sub.add_parser("import", parents=[common], help="Import a GGUF file as Ollama model")
    p_import.add_argument("file", help="Path to GGUF file")
    p_import.add_argument("-n", "--name", required=True, help="New model name")
    p_import.set_defaults(func=cmd_import)

    # --- delete ---
    p_delete = sub.add_parser("delete", parents=[common], help="Delete an Ollama model")
    p_delete.add_argument("model", help="Model name")
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    p_delete.set_defaults(func=cmd_delete)

    # --- update ---
    p_update = sub.add_parser("update", parents=[common], help="Pull latest version of a model")
    p_update.add_argument("model", help="Model name")
    p_update.set_defaults(func=cmd_update)

    # --- info ---
    p_info = sub.add_parser("info", parents=[common], help="Show detailed model information")
    p_info.add_argument("model", help="Model name")
    p_info.set_defaults(func=cmd_info)

    # --- trends ---
    p_trends = sub.add_parser("trends", parents=[common], help="Fetch model trends from ollama.com")
    p_trends.add_argument("-l", "--limit", type=int, default=100, help="Max number of models (default: 100, -1 for all)")
    p_trends.set_defaults(func=cmd_trends)

    # --- test ---
    p_test = sub.add_parser("test", parents=[common], help="Run benchmark tests on models")
    p_test.add_argument("models", help="Comma-separated model names")
    p_test.add_argument("-o", "--output", help="Export results to CSV file")
    p_test.set_defaults(func=cmd_test)

    # --- run ---
    p_run = sub.add_parser("run", parents=[common], help="Run model via vLLM or llama.cpp")
    p_run.add_argument("model", nargs="?", default="", help="Model name")
    p_run.add_argument("-b", "--backend", default="llama.cpp", choices=["vllm", "llama.cpp"])
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8080)
    p_run.add_argument("--gpu-layers", type=int, default=-1)
    p_run.add_argument("--ctx-size", type=int, default=4096)
    p_run.add_argument("--threads", type=int, default=0)
    p_run.add_argument("--batch-size", type=int, default=512)
    p_run.add_argument("--tp-size", type=int, default=1)
    p_run.add_argument("--dtype", default="auto")
    p_run.add_argument("--max-model-len", type=int, default=0)
    p_run.add_argument("--extra-args", default="", help="Extra args as space-separated string")
    p_run.add_argument("--stop", action="store_true", help="Stop running server")
    p_run.set_defaults(func=cmd_run)

    # --- resolve ---
    p_resolve = sub.add_parser("resolve", parents=[common], help="Resolve Ollama model to disk path")
    p_resolve.add_argument("model", help="Model name")
    p_resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args(argv)

    # Configure logging level based on --verbose / -v
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="[%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # --gui: launch PySide6 GUI (no subcommand needed)
    if args.gui:
        _launch_gui(verbose=getattr(args, "verbose", False))
        return

    # --web: launch Flask web dashboard (no subcommand needed)
    if args.web:
        from ollama_aid.web.main import main as web_main
        web_main(host=args.host, port=args.port)
        return

    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
