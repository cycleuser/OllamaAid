"""
OllamaAid - Command-line interface
"""

from __future__ import annotations

import argparse
import json
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
    result = fetch_trends()
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
    if args.stop:
        result = stop_backend()
        if result.success:
            print("Server stopped.")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        return

    extra = args.extra_args.split() if args.extra_args else []
    print(f"Starting {args.backend} server for {args.model}...")
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
# Main entry
# ======================================================================

def main(argv: Optional[list[str]] = None) -> None:
    from ollama_aid.__version__ import __version__

    # Shared flags available on every subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    common.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")

    parser = argparse.ArgumentParser(
        prog="ollama-aid",
        description="OllamaAid - Unified Ollama model management, trends & testing tool",
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

    # --gui: launch PySide6 GUI (no subcommand needed)
    if args.gui:
        from ollama_aid.gui.main import main as gui_main
        gui_main()
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
