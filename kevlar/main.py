from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from kevlar.config import KevlarConfig

app = typer.Typer(
    name="kevlar",
    help="Local MLX inference server with Anthropic-compatible API",
    invoke_without_command=True,
    no_args_is_help=False,
)
cache_app = typer.Typer(help="Manage KV cache on disk")
app.add_typer(cache_app, name="cache")


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@app.command()
def serve(
    model: str = typer.Option(KevlarConfig.model_path, help="HuggingFace model ID or local path"),
    host: str = typer.Option(KevlarConfig.host),
    port: int = typer.Option(KevlarConfig.port),
    cache_dir: str = typer.Option(KevlarConfig.ssd_cache_dir, help="SSD cache directory"),
    max_cache_gb: float = typer.Option(KevlarConfig.ssd_cache_max_gb, help="Max SSD cache size in GB"),
    max_tokens: int = typer.Option(KevlarConfig.default_max_tokens, help="Default max generation tokens"),
    prefill_step_size: int = typer.Option(KevlarConfig.prefill_step_size, help="Tokens per prefill chunk"),
    no_normalize: bool = typer.Option(False, "--no-normalize", is_flag=True, help="Disable header normalization"),
    haiku_port: int = typer.Option(0, help="Port of haiku subprocess (0 = disabled)"),
):
    """Start the inference server."""
    from kevlar.cli.display import print_banner, print_ready, print_model_loading, console

    config = KevlarConfig(
        model_path=model,
        host=host,
        port=port,
        ssd_cache_dir=cache_dir,
        ssd_cache_max_gb=max_cache_gb,
        default_max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        enable_header_normalization=not no_normalize,
    )
    if haiku_port:
        config.haiku_port = haiku_port
        config.enable_haiku = True

    print_banner(config)

    from kevlar.api.app import create_app
    app = create_app(config)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning", timeout_keep_alive=300)


@app.command()
def status(
    host: str = typer.Option(KevlarConfig.host),
    port: int = typer.Option(KevlarConfig.port),
):
    """Check if the server is running."""
    import httpx
    from kevlar.cli.display import print_status, print_error

    url = f"http://{host}:{port}/v1/status"
    try:
        resp = httpx.get(url, timeout=5)
        resp.raise_for_status()
        print_status(resp.json())
    except httpx.ConnectError:
        print_error(f"No server running on {host}:{port}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        raise typer.Exit(1)


@cache_app.command("clear")
def cache_clear(
    cache_dir: str = typer.Option(KevlarConfig.ssd_cache_dir, help="SSD cache directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear the SSD KV cache."""
    from kevlar.cli.display import console

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        console.print("  No cache directory found.")
        return

    entries = list(cache_path.iterdir())
    if not entries:
        console.print("  Cache is already empty.")
        return

    total_bytes = sum(
        f.stat().st_size
        for entry in entries if entry.is_dir()
        for f in entry.rglob("*") if f.is_file()
    )
    total_mb = total_bytes / 1e6

    if not force:
        confirm = typer.confirm(f"  Delete {len(entries)} cache entries ({total_mb:.0f} MB)?")
        if not confirm:
            return

    shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    console.print(f"  Cleared {len(entries)} entries ({total_mb:.0f} MB)")


def _pick_model(console) -> str:
    from kevlar.menubar_models import load_models, add_model

    models = load_models()
    console.print()
    console.print("  [bold]Available models:[/bold]")
    for i, m in enumerate(models, 1):
        console.print(f"    [dim]{i}.[/dim] {m}")
    console.print(f"    [dim]{len(models) + 1}.[/dim] [dim]Add a new model...[/dim]")
    console.print()

    while True:
        choice = console.input("  Select model [1]: ").strip()
        if not choice:
            return models[0]
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            if idx == len(models) + 1:
                new_model = console.input("  HuggingFace model ID: ").strip()
                if new_model:
                    add_model(new_model)
                    return new_model
            console.print("  [red]Invalid choice[/red]")
        except ValueError:
            # typed a model ID directly
            if "/" in choice:
                add_model(choice)
                return choice
            console.print("  [red]Enter a number or a model ID[/red]")


@app.command()
def run(
    model: str = typer.Option("", help="HuggingFace model ID or local path (interactive if omitted)"),
    host: str = typer.Option(KevlarConfig.host),
    port: int = typer.Option(KevlarConfig.port),
    cache_dir: str = typer.Option(KevlarConfig.ssd_cache_dir, help="SSD cache directory"),
    max_cache_gb: float = typer.Option(KevlarConfig.ssd_cache_max_gb, help="Max SSD cache size in GB"),
    max_tokens: int = typer.Option(KevlarConfig.default_max_tokens, help="Default max generation tokens"),
    haiku_model: str = typer.Option(KevlarConfig.haiku_model_path, help="HuggingFace model for haiku/background tasks"),
    no_haiku: bool = typer.Option(False, "--no-haiku", is_flag=True, help="Disable haiku subprocess"),
):
    """Start server and launch Claude Code against it."""
    import os
    import subprocess
    import time

    import httpx
    from kevlar.cli.display import console

    base_url = f"http://{host}:{port}"
    haiku_port = port + 1

    server_proc = None
    haiku_proc = None
    already_running = False
    try:
        resp = httpx.get(f"{base_url}/v1/status", timeout=2)
        already_running = True
        data = resp.json()
        model = data.get("model", model or "local")
        console.print(f"  Server already running on {base_url} ({model})")
    except Exception:
        pass

    if not already_running:
        if not model:
            model = _pick_model(console)

        console.print(f"  Starting server with {model}...")

        haiku_port_arg = ["--haiku-port", str(haiku_port)] if not no_haiku else []
        server_proc = subprocess.Popen(
            [
                sys.executable, "-m", "kevlar.main", "serve",
                "--model", model,
                "--host", host,
                "--port", str(port),
                "--cache-dir", cache_dir,
                "--max-cache-gb", str(max_cache_gb),
                "--max-tokens", str(max_tokens),
            ] + haiku_port_arg,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not no_haiku:
            console.print(f"  Starting haiku model ({haiku_model}) on port {haiku_port}...")
            haiku_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "kevlar.main", "serve",
                    "--model", haiku_model,
                    "--host", host,
                    "--port", str(haiku_port),
                    "--cache-dir", cache_dir,
                    "--max-cache-gb", "2",
                    "--max-tokens", "8192",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # wait for main server
        for i in range(120):
            try:
                httpx.get(f"{base_url}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            console.print("  [red]Server failed to start after 120s[/red]")
            if server_proc:
                server_proc.kill()
            if haiku_proc:
                haiku_proc.kill()
            raise typer.Exit(1)

        console.print(f"  Server ready on {base_url}")

        # wait for haiku server
        if haiku_proc:
            haiku_base = f"http://{host}:{haiku_port}"
            for i in range(120):
                try:
                    httpx.get(f"{haiku_base}/health", timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                console.print("  [yellow]Haiku server failed to start, continuing without it[/yellow]")
                haiku_proc.kill()
                haiku_proc = None

            if haiku_proc:
                console.print(f"  Haiku server ready on {haiku_base}")

    console.print("  Launching Claude Code...\n")

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)

    try:
        result = subprocess.run(["claude", "--model", model], env=env)
    except KeyboardInterrupt:
        pass
    finally:
        if haiku_proc:
            haiku_proc.terminate()
            try:
                haiku_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                haiku_proc.kill()
        if server_proc and not already_running:
            console.print("\n  Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


def main():
    app()


if __name__ == "__main__":
    main()
