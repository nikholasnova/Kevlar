from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from kevlar.config import KevlarConfig
    from kevlar.engine.generator import GenerationStats

console = Console()


def print_banner(config: KevlarConfig):
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Model", config.model_path)
    table.add_row("Endpoint", f"http://{config.host}:{config.port}/v1/messages")
    table.add_row("Cache", f"{config.ssd_cache_dir} ({config.ssd_cache_max_gb:.0f} GB max)")
    table.add_row("Normalize", "on" if config.enable_header_normalization else "off")

    panel = Panel(table, title="[bold]Kevlar v0.1.0[/bold]", border_style="dim", expand=False)
    console.print()
    console.print(panel)
    console.print()


def print_ready(config: KevlarConfig):
    url = f"http://{config.host}:{config.port}"
    console.print(f"  [bold green]Ready.[/bold green] Set [bold]ANTHROPIC_BASE_URL={url}[/bold] to use with Claude Code.")
    console.print()


def print_model_loading():
    return console.status("[bold]Loading model...", spinner="dots")


def print_request_stats(stats: GenerationStats):
    cache_pct = (stats.cache_hit_tokens / stats.prompt_tokens * 100) if stats.prompt_tokens > 0 else 0

    header = (
        f"  [bold]POST /v1/messages[/bold]  "
        f"{stats.prompt_tokens} tok  "
        f"cache hit {stats.cache_hit_tokens}/{stats.prompt_tokens} ({cache_pct:.1f}%)"
    )
    console.print(header)

    prefill_tokens = stats.prompt_tokens - stats.cache_hit_tokens
    if prefill_tokens > 0:
        console.print(
            f"    prefill  {prefill_tokens:>5} tok  "
            f"{stats.prefill_time_s:>5.1f}s  "
            f"{stats.prefill_tps:>6.0f} tok/s",
            style="dim",
        )

    console.print(
        f"    decode   {stats.completion_tokens:>5} tok  "
        f"{stats.decode_time_s:>5.1f}s  "
        f"{stats.decode_tps:>6.0f} tok/s",
        style="dim",
    )


def print_status(data: dict):
    if data.get("status") != "ok":
        console.print("[red]Server returned unexpected status[/red]")
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Status", "[green]running[/green]")
    table.add_row("Model", data.get("model", "unknown"))

    uptime = data.get("uptime_s", 0)
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)
    table.add_row("Uptime", f"{hours}h {minutes}m {seconds}s")

    cache = data.get("cache", {})
    mem_entries = cache.get("memory_entries", 0)
    mem_mb = cache.get("memory_bytes", 0) / 1e6
    table.add_row("Memory cache", f"{mem_entries} entries ({mem_mb:.0f} MB)")

    ssd_entries = cache.get("ssd_entries", 0)
    ssd_dir = cache.get("ssd_dir", "")
    table.add_row("SSD cache", f"{ssd_entries} entries in {ssd_dir}")

    panel = Panel(table, title="[bold]Kevlar Status[/bold]", border_style="dim", expand=False)
    console.print()
    console.print(panel)
    console.print()


def print_error(msg: str):
    console.print(f"  [red]{msg}[/red]")
