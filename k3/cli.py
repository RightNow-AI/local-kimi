"""``k3`` — the command line.

The startup paste block is not decoration. It is the moment somebody decides
whether this project is worth sharing, so it prints the exact lines you need for
the client you are about to run and nothing else.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, presets as presets_mod
from .presets import Preset
from .server import ServerConfig, create_app
from .toolcalls import parser_names
from .upstream import MockUpstream, Upstream, UpstreamConfig

app = typer.Typer(
    name="k3",
    help="Client presets for the K3 inference engine.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DIM = "dim"
ACCENT = "bold cyan"


def detect_shell() -> str:
    """Guess the shell so the paste block is actually pasteable."""
    if os.environ.get("SHELL"):
        return "posix"
    if os.name == "nt":
        return "powershell" if os.environ.get("PSModulePath") else "cmd"
    return "posix"


def _print_setup(preset: Preset, cfg: ServerConfig, shell: str) -> None:
    console.print(f"  [bold]{preset.title}[/bold]", highlight=False)
    console.print()
    token = cfg.auth_token or "local"
    lines = preset.setup.render_env(cfg.base_url, token, cfg.upstream.model, shell)
    for line in lines:
        # escape(): a TOML table header like [model_providers.k3] is valid rich
        # markup, and rich would eat the whole line. This block exists to be
        # pasted verbatim, so nothing in it may be interpreted.
        console.print(f"    [green]{escape(line)}[/green]", highlight=False)
    if preset.setup.launch:
        launch = preset.setup.launch.format(model=cfg.upstream.model)
        console.print(f"    [green]{escape(launch)}[/green]", highlight=False)
    if preset.setup.config_file:
        console.print()
        console.print(f"    [{DIM}]{escape(preset.setup.config_file)}[/{DIM}]", highlight=False)
        body = preset.setup.render_config(cfg.base_url, token, cfg.upstream.model)
        for line in body.rstrip().splitlines():
            console.print(f"      [green]{escape(line)}[/green]", highlight=False)
    console.print()


def print_banner(cfg: ServerConfig, shell: str) -> None:
    console.print()
    console.print(f"  [{ACCENT}]k3[/{ACCENT}] {__version__}   serving on [bold]{cfg.base_url}[/bold]", highlight=False)
    engine = "mock engine (no GPU needed)" if cfg.mock else cfg.upstream.base_url
    console.print(f"  [{DIM}]engine[/{DIM}]  {engine}   [{DIM}]model[/{DIM}] {cfg.upstream.model}", highlight=False)
    active = cfg.forced_client or f"auto-detect ({len(presets_mod.all_presets())} presets)"
    console.print(f"  [{DIM}]client[/{DIM}]  {active}", highlight=False)
    if cfg.record_dir:
        console.print(f"  [{DIM}]record[/{DIM}]  {cfg.record_dir}", highlight=False)
    console.print()

    if cfg.forced_client:
        _print_setup(presets_mod.get(cfg.forced_client), cfg, shell)
    else:
        for name in ("claude-code", "openai"):
            _print_setup(presets_mod.get(name), cfg, shell)
        console.print(f"  [{DIM}]other clients: k3 presets  ·  pin one with --client NAME[/{DIM}]")
        console.print()


@app.command()
def serve(
    client: Optional[str] = typer.Option(
        None, "--client", "-c", help="Pin a preset instead of auto-detecting per request."
    ),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port."),
    upstream: str = typer.Option(
        "http://127.0.0.1:8000/v1", help="K3 engine base URL (OpenAI-compatible)."
    ),
    upstream_api_key: Optional[str] = typer.Option(None, help="Bearer token for the engine."),
    model: str = typer.Option("k3", help="Model id sent to the engine."),
    small_model: Optional[str] = typer.Option(
        None, help="Model for clients that route background work to a cheaper model."
    ),
    api_key: Optional[str] = typer.Option(
        None, help="Require this token from clients. Omit to leave the server open."
    ),
    reasoning_field: str = typer.Option(
        "reasoning_content",
        help="Field the engine carries reasoning in: reasoning_content | inline | none.",
    ),
    mock: bool = typer.Option(False, "--mock", help="Run without an engine, for demos and tests."),
    cors_origin: list[str] = typer.Option(
        [],
        "--cors-origin",
        help="Allow browser calls from this origin. Repeatable. Off by default.",
    ),
    record: Optional[Path] = typer.Option(
        None, "--record", help="Write a cassette per request to this directory."
    ),
    record_compress: bool = typer.Option(
        False, "--record-compress", help="gzip cassettes; real client traffic is large."
    ),
    strict_routes: bool = typer.Option(
        False, help="With --client, 404 routes that client does not use."
    ),
    public_url: Optional[str] = typer.Option(None, help="URL to advertise in the paste block."),
    shell: Optional[str] = typer.Option(
        None, help="Paste-block syntax: posix | powershell | cmd. Default: guess."
    ),
    log_level: str = typer.Option("info", help="uvicorn/k3 log level."),
) -> None:
    """Serve K3 to any supported client."""
    if client:
        try:
            presets_mod.get(client)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)

    problems = presets_mod.validate()
    if problems:
        for problem in problems:
            console.print(f"[red]preset error:[/red] {problem}")
        raise typer.Exit(2)

    cfg = ServerConfig(
        host=host,
        port=port,
        forced_client=client,
        auth_token=api_key,
        record_dir=str(record) if record else None,
        record_compress=record_compress,
        strict_routes=strict_routes,
        mock=mock,
        cors_origins=list(cors_origin),
        public_url=public_url,
        upstream=UpstreamConfig(
            base_url=upstream,
            api_key=upstream_api_key,
            model=model,
            small_model=small_model,
            reasoning_field=reasoning_field,
        ),
    )

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
    )

    print_banner(cfg, shell or detect_shell())

    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level=log_level, access_log=False)


@app.command("presets")
def list_presets(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include notes and defaults."),
) -> None:
    """List the presets and what each one bundles."""
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("preset")
    table.add_column("status")
    table.add_column("dialect")
    table.add_column("tools")
    table.add_column("reasoning")
    table.add_column("routes")
    for preset in presets_mod.all_presets():
        status = "[green]stable[/green]" if preset.status == "stable" else f"[yellow]{preset.status}[/yellow]"
        table.add_row(
            preset.name,
            status,
            preset.dialect,
            preset.tool_parser,
            preset.reasoning.value,
            " ".join(preset.routes),
        )
    console.print()
    console.print(table)
    console.print()
    if verbose:
        for preset in presets_mod.all_presets():
            console.print(f"  [bold]{preset.name}[/bold] — {preset.title}")
            if preset.notes:
                console.print(f"    [{DIM}]{preset.notes}[/{DIM}]")
            d = preset.defaults
            console.print(
                f"    [{DIM}]defaults: max_tokens={d.max_tokens} temperature={d.temperature} "
                f"effort={d.reasoning_effort}[/{DIM}]"
            )
            console.print()


@app.command()
def detect(
    path: str = typer.Option("/v1/messages", help="Request path."),
    header: list[str] = typer.Option([], "--header", "-H", help="Repeatable 'Name: value'."),
) -> None:
    """Show which preset a request would resolve to, and why."""
    from .detect import detect as detect_fn

    headers: dict[str, str] = {}
    for item in header:
        if ":" not in item:
            console.print(f"[red]bad header {item!r}; expected 'Name: value'[/red]")
            raise typer.Exit(2)
        name, _, value = item.partition(":")
        headers[name.strip().lower()] = value.strip()

    result = detect_fn(path, headers, None)
    console.print()
    console.print(f"  preset   [bold]{result.preset.name}[/bold]  ({result.preset.title})")
    console.print(f"  why      {result.reason}")
    console.print(f"  score    {result.score}{'  (route fallback)' if result.fallback else ''}")
    console.print(f"  dialect  {result.preset.dialect}")
    console.print()


@app.command()
def doctor(
    upstream: str = typer.Option("http://127.0.0.1:8000/v1", help="K3 engine base URL."),
    upstream_api_key: Optional[str] = typer.Option(None),
    mock: bool = typer.Option(False, "--mock"),
) -> None:
    """Check presets, parsers, and whether the engine is reachable."""
    console.print()
    problems = presets_mod.validate()
    if problems:
        for problem in problems:
            console.print(f"  [red]x[/red] {problem}")
    else:
        console.print(f"  [green]ok[/green] {len(presets_mod.all_presets())} presets validate")
    console.print(f"  [green]ok[/green] tool parsers: {', '.join(parser_names())}")

    cfg = UpstreamConfig(base_url=upstream, api_key=upstream_api_key)
    engine = MockUpstream(cfg) if mock else Upstream(cfg)

    async def check() -> tuple[bool, str]:
        try:
            return await engine.health()
        finally:
            await engine.aclose()

    ok, detail = asyncio.run(check())
    if ok:
        console.print(f"  [green]ok[/green] engine reachable at {upstream} ({detail})")
    else:
        console.print(f"  [red]x[/red] engine unreachable at {upstream}: {detail}")
        console.print(f"    [{DIM}]start the engine, or run `k3 serve --mock` to try the presets without one[/{DIM}]")
    console.print()
    raise typer.Exit(0 if ok and not problems else 1)


@app.command()
def replay(
    directory: Path = typer.Argument(..., help="Directory of cassettes from `k3 serve --record`."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every diff."),
) -> None:
    """Replay recorded client traffic and report what changed."""
    from .record import load_cassettes
    from .replay import replay_all

    cassettes = load_cassettes(directory)
    if not cassettes:
        console.print(f"[yellow]no cassettes in {directory}[/yellow]")
        raise typer.Exit(1)

    results = asyncio.run(replay_all(cassettes))
    failures = [r for r in results if not r.ok]

    console.print()
    for result in results:
        mark = "[green]pass[/green]" if result.ok else "[red]FAIL[/red]"
        console.print(f"  {mark}  {result.name}", highlight=False)
        if not result.ok or verbose:
            for diff in result.diffs[:10]:
                console.print(f"          [{DIM}]{diff}[/{DIM}]", highlight=False)
    console.print()
    console.print(f"  {len(results) - len(failures)}/{len(results)} cassettes match")
    console.print()
    raise typer.Exit(1 if failures else 0)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
