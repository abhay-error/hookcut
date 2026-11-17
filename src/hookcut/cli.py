"""Command line: `hookcut clip <url>`, `hookcut serve`, `hookcut doctor`."""

from __future__ import annotations

import logging
import platform
import sys

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from . import __version__, config, ffmpeg, llm, pipeline
from .stages import transcribe

app = typer.Typer(add_completion=False, help="Turn long videos into short vertical clips, locally.")
console = Console()


@app.command()
def clip(
    target: str = typer.Argument(..., help="A YouTube (or other) URL, or a local video file."),
    clips: int = typer.Option(0, "--clips", "-n", help="How many clips (default from config)."),
    model: str = typer.Option("", "--model", "-m", help="LLM model, e.g. qwen3:14b."),
    think: bool = typer.Option(None, "--think/--no-think", help="Let the model reason first."),
    no_captions: bool = typer.Option(False, "--no-captions", help="Skip burned-in captions."),
    centre: bool = typer.Option(False, "--centre", help="Centre-crop; skip face tracking."),
    whisper: str = typer.Option("", "--whisper", help="Whisper model, e.g. small, large-v3."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Find the best moments in a video and render them as vertical clips."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = config.load()
    if clips:
        settings.clips.max_clips = clips
    if model:
        settings.llm.model = model
    if think is not None:
        settings.llm.think = think
    if no_captions:
        settings.render.captions = False
    if centre:
        settings.render.face_tracking = False
    if whisper:
        settings.whisper.model = whisper

    job = pipeline.new_job(target, settings)
    columns = (TextColumn("{task.description}"), BarColumn(),
               TextColumn("{task.percentage:>3.0f}%"))
    with Progress(*columns, console=console, transient=True) as bar:
        task = bar.add_task("Starting", total=1.0)

        def update(j: pipeline.Job) -> None:
            bar.update(task, completed=j.progress, description=j.message or j.stage)

        try:
            result = pipeline.Runner(job, on_update=update).run()
        except KeyboardInterrupt:
            console.print("[yellow]Stopped.[/yellow]")
            raise typer.Exit(130) from None

    if result.status == "failed":
        console.print(f"[red]Failed during {result.stage}:[/red] {result.error}")
        raise typer.Exit(1)
    if not result.clips:
        console.print(result.message)
        return

    out = pipeline.job_dir(result.id)
    table = Table(title=result.title, title_justify="left")
    for col in ("#", "Score", "Length", "Title", "Why"):
        table.add_column(col)
    for c in result.clips:
        h = c.highlight
        table.add_row(str(c.index), str(h.score), f"{h.duration:.0f}s", h.title, h.reason)
    console.print(table)
    console.print(f"Clips saved in [bold]{out / 'clips'}[/bold]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="127.0.0.1 keeps it private to this computer."),
    port: int = typer.Option(8765),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Start the web app."""
    import threading
    import webbrowser

    import uvicorn

    from .web.app import create_app

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(f"hookcut is running at [bold]{url}[/bold]  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def doctor() -> None:
    """Check that everything needed is installed."""
    settings = config.load()
    ok, bad = "[green]OK[/green]", "[red]MISSING[/red]"
    table = Table(show_header=False)
    table.add_row("hookcut", ok, __version__)
    py = sys.version.split()[0]
    table.add_row("Python", ok, f"{py} on {platform.system()} {platform.machine()}")

    problems: list[str] = []
    try:
        path = ffmpeg.ffmpeg()
        table.add_row("ffmpeg", ok, path)
        if ffmpeg.has_filter("ass"):
            table.add_row("Captions (libass)", ok, "")
        else:
            table.add_row("Captions (libass)", bad, "this ffmpeg can't burn in captions")
            problems.append("Install an ffmpeg with libass. macOS: `brew install ffmpeg-full` and "
                            "set HOOKCUT_FFMPEG=$(brew --prefix ffmpeg-full)/bin. "
                            "Windows/Linux: use a 'full' build from ffmpeg.org.")
        table.add_row("Video encoder", ok, ffmpeg.pick_encoder(settings.render.encoder))
    except ffmpeg.FFmpegError as exc:
        table.add_row("ffmpeg", bad, "")
        problems.append(str(exc))

    table.add_row("Transcription", ok, f"{transcribe.pick_backend(settings.whisper)} "
                                        f"({settings.whisper.model})")

    models = llm.list_models(settings.llm)
    if not models:
        table.add_row("LLM server", bad, settings.llm.base_url)
        problems.append("Start your model server (Ollama: `ollama serve`).")
    elif settings.llm.model not in models:
        table.add_row("LLM model", bad, f"{settings.llm.model} not installed")
        problems.append(f"Pull it: `ollama pull {settings.llm.model}` "
                        f"(installed: {', '.join(models[:5])})")
    else:
        table.add_row("LLM model", ok, settings.llm.model)

    console.print(table)
    if problems:
        console.print("\n[bold]To fix:[/bold]")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(1)
    console.print("\n[green]Ready to clip.[/green]")


@app.command()
def jobs() -> None:
    """List recent jobs."""
    table = Table()
    for col in ("ID", "Status", "Clips", "Title"):
        table.add_column(col)
    for j in pipeline.list_jobs()[:20]:
        table.add_row(j.id, j.status, str(len(j.clips)), j.title or j.target)
    console.print(table)


@app.command("config")
def show_config(
    set_: list[str] = typer.Option([], "--set", help="key=value, e.g. llm.model=qwen3:8b"),
) -> None:
    """Show settings, or change them with --set section.key=value."""
    settings = config.load()
    data = settings.model_dump()
    for item in set_:
        key, _, value = item.partition("=")
        section, _, field = key.partition(".")
        if section not in data or field not in data[section]:
            console.print(f"[red]Unknown setting {key}[/red]")
            raise typer.Exit(1)
        data[section][field] = value
    if set_:
        settings = config.Settings.model_validate(data)
        config.save(settings)
        console.print(f"Saved to {config.config_path()}")
    console.print_json(settings.model_dump_json())


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

