from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .report.generator import generate_site
from .server import create_app

app = typer.Typer(add_completion=False, help="Generate and serve a local bank-style FX morning report.")


@app.command()
def generate(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, help="Path to config.yaml"),
):
    """Generate HTML pages + charts under output.site_dir."""
    cfg = load_config(config)
    generate_site(cfg)
    typer.echo(f"Generated site at: {cfg.site_dir}")


@app.command()
def serve(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c", exists=True, help="Path to config.yaml"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Port"),
):
    """Serve the generated site locally with Flask."""
    cfg = load_config(config)
    site_dir = cfg.site_dir
    app_ = create_app(cfg)
    typer.echo(f"Serving {site_dir} with dynamic generation at http://{host}:{port}")
    app_.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    app()
