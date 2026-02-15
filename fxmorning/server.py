from __future__ import annotations

from pathlib import Path

from flask import Flask, send_from_directory


from .report.generator import generate_site
from .config import AppConfig

def create_app(cfg: AppConfig) -> Flask:
    site_dir = Path(cfg.site_dir).resolve()
    app = Flask(__name__, static_folder=str(site_dir / "static"), static_url_path="/static")

    def sync_site():
        print("Dynamic Request: Regenerating FX Morning Report...")
        generate_site(cfg)

    @app.route("/")
    def home():
        sync_site()
        return send_from_directory(site_dir, "index.html")

    @app.route("/daily")
    def daily():
        sync_site()
        return send_from_directory(site_dir, "daily.html")

    @app.route("/weekly")
    def weekly():
        sync_site()
        return send_from_directory(site_dir, "weekly.html")

    @app.route("/crypto")
    def crypto():
        sync_site()
        return send_from_directory(site_dir, "crypto.html")

    # also allow direct file names
    @app.route("/<path:filename>")
    def files(filename: str):
        # Only regenerate for .html files to avoid redundant fetches for static assets
        if filename.endswith(".html"):
            sync_site()
        return send_from_directory(site_dir, filename)

    return app
