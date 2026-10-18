"""Render docs/architecture.html to docs/architecture.png (headless Chromium).

usage: python docs/render_architecture.py
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1640, "height": 1000}, device_scale_factor=1.5)
        page.goto((HERE / "architecture.html").as_uri())
        page.locator("#canvas").screenshot(path=str(HERE / "architecture.png"))
        browser.close()
    print("wrote docs/architecture.png")


if __name__ == "__main__":
    main()
