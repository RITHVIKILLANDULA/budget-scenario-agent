"""Grab the UI screenshots that go in the README.

Drives a headless Chrome over the DevTools protocol. Chrome's plain
--screenshot flag freezes the clock, and Streamlit never finishes its
websocket handshake under a frozen clock, so you get a skeleton loader
instead of an app. Hence the long way round.

    .venv/bin/streamlit run app.py --server.port 8512 --server.headless true
    .venv/bin/python scripts/screenshot.py
"""

from __future__ import annotations

import base64
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from urllib.parse import quote

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "img"


def _targets() -> list[dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=5) as response:
        return json.load(response)


def _launch(profile: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
            "--no-first-run", "--force-color-profile=srgb", "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class Session:
    def __init__(self, url: str):
        from websockets.sync.client import connect

        self.socket = connect(url, max_size=64 * 1024 * 1024)
        self.counter = 0

    def call(self, method: str, **params):
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.counter:
                return message.get("result", {})

    def close(self):
        self.socket.close()


SCROLL_TO_HEADING = """
(() => {
  const heading = [...document.querySelectorAll('h3, h2')]
    .find(h => h.textContent.trim().startsWith(%s));
  if (!heading) return false;
  heading.scrollIntoView({block: 'start'});
  return true;
})()
"""


def capture(session: Session, url: str, path: pathlib.Path, width: int, height: int,
            settle: float = 8.0, ready_js: str | None = None,
            scroll_to: str | None = None) -> None:
    session.call(
        "Emulation.setDeviceMetricsOverride",
        width=width, height=height, deviceScaleFactor=2, mobile=False,
    )
    session.call("Emulation.setEmulatedMedia", media="screen",
                 features=[{"name": "prefers-color-scheme", "value": "light"}])
    session.call("Page.navigate", url=url)
    deadline = time.time() + 40
    while time.time() < deadline:
        time.sleep(1.0)
        check = session.call(
            "Runtime.evaluate",
            expression=ready_js or "!!document.querySelector('[data-testid=\"stMetric\"]')",
            returnByValue=True,
        )
        if check.get("result", {}).get("value"):
            break
    time.sleep(settle)
    if scroll_to:
        session.call(
            "Runtime.evaluate",
            expression=SCROLL_TO_HEADING % json.dumps(scroll_to),
            returnByValue=True,
        )
        time.sleep(1.5)
    shot = session.call("Page.captureScreenshot", format="png")
    path.write_bytes(base64.b64decode(shot["data"]))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


def main() -> int:
    if not pathlib.Path(CHROME).exists():
        print(f"no Chrome at {CHROME}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="bsa-chrome-")
    process = _launch(profile)
    try:
        for _ in range(30):
            try:
                targets = _targets()
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("chrome did not come up", file=sys.stderr)
            return 1

        session = Session(page["webSocketDebuggerUrl"])
        session.call("Page.enable")
        session.call("Runtime.enable")
        base = "http://localhost:8512"
        capture(
            session,
            base + "/?q=" + quote("what happens if we cut contractor spend 15% in Q3"),
            OUT / "ui_scenario.png", 1460, 1620,
        )
        capture(
            session,
            base + "/?q=" + quote("cut the Atlantis team by 10%"),
            OUT / "ui_rejected.png", 1460, 860,
            ready_js="!!document.querySelector('[data-testid=\"stAlertContainer\"]')",
        )
        capture(
            session,
            base
            + "/?q=" + quote("freeze travel for the year")
            + "&a=" + quote("what happens if we cut contractor spend 15% in Q3")
            + "&b=" + quote("move 30% of contractor spend into software"),
            OUT / "ui_compare.png", 1460, 1180,
            scroll_to="Compare two scenarios",
        )
        session.close()
    finally:
        process.terminate()
        shutil.rmtree(profile, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
