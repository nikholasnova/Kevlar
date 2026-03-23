from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import rumps

from kevlar.config import KevlarConfig
from kevlar.menubar_models import add_model, load_models, remove_model

KEVLAR_EXECUTABLE = sys.executable


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


class KevlarApp(rumps.App):
    def __init__(self):
        super().__init__("K", quit_button=None)
        self.config = KevlarConfig()
        self.server_process = None
        self.current_model = None

        models = load_models()
        if models:
            self.current_model = models[0]

        self.menu = [
            rumps.MenuItem("Stopped"),
            rumps.MenuItem(""),
            None,
            rumps.MenuItem("Start Server", callback=self._start_server),
            rumps.MenuItem("Stop Server", callback=self._stop_server),
            None,
            self._build_model_submenu(),
            None,
            rumps.MenuItem("Copy API URL", callback=self._copy_url),
            rumps.MenuItem("Open Cache Dir", callback=self._open_cache_dir),
            rumps.MenuItem("Clear Cache", callback=self._clear_cache),
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

    def _build_model_submenu(self) -> rumps.MenuItem:
        sub = rumps.MenuItem("Switch Model")
        for m in load_models():
            label = _short_name(m)
            if m == self.current_model:
                label += "  [active]"
            sub.add(rumps.MenuItem(label, callback=lambda _, mid=m: self._switch_model(mid)))
        sub.add(rumps.separator)
        sub.add(rumps.MenuItem("Add Model...", callback=self._add_model))
        sub.add(rumps.MenuItem("Remove Model...", callback=self._remove_model))
        return sub

    def _rebuild_model_submenu(self):
        old = self.menu.get("Switch Model")
        if old:
            old.clear()
            for m in load_models():
                label = _short_name(m)
                if m == self.current_model:
                    label += "  [active]"
                old.add(rumps.MenuItem(label, callback=lambda _, mid=m: self._switch_model(mid)))
            old.add(rumps.separator)
            old.add(rumps.MenuItem("Add Model...", callback=self._add_model))
            old.add(rumps.MenuItem("Remove Model...", callback=self._remove_model))

    @rumps.timer(5)
    def _poll_status(self, _):
        try:
            status_item = self.menu.get("Stopped")
            stats_item = self.menu.get("")

            # Check if server is actually running via HTTP first
            server_running = False
            try:
                url = f"http://{self.config.host}:{self.config.port}/v1/status"
                resp = httpx.get(url, timeout=3)
                resp.raise_for_status()
                data = resp.json()
                server_running = True
            except Exception:
                pass

            if server_running:
                # Check if our tracked process is still alive
                if self.server_process is not None and self.server_process.poll() is not None:
                    self.server_process = None

                model_name = _short_name(data.get("model", "unknown"))
                if status_item:
                    status_item.title = f"Running - {model_name}"

                cache = data.get("cache", {})
                mem_entries = cache.get("memory_entries", 0)
                mem_mb = cache.get("memory_bytes", 0) / 1e6
                if stats_item:
                    stats_item.title = f"Cache: {mem_entries} entries ({mem_mb:.0f} MB)"
                return

            # No server detected via HTTP
            if self.server_process is not None:
                # We tracked a process but it's dead
                self.server_process = None
                if status_item:
                    status_item.title = "Stopped (crashed)"
                if stats_item:
                    stats_item.title = ""
                return

            if status_item:
                status_item.title = "Stopped"
            if stats_item:
                stats_item.title = ""
        except Exception:
            pass

    def _start_server(self, _):
        if self.server_process and self.server_process.poll() is None:
            rumps.alert("Server already running")
            return

        if not self.current_model:
            rumps.alert("No model selected")
            return

        cmd = [
            KEVLAR_EXECUTABLE, "-m", "kevlar.main", "serve",
            "--model", self.current_model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--cache-dir", self.config.ssd_cache_dir,
        ]

        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _stop_server(self, _):
        if self.server_process is None:
            return

        self.server_process.terminate()
        try:
            self.server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
        self.server_process = None

    def _switch_model(self, model_id: str):
        was_running = self.server_process and self.server_process.poll() is None
        if was_running:
            self._stop_server(None)

        self.current_model = model_id
        self._rebuild_model_submenu()

        if was_running:
            self._start_server(None)

    def _add_model(self, _):
        window = rumps.Window(
            message="Enter HuggingFace model ID:",
            title="Add Model",
            default_text="mlx-community/",
            ok="Add",
            cancel="Cancel",
            dimensions=(400, 24),
        )
        response = window.run()
        if response.clicked and response.text.strip():
            add_model(response.text.strip())
            self._rebuild_model_submenu()

    def _remove_model(self, _):
        models = load_models()
        if len(models) <= 1:
            rumps.alert("Need at least one model")
            return

        window = rumps.Window(
            message="Enter model name to remove:\n\n" + "\n".join(f"  {m}" for m in models),
            title="Remove Model",
            ok="Remove",
            cancel="Cancel",
            dimensions=(400, 24),
        )
        response = window.run()
        if response.clicked and response.text.strip():
            text = response.text.strip()
            match = next((m for m in models if text in m), None)
            if match:
                if match == self.current_model:
                    rumps.alert("Can't remove the active model. Switch first.")
                    return
                remove_model(match)
                self._rebuild_model_submenu()

    def _copy_url(self, _):
        url = f"http://{self.config.host}:{self.config.port}"
        subprocess.run(["pbcopy"], input=url.encode(), check=True)
        rumps.notification("Kevlar", "Copied to clipboard", url)

    def _open_cache_dir(self, _):
        cache_dir = self.config.ssd_cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", cache_dir])

    def _clear_cache(self, _):
        cache_path = Path(self.config.ssd_cache_dir)
        if not cache_path.exists() or not any(cache_path.iterdir()):
            rumps.notification("Kevlar", "Cache already empty", "")
            return

        resp = rumps.alert("Clear all cached KV states?", ok="Clear", cancel="Cancel")
        if resp == 1:
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            rumps.notification("Kevlar", "Cache cleared", "")

    def _quit(self, _):
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        rumps.quit_application()


def main():
    KevlarApp().run()


if __name__ == "__main__":
    main()
