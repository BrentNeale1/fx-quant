"""
Lightweight threaded HTTP health server for Railway liveness checks.

Runs as a daemon thread so it never blocks the main trading engine loop.
"""
import json
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

_start_time = time.time()

STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "live_state.json",
)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "uptime_seconds": round(time.time() - _start_time, 1),
            })
            self._respond(200, body)
        elif self.path == "/state":
            try:
                with open(STATE_FILE) as f:
                    body = f.read()
            except FileNotFoundError:
                body = json.dumps({"error": "state file not found"})
            self._respond(200, body)
        else:
            self._respond(404, json.dumps({"error": "not found"}))

    def _respond(self, code: int, body: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        pass  # suppress request logs to keep engine output clean


def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Health server listening on :{port}")
