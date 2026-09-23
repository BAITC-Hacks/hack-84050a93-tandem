"""Serve only this run's generated HTML on loopback; no directory/file browsing."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
record = json.loads((ROOT / "reports/final_acceptance/automatic_checks.json").read_text(encoding="utf-8"))
html = (ROOT / record["fresh_html"]).read_bytes()


class ReportOnly(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), ReportOnly)
print(f"http://127.0.0.1:{server.server_port}/", flush=True)
server.serve_forever()
