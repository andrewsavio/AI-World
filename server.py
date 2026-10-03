"""AI World server.

Serves the world on http://localhost:8000 (Ollama only talks to pages served from localhost), keeps the
notebooks (professor-memory.json, students-memory.json) because a web page can't write files by itself, and hosts the
students' neural brains (brain.py, PyTorch). Listens on this computer only (127.0.0.1).
"""
import json
import pathlib
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).parent
NOTEBOOKS = {'/memory': ROOT / 'professor-memory.json', '/memory/students': ROOT / 'students-memory.json'}
MAX_BYTES = 5_000_000
brain = None  # the PyTorch brains; loaded in the background so the page opens at once
BRAIN_ERROR = None


def load_brain():
    global brain, BRAIN_ERROR
    try:
        import brain as module
        brain = module
    except Exception as e:  # PyTorch missing: the world falls back to the brains written in JavaScript
        BRAIN_ERROR = str(e)


threading.Thread(target=load_brain, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'  # keep-alive: the page makes many small calls, so reuse one connection instead of opening a new one each time
    timeout = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/brain/info':
            return self.reply(200, json.dumps(brain.handle('/brain/info', {}) if brain else {'engine': None, 'loading': BRAIN_ERROR is None, 'error': BRAIN_ERROR}).encode())
        if url.path == '/brain/view' and brain:
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                return self.reply(200, json.dumps(brain.handle('/brain/view', {'id': int(q['id']), 'skill': int(q['skill']), 'px': q['px'], 'py': q['py']})).encode())
            except (KeyError, ValueError):
                return self.reply(404, b'{"error": "unknown brain"}')
        if self.path in NOTEBOOKS:
            file = NOTEBOOKS[self.path]
            return self.reply(200, file.read_bytes() if file.exists() else b'{}')
        super().do_GET()

    def do_POST(self):
        # Only the world's own page may change anything (browsers always send Origin on cross-site posts). curl sends none.
        if self.headers.get('Origin') not in (None, 'http://localhost:8000'):
            return self.reply(403, b'{"error": "forbidden"}')
        if self.path == '/shutdown':
            if self.headers.get('Origin') != 'http://localhost:8000':
                return self.reply(403, b'{"error": "forbidden"}')
            self.reply(200, b'{"ok": true}')
            return threading.Thread(target=self.server.shutdown).start()
        if self.path.startswith(('/brain/', '/policy/')):
            if not brain:
                return self.reply(503, b'{"error": "brains not loaded"}')
            size = int(self.headers.get('Content-Length', 0))
            if size > MAX_BYTES:
                return self.reply(413, b'{"error": "too big"}')
            try:
                return self.reply(200, json.dumps(brain.handle(self.path, json.loads(self.rfile.read(size)))).encode())
            except (KeyError, ValueError, IndexError, TypeError) as e:
                return self.reply(400, json.dumps({'error': f'{type(e).__name__}: {e}'}).encode())
        if self.path not in NOTEBOOKS:
            return self.reply(404, b'{"error": "not found"}')
        size = int(self.headers.get('Content-Length', 0))
        if size > MAX_BYTES:
            return self.reply(413, b'{"error": "notebook too big"}')
        body = self.rfile.read(size)
        try:
            json.loads(body)
        except ValueError:
            return self.reply(400, b'{"error": "not JSON"}')
        file = NOTEBOOKS[self.path]
        tmp = file.with_suffix('.tmp')  # write-then-rename so a crash never leaves half a notebook
        tmp.write_bytes(body)
        tmp.replace(file)
        self.reply(200, b'{"ok": true}')

    def reply(self, code, body):
        self.send_response(code)
        if code >= 400:  # an error may leave an unread request body, so don't reuse this connection
            self.close_connection = True
            self.send_header('Connection', 'close')
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if sys.stderr is not None and '/memory' not in self.path:  # under pythonw (background) there is no console to log to
            super().log_message(fmt, *args)


if __name__ == '__main__':
    print('AI World running at http://localhost:8000  (close this window or press Ctrl+C to stop)')
    if '--no-browser' not in sys.argv:
        threading.Timer(1, webbrowser.open, ['http://localhost:8000']).start()
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
