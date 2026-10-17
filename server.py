"""AI World server.

Serves the world on http://localhost:8000 and hosts the students' neural brains (brain.py, PyTorch).
Local-first: it listens on this computer only (127.0.0.1) and accepts writes only from its own page.
To let a copy of the page hosted elsewhere talk to a server you run, set AIWORLD_ORIGINS=https://your-site (comma separated)
and, if it must be reachable from other machines, AIWORLD_HOST=0.0.0.0. There is one world per server, so it is meant for one viewer.
Python standard library only (plus PyTorch inside brain.py).
"""
import json
import os
import pathlib
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).parent
HOST = os.environ.get('AIWORLD_HOST', '127.0.0.1')
PORT = int(os.environ.get('AIWORLD_PORT', 8000))  # e.g. to try a second world without stopping the first
HOME = f'http://localhost:{PORT}'
ALLOWED = {HOME, f'http://127.0.0.1:{PORT}', *filter(None, os.environ.get('AIWORLD_ORIGINS', '').split(','))}
MAX_BYTES = 5_000_000
brain = None  # the PyTorch brains; loaded in the background so the page opens at once
BRAIN_ERROR = None
STOPPING = False  # set by the tray launcher's "Stop world": the page sees it, saves the day, then shuts the server down


def load_brain():
    global brain, BRAIN_ERROR
    try:
        import brain as module
        brain = module
    except Exception as e:  # PyTorch missing: the page shows a banner explaining what to install
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
            info = brain.handle('/brain/info', {}) if brain else {'engine': None, 'loading': BRAIN_ERROR is None, 'error': BRAIN_ERROR}
            return self.reply(200, json.dumps({**info, 'stopping': STOPPING}).encode())
        if url.path == '/brain/view' and brain:
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                return self.reply(200, json.dumps(brain.handle('/brain/view', {'id': int(q['id']), 'skill': int(q['skill']), 'px': q['px'], 'py': q['py']})).encode())
            except (KeyError, ValueError):
                return self.reply(404, b'{"error": "unknown brain"}')
        super().do_GET()

    def do_POST(self):
        global STOPPING
        # Only the world's own page may change anything (browsers always send Origin on cross-site posts). curl sends none.
        if self.headers.get('Origin') not in (None, *ALLOWED):
            return self.reply(403, b'{"error": "forbidden"}')
        if self.path == '/world/stop':  # polite stop: ask the page to save first (it then calls /shutdown)
            STOPPING = True
            return self.reply(200, b'{"ok": true}')
        if self.path == '/shutdown':
            if self.headers.get('Origin') != HOME:
                return self.reply(403, b'{"error": "forbidden"}')
            self.reply(200, b'{"ok": true}')
            return threading.Thread(target=self.server.shutdown).start()
        if self.path.startswith(('/brain/', '/policy/', '/world/')):
            if not brain:
                return self.reply(503, b'{"error": "brains not loaded"}')
            size = int(self.headers.get('Content-Length', 0))
            if size > MAX_BYTES:
                return self.reply(413, b'{"error": "too big"}')
            try:
                return self.reply(200, json.dumps(brain.handle(self.path, json.loads(self.rfile.read(size)))).encode())
            except (KeyError, ValueError, IndexError, TypeError) as e:
                return self.reply(400, json.dumps({'error': f'{type(e).__name__}: {e}'}).encode())
        self.reply(404, b'{"error": "not found"}')

    def end_headers(self):  # always check for a newer page: a stale cached index.html would run yesterday's world
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def reply(self, code, body):
        self.send_response(code)
        if self.headers.get('Origin') in ALLOWED:  # lets an allowed hosted copy of the page read the answers
            self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
        if code >= 400:  # an error may leave an unread request body, so don't reuse this connection
            self.close_connection = True
            self.send_header('Connection', 'close')
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if sys.stderr is not None and '/brain/' not in self.path and '/policy/' not in self.path:  # under pythonw there is no console to log to
            super().log_message(fmt, *args)


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind the same port and silently split the page's requests between two
    # sets of brains; refusing the bind makes a second start fail loudly instead.
    allow_reuse_address = os.name != 'nt'


if __name__ == '__main__':
    print(f'AI World running at {HOME}  (close this window or press Ctrl+C to stop)')
    if '--no-browser' not in sys.argv:
        threading.Timer(1, webbrowser.open, [HOME]).start()
    Server((HOST, PORT), Handler).serve_forever()
