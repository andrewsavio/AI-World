"""AI World background launcher: a tray icon that keeps the world running.

Starts server.py, opens the world in its own window (Edge or Chrome in app mode, with background throttling
switched off so the simulation keeps running when the window is minimized), and gives you a tray menu:
Open dashboard / Stop world. Needs: pip install pystray pillow
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request

import pystray
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).parent
URL = 'http://localhost:8000'
BROWSERS = [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe']
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

server = subprocess.Popen([sys.executable, str(ROOT / 'server.py'), '--no-browser'], cwd=ROOT, creationflags=NO_WINDOW)
window = None


def open_dashboard(*_):
    global window
    if window and window.poll() is None:
        return
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        return os.startfile(URL)
    profile = pathlib.Path(tempfile.gettempdir()) / 'ai-world-profile'  # own profile, so the flags always apply
    window = subprocess.Popen([browser, f'--app={URL}', f'--user-data-dir={profile}', '--disable-background-timer-throttling',
                               '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'])


def stop(icon, *_):
    if window and window.poll() is None:
        window.terminate()
    server.terminate()
    icon.stop()


def alive():
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except OSError:
        return False


def watch(icon):  # if the world is stopped from its own Stop button, the tray icon goes away too
    icon.visible = True
    for _ in range(50):
        if alive():
            break
        time.sleep(0.2)
    open_dashboard()
    while server.poll() is None:
        time.sleep(1)
    icon.stop()


image = Image.new('RGB', (64, 64), '#0d1117')
draw = ImageDraw.Draw(image)
draw.ellipse((8, 8, 56, 56), fill='#3fb950')
draw.ellipse((20, 14, 44, 38), fill='#58a6ff')
menu = pystray.Menu(pystray.MenuItem('Open dashboard', open_dashboard, default=True), pystray.MenuItem('Stop world', stop))
pystray.Icon('ai-world', image, 'AI World (running)', menu).run(setup=watch)
