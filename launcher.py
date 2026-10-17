"""AI World background launcher: a tray icon that keeps the world running.

Starts server.py, opens the world in its own window (Edge or Chrome in app mode, with background throttling
switched off so the simulation keeps running when the window is minimized), and gives you a tray menu:
Open dashboard / Stop world. While it runs, Windows is asked not to go to sleep (the screen may still turn off), so the
world can live for days; "Stop world" lets the page save the day first. With --quiet (used when Windows starts it at
login) the world window opens minimized: the students keep living (the simulation runs on its own timer) and nothing is drawn
until you click the tray icon. Needs: pip install pystray pillow
"""
import ctypes
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

if sys.platform == 'win32':  # keep the PC awake while the world lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released when this process ends
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

server = None  # started below, unless a world is already running
window = None


def world_windows():
    """Handles of the visible world windows (title 'AI World Tycoon')."""
    user32, found = ctypes.windll.user32, []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def check(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, title, n + 1)
            if title.value == 'AI World Tycoon':
                found.append(hwnd)
        return True
    user32.EnumWindows(check, 0)
    return found


def close_world_windows():
    for hwnd in world_windows():
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE


# Edge/Chrome often hand a new window to a browser process that is already running and exit at once, so the window is found by
# its title rather than through the process this launcher started.
def open_dashboard(*_):
    global window
    if world_windows():  # already open (perhaps minimized): bring it back
        for hwnd in world_windows():
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        return
    browser = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not browser:
        return os.startfile(URL)
    profile = pathlib.Path(tempfile.gettempdir()) / 'ai-world-profile'  # own profile, so the flags always apply
    fresh = f'{URL}/?v={int((ROOT / "index.html").stat().st_mtime)}'  # a new address whenever the page changes, so no old cached copy is used
    window = subprocess.Popen([browser, f'--app={fresh}', f'--user-data-dir={profile}', '--disable-background-timer-throttling',
                               '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'])


def stop(icon, *_):
    try:  # ask the page to save, then it shuts the server down itself
        urllib.request.urlopen(urllib.request.Request(URL + '/world/stop', data=b'{}', method='POST'), timeout=3)
        server.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    close_world_windows()
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
    if '--quiet' in sys.argv:  # tuck the window away without stealing focus, and keep it tucked while the page finishes loading
        for _ in range(80):
            for hwnd in world_windows():
                if not ctypes.windll.user32.IsIconic(hwnd):
                    ctypes.windll.user32.ShowWindow(hwnd, 7)  # SW_SHOWMINNOACTIVE
            time.sleep(0.5)
    while server.poll() is None:
        time.sleep(1)
    close_world_windows()  # the world stopped itself (its own button): close its window too
    icon.stop()


if alive():  # a world is already running (started at login, say): show it rather than start a second server beside it
    open_dashboard()
    sys.exit()
server = subprocess.Popen([sys.executable, str(ROOT / 'server.py'), '--no-browser'], cwd=ROOT, creationflags=NO_WINDOW)

image = Image.new('RGB', (64, 64), '#0d1117')
draw = ImageDraw.Draw(image)
draw.ellipse((8, 8, 56, 56), fill='#3fb950')
draw.ellipse((20, 14, 44, 38), fill='#58a6ff')
menu = pystray.Menu(pystray.MenuItem('Open dashboard', open_dashboard, default=True), pystray.MenuItem('Stop world', stop))
pystray.Icon('ai-world', image, 'AI World (running)', menu).run(setup=watch)
