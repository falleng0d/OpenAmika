"""OpenAmika launcher — one screen for the whole stack.

  python launch.py                 grace cleanup → pre-flight → services → components → browser
  python launch.py --check         validate config, environments, imports and files; start nothing
                                   (one OK / -- / !! line per check, then "result: ready" or what to fix)
  python launch.py --no-browser    same as the default, without opening the web UI
  python launch.py --pet           same, with her desktop pet window (pet.py: only her, transparent, always on top)
                                   instead of the web UI in the browser; needs pywebview (brain/requirements-pet.txt)
  python launch.py --wipe-memory   wipe her memory, backup first (Setup → Wipe memory, for scripts): through the
                                   running stack, or on disk when she is not running; --force stops what she is doing
  python launch.py --diagnose      one zip in logs/ for a help request: the pre-flight report, the last 200 lines of
                                   each log, settings.json without keys, Python/driver versions, free disk and GPU memory

The console is a single dashboard that redraws in place: one row per service. A spinning marker means
it is still launching, or up and working right now (its log is moving); a steady light means it is idle
— green up, yellow degraded/late, red failed — with a one-line description. Full output of every
process stays in logs/<name>.log; nothing scrolls here.

What happens
  1. Grace: an earlier instance of THIS folder is stopped first — its launcher, brain, speech, embedder
     and vision (found by script path, like Stop_OpenAmika.ps1) and the services that launcher started
     (recorded in logs/instance.json; matched by pid AND executable, so a reused pid is never hit).
     Nothing else on the machine is touched.
  2. Pre-flight: one light per check — ports (tested with a bind: a port of hers that Windows reserved for
     Hyper-V / WSL2 / Docker Desktop moves to the next free port outside `netsh interface ipv4 show
     excludedportrange protocol=tcp`, written to settings.json, and the dashboard shows the new address),
     Python environments, files, API keys, her clock, the TTS
     server, the tool servers (agents, smart home) and the services below. Red never blocks the launch:
     this is an open project, so you get to see what breaks. The report is kept in logs/preflight.json
     and shown on the Setup page; the dashboard lists what is not green.
  3. Start: the services in settings.json "services" (TTS, MQTT broker, zigbee2mqtt…) and the agents in
     config/tools.json "autostart" (Hermes, nanoMuse), then embedder → brain → vision → speech. Every
     process runs hidden. A service that is already running (started by hand) is used as is and left alone.
     The smart-home services (MQTT broker, zigbee2mqtt) are held back until the IoT safety statement
     (docs/IOT_DISCLAIMER.md) is accepted on the Tools page; accepted while running, they start then.
  4. Open the web UI (with --pet: her desktop pet window, pet.py). Ctrl+C stops everything this launcher started.

The optional local embedding service (brain/embed_server.py, BAAI/bge-m3 cut to 768 dims) starts before
the brain when settings.json has an `embedder` block; the brain reaches it through AMIKA_EMBEDDING_URL.
"""
import argparse
import errno
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
APP = 'OpenAmika'
LOGS = ROOT / 'logs'
INSTANCE = LOGS / 'instance.json'
PREFLIGHT = LOGS / 'preflight.json'
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
CORE_SCRIPTS = ('launch.py', 'brain/webui.py', 'speech/api.py', 'brain/embed_server.py', 'vision/vlm_service.py',
                'messengers/service.py', 'pet.py')
AGENT_HEALTH = {'hermes': ('HERMES_URL', '/health'), 'nanomuse': ('NANOMUSE_URL', '/api/health')}
# the smart-home chain is several processes but one thing to the person: one row, one light
IOT_SERVICES = ('mqtt', 'zigbee2mqtt')
IOT_CHECKS = ('service mqtt', 'service zigbee2mqtt', 'smart home broker', 'Zigbee dongle')
IOT_STATEMENT_CHECK = 'IoT safety statement'
IOT_RULES_PATH = Path(__file__).resolve().parent / 'brain' / 'iot_disclaimer.py'
_IOT_RULES = {}
LEVEL_RANK = {'ok': 0, 'warn': 1, 'err': 2}
BUSY_WINDOW_S = 2.5          # a log line newer than this = "working"
# log lines that are not activity (health polls and the like); the log files keep everything
NOISE = {
    'brain': re.compile(r'HTTP/1\.1" 200|Processing request of type|HTTP Request: GET .*(health|threads)|cleared \d+ unprotected turn'),
    'speech': re.compile(r'GET /api/build_f|GET /health|GET /api/tts_probe|GET /api/preflight|/api/evo/|HTTP/1\.1" 200|\[lang\] expected|connection (open|closed)|WebSocket'),
    'hermes': re.compile(r'aiohttp\.access|auxiliary_client|auxiliary_unavailable'),
    'nanomuse': re.compile(r'GET /api/health|GET /api/threads'),
    'zigbee2mqtt': re.compile(r'MQTT publish: topic .zigbee2mqtt/bridge/(state|info|devices|groups|definitions|extensions|logging)'),
    'tts': re.compile(r'GET /ping|GET /api/characters|HTTP/1\.1" 200'),
}

# ── console primitives ───────────────────────────────────────────────────────────────────
RESET, BOLD, DIM = '\x1b[0m', '\x1b[1m', '\x1b[2m'
COLOR = {'ok': '\x1b[32m', 'warn': '\x1b[33m', 'err': '\x1b[31m', 'off': '\x1b[90m', 'info': '\x1b[36m'}
SPINNER = '◐◓◑◒'
_print_lock = threading.Lock()


def enable_vt():
    if os.name != 'nt':
        return
    try:
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def log(text):
    """The launcher's own trail (logs/launcher.log): what it stopped, started and saw die — the dashboard
    shows only the present state."""
    try:
        LOGS.mkdir(exist_ok=True)
        with open(LOGS / 'launcher.log', 'a', encoding='utf-8') as handle:
            handle.write(f'{datetime.now():%Y-%m-%d %H:%M:%S} [{os.getpid()}] {text}\n')
    except OSError:
        pass


def dot(level):
    return f"{COLOR.get(level, '')}{'○' if level == 'off' else '●'}{RESET}"


def display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else (0 if unicodedata.combining(ch) else 1) for ch in text)


def fit(text, width):
    """Cut plain text to a display width (CJK counts double) with an ellipsis."""
    text = ' '.join(str(text or '').split())
    if display_width(text) <= width:
        return text
    out, used = [], 0
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
        if used + w > max(0, width - 1):
            break
        out.append(ch)
        used += w
    return ''.join(out) + '…'


def pad(text, width):
    return text + ' ' * max(0, width - display_width(text))


# ── small probes ─────────────────────────────────────────────────────────────────────────
def port_open(port, host='127.0.0.1', timeout=0.5):
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def bind_state(port, host='127.0.0.1'):
    """'' when a server could bind the port right now (a plain bind, as hers do), 'reserved' when Windows refuses it
    (WinError 10013: the port lies in a range Windows keeps for Hyper-V, WSL2 or Docker Desktop), 'in use' when a
    program holds it, else the error. A connect cannot tell a reserved port from a free one: nothing answers on either."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, int(port)))
        return ''
    except OSError as error:
        code = getattr(error, 'winerror', None)
        if code == 10013 or (code is None and error.errno == errno.EACCES):
            return 'reserved'
        if code == 10048 or error.errno == errno.EADDRINUSE:
            return 'in use'
        return f'{type(error).__name__}: {error}'
    finally:
        sock.close()


def parse_excluded_ranges(text):
    """[(first, last)] from `netsh interface ipv4 show excludedportrange protocol=tcp` (any display language: the
    numbers are the same; a '*' marks an administered range)."""
    ranges = []
    for line in (text or '').splitlines():
        match = re.match(r'\s*(\d{1,5})\s+(\d{1,5})\s*\*?\s*$', line)
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            if 0 < first <= last <= 65535:
                ranges.append((first, last))
    return ranges


def excluded_port_ranges():
    """The TCP port ranges Windows keeps for itself right now ([] elsewhere, or when netsh cannot be asked). They
    are chosen again at every boot, so a port can be fine one day and reserved the next."""
    if os.name != 'nt':
        return []
    try:
        out = subprocess.run(['netsh', 'interface', 'ipv4', 'show', 'excludedportrange', 'protocol=tcp'],
                             capture_output=True, text=True, errors='replace', timeout=20, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_excluded_ranges(out)


def range_of(port, ranges):
    return next((r for r in ranges if r[0] <= int(port) <= r[1]), None)


def next_free_port(port, ranges, avoid, state=None):
    """The next port above `port` that lies outside every excluded range, is not in `avoid` and binds now; None when
    there is none up to 65535."""
    state = state or bind_state
    candidate = int(port) + 1
    while candidate <= 65535:
        hit = range_of(candidate, ranges)
        if hit:
            candidate = hit[1] + 1
            continue
        if candidate not in avoid and state(candidate) == '':
            return candidate
        candidate += 1
    return None


# her own ports in settings.json: (pre-flight row, keys in settings.json); the voice server's is in tts_url and
# services.tts (its --port and its health URL), and is hers to move only when services.tts starts it on that port
PORT_SETTINGS = (('web/speech port', ('chat_port',)), ('brain port', ('brain_port',)),
                 ('messengers port', ('messengers_port',)), ('embedder port', ('embedder', 'port')),
                 ('vision port', ('vision', 'port')))


def own_port_settings(raw):
    """[(row, keys, port)] of the ports her own processes bind, read from settings.json as written."""
    out = []
    for label, keys in PORT_SETTINGS:
        holder = raw
        for key in keys[:-1]:
            holder = holder.get(key) if isinstance(holder, dict) else None
        if not isinstance(holder, dict):
            continue
        if keys[0] == 'vision' and not holder.get('enabled', True):
            continue
        port = holder.get(keys[-1]) or (5215 if keys == ('messengers_port',) else None)
        if isinstance(port, (int, str)) and str(port).isdigit():
            out.append((label, keys, int(port)))
    tts_port = urlsplit(str(raw.get('tts_url') or '')).port
    tts = (raw.get('services') or {}).get('tts') if isinstance(raw.get('services'), dict) else None
    command = [str(c) for c in (tts or {}).get('command') or []]
    if tts_port and tts and tts.get('enabled', True) and any(
            command[i] == '--port' and command[i + 1] == str(tts_port) for i in range(len(command) - 1)):
        out.append(('TTS port', ('tts_url',), tts_port))
    return out


def set_port(raw, keys, old, new):
    """Write a new port into settings.json's dict: the key itself, or for the voice server tts_url, services.tts's
    --port and its health URL."""
    if keys == ('tts_url',):
        raw['tts_url'] = re.sub(rf':{old}(?=/|$)', f':{new}', str(raw['tts_url']), count=1)
        tts = raw['services']['tts']
        command = tts['command']
        for i in range(len(command) - 1):
            if str(command[i]) == '--port' and str(command[i + 1]) == str(old):
                command[i + 1] = str(new)
        if isinstance(tts.get('health'), str):
            tts['health'] = re.sub(rf':{old}(?=/|$)', f':{new}', tts['health'], count=1)
        return
    holder = raw
    for key in keys[:-1]:
        holder = holder[key]
    holder[keys[-1]] = new


def move_reserved_ports(path=None, state=None, ranges=None, taken=None):
    """Her ports that Windows has reserved move to the next free port outside the excluded ranges, written to
    settings.json (a copy of the old file is kept as settings.json.bak). Only a reserved port moves: one that a program
    holds stays, and its pre-flight row names that program. Returns [(row, keys, old, new or None, range)]."""
    path = Path(path or ROOT / 'settings.json')
    state = state or bind_state
    try:
        text = path.read_text('utf-8-sig')
        raw = json.loads(text)
    except (OSError, ValueError):
        return []
    entries = own_port_settings(raw)
    reserved = [(label, keys, port) for label, keys, port in entries if state(port) == 'reserved']
    if not reserved:
        return []
    ranges = excluded_port_ranges() if ranges is None else ranges
    avoid = {port for _, _, port in entries} | set(listeners() if taken is None else taken)
    moves = []
    for label, keys, port in reserved:
        new = next_free_port(port, ranges, avoid, state)
        if new:
            set_port(raw, keys, port, new)
            avoid.add(new)
        moves.append((label, keys, port, new, range_of(port, ranges)))
        log(f'{label} {port} is reserved by Windows' + (f': moved to {new} in settings.json' if new else
                                                          ': no free port found above it'))
    if any(m[3] for m in moves):
        try:
            path.with_name(path.name + '.bak').write_text(text, 'utf-8')
        except OSError:
            pass
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + '\n', 'utf-8')
    return moves


def port_move_row(move):
    """(level, row, detail) for one reserved port: moved (warn, so the dashboard shows it), or stuck (err)."""
    label, keys, old, new, hit = move
    where = f'excluded range {hit[0]}-{hit[1]}' if hit else 'WinError 10013'
    key = 'tts_url and services.tts' if keys == ('tts_url',) else '.'.join(keys)
    if new:
        page = f'; her page is now http://localhost:{new}' if keys == ('chat_port',) else ''
        return 'warn', label, (f'{old} is reserved by Windows ({where}: Hyper-V, WSL2 or Docker Desktop) — moved to {new} '
                               f'in settings.json ({key}){page}')
    return 'err', label, (f'{old} is reserved by Windows ({where}) and no free port was found above it — set {key} in '
                          'settings.json to a port outside `netsh interface ipv4 show excludedportrange protocol=tcp`')


def http_json(url, timeout=2):
    try:
        with urlopen(url, timeout=timeout) as response:
            body = response.read()
    except Exception:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return {'_text': body.decode('utf-8', 'replace')}


def resolve(path):
    path = Path(str(path))
    return path if path.is_absolute() else ROOT / path


def read_env(path):
    values = {}
    try:
        for raw in Path(path).read_text('utf-8', errors='replace').splitlines():
            text = raw.strip()
            if not text or text.startswith('#') or '=' not in text:
                continue
            key, value = text.split('=', 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


# ── processes on this machine (Windows: one PowerShell call; elsewhere nothing to clean) ──
def list_processes():
    if os.name != 'nt':
        return []
    script = ('Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine '
              '| ConvertTo-Json -Compress')
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True,
                             timeout=40, creationflags=NO_WINDOW).stdout
        data = json.loads(out or '[]')
    except Exception as error:
        log(f'could not list processes ({type(error).__name__}): a previous instance may still be running unseen')
        return []
    if isinstance(data, dict):
        data = [data]
    return [{'pid': int(d.get('ProcessId') or 0), 'ppid': int(d.get('ParentProcessId') or 0),
             'name': str(d.get('Name') or ''), 'exe': str(d.get('ExecutablePath') or ''),
             'cmd': str(d.get('CommandLine') or '')} for d in data if isinstance(d, dict)]


def listeners():
    """Port -> PID for listeners that can conflict with our IPv4 loopback services."""
    if os.name != 'nt':
        return {}
    script = 'Get-NetTCPConnection -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json -Compress'
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True,
                             timeout=30, creationflags=NO_WINDOW).stdout
        data = json.loads(out or '[]')
    except Exception:
        return {}
    if isinstance(data, dict):
        data = [data]
    taken = {}
    # Prefer the IPv4 listener if a dual-stack wildcard and loopback share a port.
    for d in sorted((row for row in data if isinstance(row, dict)), key=lambda row: row.get('LocalAddress') != '::'):
        if d.get('LocalPort') and d.get('LocalAddress') in ('127.0.0.1', '0.0.0.0', '::', '::ffff:127.0.0.1'):
            taken[int(d['LocalPort'])] = int(d['OwningProcess'])
    return taken


_CLOSE = {'services': [], 'handler': None}


def stop_started_fast():
    """Stop every service this launcher started, all at once, within the seconds Windows allows on a window
    close; then drop the instance record so the next launcher does not chase dead pids."""
    victims = [s for s in _CLOSE['services'] if s.started_by_me and s.proc is not None and s.proc.poll() is None]
    threads = [threading.Thread(target=kill_tree, args=(s.proc.pid,), daemon=True) for s in victims]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 4.0
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    try:
        INSTANCE.unlink()
    except OSError:
        pass
    log(f'window closed: stopped {len(victims)} service(s) started here')


def install_close_handler():
    """Window close, log-off or shutdown: stop what was started here instead of leaving hidden services
    running where the next launcher cannot see them. Ctrl+C keeps its normal path."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        from ctypes import wintypes
        prototype = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def handler(event):
            if event in (2, 5, 6):          # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                stop_started_fast()
                return True
            return False
        _CLOSE['handler'] = prototype(handler)       # kept referenced: Windows calls it later
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_CLOSE['handler'], True)
    except Exception as error:
        log(f'close handler not installed ({type(error).__name__}): closing the window leaves services running')


def kill_tree(pid):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, creationflags=NO_WINDOW)
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass


def serial_dongle():
    """The Zigbee dongle's COM port on Windows (Silicon Labs CP210x — the SONOFF Dongle Plus MG24), or
    ('', reason) when there is none / no driver yet."""
    if os.name != 'nt':
        return '', 'not on Windows'
    script = ("$p = Get-CimInstance Win32_SerialPort -EA SilentlyContinue | Where-Object { $_.PNPDeviceID -like 'USB\\VID_10C4&PID_EA60*' } | Select-Object -First 1;"
              "if ($p) { 'PORT ' + $p.DeviceID } else { $d = Get-PnpDevice -PresentOnly -EA SilentlyContinue | Where-Object { $_.InstanceId -like 'USB\\VID_10C4&PID_EA60*' } | Select-Object -First 1;"
              "if ($d) { 'NODRIVER ' + $d.Status } else { 'NONE' } }")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, text=True,
                             timeout=30, creationflags=NO_WINDOW).stdout.strip()
    except Exception:
        return '', 'could not query devices'
    if out.startswith('PORT '):
        return out[5:].strip(), ''
    if out.startswith('NODRIVER'):
        return '', 'plugged in, no COM port — CP210x driver missing (run IoT\\Setup_IoT_admin.bat)'
    return '', 'no Zigbee dongle (CP210x) found on this PC'


# ── grace: stop an earlier instance of this folder ───────────────────────────────────────
def launcher_target(command):
    """The launch.py a command line runs, as a normalised absolute path, or '' when it is relative or
    cannot be read (then it may be this install's own launcher, started as a relative "launch.py")."""
    try:
        tokens = shlex.split(command or '', posix=False)
    except ValueError:
        return ''
    for token in tokens:
        token = token.strip('"\'')
        if token.lower().endswith('launch.py') and os.path.isabs(token):
            return os.path.normcase(os.path.abspath(token))
    return ''


def roles_all_local():
    """True when every LLM role selected in config/llm_roles.json runs on Ollama (no cloud key needed)."""
    try:
        doc = json.loads((ROOT / 'config' / 'llm_roles.json').read_text('utf-8-sig'))
        picks = list((doc.get('selections') or {}).values())
    except Exception:
        return False
    return bool(picks) and all(isinstance(k, str) and ':ollama:' in k for k in picks)


OLLAMA_DEFAULT = 'http://127.0.0.1:11434'


def ollama_roles():
    """{model: [roles]} for the LLM roles that run on Ollama: config/llm_roles.json selections "<role>:ollama:<model>"."""
    try:
        picks = json.loads((ROOT / 'config' / 'llm_roles.json').read_text('utf-8-sig')).get('selections') or {}
    except Exception:
        return {}
    out = {}
    for role, key in picks.items() if isinstance(picks, dict) else ():
        if isinstance(key, str) and ':ollama:' in key:
            out.setdefault(key.split(':ollama:', 1)[1], []).append(str(role))
    return out


def ollama_base():
    """The Ollama server the brain uses: config/ollama.json base_url (Setup → Brain), else the local default."""
    try:
        url = str(json.loads((ROOT / 'config' / 'ollama.json').read_text('utf-8-sig')).get('base_url') or '')
    except Exception:
        url = ''
    url = url.strip().rstrip('/')
    return url if re.fullmatch(r'https?://[A-Za-z0-9._\-\[\]]+(?::\d{1,5})?', url) else OLLAMA_DEFAULT


def ollama_checks():
    """[(level, name, detail)] when a role runs on Ollama: the server answers (GET /api/version) and every model in
    use is pulled (GET /api/tags). Nothing when no role uses Ollama."""
    models = ollama_roles()
    if not models:
        return []
    base = ollama_base()
    version = http_json(base + '/api/version', timeout=3)
    if not isinstance(version, dict) or '_text' in version:
        roles = sorted({r for rs in models.values() for r in rs})
        return [('err', 'Ollama', f'not answering at {base} — start Ollama (or fix its address on Setup → Brain); '
                                  f'{", ".join(roles)} run{"s" if len(roles) == 1 else ""} on it')]
    tags = http_json(base + '/api/tags', timeout=5)
    pulled = {str(m.get('name') or m.get('model') or '') for m in (tags or {}).get('models') or [] if isinstance(m, dict)} \
        if isinstance(tags, dict) else set()
    out = [('ok', 'Ollama', f"{version.get('version') or '?'} at {base}")]
    for model, roles in models.items():
        if model in pulled:
            out.append(('ok', f'Ollama model {model}', f'pulled · {", ".join(roles)}'))
        else:
            out.append(('err', f'Ollama model {model}', f'not pulled — ollama pull {model} ({", ".join(roles)} '
                                                       f'run{"s" if len(roles) == 1 else ""} on it)'))
    return out


def gpu_info():
    """[{name, driver, total_mb, used_mb, free_mb}] for each NVIDIA GPU (nvidia-smi); [] without one."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15,
                             creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in (out or '').splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 5:
            continue
        try:
            gpus.append({'name': parts[0], 'driver': parts[1], 'total_mb': int(float(parts[2])),
                         'used_mb': int(float(parts[3])), 'free_mb': int(float(parts[4]))})
        except ValueError:
            continue
    return gpus


# Her own parts on the GPU (speech recognition, the embedder, her voice) and what her eyes (vision) add, in GB: the
# README's hardware table. A local model has to fit next to them.
HER_GPU_GB, HER_EYES_GB = 6.0, 4.0


def model_fit_check(vision_on, gpus=None):
    """(level, detail): does the biggest local Ollama model in use fit this PC's GPU next to her own parts? None when
    no role runs on an Ollama of this PC, a model's size is unknown (Setup → Brain → Detect records it) or there is no
    NVIDIA GPU to judge by. Weights plus about 1.5 GB of context cache, plus her parts (item 33: "the model does not fit")."""
    models = ollama_roles()
    if not models or (urlsplit(ollama_base()).hostname or '') not in ('127.0.0.1', 'localhost', '::1'):
        return None
    try:
        rows = json.loads((ROOT / 'config' / 'ollama.json').read_text('utf-8-sig')).get('models') or []
    except Exception:
        return None
    sizes = {r.get('name'): r.get('size_gb') for r in rows if isinstance(r, dict)}
    known = {name: float(sizes[name]) for name in models
             if isinstance(sizes.get(name), (int, float)) and not isinstance(sizes.get(name), bool)}
    gpus = gpu_info() if gpus is None else gpus
    if not known or not gpus:
        return None
    total = max(g['total_mb'] for g in gpus) / 1024
    mine = HER_GPU_GB + (HER_EYES_GB if vision_on else 0.0)
    name, size = max(known.items(), key=lambda item: item[1])
    if size + 1.5 + mine <= total:
        return 'ok', f'{name} ({size:g} GB) fits next to her own parts (~{mine:g} GB) on this {total:.0f} GB GPU'
    return 'warn', (f'{name} ({size:g} GB) does not fit next to her own parts (~{mine:g} GB) on this {total:.0f} GB GPU: '
                    'Ollama moves part of it to the CPU (very slow) or fails, and she does not answer — use OpenRouter '
                    '(Setup → Brain), or Ollama on a PC with a bigger GPU (its address on Setup → Brain)')


def her_clock(zone_info=None):
    """(level, detail) for her clock: config/build_f.json "timezone". The brain needs a time-zone database to start at
    all (it builds its fallback zone, Asia/Tokyo, at import), and Windows has none of its own: the tzdata package
    (brain/requirements.txt) is it. zone_info: zoneinfo.ZoneInfo (tests pass a stand-in)."""
    if zone_info is None:
        from zoneinfo import ZoneInfo as zone_info
    try:
        zone = str(json.loads((ROOT / 'config' / 'build_f.json').read_text('utf-8-sig')).get('timezone') or 'Asia/Tokyo')
    except Exception as error:
        return 'warn', f'config/build_f.json cannot be read ({error}) — the brain falls back to Asia/Tokyo'
    try:
        now = datetime.now(zone_info(zone))
        return 'ok', f'{zone} · {now:%a %H:%M} {now.tzname()}'
    except Exception as error:
        problem = error
    try:
        zone_info('Asia/Tokyo')
    except Exception:
        python = shown(sys.executable)
        return 'err', (f'{python} has no time-zone database (the tzdata package), so the brain cannot start — run the '
                       f'setup again ({SETUP_CORE}), or {python} -m pip install tzdata')
    return 'warn', f'build_f.json timezone {zone!r} is not a time zone ({problem}) — the brain falls back to Asia/Tokyo'


def messengers_port(cfg):
    """The Discord / Telegram service's port: settings.json "messengers_port" (default 5215)."""
    try:
        return int(cfg.get('messengers_port') or 5215)
    except (TypeError, ValueError):
        return 5215


def own_pids(procs):
    """pids of this folder's running OpenAmika: its core scripts by path on the command line (as grace_cleanup and
    Stop_OpenAmika.ps1 find them) and the services this folder's launcher recorded in logs/instance.json."""
    pids = {p['pid'] for p in procs if p['cmd'] and any(str(ROOT / s).lower() in p['cmd'].lower() for s in CORE_SCRIPTS)}
    recorded = []
    try:
        if INSTANCE.exists():
            recorded = json.loads(INSTANCE.read_text('utf-8')).get('processes') or []
    except Exception:
        pass
    by_pid = {p['pid']: p for p in procs}
    for entry in recorded:
        live = by_pid.get(int(entry.get('pid') or 0)) if isinstance(entry, dict) else None
        exe = str(entry.get('exe') or '').lower() if live else ''
        if live and exe and (live['exe'].lower() == exe or exe in live['cmd'].lower()):
            pids.add(live['pid'])
    return pids


def grace_cleanup(procs):
    me, parent = os.getpid(), os.getppid()
    victims = {}
    try:
        previous = json.loads(INSTANCE.read_text('utf-8')) if INSTANCE.exists() else {}
    except Exception:
        previous = {}
    by_pid = {p['pid']: p for p in procs}
    # the previous launcher itself (its console may have been started with a relative "launch.py")
    old_launcher = by_pid.get(int(previous.get('launcher_pid') or 0))
    if old_launcher and old_launcher['pid'] not in (me, parent) and old_launcher['name'].lower().startswith('python') \
            and 'launch.py' in old_launcher['cmd'].lower():
        victims[old_launcher['pid']] = 'launcher'
    for entry in previous.get('processes') or []:
        live = by_pid.get(int(entry.get('pid') or 0))
        exe = str(entry.get('exe') or '').lower()
        if live and live['pid'] not in (me, parent) and exe and (live['exe'].lower() == exe or exe in live['cmd'].lower()):
            victims[live['pid']] = entry.get('name') or 'service'
    my_exe = str(Path(sys.executable).resolve()).lower()
    own_launcher = os.path.normcase(str(ROOT / 'launch.py'))
    for p in procs:
        if p['pid'] in (me, parent) or not p['name'].lower().startswith('python'):
            continue
        command = p['cmd'].lower()
        if any(str(ROOT / script).lower() in command for script in CORE_SCRIPTS):
            victims.setdefault(p['pid'], 'core')
        elif 'launch.py' in command and p['exe'].lower() == my_exe and launcher_target(p['cmd']) in ('', own_launcher):
            # another launcher run from this install's own brain python (started with a relative "launch.py");
            # a launcher of a different checkout that happens to use the same Python is left alone
            victims.setdefault(p['pid'], 'launcher')
    for pid, name in victims.items():
        log(f'grace: stopping previous {name} (pid {pid})')
        kill_tree(pid)
    if victims:
        time.sleep(2.0)
    try:
        INSTANCE.unlink()
    except OSError:
        pass
    return victims


# ── what to run ──────────────────────────────────────────────────────────────────────────
class Service:
    def __init__(self, name, cmd, cwd=None, env=None, health=None, port=None, expect_app=False, expect_ok=False,
                 timeout=60, kind='service', log=None, after=None, url=None, live_light=False):
        self.name, self.cmd, self.cwd = name, [str(c) for c in cmd], str(cwd or ROOT)
        self.env, self.health, self.port = dict(env or {}), health, port
        self.expect_app, self.expect_ok, self.timeout, self.kind = expect_app, expect_ok, timeout, kind
        self.log = log or LOGS / f'{name}.log'
        self.after, self.url = after, url
        self.live_light = live_light  # its health JSON carries {"light": {level, detail}}: polled while up, shown as its row
        self.live = None
        self.proc = None
        self.started_by_me = False
        self.state = 'off'          # off | starting | up | external | failed | timeout | exited
        self.detail = ''
        self.started_at = None
        self.last_line = ''         # latest meaningful log line (LogWatch)
        self.last_activity = 0.0    # monotonic time of that line
        self.why = ''               # why it is not up, in words (start_failure; item 33)
        self.caveat = ''            # up, but she cannot answer: why (the brain row; brain_caveat)
        self.exe = shutil.which(self.cmd[0]) or self.cmd[0]

    def is_up(self):
        if self.health:
            data = http_json(self.health)
            if data is None:
                return False
            if self.live_light:
                self.live = data.get('light') if isinstance(data.get('light'), dict) else None
            if self.expect_app:
                return data.get('app') == APP
            if self.expect_ok:
                return bool(data.get('ok'))
            return True
        if self.port:
            return port_open(self.port)
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        self.detail = ''
        if self.is_up():
            self.state, self.detail = 'external', 'running already — started elsewhere, left alone at exit'
            return
        LOGS.mkdir(exist_ok=True)
        handle = open(self.log, 'ab')
        handle.write(f"\n--- {datetime.now():%Y-%m-%d %H:%M:%S} {APP} starting {self.name}: {' '.join(self.cmd)}\n".encode('utf-8'))
        handle.flush()
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1', **self.env}
        kwargs = dict(cwd=self.cwd, env=env, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        if os.name == 'nt':
            kwargs['creationflags'] = NO_WINDOW
        try:
            self.proc = subprocess.Popen(self.cmd, **kwargs)
        except OSError as error:
            self.state, self.detail = 'failed', f'could not start: {error}'
            log(f'{self.name}: could not start: {error}')
            return
        self.started_by_me = True
        self.started_at = time.monotonic()
        self.state = 'starting'
        log(f'{self.name}: started pid {self.proc.pid}: {" ".join(self.cmd)}')

    def poll(self):
        """Advance a starting/up service: exited → failed/exited; starting and answering → up; late → timeout."""
        if self.proc is not None and self.started_by_me and self.proc.poll() is not None and self.state in ('starting', 'up', 'timeout'):
            was_up = self.state == 'up'
            self.state = 'exited' if was_up else 'failed'
            self.detail = f'exited with code {self.proc.returncode} — see logs/{self.log.name}'
            log(f'{self.name}: {self.detail}')
            return
        if self.state in ('starting', 'timeout') and self.is_up():
            self.state, self.detail = 'up', ''
            log(f'{self.name}: up')
        elif self.state == 'starting' and self.started_at and time.monotonic() - self.started_at > self.timeout:
            self.state, self.detail = 'timeout', f'no answer after {int(self.timeout)}s — still running, see logs/{self.log.name}'
            log(f'{self.name}: {self.detail}')

    def wait_up(self, timeout=None, tick=None):
        deadline = time.monotonic() + (timeout or self.timeout)
        while self.state == 'starting' and time.monotonic() < deadline:
            self.poll()
            if tick:
                tick()
            time.sleep(0.4)
        self.poll()
        return self.state in ('up', 'external')

    def stop(self):
        if not self.started_by_me or self.proc is None or self.proc.poll() is not None:
            return
        if self.kind == 'core':
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        else:
            kill_tree(self.proc.pid)

    def busy(self):
        return self.state == 'up' and time.monotonic() - self.last_activity < BUSY_WINDOW_S

    def light(self, frame):
        """(marker, level, description) for the dashboard row."""
        if self.state == 'starting':
            age = int(time.monotonic() - (self.started_at or time.monotonic()))
            tail = f' · {self.last_line}' if self.last_line else ''
            return f"{COLOR['warn']}{SPINNER[frame % len(SPINNER)]}{RESET}", 'warn', f'starting… {age}s{tail}'
        if self.state == 'up':
            if self.busy():
                return f"{COLOR['ok']}{SPINNER[frame % len(SPINNER)]}{RESET}", 'ok', f'working · {self.last_line}'
            if self.live and self.live.get('level') in ('ok', 'warn', 'err'):
                return dot(self.live['level']), self.live['level'], str(self.live.get('detail') or '')
            if self.caveat:
                return dot('warn'), 'warn', f'up, but she cannot answer: {self.caveat}'
            where = self.url or self.health or (f'port {self.port}' if self.port else '')
            pid = f'pid {self.proc.pid}' if self.proc is not None else ''
            return dot('ok'), 'ok', ' · '.join(x for x in ('ready', pid, where) if x)
        why = f' — {self.why}' if self.why else ''
        if self.state == 'external':
            if self.caveat:
                return dot('warn'), 'warn', f'{self.detail} — but she cannot answer: {self.caveat}'
            return dot('ok'), 'ok', self.detail
        if self.state == 'timeout':
            return dot('warn'), 'warn', self.detail + why
        if self.state in ('failed', 'exited'):
            return dot('err'), 'err', self.detail + why
        return dot('off'), 'off', self.detail or self.why or 'not started'


def prime_dongle(port):
    """Reset the Zigbee dongle into its Zigbee firmware before zigbee2mqtt opens it.

    The dongle's DTR/RTS pins drive its reset/boot lines. A zigbee2mqtt that was killed hard can leave it in the
    Gecko bootloader, and the next open (DTR released, see IoT/patch_herdsman_windows.py) causes no line change,
    so nothing resets it and ASH never answers. Pulsing both lines resets the chip; if the bootloader menu answers,
    its "run" command starts the firmware. Returns a short note for the log; never raises; skips without pyserial."""
    try:
        import serial
    except ImportError:
        return 'pyserial not installed in the brain python — skipped'
    rst = bytes([0x1A, 0xC0, 0x38, 0xBC, 0x7E])            # ASH RST (after the cancel byte)
    try:
        with serial.Serial(port, 115200, timeout=1.0, write_timeout=1.0, xonxoff=True) as link:
            for attempt in range(2):
                link.dtr, link.rts = True, True
                time.sleep(0.6)
                link.reset_input_buffer()
                link.write(rst)
                time.sleep(1.0)
                answer = link.read(300)
                if b'\xc1' in answer and b'Bootloader' not in answer:
                    link.dtr, link.rts = False, False                   # how zigbee2mqtt will find the lines
                    time.sleep(0.3)
                    return f'dongle answered the reset on {port}' + (' after leaving the bootloader' if attempt else '')
                if b'Bootloader' in answer or b'upload gbl' in answer:
                    link.write(b'2\r\n')                                # bootloader menu: 2 = run the application
                    time.sleep(1.5)
                    continue
                link.dtr, link.rts = False, False
                time.sleep(0.6)
            return f'no reply from the dongle on {port} (zigbee2mqtt may still manage)'
    except Exception as error:
        return f'could not probe {port}: {str(error)[:120]}'


def iot_light(parts, frame):
    """One (marker, level, description) for the smart-home chain: the worst of its parts, spinning while any
    part is launching or working."""
    if any(s.state == 'starting' for s in parts):
        names = ', '.join(s.name for s in parts if s.state == 'starting')
        return f"{COLOR['warn']}{SPINNER[frame % len(SPINNER)]}{RESET}", 'warn', f'starting… {names}'
    busy = next((s for s in parts if s.busy()), None)
    if busy:
        return f"{COLOR['ok']}{SPINNER[frame % len(SPINNER)]}{RESET}", 'ok', f'working · {busy.name}: {busy.last_line}'
    broken = [s for s in parts if s.state in ('failed', 'exited')]
    if broken:
        return dot('err'), 'err', ' · '.join(f'{s.name}: {s.detail}' for s in broken)
    late = [s for s in parts if s.state == 'timeout']
    if late:
        return dot('warn'), 'warn', ' · '.join(f'{s.name}: {s.detail}' for s in late)
    if all(s.state in ('up', 'external') for s in parts):
        z2m = next((s for s in parts if s.name == 'zigbee2mqtt'), None)
        where = (z2m.health or '').replace('http://127.0.0.1', 'http://localhost') if z2m else ''
        return dot('ok'), 'ok', 'ready · broker + zigbee2mqtt' + (f' · {where}' if where else '')
    return dot('off'), 'off', 'not started'


class LogWatch(threading.Thread):
    """Follows a service's log: keeps its latest meaningful line and when it arrived (drives "working")."""

    def __init__(self, service):
        super().__init__(daemon=True, name=f'watch-{service.name}')
        self.service = service
        self.stop_flag = threading.Event()
        self.noise = NOISE.get(service.name)

    def run(self):
        path = self.service.log
        position = path.stat().st_size if path.exists() else 0
        buffer = b''
        while not self.stop_flag.is_set():
            try:
                size = path.stat().st_size
            except OSError:
                time.sleep(0.5)
                continue
            if size < position:
                position = 0
            if size > position:
                with open(path, 'rb') as handle:
                    handle.seek(position)
                    buffer += handle.read(size - position)
                position = size
                *lines, buffer = buffer.split(b'\n')
                for raw in lines:
                    text = raw.decode('utf-8', 'replace').rstrip('\r').strip()
                    if not text or text.startswith('--- ') or (self.noise and self.noise.search(text)):
                        continue
                    self.service.last_line = re.sub(r'\x1b\[[0-9;]*m', '', text)
                    self.service.last_activity = time.monotonic()
            time.sleep(0.25)


# ── settings ─────────────────────────────────────────────────────────────────────────────
def embedder_settings(cfg):
    emb = cfg.get('embedder')
    if not emb:
        return None
    for key in ('python', 'port', 'model', 'tag', 'dims'):
        if key not in emb:
            raise RuntimeError(f'settings.json embedder block is missing {key!r}')
    return emb


def vision_settings(cfg):
    vis = cfg.get('vision')
    if not vis or not vis.get('enabled', True):
        return None
    for key in ('python', 'port', 'model'):
        if key not in vis:
            raise RuntimeError(f'settings.json vision block is missing {key!r}')
    return vis


def load_tools():
    path = ROOT / 'config' / 'tools.json'
    if not path.is_file():
        return None, None           # optional: no tools.json = tool layer off (the Tools page creates it)
    try:
        return json.loads(path.read_text('utf-8-sig')), None
    except ValueError as error:
        return None, f'config/tools.json is not valid JSON: {error}'


def configured_services(cfg):
    """settings.json "services": {name: {enabled, command, cwd, env, health|port, timeout_s, after}}."""
    out = []
    for name, spec in (cfg.get('services') or {}).items():
        if not isinstance(spec, dict) or not spec.get('enabled', True) or not spec.get('command'):
            continue
        cmd = list(spec['command'])
        cmd[0] = str(resolve(cmd[0])) if ('/' in cmd[0] or '\\' in cmd[0]) and not Path(cmd[0]).is_absolute() else cmd[0]
        env = {k: (str(resolve(v)) if isinstance(v, str) and v.startswith(('IoT', 'Hermes', 'Nanomuse', 'vendor', 'logs', 'servers')) else str(v))
               for k, v in (spec.get('env') or {}).items()}
        out.append(Service(name, cmd, cwd=resolve(spec.get('cwd') or '.'), env=env, health=spec.get('health'),
                           port=spec.get('port'), timeout=int(spec.get('timeout_s', 60)), after=spec.get('after'),
                           url=spec.get('url')))
    return out


def iot_rules():
    """brain/iot_disclaimer.py (standard library only), loaded by its path: the IoT safety statement's version, what
    counts as IoT and whether config/tools.json records the acceptance. None when it cannot be loaded; then no
    smart-home service starts."""
    if 'module' not in _IOT_RULES:
        try:
            spec = importlib.util.spec_from_file_location('openamika_iot_disclaimer', IOT_RULES_PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as error:
            log(f'IoT safety rules unreadable ({type(error).__name__}: {error}): the smart home stays off')
            module = None
        _IOT_RULES['module'] = module
    return _IOT_RULES['module']


def iot_accepted(tools):
    """True when config/tools.json records the acceptance of the current IoT safety statement."""
    rules = iot_rules()
    return bool(rules and rules.accepted(tools or {}))


def is_smart_home_server(name, scfg):
    rules = iot_rules()
    return rules.is_smart_home(name, scfg) if rules else name == 'iot'


def hold_iot_services(services, tools):
    """(services to start, smart-home services held back): the MQTT broker and zigbee2mqtt wait until the IoT safety
    statement (docs/IOT_DISCLAIMER.md) is accepted on the Tools page."""
    if iot_accepted(tools):
        return list(services), []
    rules = iot_rules()
    held = [s for s in services if (rules.is_iot_service(s.name, s.cmd, s.cwd) if rules else s.name in IOT_SERVICES)]
    return [s for s in services if s not in held], held


def start_plan(cfg, tools):
    """What the launcher starts before the core: (services, smart-home services held back, agents). The MQTT broker
    and zigbee2mqtt, and a smart-home server's autostart, wait for the IoT safety statement to be accepted."""
    services, held = hold_iot_services(configured_services(cfg), tools)
    agents = agent_services(tools) if tools and tools.get('enabled') else []
    return services, held, agents


def agent_bridge(name, scfg):
    """Which agent bridge a config/tools.json entry runs: 'hermes' or 'nanomuse' (the word after `-m openamika_agents`
    in its args, else its name), or None for any other server. The launcher's health gate and checks follow it, so an
    entry may be named as the person likes."""
    args = [str(a) for a in ((scfg or {}).get('args') or [])]
    if 'openamika_agents' in args:
        i = args.index('openamika_agents')
        if i + 1 < len(args) and args[i + 1] in AGENT_HEALTH:
            return args[i + 1]
    return name if name in AGENT_HEALTH else None


def autostart_spec(raw):
    """{"command", "args", "cwd", "env"} from an entry's `autostart`: one command line (quotes keep a path with spaces
    together, as in the brain's f_tools.autostart_config) or such an object; None when unset."""
    if isinstance(raw, str):
        try:
            parts = shlex.split(raw.strip(), posix=False)
        except ValueError:
            parts = raw.strip().split()
        parts = [p.strip('"') for p in parts]
        raw = {'command': parts[0], 'args': parts[1:]} if parts else None
    if not isinstance(raw, dict) or not str(raw.get('command') or '').strip():
        return None
    env = raw.get('env') if isinstance(raw.get('env'), dict) else {}
    return {'command': str(raw['command']).strip(), 'args': [str(a) for a in (raw.get('args') or [])],
            'cwd': str(raw.get('cwd') or '') or None, 'env': {str(k): str(v) for k, v in env.items()}}


def is_absolute(path):
    return Path(path).is_absolute() or bool(re.match(r'^[A-Za-z]:[\\/]', path)) or path.startswith(('\\\\', '//'))


def launch_path(value, command=False):
    """An autostart path as it is run: %VARIABLES% expanded (%LOCALAPPDATA%\\hermes\\bin\\hermes.exe), and a relative
    path taken from this OpenAmika folder. A bare command name (`hermes`) is left to the PATH. The brain does the same
    (f_tools.launch_path), so the launcher and the brain start the same program."""
    text = os.path.expandvars(str(value).strip())
    if command and not ('/' in text or '\\' in text):
        return text
    return text if is_absolute(text) else str(ROOT / text)


def agent_services(tools):
    """Agents from config/tools.json autostart entries, so they come up here, before the brain needs them.
    A smart-home server's autostart waits for the IoT safety statement like the services do."""
    out = []
    accepted = iot_accepted(tools)
    for name, scfg in ((tools or {}).get('mcpServers') or {}).items():
        if not isinstance(scfg, dict) or not scfg.get('enabled') or not scfg.get('autostart'):
            continue
        if not accepted and is_smart_home_server(name, scfg):
            continue
        auto = autostart_spec(scfg['autostart'])
        if auto is None:
            continue
        env_key, path = AGENT_HEALTH.get(agent_bridge(name, scfg), (None, None))
        base = ((scfg.get('env') or {}).get(env_key) or '').rstrip('/') if env_key else ''
        health = base + path if base and path else None
        cmd = [launch_path(auto['command'], command=True), *(os.path.expandvars(a) for a in auto['args'])]
        out.append(Service(name, cmd, cwd=launch_path(auto['cwd']) if auto['cwd'] else ROOT,
                           env=auto['env'], health=health, timeout=90, kind='agent',
                           log=LOGS / f'agent-{name}.log', url=base or None))
    return out


# ── her voice server, her voice and her Live2D model: file checks only (nothing heavy is imported) ───────────
SETUP_CORE = r'powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1'
# Windows' "The filename or extension is too long" (error 206; localized, also inside a DLL load failure): a folder path
# too long for what lies deep in the environments. Only a shorter folder fixes it (scripts/setup_tts.py too_long_hint).
TOO_LONG = re.compile(r'(?i)filename or extension is too long|winerror 206|ファイル名または拡張子が長すぎます|文件名或扩展名太长')
TTS_ENGINE = ROOT / 'vendor' / 'tts-engine'
TTS_PRETRAINED = ('chinese-hubert-base/pytorch_model.bin', 'chinese-roberta-wwm-ext-large/pytorch_model.bin',
                  'sv/pretrained_eres2netv2w24s4ep4.ckpt', 'fast_langdetect/lid.176.bin')
AVATAR_MODELS = ROOT / 'avatar' / 'models'


def shown(path):
    """A path as the person types it: relative to this folder when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


def setup_tts_command(cfg):
    """The command that installs her voice and her Live2D model (scripts/setup_tts.py), as the person types it."""
    brain = cfg.get('brain_python')
    return f"{shown(brain) if brain and Path(brain).is_file() else 'python'} {os.path.join('scripts', 'setup_tts.py')}"


def voice_folder(folder):
    """(level, detail) for a voice in the voice server's Speakers/ folder: character.json, the weights it names,
    references/emotions.json and its clips."""
    try:
        meta = json.loads((folder / 'character.json').read_text('utf-8-sig'))
    except (OSError, ValueError):
        return 'err', f'{shown(folder)} is not installed (no character.json)'
    cid = meta.get('character_id') or folder.name
    for key, default in (('llm_model', f'{cid}.ckpt'), ('vocoder_model', f'{cid}.pth')):
        if not (folder / str(meta.get(key) or default)).is_file():
            return 'err', f'{shown(folder)}: {meta.get(key) or default} is missing'
    try:
        table = json.loads((folder / 'references' / 'emotions.json').read_text('utf-8-sig'))
    except (OSError, ValueError):
        return 'err', f'{shown(folder)}: references/emotions.json is missing'
    tags = {k: v for k, v in table.items() if not str(k).startswith('_') and isinstance(v, dict)}
    clips = sum(1 for v in tags.values() if (folder / 'references' / str(v.get('audio', ''))).is_file())
    if not clips:
        return 'err', f'{shown(folder)}: no reference clips'
    return ('ok' if clips == len(tags) else 'warn'), f'{folder.name} · {clips}/{len(tags)} emotion clips'


def avatar_folders():
    """[(folder, its .model3.json, role, motions)] for every Live2D model in avatar/models/, in name order (the order
    of the gateway's /api/avatar/models). role is "agent" when banks.json says so (the paper doll), else None."""
    out = []
    for folder in sorted(AVATAR_MODELS.iterdir()) if AVATAR_MODELS.is_dir() else []:
        if not folder.is_dir():
            continue
        model3 = sorted(folder.glob('*.model3.json')) or sorted(folder.rglob('*.model3.json'))
        if not model3:
            continue
        try:
            role = (json.loads((folder / 'banks.json').read_text('utf-8-sig')).get('config') or {}).get('role')
        except (OSError, ValueError, AttributeError):
            role = None
        out.append((folder, model3[0], 'agent' if role == 'agent' else None, len(list(folder.rglob('*.motion3.json')))))
    return out


def live2d_models():
    """(level, detail) for the Live2D models she can wear (avatar/models/<name>/ with a .model3.json; an agent
    avatar does not count). None is an error: the Use page's stage stays empty."""
    found = [f"{folder.name} ({model3.name}, {motions} motions)"
             for folder, model3, role, motions in avatar_folders() if role != 'agent']
    if not found:
        return 'err', 'no Live2D model in avatar/models/: the stage stays empty'
    return 'ok', ' · '.join(found)


def agents_switched_on(tools):
    """True when config/tools.json has the tool layer on and a task agent (kind "agent", or an openamika_agents bridge)
    switched on: then the Use page seats the paper doll beside her."""
    if not (tools and tools.get('enabled')):
        return False
    return any(isinstance(scfg, dict) and scfg.get('enabled') and (scfg.get('kind') == 'agent' or agent_bridge(name, scfg))
               for name, scfg in (tools.get('mcpServers') or {}).items())


def paper_doll(cfg, tools):
    """(level, detail) for the paper doll that stands in for her helpers (the first model whose banks.json says
    "role": "agent"; scripts/setup_tts.py installs MekaHime's into avatar/models/Host), or (None, None) when there is
    none and no task agent is on, so nobody misses it."""
    on = agents_switched_on(tools)
    dolls = [(folder, motions) for folder, _, role, motions in avatar_folders() if role == 'agent']
    if dolls:
        folder, motions = dolls[0]
        return 'ok', (f'{folder.name} · {motions} motions — ' + ('stands in for her helpers on the Use page' if on else
                      'shown beside her while a task agent is switched on (Tools page)'))
    if on:
        return 'warn', (f'a task agent is switched on, but avatar/models/ has no paper doll (a model whose banks.json says '
                        f'"role": "agent"): her helpers get no stand-in and no speech bubbles on the Use page — run '
                        f'{setup_tts_command(cfg)} (it installs the paper doll with her Live2D model)')
    return None, None


HERMES_KEY_MIN = 16      # Hermes refuses to open its API server with a shorter API_SERVER_KEY, or a placeholder


def hermes_key_usable(key):
    """Hermes's own rule (hermes_cli.auth.has_usable_secret, min_length=16), in short: 16 characters or more and not a
    placeholder such as `your_api_key_here` or `xxxx`."""
    key = (key or '').strip()
    low = key.lower()
    return (len(key) >= HERMES_KEY_MIN and low not in ('changeme', 'placeholder', 'example', 'dummy')
            and not (low.startswith('your_') and low.endswith('_here'))
            and not re.fullmatch(r'(?:sk-|ghp_|hf_)?[x\s_-]+', low))


def hermes_home(scfg, agent):
    """The Hermes home whose .env holds API_SERVER_KEY: HERMES_HOME in the autostart env, the autostart cwd, the folder
    above the autostart's bin\\hermes.exe, this console's HERMES_HOME, then Hermes's default %LOCALAPPDATA%\\hermes;
    the first that has a .env. None when none has (Hermes elsewhere, or not installed yet)."""
    auto = autostart_spec((scfg or {}).get('autostart')) or {}
    candidates = []
    if (auto.get('env') or {}).get('HERMES_HOME'):
        candidates.append(launch_path(auto['env']['HERMES_HOME']))
    if agent is not None and auto.get('cwd'):
        candidates.append(agent.cwd)
    if agent is not None and Path(agent.cmd[0]).parent.name.lower() == 'bin':
        candidates.append(str(Path(agent.cmd[0]).parent.parent))
    if os.environ.get('HERMES_HOME'):
        candidates.append(os.environ['HERMES_HOME'])
    if os.environ.get('LOCALAPPDATA'):
        candidates.append(str(Path(os.environ['LOCALAPPDATA']) / 'hermes'))
    return next((Path(c) for c in candidates if (Path(c) / '.env').is_file()), None)


def hermes_checks(scfg, agent):
    """[(level, name, detail)]: does Hermes's API server get a key it accepts, and is it the one she sends? Read from the
    Hermes home's .env (hermes_home); the key itself is never shown."""
    home = hermes_home(scfg, agent)
    if home is None:
        return []
    # Hermes reads its .env over the environment it was started with (which the launcher passes on)
    key = (read_env(home / '.env').get('API_SERVER_KEY') or os.environ.get('API_SERVER_KEY') or '').strip()
    ours = str((scfg.get('env') or {}).get('HERMES_API_KEY') or '').strip()
    where = shown(home / '.env')
    if not key:
        try:
            yaml_text = (home / 'config.yaml').read_text('utf-8', errors='replace')
        except OSError:
            yaml_text = ''
        if re.search(r'(?m)^\s*api_server\s*:', yaml_text):
            return []                     # set in config.yaml (gateway.api_server) instead: not checked here
    if not hermes_key_usable(key):
        return [('warn', 'hermes key', f'API_SERVER_KEY in {where} is missing, shorter than {HERMES_KEY_MIN} characters or '
                                       f'a placeholder, so Hermes never opens its API server and she cannot reach it — '
                                       f'set a long random key there and the same one as HERMES_API_KEY in '
                                       f'config/tools.json (docs/TOOLS.md, "Set up Hermes Agent")')]
    if not ours or ours.upper().startswith('PASTE') or (ours.startswith('<') and ours.endswith('>')):   # a placeholder
        return [('warn', 'hermes key', f'HERMES_API_KEY in config/tools.json is not set yet: put the API_SERVER_KEY from '
                                       f'{where} there — Hermes rejects her without it')]
    if key != ours:
        return [('warn', 'hermes key', f'tools.json HERMES_API_KEY differs from API_SERVER_KEY in {where} — Hermes will '
                                       f'reject her')]
    return [('ok', 'hermes key', f'HERMES_API_KEY is the API_SERVER_KEY in {where}')]


def voice_checks(cfg, services):
    """[(level, name, detail)]: the voice server (installed? running?), her voice files and her Live2D model, each
    with the command that fixes it."""
    out = []
    fix = setup_tts_command(cfg)
    tts_url = str(cfg.get('tts_url') or '').rstrip('/')
    tts = next((s for s in services if s.name == 'tts'), None)
    local = tts is not None and Path(tts.cwd).resolve() == TTS_ENGINE.resolve()
    exe_ok = tts is not None and (Path(tts.exe).is_file() or bool(shutil.which(tts.cmd[0])))
    gsv = TTS_ENGINE / 'GPT-SoVITS' / 'GPT_SoVITS'
    missing = [] if not local else (
        ['GPT-SoVITS'] if not (gsv / 'TTS_infer_pack' / 'TTS.py').is_file() else
        ['its pretrained models'] if any(not (gsv / 'pretrained_models' / p).is_file() for p in TTS_PRETRAINED) else [])
    voice = str(cfg.get('tts_character') or 'MekaHime_Amika')
    if http_json(tts_url + '/ping') is not None:
        # something answers on her voice port: it is used as it is, so it must be a voice server that has her voice
        listing = http_json(tts_url + '/api/characters')
        voices = listing.get('characters') if isinstance(listing, dict) else None
        if isinstance(voices, dict) and voice in voices:
            out.append(('ok', 'TTS (voice)', f'{tts_url} answers · voice {voice}'))
        else:
            port = urlsplit(tts_url).port or tts_url
            out.append(('err', 'TTS (voice)', f'port {port} answers but has no voice {voice} (another voice server, '
                                              f"OpenSakura's for one?) — stop it, or move this one to another port: "
                                              f'tts_url and services.tts (--port, health) in settings.json'))
    elif tts is None:
        out.append(('warn', 'TTS (voice)', f'nothing answers at {tts_url} and no services.tts in settings.json — she will not speak'))
    elif local and not exe_ok:
        out.append(('err', 'TTS (voice)', f'her voice server is not installed — run {fix}'))
    elif local and missing:
        out.append(('err', 'TTS (voice)', f'her voice server is incomplete ({missing[0]} missing) — run {fix} again'))
    elif exe_ok:
        out.append(('warn', 'TTS (voice)', f'not running — will be started ({shown(tts.cmd[0])})'))
    else:
        out.append(('err', 'TTS (voice)', f'not running and its command is missing: {tts.cmd[0]}'))
    if local:
        level, detail = voice_folder(TTS_ENGINE / 'Speakers' / voice)
        out.append((level, 'her voice', detail if level == 'ok' else f'{detail} — run {fix} (type yes to accept her license)'))
    level, detail = live2d_models()
    out.append((level, 'Live2D model', detail if level == 'ok' else
                f'{detail} — run {fix} (her voice and her Live2D model come together)'))
    return out


# ── pre-flight ───────────────────────────────────────────────────────────────────────────
class Report:
    def __init__(self, printer=None):
        self.checks = []
        self.printer = printer

    def add(self, level, name, detail):
        self.checks.append({'level': level, 'name': name, 'detail': detail})
        if self.printer:
            self.printer(level, name, detail)

    def count(self, level):
        return sum(1 for c in self.checks if c['level'] == level)

    @staticmethod
    def _folded(parts):
        level = max((c['level'] for c in parts), key=lambda lv: LEVEL_RANK.get(lv, 0))
        bad = [f"{c['name']}: {c['detail']}" for c in parts if c['level'] != 'ok']
        return level, (' · '.join(bad) if bad else 'broker, zigbee2mqtt and the Zigbee dongle are ready')

    def fold_iot(self):
        """Collapse the smart-home checks (broker, zigbee2mqtt, dongle) into one 'smart home' line."""
        parts = [c for c in self.checks if c['name'] in IOT_CHECKS]
        if not parts:
            return
        level, detail = self._folded(parts)
        first = min(self.checks.index(c) for c in parts)
        self.checks = [c for c in self.checks if c['name'] not in IOT_CHECKS]
        self.checks.insert(first, {'level': level, 'name': 'smart home', 'detail': detail, 'parts': parts})

    @staticmethod
    def _service_for(check_name):
        """Which service a 'not running — will be started' check was waiting on."""
        if check_name.startswith('service '):
            return check_name[len('service '):]
        return {'smart home broker': 'mqtt', 'TTS (voice)': 'tts'}.get(check_name) or \
            (check_name[len('agent '):] if check_name.startswith('agent ') else None)

    def resolve(self, services):
        """After the start phase: a check that only said 'will be started' is green once that service is up.
        The pre-flight is a snapshot from before anything ran; this keeps it from nagging about services that
        the launcher has since brought up."""
        up = {s.name for s in services if s.state in ('up', 'external')}

        def settle(check):
            if 'will be started' in check.get('detail', '') and self._service_for(check['name']) in up:
                check['level'], check['detail'] = 'ok', f"{self._service_for(check['name'])} started by the launcher"

        for check in self.checks:
            if check.get('parts'):
                for part in check['parts']:
                    settle(part)
                check['level'], check['detail'] = self._folded(check['parts'])
            else:
                settle(check)

    def core_states(self, services):
        """One check per core component that did not come up, saying why ("brain start  the port is taken: …"):
        logs/preflight.json reaches the Setup page and the health light, so the reason shows in the browser too.
        A component that came up drops its check (item 33)."""
        names = {f'{s.name} start' for s in services if s.kind == 'core'}
        self.checks = [c for c in self.checks if c['name'] not in names]
        for svc in services:
            if svc.kind == 'core' and svc.state in ('failed', 'exited', 'timeout'):
                self.checks.append({'level': 'warn' if svc.state == 'timeout' else 'err', 'name': f'{svc.name} start',
                                    'detail': svc.why or svc.detail or svc.state})

    def iot_accepted(self, version, started):
        """The IoT safety statement was accepted while the launcher runs: its line turns green."""
        for check in self.checks:
            if check['name'] == IOT_STATEMENT_CHECK:
                check['level'] = 'ok'
                check['detail'] = f'accepted (version {version})' + (f' · started {", ".join(started)}' if started else '')

    def save(self):
        LOGS.mkdir(exist_ok=True)
        PREFLIGHT.write_text(json.dumps({'at': datetime.now().isoformat(timespec='seconds'), 'checks': self.checks,
                                         'summary': {k: self.count(k) for k in ('ok', 'warn', 'err')}},
                                        ensure_ascii=False, indent=2), 'utf-8')


def preflight(cfg, emb, vis, tools, tools_error, services, agents, procs, deep=False, printer=None, held_iot=(),
              port_moves=()):
    """held_iot: the smart-home services held back because the IoT safety statement was not accepted. port_moves:
    what move_reserved_ports did before settings.json was read (a row each, in place of that port's own row)."""
    report = Report(printer)
    for label, python in (('brain python', cfg.get('brain_python')), ('speech python', cfg.get('speech_python')),
                          ('embedder python', (emb or {}).get('python')), ('vision python', (vis or {}).get('python'))):
        if python is None:
            continue
        report.add('ok' if Path(python).is_file() else 'err', label, python if Path(python).is_file() else
                   f'missing: {python} — run {SETUP_CORE} (or fix the path in settings.json)')
    local_only = roles_all_local()
    speech_backend = cfg.get('speech_backend', 'local').strip().lower()
    needed = ['config/word_replace.json'] + (['brain/embed_server.py'] if emb else [])
    if speech_backend == 'local':
        needed.append('speech/models/model.pt')
    hints = {'speech/models/model.pt': f'speech recognition model (SenseVoiceSmall, ~940 MB): run {SETUP_CORE}'}
    missing = [p for p in needed if not (ROOT / p).is_file()]
    report.add('ok' if not missing else 'err', 'required files',
               'all present' if not missing else ', '.join(f'{p} ({hints.get(p, "missing")})' for p in missing))
    env = read_env(ROOT / 'brain' / '.env')
    if speech_backend == 'openrouter':
        stt_key = (os.environ.get('OPENROUTER_STT_API_KEY') or env.get('OPENROUTER_STT_API_KEY')
                   or os.environ.get('OPENROUTER_API_KEY') or env.get('OPENROUTER_API_KEY'))
        report.add('ok' if stt_key else 'err', 'speech backend',
                   f"OpenRouter · {cfg.get('speech_openrouter_model', 'microsoft/mai-transcribe-2')} (local VAD)" if stt_key
                   else 'OpenRouter transcription needs OPENROUTER_STT_API_KEY (or OPENROUTER_API_KEY) in brain/.env')
    elif speech_backend != 'local':
        report.add('err', 'speech backend', 'speech_backend must be local or openrouter in settings.json')
    key = env.get('OPENROUTER_API_KEY', '')
    if key and not key.startswith('sk-or-v1-your'):
        others = [k for k in ('DASHSCOPE_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY', 'CEREBRAS_API_KEY', 'ANTHROPIC_API_KEY') if env.get(k)]
        report.add('ok', 'API keys', f'OpenRouter key set (…{key[-4:]})' + (f'; also {", ".join(others)}' if others else ''))
    elif local_only:
        report.add('ok', 'API keys', 'no OpenRouter key, none needed: every role runs on Ollama (Setup → Brain)')
    else:
        report.add('warn', 'API keys', 'no OpenRouter key yet — paste it in the first-run wizard on the Use page (or put '
                                       'it after OPENROUTER_API_KEY= in brain/.env); until then she cannot answer. '
                                       'Fully local needs a model of 27B or more: switch the roles to Ollama on '
                                       'Setup → Brain')
    for level, name, detail in ollama_checks():
        report.add(level, name, detail)
    fit_check = model_fit_check(vis is not None)
    if fit_check:
        report.add(fit_check[0], 'local model fit', fit_check[1])
    level, detail = her_clock()
    report.add(level, 'her clock', detail)
    taken = listeners()
    names = {p['pid']: p['name'] for p in procs}
    unreadable = {p['pid'] for p in procs if not p['cmd']}      # a medium-integrity process cannot read an elevated one's command line
    ours = own_pids(procs)
    moved = {m[0]: m for m in port_moves}
    for label, port in (('brain port', cfg.get('brain_port')), ('web/speech port', cfg.get('chat_port')),
                        ('embedder port', (emb or {}).get('port')), ('vision port', (vis or {}).get('port')),
                        ('messengers port', messengers_port(cfg))):
        if port is None:
            continue
        if label in moved:
            report.add(*port_move_row(moved[label]))
            continue
        pid = taken.get(int(port))
        if pid and pid in ours:
            report.add('ok', label, f'{port} held by this OpenAmika (running)')
        elif pid and names.get(pid, '').lower().startswith('python') and pid in unreadable:
            report.add('err', label, f'{port} is held by pid {pid} (python.exe) started from an administrator window, which this '
                                     f'launcher cannot see into or stop — close that window, or run Stop_OpenAmika.bat as administrator, then start again')
        elif pid:
            report.add('err', label, f'{port} is in use by pid {pid} ({names.get(pid, "?")}) — that component cannot start; change settings.json or stop it')
        else:
            # a bind, not a connect: a port Windows reserved answers nothing, like a free one, yet no server can bind it
            state = bind_state(int(port))
            if not state:
                report.add('ok', label, f'{port} free')
            elif state == 'reserved':
                report.add(*port_move_row((label, dict(PORT_SETTINGS).get(label, (label,)), int(port), None,
                                           range_of(port, excluded_port_ranges()))))
            elif state == 'in use':
                report.add('err', label, f'{port} is in use by another program — that component cannot start; change '
                                         'settings.json or stop it')
            else:
                report.add('warn', label, f'{port}: {state}')
    if 'TTS port' in moved:
        report.add(*port_move_row(moved['TTS port']))
    for level, name, detail in voice_checks(cfg, services):
        report.add(level, name, detail)
    level, detail = paper_doll(cfg, None if tools_error else tools)
    if level:
        report.add(level, 'paper doll', detail)
    for svc in services:
        if svc.name == 'tts':
            continue
        exe_ok = Path(svc.exe).is_file() or bool(shutil.which(svc.cmd[0]))
        if svc.is_up():
            report.add('ok', f'service {svc.name}', 'already running')
        elif not exe_ok:
            hint = ' — run IoT\\Setup_IoT_admin.bat' if svc.name == 'mqtt' else ''
            report.add('err', f'service {svc.name}', f'command missing: {svc.cmd[0]}{hint}')
        elif not Path(svc.cwd).is_dir():
            report.add('err', f'service {svc.name}', f'folder missing: {svc.cwd}')
        else:
            report.add('warn', f'service {svc.name}', 'not running — will be started')
    held_servers = []                 # smart-home tool servers switched on but held off by the IoT safety statement
    iot_ok = iot_accepted(tools)
    if tools_error:
        report.add('err', 'tools', tools_error)
    elif tools and tools.get('enabled'):
        for name, scfg in (tools.get('mcpServers') or {}).items():
            if not isinstance(scfg, dict) or not scfg.get('enabled'):
                continue
            if not iot_ok and is_smart_home_server(name, scfg):
                held_servers.append(name)
                continue
            agent = next((a for a in agents if a.name == name), None)
            bridge = agent_bridge(name, scfg)
            if bridge:
                env_key, path = AGENT_HEALTH[bridge]
                base = ((scfg.get('env') or {}).get(env_key) or '').rstrip('/')
                if base and http_json(base + path) is not None:
                    report.add('ok', f'agent {name}', f'online at {base}')
                elif agent and not (Path(agent.exe).is_file() or shutil.which(agent.cmd[0])):
                    report.add('err', f'agent {name}', f'not running, and its start command does not exist: '
                                                       f'{agent.cmd[0]} — install it, or fix "autostart" in config/tools.json')
                elif agent:
                    report.add('warn', f'agent {name}', f'not running — will be started ({agent.cmd[0]})')
                else:
                    report.add('err', f'agent {name}', f'not running at {base or "?"} and no autostart configured (Tools page)')
                if bridge == 'hermes':
                    for level, check, detail in hermes_checks(scfg, agent):
                        report.add(level, check, detail)
            elif name == 'iot':
                port = int((scfg.get('env') or {}).get('AMIKA_MQTT_PORT') or 1883)
                mqtt = next((s for s in services if s.name == 'mqtt'), None)
                if port_open(port):
                    report.add('ok', 'smart home broker', f'MQTT on port {port}')
                elif mqtt:
                    report.add('warn', 'smart home broker', f'MQTT port {port} closed — will be started')
                else:
                    report.add('err', 'smart home broker', f'MQTT port {port} closed and no services.mqtt — add mosquitto (IoT\\Setup_IoT_admin.bat)')
                com, why = serial_dongle()
                report.add('ok' if com else 'err', 'Zigbee dongle', com if com else why)
            else:
                report.add('ok', f'tool {name}', 'configured (' + ('http' if scfg.get('url') else 'stdio') + ')')
    else:
        report.add('ok', 'tools', 'tool layer disabled' if tools is not None
                   else 'tool layer off (no config/tools.json yet; the Tools page creates it)')
    if held_iot or held_servers:
        rules = iot_rules()
        what = ', '.join([*(s.name for s in held_iot), *held_servers])
        report.add('warn', IOT_STATEMENT_CHECK,
                   f'not accepted (version {rules.IOT_DISCLAIMER_VERSION if rules else "?"}): {what} held off — turn '
                   f'IoT on from the Tools page to read and accept docs/IOT_DISCLAIMER.md')
    if deep:
        for key, packages in (
            ('brain_python', ['flask', 'flask_cors', 'flask_socketio', 'requests', 'dotenv', 'google.genai', 'tzdata']),
            ('speech_python', ['torch', 'torchaudio', 'soundfile', 'numpy', 'fastapi', 'uvicorn', 'funasr', 'modelscope', 'silero_vad', 'socketio', 'aiohttp', 'multipart']),
        ):
            code = ('import importlib.util; names=' + repr(packages) +
                    '; missing=[n for n in names if importlib.util.find_spec(n) is None]; print(",".join(missing))')
            if not cfg.get(key) or not Path(cfg[key]).is_file():
                continue                              # its own row already says it is missing
            try:
                out = subprocess.run([cfg[key], '-c', code], capture_output=True, text=True, timeout=120, creationflags=NO_WINDOW).stdout.strip()
                report.add('ok' if not out else 'err', f'{key} imports', 'all importable' if not out else f'missing modules: {out}')
            except Exception as error:
                report.add('err', f'{key} imports', f'could not run: {error}')
        for label, block, cmd in (('embedder check', emb, ['brain/embed_server.py', '--check', '--model', (emb or {}).get('model', '')]),
                                  ('vision check', vis, ['vision/vlm_service.py', '--check', '--device', (vis or {}).get('device', 'cuda')])):
            if not block or not Path(block['python']).is_file():
                continue                              # off, or its python row already says it is missing
            try:
                result = subprocess.run([block['python'], '-X', 'utf8', str(ROOT / cmd[0]), *cmd[1:]], capture_output=True,
                                        text=True, encoding='utf-8', errors='replace', timeout=300, creationflags=NO_WINDOW)
            except (OSError, subprocess.SubprocessError) as error:
                report.add('err', label, f'could not run {block["python"]}: {error} — run {SETUP_CORE}')
                continue
            # its last line says it all (a traceback ends with the error); long paths are cut at the end, not the start
            lines = [ln.strip() for ln in (result.stdout or result.stderr or '').splitlines() if ln.strip()]
            text = fit(lines[-1], 240) if lines else 'ok'
            if result.returncode != 0:
                if TOO_LONG.search(result.stdout + result.stderr):
                    report.add('err', label, f'{text} — the folder path is too long for Windows: move OpenAmika to a '
                                             'short folder such as C:\\OpenAmika and run the setup there')
                else:
                    report.add('err', label, f'{text} — run {SETUP_CORE}')
            elif 'not cached yet' in text:
                report.add('warn', label, f'{text} (or run {SETUP_CORE} to download it now)')
            else:
                report.add('ok', label, text)
    report.fold_iot()
    return report


# ── the dashboard ────────────────────────────────────────────────────────────────────────
class Dashboard(threading.Thread):
    """Redraws one screen in place: header, pre-flight (what is not green), one row per service."""

    def __init__(self, state):
        super().__init__(daemon=True, name='dashboard')
        self.state = state
        self.stop_flag = threading.Event()
        self.frame = 0
        self.out = sys.stdout

    def run(self):
        self.out.write('\x1b[?25l\x1b[2J')
        self.out.flush()
        while not self.stop_flag.is_set():
            try:
                self.draw()
            except Exception:
                pass
            self.frame += 1
            time.sleep(0.12)
        try:
            self.draw()
        finally:
            self.out.write('\x1b[?25h\n')
            self.out.flush()

    def compose(self, width):
        st = self.state
        rows = []
        clock = f'{datetime.now():%H:%M:%S}'
        phase = st.get('phase', '')
        head = f" {BOLD}{APP}{RESET}  {st.get('url', '')}  {DIM}{clock}{RESET}"
        rows.append((head, pad(f' {APP}  {st.get("url", "")}  {clock}', 0)))
        rows.append((f" {DIM}{phase}{RESET}", ' ' + phase))
        grace = st.get('grace')
        if grace is not None:
            text = ('stopped the previous instance: ' + ', '.join(sorted(set(grace.values())))) if grace else 'no previous instance running'
            rows.append((f"  {dot('ok')} grace            {text}", f'  * grace            {text}'))
        report = st.get('report')
        if report is not None:
            summary = (f"  {dot('ok')} {report.count('ok')} ok   {dot('warn')} {report.count('warn')} warning(s)   "
                       f"{dot('err')} {report.count('err')} problem(s)   {DIM}logs/preflight.json · Setup page{RESET}")
            rows.append((f" {BOLD}Pre-flight{RESET}", ' Pre-flight'))
            rows.append((summary, '  pre-flight summary'))
            for check in report.checks:
                if check['level'] == 'ok':
                    continue
                name = fit(check['name'], 16)
                rows.append((f"  {dot(check['level'])} {pad(name, 16)} {fit(check['detail'], width - 23)}", f"  * {name} {check['detail']}"))
        services = st.get('services', [])
        iot = [s for s in services if s.kind == 'service' and s.name in IOT_SERVICES]
        groups = (('Services', [s for s in services if s.kind == 'service' and s.name not in IOT_SERVICES]),
                  ('Agents', [s for s in services if s.kind == 'agent']),
                  ('Core', [s for s in services if s.kind == 'core']))
        for title, items in groups:
            if not items and not (title == 'Services' and iot):
                continue
            rows.append((f" {BOLD}{title}{RESET}", ' ' + title))
            for svc in items:
                marker, _level, desc = svc.light(self.frame)
                rows.append((f"  {marker} {pad(fit(svc.name, 16), 16)} {fit(desc, width - 23)}", f'  * {svc.name} {desc}'))
            if title == 'Services' and iot:
                marker, _level, desc = iot_light(iot, self.frame)
                rows.append((f"  {marker} {pad('smart home', 16)} {fit(desc, width - 23)}", f'  * smart home {desc}'))
        rows.append(('', ''))
        rows.append((f" {DIM}Ctrl+C stops everything started here · full output: logs\\<name>.log · spinning = launching or working{RESET}", ''))
        return rows

    def draw(self):
        width, height = shutil.get_terminal_size((110, 32))
        rows = self.compose(max(60, width))
        visible = rows[:max(1, height - 1)]
        frame = '\x1b[H' + '\n'.join(text + '\x1b[K' for text, _plain in visible) + '\x1b[J'
        with _print_lock:
            self.out.write(frame)
            self.out.flush()


def plain_summary(state):
    """After the dashboard closes (or when stdout is not a console): the final state as plain lines."""
    report = state.get('report')
    lines = [f"{APP}  {state.get('url', '')}  {datetime.now():%Y-%m-%d %H:%M:%S}"]
    if report is not None:
        lines.append(f"pre-flight: {report.count('ok')} ok · {report.count('warn')} warning(s) · {report.count('err')} problem(s)")
        lines += [f"  [{c['level']}] {c['name']}: {c['detail']}" for c in report.checks if c['level'] != 'ok']
    for svc in state.get('services', []):
        _marker, level, desc = svc.light(0)
        lines.append(f"  [{level}] {svc.name}: {desc}")
    return '\n'.join(lines)


class PageOpener:
    """Opens the web UI once, when the speech gateway answers: right after the start, or later, when a slow first start
    (new environments spend a minute or more on their first imports) answers after its start timeout."""

    def __init__(self, url, speech, enabled=True, opener=None):
        self.url, self.speech, self.done = url, speech, not enabled
        self.opener = opener or webbrowser.open

    def check(self):
        """Open the page now if speech is up and it was not opened yet; True when it did."""
        if self.done or self.speech.state != 'up':
            return False
        self.done = True
        self.opener(self.url)
        return True


# ── her desktop pet (launch.py --pet) ────────────────────────────────────────────────────
PET_REQUIREMENTS = os.path.join('brain', 'requirements-pet.txt')


def pet_check():
    """(level, detail) of the 'pet window' pre-flight line: pet.py runs in this Python (the brain's) and needs pywebview
    in it, on Windows. Without it the page opens in the browser as usual."""
    if os.name != 'nt':
        return 'err', 'her desktop pet window is Windows-only; the page opens in your browser instead'
    if importlib.util.find_spec('webview') is None:
        return 'err', (f'pywebview is not installed: {shown(sys.executable)} -m pip install -r {PET_REQUIREMENTS} '
                       '(the page opens in your browser until then)')
    return 'ok', 'pywebview found: her window opens when speech is up (pet.py)'


class PetWindow(Service):
    """Her desktop pet window (pet.py), opened in place of the browser page once speech answers and stopped with the
    rest. Closing it is no failure: she keeps running, and pet.py opens it again."""

    def __init__(self, url):
        super().__init__('pet window', [sys.executable, '-X', 'utf8', ROOT / 'pet.py', '--url', url], kind='core',
                         log=LOGS / 'pet.log', url=url)
        self.detail = 'opens when speech is up'

    def is_up(self):
        return self.proc is not None and self.proc.poll() is None

    def open(self, _url=None):
        self.start()

    def poll(self):
        if self.proc is not None and self.proc.poll() is not None and self.state in ('starting', 'up'):
            code = self.proc.returncode
            if code == 0:
                self.state, self.detail = 'off', f'closed; she keeps running. Open it again: {shown(sys.executable)} pet.py'
            else:
                self.state, self.detail = 'failed', f'exited with code {code} — see logs/{self.log.name}'
            log(f'{self.name}: {self.detail}')
        elif self.state == 'starting' and self.is_up():
            self.state, self.detail = 'up', ''
            log(f'{self.name}: up')

    def stop(self):
        if self.started_by_me and self.proc is not None and self.proc.poll() is None:
            kill_tree(self.proc.pid)         # with its WebView2 processes; its position is saved as it moves


# ── why a component did not start (item 33) ─────────────────────────────────────────────
# A red row used to say only "exited with code 1" (or "not started"), and the help desk asked why every time. The
# pre-flight's own verdict on the component comes first (its port, its Python), then what its log says since its last
# start: a missing module, a port held by someone else, a model that does not fit, a key that is missing.
CORE_CHECKS = {'brain': ('brain python', 'brain port'), 'speech': ('speech python', 'web/speech port'),
               'embedder': ('embedder python', 'embedder port', 'embedder check'),
               'vision': ('vision python', 'vision port', 'vision check'),
               'messengers': ('speech python', 'messengers port')}
START_FAILURES = (
    (re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'"),
     'the environment is broken: the Python module {0} is missing — run the setup again ({setup})'),
    (re.compile(r'DLL load failed[^\n]*|ImportError: [^\n]*'),
     'the environment is broken: {match} — run the setup again ({setup})'),
    (re.compile(r'ZoneInfoNotFoundError|No time zone found'),
     'the environment is broken: no time-zone database (tzdata) — run the setup again ({setup})'),
    (re.compile(r'Errno 10048|Errno 10013|Errno 98\b|address already in use|only one usage of each socket address'
                r'|error while attempting to bind', re.I),
     'the port is taken: another program (or an earlier OpenAmika) holds it — run Stop_OpenAmika.bat, or change '
     'the port in settings.json'),
    (re.compile(r'requires more system memory|CUDA out of memory|OutOfMemoryError|out of memory|cudaMalloc'
                r'|not enough memory', re.I),
     'the model does not fit in memory: {match} — use OpenRouter (Setup → Brain), close other programs that use '
     'the graphics card, or turn her eyes off ("vision": {{"enabled": false}} in settings.json)'),
    (re.compile(r'A key and both mandatory metering callbacks are required|OPENROUTER_API_KEY|HTTP 401', re.I),
     'the key is missing or refused: paste your OpenRouter key (first-run wizard, or Setup → Brain)'),
)


def log_since_start(name):
    """logs/<name>.log since the launcher's last '--- … starting <name>:' line (its newest start), as lines."""
    try:
        with open(LOGS / f'{name}.log', 'rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 262144))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        return []
    cut = text.rfind(f' starting {name}: ')
    if cut >= 0:
        text = text[text.rfind('\n', 0, cut) + 1:]
    return text.splitlines()


def start_failure(svc, report=None):
    """Why core component svc is not up, in words ('' when nothing specific is known)."""
    checks = {c['name']: c for c in (report.checks if report is not None else [])}
    for name in CORE_CHECKS.get(svc.name, ()):
        check = checks.get(name)
        if check and check['level'] == 'err':
            what = 'the port is taken' if name.endswith('port') else 'the environment is broken'
            return f"{what}: {check['detail']}"
    lines = log_since_start(svc.name)
    text = '\n'.join(lines)
    for pattern, words in START_FAILURES:
        match = pattern.search(text)
        if match:
            return words.format(*match.groups(), setup=SETUP_CORE, match=fit(match.group(0), 90))
    tail = [line.strip() for line in lines if line.strip() and not line.startswith('---')]
    return f'its log ends with: {fit(tail[-1], 160)}' if tail else ''


def brain_caveat(report=None):
    """What keeps a running brain from answering: no OpenRouter key for a cloud role, Ollama down or a model not
    pulled, a local model that does not fit ('' when nothing). The key is read again each time (the wizard saves it
    while the launcher runs)."""
    if not roles_all_local():
        key = read_env(ROOT / 'brain' / '.env').get('OPENROUTER_API_KEY', '')
        if not key or key.startswith('sk-or-v1-your'):
            return 'the key is missing: paste your OpenRouter key (first-run wizard, or Setup → Brain)'
    for check in (report.checks if report is not None else []):
        if (check['name'] == 'Ollama' or check['name'].startswith('Ollama model')) and check['level'] == 'err':
            return check['detail']
        if check['name'] == 'local model fit' and check['level'] != 'ok':
            return 'the model does not fit: ' + check['detail']
    return ''


# ── launch.py --wipe-memory (item 49) ─────────────────────────────────────────────────────
WIPE_WAIT_S = 12


def _post_json(url, body, headers, timeout=120):
    """(status, payload) of a JSON POST; status 0 when nothing answered."""
    request = Request(url, data=json.dumps(body).encode('utf-8'), method='POST',
                      headers={'Content-Type': 'application/json', **headers})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b'{}')
    except HTTPError as error:
        try:
            payload = json.loads(error.read() or b'{}')
        except ValueError:
            payload = {}
        return error.code, payload if isinstance(payload, dict) else {}
    except (URLError, OSError, ValueError):
        return 0, {}


def wipe_memory(cfg, force=False, out=print):
    """Wipe her memory, backup first: Setup → Danger zone → Wipe memory, for scripts. While she runs it goes through
    her own wipe — the speech gateway (which also disconnects the stage), else the brain — waiting up to WIPE_WAIT_S
    for her to finish; force stops what she is doing first. When she is not running, the same transaction runs on
    disk (brain/f_reset.wipe_offline). Returns the exit code: 0 wiped, 2 she is busy, 1 failed."""
    brain_port, chat_port = int(cfg['brain_port']), int(cfg['chat_port'])
    body = {'confirmation': 'WIPE', **({'force': True} if force else {'wait_s': WIPE_WAIT_S})}
    out(f'{APP}: wiping her memory (a backup is taken first)' + (' — forced: she stops what she is doing' if force else '') + '…')
    if (http_json(f'http://127.0.0.1:{brain_port}/api/evo/health') or {}).get('app') == APP:
        gateway = f'http://127.0.0.1:{chat_port}'
        if (http_json(gateway + '/health') or {}).get('app') == APP:
            status, payload = _post_json(gateway + '/api/build_f/wipe', body,
                                         {'Origin': gateway, 'X-Build-F-Action': 'wipe-memory'})
        else:
            status, payload = _post_json(f'http://127.0.0.1:{brain_port}/api/build_f/wipe', body,
                                         {'X-Build-F-Action': 'wipe-memory'})
        if status == 200 and payload.get('ok'):
            out(f"Memory wiped. Backup: {shown(ROOT / 'memory_f' / 'backups' / str(payload.get('backup_id')))}")
            return 0
        detail = payload.get('detail', payload.get('error'))
        message = detail.get('message') if isinstance(detail, dict) else detail
        if status == 409:
            out(message or 'Amika is busy.')
            if not force:
                out('Run it again with --force to stop what she is doing and wipe anyway (the backup is still taken).')
            return 2
        out(f'The wipe failed (HTTP {status or "no answer"}): {message or "see logs/brain.log"}')
        return 1
    if port_open(brain_port):
        out(f'Something holds the brain port {brain_port} but does not answer like her brain: stop OpenAmika '
            '(Stop_OpenAmika.bat), then run this again.')
        return 1
    if str(ROOT / 'brain') not in sys.path:
        sys.path.insert(0, str(ROOT / 'brain'))
    try:
        import f_reset
        result = f_reset.wipe_offline(ROOT)
    except Exception as error:
        out(f'The wipe failed: {error}')
        return 1
    retired = result.get('retired_requests')
    out(f"She is not running, so it was done on disk. Memory wiped. Backup: {shown(result['backup_dir'])}"
        + (f' ({retired} unfinished request(s) retired)' if retired else ''))
    return 0


# ── launch.py --diagnose (item 33) ────────────────────────────────────────────────────────
SECRET_NAME = re.compile(r'key|token|secret|passw|auth|cookie', re.I)
SECRET_TEXT = re.compile(r'sk-or-v1-[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{20,}'
                         r'|Bearer\s+[A-Za-z0-9._~+/=-]{8,}|[MN][A-Za-z\d_-]{23,25}\.[\w-]{6}\.[\w-]{27,}'
                         r'|\b\d{8,10}:[A-Za-z0-9_-]{35}\b')


def known_secrets():
    """Secret values this install holds (brain/.env, the bots' tokens, the helpers' keys), cut out of anything shared."""
    values = {v for v in read_env(ROOT / 'brain' / '.env').values() if len(v) >= 8}

    def walk(node, name=''):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, str(key))
        elif isinstance(node, list):
            for value in node:
                walk(value, name)
        elif isinstance(node, str) and len(node) >= 8 and SECRET_NAME.search(name):
            values.add(node)
    for rel in ('config/messengers.json', 'config/tools.json'):
        try:
            walk(json.loads((ROOT / rel).read_text('utf-8-sig')))
        except (OSError, ValueError):
            pass
    for name in ('HF_TOKEN', 'OPENROUTER_API_KEY', 'API_SERVER_KEY'):
        if len(os.environ.get(name, '')) >= 8:
            values.add(os.environ[name])
    return sorted(values, key=len, reverse=True)


def redact(text, secrets):
    for value in secrets:
        text = text.replace(value, '<removed>')
    return SECRET_TEXT.sub('<removed>', text)


def scrub(node):
    """A settings document with every value under a key-like name (key, token, secret, password…) removed."""
    if isinstance(node, dict):
        return {k: ('<removed>' if SECRET_NAME.search(str(k)) and isinstance(v, (str, int, float)) and v != ''
                    else scrub(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    return node


def tail_lines(path, count=200):
    try:
        with open(path, 'rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 1048576))
            return handle.read().decode('utf-8', 'replace').splitlines()[-count:]
    except OSError:
        return []


def run_python(python, code, timeout=120):
    """What `python -c code` prints (its last error line when it fails)."""
    if not python or not Path(python).is_file():
        return f'missing: {python}'
    try:
        result = subprocess.run([python, '-c', code], capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as error:
        return f'could not run: {error}'
    lines = [ln for ln in ((result.stdout or '') + '\n' + (result.stderr or '')).splitlines() if ln.strip()]
    return (lines[0] if result.returncode == 0 else lines[-1]) if lines else '?'


def system_report(cfg, emb, vis):
    """Python and driver versions, the GPUs and their memory, free disk, the models she runs on (no secrets)."""
    lines = [f'{APP} diagnose {datetime.now().isoformat(timespec="seconds")}', f'folder: {ROOT}',
             f'os: {platform.platform()}', f'launcher python: {sys.version.split()[0]} ({sys.executable})']
    try:
        rev = subprocess.run(['git', '-C', str(ROOT), 'log', '-1', '--format=%h %cs %s'], capture_output=True, text=True,
                             timeout=15, creationflags=NO_WINDOW).stdout.strip()
        lines.append(f'checkout: {rev or "not a git clone"}')
    except (OSError, subprocess.SubprocessError):
        lines.append('checkout: git not found')
    cfg = cfg or {}
    for label, holder, key in (('brain python', cfg, 'brain_python'), ('speech python', cfg, 'speech_python'),
                               ('embedder python', emb or {}, 'python'), ('vision python', vis or {}, 'python')):
        if holder.get(key):
            lines.append(f'{label}: {run_python(holder[key], "import sys; print(sys.version.split()[0])")} ({holder[key]})')
    if cfg.get('speech_python'):
        lines.append('speech torch: ' + run_python(cfg['speech_python'], 'import torch; print(torch.__version__, "cuda",'
                                                    ' torch.version.cuda, "available", torch.cuda.is_available())'))
    gpus = gpu_info()
    for gpu in gpus:
        lines.append(f"gpu: {gpu['name']} · driver {gpu['driver']} · memory {gpu['total_mb']} MB total, "
                     f"{gpu['used_mb']} MB used, {gpu['free_mb']} MB free")
    if not gpus:
        lines.append('gpu: no NVIDIA GPU answered (nvidia-smi)')
    try:
        disk = shutil.disk_usage(ROOT)
        lines.append(f'disk: {disk.free / 2**30:.1f} GB free of {disk.total / 2**30:.1f} GB ({ROOT.anchor})')
    except OSError as error:
        lines.append(f'disk: {error}')
    try:
        picks = json.loads((ROOT / 'config' / 'llm_roles.json').read_text('utf-8-sig')).get('selections') or {}
        lines.append('models: ' + ', '.join(f'{role} {key}' for role, key in picks.items()))
    except (OSError, ValueError, AttributeError):
        lines.append('models: config/llm_roles.json unreadable')
    if ollama_roles():
        loaded = http_json(ollama_base() + '/api/ps', timeout=3)
        rows = (loaded or {}).get('models') if isinstance(loaded, dict) else None
        lines.append('ollama loaded: ' + (', '.join(f"{m.get('name')} {m.get('size', 0) / 1e9:.1f} GB, "
                                                    f"{m.get('size_vram', 0) / 1e9:.1f} GB on the GPU"
                                                    for m in rows if isinstance(m, dict)) if rows else 'nothing (or not answering)'))
    return '\n'.join(lines) + '\n'


def diagnose(cfg, emb=None, vis=None, tools=None, tools_error=None, services=(), agents=(), held_iot=(), out=print):
    """python launch.py --diagnose: one zip in logs/ to attach to a help request — the pre-flight run now (the report of
    her last start too), the last 200 lines of each log, settings.json without keys, Python/driver versions, free disk
    and GPU memory. No .env, and every secret value this install holds is cut from all of it. Returns its path."""
    LOGS.mkdir(exist_ok=True)
    path = LOGS / f'diagnose-{datetime.now():%Y%m%d-%H%M%S}.zip'
    secrets = known_secrets()
    out(f'{APP}: collecting a diagnose zip (the pre-flight runs again; nothing is started)…')
    if cfg is not None:
        try:
            report = preflight(cfg, emb, vis, tools, tools_error, list(services), list(agents), list_processes(),
                               deep=True, held_iot=held_iot)
            now = {'at': datetime.now().isoformat(timespec='seconds'), 'checks': report.checks,
                   'summary': {k: report.count(k) for k in ('ok', 'warn', 'err')}}
        except Exception as error:
            now = {'error': f'the pre-flight failed: {type(error).__name__}: {error}'}
    else:
        now = {'error': 'settings.json not found: the setup has not finished (README, "Set her up")'}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('README.txt', f'{APP} diagnose bundle (launch.py --diagnose). preflight-now.json: the checks run '
                                      'just now; preflight-last-start.json: what her last start saw; logs/: the last 200 '
                                      'lines of each log; settings.json: without keys; system.txt: versions, GPU, disk. '
                                      'brain/.env is never included, and known secrets are replaced by <removed>.\n')
        bundle.writestr('preflight-now.json', redact(json.dumps(now, ensure_ascii=False, indent=2), secrets))
        if PREFLIGHT.is_file():
            bundle.writestr('preflight-last-start.json', redact(PREFLIGHT.read_text('utf-8', errors='replace'), secrets))
        for log in sorted(LOGS.glob('*.log')):
            bundle.writestr(f'logs/{log.name}', redact('\n'.join(tail_lines(log)) + '\n', secrets))
        try:
            settings = json.loads((ROOT / 'settings.json').read_text('utf-8-sig'))
            bundle.writestr('settings.json', redact(json.dumps(scrub(settings), ensure_ascii=False, indent=2), secrets))
        except (OSError, ValueError) as error:
            bundle.writestr('settings.json', f'unreadable: {error}\n')
        bundle.writestr('system.txt', redact(system_report(cfg, emb, vis), secrets))
    return path


# ── main ─────────────────────────────────────────────────────────────────────────────────
def core_services(cfg, emb, vis):
    ports = [int(cfg['brain_port']), int(cfg['chat_port'])]
    texts = f'http://127.0.0.1:{messengers_port(cfg)}'          # the Discord / Telegram service
    env = dict(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               HANE_HOST='127.0.0.1', HANE_PORT=str(ports[0]),
               AMIKA_CHAT_URL=f'http://localhost:{ports[1]}',
               HANE_URL=f'http://127.0.0.1:{ports[0]}',
               AMIKA_LOCAL_MODEL_URL=cfg['local_model_url'],
               YUMEHA_HOST='127.0.0.1', YUMEHA_PORT=str(ports[1]),
               YUMEHA_MODEL=str(ROOT / 'speech/models'), YUMEHA_DEVICE=cfg['speech_device'],
                YUMEHA_LANGUAGE=cfg.get('speech_language', 'auto'),
                AMIKA_STT_BACKEND=cfg.get('speech_backend', 'local'),
                AMIKA_STT_MODEL=cfg.get('speech_openrouter_model', 'microsoft/mai-transcribe-2'),
                AMIKA_STT_TIMEOUT_S=str(cfg.get('speech_timeout_s', 30)),
               AMIKA_TTS_URL=cfg['tts_url'], AMIKA_TTS_CHARACTER=cfg['tts_character'],
               AMIKA_TTS_TRANSPORT=cfg.get('tts_transport', 'local'),
               AMIKA_TTS_TIMEOUT_S=str(cfg.get('tts_timeout_s', 240)),
               AMIKA_MESSENGERS_PORT=str(messengers_port(cfg)), AMIKA_MESSENGERS_URL=texts,
               AMIKA_PET_PYTHON=sys.executable)     # the Use page's "Desktop pet" button: the Python of --pet
    if emb:
        env['AMIKA_EMBEDDING_URL'] = f"http://127.0.0.1:{int(emb['port'])}"
    if vis:
        env['AMIKA_VLM_URL'] = f"http://127.0.0.1:{int(vis['port'])}"
    out = []
    if emb:
        out.append(Service('embedder', [emb['python'], '-u', str(ROOT / 'brain/embed_server.py'),
                                        '--port', str(int(emb['port'])), '--model', emb['model'], '--tag', emb['tag'], '--dims', str(int(emb['dims'])),
                                        '--dtype', emb.get('dtype', 'fp16'), '--device', emb.get('device', 'cuda')],
                           cwd=ROOT / 'brain', env=env, health=f"http://127.0.0.1:{int(emb['port'])}/health", expect_ok=True,
                           timeout=int(emb.get('load_timeout_s', 240)), kind='core', url=f"http://127.0.0.1:{int(emb['port'])}"))
    out.append(Service('brain', [cfg['brain_python'], '-u', str(ROOT / 'brain/webui.py')], cwd=ROOT / 'brain', env=env,
                       health=f'http://127.0.0.1:{ports[0]}/api/evo/health', expect_app=True, timeout=90, kind='core',
                       url=f'http://127.0.0.1:{ports[0]}'))
    if vis:
        out.append(Service('vision', [vis['python'], '-u', str(ROOT / 'vision/vlm_service.py'),
                                      '--port', str(int(vis['port'])), '--model', vis['model'],
                                      '--feed-url', f"ws://127.0.0.1:{ports[1]}/ws/vlm_feed",
                                      '--max-tokens', str(int(vis.get('max_tokens', 60))),
                                      '--max-dps', str(float(vis.get('max_dps', 1.0))),
                                      '--device', vis.get('device', 'cuda')],
                           cwd=ROOT / 'vision', env=env, health=f"http://127.0.0.1:{int(vis['port'])}/api/health", expect_ok=True,
                           timeout=int(vis.get('load_timeout_s', 300)), kind='core', url=f"http://127.0.0.1:{int(vis['port'])}"))
    speech_env = dict(env)
    if cfg.get('speech_backend', 'local').strip().lower() == 'openrouter':
        secrets = read_env(ROOT / 'brain' / '.env')
        speech_env['OPENROUTER_STT_API_KEY'] = (os.environ.get('OPENROUTER_STT_API_KEY') or secrets.get('OPENROUTER_STT_API_KEY')
                                               or os.environ.get('OPENROUTER_API_KEY') or secrets.get('OPENROUTER_API_KEY', ''))
    out.append(Service('speech', [cfg['speech_python'], '-u', str(ROOT / 'speech/api.py')], cwd=ROOT / 'speech', env=speech_env,
                       health=f'http://127.0.0.1:{ports[1]}/health', expect_app=True, timeout=90, kind='core',
                       url=f'http://localhost:{ports[1]}'))
    # text channels (Discord / Telegram bots): its light says which bots are online, or that none is set up yet
    out.append(Service('messengers', [cfg['speech_python'], '-u', str(ROOT / 'messengers/service.py')], cwd=ROOT / 'messengers',
                       env=env, health=f'{texts}/health', expect_app=True, timeout=60, kind='core',
                       url=texts, live_light=True))
    return out


def write_instance(all_services):
    LOGS.mkdir(exist_ok=True)
    INSTANCE.write_text(json.dumps({
        'app': APP, 'launcher_pid': os.getpid(), 'started_at': datetime.now().isoformat(timespec='seconds'),
        'processes': [{'name': s.name, 'pid': s.proc.pid, 'exe': s.exe, 'cmd': ' '.join(s.cmd), 'kind': s.kind}
                      for s in all_services if s.started_by_me and s.proc is not None and s.proc.poll() is None]},
        ensure_ascii=False, indent=2), 'utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--check', action='store_true', help='validate config, environments, imports and files; start nothing')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--pet', action='store_true', help='open her desktop pet window (pet.py) instead of the browser page')
    parser.add_argument('--wipe-memory', action='store_true',
                        help='wipe her memory, a backup first (Setup → Wipe memory, for scripts); waits a few seconds '
                             'if she is busy')
    parser.add_argument('--force', action='store_true', help='with --wipe-memory: stop what she is doing and wipe anyway')
    parser.add_argument('--diagnose', action='store_true',
                        help='write one zip in logs/ for a help request (pre-flight, log tails, settings without keys, '
                             'versions, disk, GPU) and print its path')
    args = parser.parse_args()
    enable_vt()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    settings_path = ROOT / 'settings.json'
    if not settings_path.is_file():
        if args.diagnose:          # the most common first-day problem is exactly this: the zip still helps
            print(f'Diagnose zip: {diagnose(None)}')
            return
        raise RuntimeError(f'settings.json not found — run the setup first: {SETUP_CORE} (README, "Set her up"); it '
                           'creates settings.json')
    # a port of hers that Windows reserved (Hyper-V, WSL2, Docker Desktop) moves before anything reads settings.json
    port_moves = move_reserved_ports(settings_path)
    cfg = json.loads(settings_path.read_text(encoding='utf-8-sig'))
    # interpreter paths may be relative to this folder (envs\brain\Scripts\python.exe from scripts/setup_core.py)
    for holder in (cfg, cfg.get('embedder') or {}, cfg.get('vision') or {}):
        for key in ('brain_python', 'speech_python', 'python'):
            if isinstance(holder.get(key), str) and holder[key].strip():
                holder[key] = str(resolve(holder[key]))
    if args.wipe_memory:
        sys.exit(wipe_memory(cfg, force=args.force))
    emb = embedder_settings(cfg)
    vis = vision_settings(cfg)
    ports = ([int(cfg['brain_port']), int(cfg['chat_port']), messengers_port(cfg)] + ([int(emb['port'])] if emb else [])
             + ([int(vis['port'])] if vis else []))
    if len(set(ports)) != len(ports) or any(not 1024 <= p <= 65535 for p in ports):
        raise RuntimeError('Use distinct ports between 1024 and 65535.')
    tools, tools_error = load_tools()
    # the MQTT broker and zigbee2mqtt wait for the IoT safety statement to be accepted (Tools page)
    services, held_iot, agents = start_plan(cfg, tools)
    url = f"http://localhost:{cfg['chat_port']}"

    if args.diagnose:
        path = diagnose(cfg, emb, vis, tools, tools_error, services, agents, held_iot)
        print(f'Diagnose zip: {path}')
        print('Attach it to your help request (Discord or a GitHub issue). It holds no keys: brain/.env is not in it.')
        return

    if args.check:
        color = sys.stdout.isatty()
        marks = {'ok': 'OK', 'warn': '--', 'err': '!!'}

        def printer(level, name, detail):
            mark = marks.get(level, '--')
            print(f"  {COLOR.get(level, '') + mark + RESET if color else mark} {name:<22} {detail}")
        print(f'{APP} pre-flight  {ROOT}')
        report = preflight(cfg, emb, vis, tools, tools_error, services, agents, list_processes(), deep=True, printer=printer,
                           held_iot=held_iot, port_moves=port_moves)
        if args.pet:
            level, detail = pet_check()
            report.add(level, 'pet window', detail)
        report.save()
        print(f"  {report.count('ok')} ok · {report.count('warn')} warning(s) · {report.count('err')} problem(s). "
              "No models loaded and no API requests made.")
        print(f'  her page: {url}' + ('  (moved: Windows reserved the old port)' if any(
            m[0] == 'web/speech port' and m[3] for m in port_moves) else ''))
        print('result: ready' if not report.count('err') else 'result: fix the !! lines above (-- lines are informational)')
        sys.exit(1 if report.count('err') else 0)

    cores = core_services(cfg, emb, vis)
    everything = services + agents + cores
    _CLOSE['services'] = everything
    install_close_handler()
    state = {'url': url, 'phase': 'grace: looking for an earlier instance…', 'services': everything}
    interactive = sys.stdout.isatty()
    dashboard = Dashboard(state) if interactive else None
    if dashboard:
        dashboard.start()
    watchers = []
    try:
        log(f'launcher started ({ROOT})')
        procs = list_processes()
        state['grace'] = grace_cleanup(procs)
        state['phase'] = 'pre-flight…'
        procs = list_processes()
        report = preflight(cfg, emb, vis, tools, tools_error, services, agents, procs, held_iot=held_iot,
                           port_moves=port_moves)
        pet = None
        if args.pet and not args.no_browser:
            level, detail = pet_check()
            report.add(level, 'pet window', detail)
            if level == 'ok':
                pet = PetWindow(url)
                everything.append(pet)
        report.save()
        state['report'] = report

        z2m = next((s for s in services if s.name == 'zigbee2mqtt'), None)
        dongle_port = ''
        if z2m:
            dongle_port, _why = serial_dongle()
            if dongle_port:
                z2m.env.setdefault('ZIGBEE2MQTT_CONFIG_SERIAL_PORT', dongle_port)
                if not z2m.is_up():
                    state['phase'] = 'resetting the Zigbee dongle…'
                    log(f'zigbee2mqtt: priming the dongle: {prime_dongle(dongle_port)}')

        def tick():
            for svc in everything:
                svc.poll()

        # services and agents first, in parallel, so the brain finds its helpers already up
        for svc in services + agents:
            state['phase'] = f'starting {svc.name}…'
            if svc.after:
                dep = next((s for s in services if s.name == svc.after), None)
                if dep is not None and dep.state == 'starting':
                    dep.wait_up(tick=tick)
            svc.start()
            if svc.state == 'starting':
                watchers.append(LogWatch(svc))
                watchers[-1].start()
        def explain():
            """Each core row that is not up says why; an up brain that cannot answer says that (item 33)."""
            for svc in cores:
                svc.why = start_failure(svc, report) if svc.state in ('failed', 'exited', 'timeout') else ''
                if svc.name == 'brain':
                    svc.caveat = brain_caveat(report) if svc.state in ('up', 'external') else ''
            report.core_states(cores)

        # the core, in order, each waiting for its health check — a failure is a red row, not a stop
        for svc in cores:
            state['phase'] = f'starting {svc.name}…'
            svc.start()
            if svc.state == 'starting':
                watchers.append(LogWatch(svc))
                watchers[-1].start()
                svc.wait_up(tick=tick)
            explain()
        state['phase'] = 'waiting for services…'
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and any(s.state == 'starting' for s in services + agents):
            tick()
            time.sleep(0.4)
        report.resolve(everything)       # "will be started" items that did start are green now, here and on the Setup page
        explain()
        report.save()
        write_instance(everything)
        page = PageOpener(url, next(s for s in cores if s.name == 'speech'), enabled=not args.no_browser,
                          opener=pet.open if pet else None)
        page.check()
        state['phase'] = 'running'
        last_instance = time.monotonic()
        retried = set()
        last_live = 0.0
        report_starts = [c for c in report.checks if c['name'].endswith(' start')]
        last_iot_check = 0.0
        iot_waiting = any(c['name'] == IOT_STATEMENT_CHECK and c['level'] != 'ok' for c in report.checks)
        while True:
            time.sleep(0.5)
            before = [(s.name, s.state) for s in everything]
            tick()
            if page.check():                 # speech answered after its start timeout (a first start): open it now
                log(f'speech: up after its start timeout; opened {url}')
            # the IoT safety statement accepted on the Tools page while this runs: start the held smart-home services
            if iot_waiting and time.monotonic() - last_iot_check > 3:
                last_iot_check = time.monotonic()
                if iot_accepted(load_tools()[0]):
                    iot_waiting = False
                    late, held_iot = sorted(held_iot, key=lambda s: 1 if s.after else 0), []
                    log('IoT safety statement accepted' + (': starting ' + ', '.join(s.name for s in late) if late else ''))
                    z2m = next((s for s in late if s.name == 'zigbee2mqtt'), None)
                    if z2m:
                        dongle_port, _why = serial_dongle()
                        if dongle_port:
                            z2m.env.setdefault('ZIGBEE2MQTT_CONFIG_SERIAL_PORT', dongle_port)
                            if not z2m.is_up():
                                log(f'zigbee2mqtt: priming the dongle: {prime_dongle(dongle_port)}')
                    for svc in late:
                        services.append(svc)
                        everything.append(svc)
                        if svc.after:
                            dep = next((s for s in services if s.name == svc.after), None)
                            if dep is not None and dep.state == 'starting':
                                dep.wait_up(tick=tick)
                        svc.start()
                        if svc.state == 'starting':
                            watchers.append(LogWatch(svc))
                            watchers[-1].start()
                    report.iot_accepted(iot_rules().IOT_DISCLAIMER_VERSION, [s.name for s in late])
                    report.save()
            if time.monotonic() - last_live > 5:         # rows whose light comes from their own health JSON
                last_live = time.monotonic()
                for svc in everything:
                    if svc.live_light and svc.state == 'up' and not svc.is_up():
                        svc.state, svc.detail = 'exited', f'stopped answering at {svc.health}'
                        log(f'{svc.name}: {svc.detail}')
                explain()                        # e.g. the key pasted in the wizard: the brain row turns green
                if [c for c in report.checks if c['name'].endswith(' start')] != report_starts:
                    report_starts = [c for c in report.checks if c['name'].endswith(' start')]
                    report.save()
            # a service that died while coming up gets one more try (the Zigbee dongle, for one, sometimes needs
            # a second reset); the core components do not — their logs say why they stopped
            for svc in services:
                if svc.state == 'failed' and svc.started_by_me and svc.name not in retried and svc.started_at \
                        and time.monotonic() - svc.started_at < 180:
                    retried.add(svc.name)
                    if svc.name == 'zigbee2mqtt' and dongle_port:
                        log(f'zigbee2mqtt: retry — priming the dongle again: {prime_dongle(dongle_port)}')
                    else:
                        log(f'{svc.name}: retrying once')
                    time.sleep(3.0)
                    svc.start()
                    if svc.state == 'starting':
                        watchers.append(LogWatch(svc))
                        watchers[-1].start()
            changed = before != [(s.name, s.state) for s in everything]
            if changed or time.monotonic() - last_instance > 60:
                if changed:
                    report.resolve(everything)       # e.g. zigbee2mqtt came up on its retry: its pre-flight line turns green
                    explain()                        # a core component that died says why, here and on the Setup page
                    report_starts = [c for c in report.checks if c['name'].endswith(' start')]
                    report.save()
                write_instance(everything)
                last_instance = time.monotonic()
    except KeyboardInterrupt:
        state['phase'] = 'stopping everything started here…'
        log('Ctrl+C: stopping everything started here')
    finally:
        for watcher in watchers:
            watcher.stop_flag.set()
        for svc in reversed(everything):
            svc.stop()
        try:
            INSTANCE.unlink()
        except OSError:
            pass
        if dashboard:
            dashboard.stop_flag.set()
            dashboard.join(timeout=2)
        print(plain_summary(state))
        print(f'{APP} stopped.')


def keep_window_open(code, failed):
    """Start_OpenAmika.bat runs the launcher in a window of its own (setting OPENAMIKA_PAUSE_ON_ERROR=1), and that
    window closes the moment the launcher exits. After a failure: print the log's path last and wait for Enter, so
    the error can be read. A launcher that is stopped (Ctrl+C, Stop_OpenAmika, a new start) exits 0 or is ended from
    outside, and its window closes as before."""
    from_bat = os.environ.get('OPENAMIKA_PAUSE_ON_ERROR') == '1'
    if not code or not (failed or from_bat):
        return
    print(f'{APP} stopped with exit code {code}. The launcher log: {LOGS / "launcher.log"} '
          f'(each part of her writes its own logs\\<name>.log next to it)', file=sys.stderr, flush=True)
    if from_bat and sys.stdin is not None and sys.stdin.isatty():
        try:
            input('Press Enter to close this window.')
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == '__main__':
    code, failed = 0, False
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    except Exception as error:
        log(f'could not start: {type(error).__name__}: {error}\n{traceback.format_exc()}')
        print(f'{APP} could not start: {error}', file=sys.stderr)
        code, failed = 1, True
    keep_window_open(code, failed)
    sys.exit(code)
