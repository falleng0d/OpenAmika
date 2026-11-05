"""launch.py pre-flight on a fresh checkout: it must never crash, and it must name what is missing and how to get it,
her voice and her Live2D model above all. Offline: nothing is started and no port of a running stack is touched."""
import contextlib
import http.server
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("openamika_launch_preflight", ROOT / "launch.py")
launch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launch)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def example_cfg(tmp_path):
    cfg = json.loads((ROOT / "settings.example.json").read_text("utf-8"))
    for holder, key in ((cfg, "brain_python"), (cfg, "speech_python"), (cfg["embedder"], "python"), (cfg["vision"], "python")):
        holder[key] = str(tmp_path / "envs" / "missing" / "python.exe")
    cfg["tts_url"] = f"http://127.0.0.1:{free_port()}"             # never the owner's voice server
    return cfg


@pytest.mark.parametrize("language", [None, "en"])
def test_speech_recognition_language_reaches_child_environment(language):
    cfg = json.loads((ROOT / "settings.example.json").read_text("utf-8"))
    cfg.pop("speech_language", None)
    if language is not None:
        cfg["speech_language"] = language
    speech = next(service for service in launch.core_services(cfg, None, None) if service.name == "speech")
    assert speech.env["YUMEHA_LANGUAGE"] == (language or "auto")


@pytest.mark.parametrize("transport", [None, "modal"])
def test_voice_transport_and_cloud_budget_reach_gateway(transport):
    cfg = json.loads((ROOT / "settings.example.json").read_text("utf-8"))
    cfg.pop("tts_transport", None)
    cfg.pop("tts_timeout_s", None)
    if transport is not None:
        cfg.update(tts_transport=transport, tts_timeout_s=210)
    speech = next(service for service in launch.core_services(cfg, None, None) if service.name == "speech")
    assert speech.env["AMIKA_TTS_TRANSPORT"] == (transport or "local")
    assert speech.env["AMIKA_TTS_TIMEOUT_S"] == ("210" if transport else "240")


@pytest.mark.parametrize("dedicated", [True, False])
def test_remote_transcription_key_and_settings_reach_only_the_speech_child(tmp_path, monkeypatch, dedicated):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_STT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain/.env").write_text("OPENROUTER_API_KEY=shared-test-key\n" + (
        "OPENROUTER_STT_API_KEY=speech-test-key\n" if dedicated else ""), "utf-8")
    cfg = json.loads((ROOT / "settings.example.json").read_text("utf-8"))
    cfg.update(speech_backend="openrouter", speech_openrouter_model="elevenlabs/scribe-v2", speech_timeout_s=17)
    services = launch.core_services(cfg, None, None)
    speech = next(s for s in services if s.name == "speech")
    assert speech.env["OPENROUTER_STT_API_KEY"] == ("speech-test-key" if dedicated else "shared-test-key")
    assert (speech.env["AMIKA_STT_BACKEND"], speech.env["AMIKA_STT_MODEL"], speech.env["AMIKA_STT_TIMEOUT_S"]) == (
        "openrouter", "elevenlabs/scribe-v2", "17",
    )
    assert all("OPENROUTER_STT_API_KEY" not in s.env for s in services if s.name != "speech")


@pytest.fixture()
def fresh(tmp_path, monkeypatch):
    """launch.py looking at an empty checkout: no environments, no voice, no models."""
    monkeypatch.setattr(launch, "TTS_ENGINE", tmp_path / "vendor" / "tts-engine")
    monkeypatch.setattr(launch, "AVATAR_MODELS", tmp_path / "avatar" / "models")
    monkeypatch.setattr(launch, "INSTANCE", tmp_path / "logs" / "instance.json")
    monkeypatch.setattr(launch, "listeners", lambda: {})
    monkeypatch.setattr(launch, "bind_state", lambda port, host="127.0.0.1": "")   # never this machine's own ports
    monkeypatch.delenv("HERMES_HOME", raising=False)                    # never this machine's own Hermes home
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    return tmp_path


@contextlib.contextmanager
def serve(port, routes):
    """A stand-in on 127.0.0.1:port answering GET <path> with routes[path] (a dict as JSON, a str as text)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            data = (json.dumps(body) if isinstance(body, dict) else body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def checks(report):
    return {c["name"]: (c["level"], c["detail"]) for c in report.checks}


def services_for(cfg, engine):
    cfg = json.loads(json.dumps(cfg))
    cfg["services"]["tts"]["cwd"] = str(engine)
    cfg["services"]["tts"]["command"][0] = str(engine / ".venv" / "Scripts" / "python.exe")
    return cfg, launch.start_plan(cfg, None)


def test_nothing_installed_is_named_with_its_fix_and_never_crashes(fresh):
    cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor" / "tts-engine")
    emb, vis = cfg["embedder"], {**cfg["vision"], "enabled": True}
    report = launch.preflight(cfg, emb, vis, None, None, services, agents, [], deep=True, held_iot=held)
    got = checks(report)
    assert got["brain python"][0] == "err" and "setup_windows.ps1" in got["brain python"][1]
    assert got["TTS (voice)"] == ("err", "her voice server is not installed — run python " + str(Path("scripts/setup_tts.py")))
    assert got["her voice"][0] == "err" and "type yes to accept her license" in got["her voice"][1]
    assert got["Live2D model"][0] == "err" and "setup_tts.py" in got["Live2D model"][1]
    assert "embedder check" not in got and "vision check" not in got     # their python rows already say it


def test_a_missing_key_is_a_warning_the_wizard_can_fix(fresh, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", fresh)                           # no brain/.env, default (OpenRouter) roles
    cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor" / "tts-engine")
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["API keys"][0] == "warn" and "wizard" in got["API keys"][1]


def test_remote_mode_does_not_require_sensevoice_but_requires_a_key(fresh, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", fresh)
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_STT_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (fresh / "config").mkdir()
    (fresh / "config/word_replace.json").write_text("{}", "utf-8")
    cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor/tts-engine")
    cfg["speech_backend"] = "openrouter"
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["required files"] == ("ok", "all present")
    assert got["speech backend"][0] == "err"
    monkeypatch.setenv("OPENROUTER_STT_API_KEY", "test-key-never-log-this")
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["speech backend"][0] == "ok"
    assert "test-key-never-log-this" not in str(got)


def install_voice(engine, clips=2, missing_clip=False):
    voice = engine / "Speakers" / "MekaHime_Amika"
    (voice / "references").mkdir(parents=True)
    (voice / "character.json").write_text(json.dumps({"character_id": "MekaHime_Amika"}), "utf-8")
    for name in ("MekaHime_Amika.ckpt", "MekaHime_Amika.pth"):
        (voice / name).write_bytes(b"w")
    table = {f"tag{i}": {"audio": f"clip{i}.wav"} for i in range(clips)}
    (voice / "references" / "emotions.json").write_text(json.dumps(table), "utf-8")
    for i in range(clips - (1 if missing_clip else 0)):
        (voice / "references" / f"clip{i}.wav").write_bytes(b"RIFF")
    venv = engine / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_bytes(b"")
    gsv = engine / "GPT-SoVITS" / "GPT_SoVITS"
    for rel in ["TTS_infer_pack/TTS.py", *("pretrained_models/" + p for p in launch.TTS_PRETRAINED)]:
        (gsv / rel).parent.mkdir(parents=True, exist_ok=True)
        (gsv / rel).write_bytes(b"x")


def install_model(models, name="Amika", role=None):
    folder = models / name
    folder.mkdir(parents=True)
    (folder / f"{name}V4.model3.json").write_text("{}", "utf-8")
    (folder / "idle1.motion3.json").write_text("{}", "utf-8")
    if role:
        (folder / "banks.json").write_text(json.dumps({"config": {"role": role}}), "utf-8")


def test_installed_voice_and_model_are_green(fresh):
    engine = fresh / "vendor" / "tts-engine"
    install_voice(engine)
    install_model(fresh / "avatar" / "models")
    cfg, (services, held, agents) = services_for(example_cfg(fresh), engine)
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["TTS (voice)"][0] == "warn" and "will be started" in got["TTS (voice)"][1]
    assert got["her voice"] == ("ok", "MekaHime_Amika · 2/2 emotion clips")
    assert got["Live2D model"] == ("ok", "Amika (AmikaV4.model3.json, 1 motions)")


def test_an_incomplete_voice_server_and_an_agent_only_stage(fresh):
    engine = fresh / "vendor" / "tts-engine"
    install_voice(engine, missing_clip=True)
    (engine / "GPT-SoVITS" / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt").unlink()
    install_model(fresh / "avatar" / "models", "Host", role="agent")         # her helpers' stand-in is not her body
    cfg, (services, held, agents) = services_for(example_cfg(fresh), engine)
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["TTS (voice)"][0] == "err" and "pretrained models missing" in got["TTS (voice)"][1]
    assert got["her voice"][0] == "warn" and "1/2 emotion clips" in got["her voice"][1]
    assert got["Live2D model"][0] == "err"


def test_a_voice_server_elsewhere_is_not_checked_for_files(fresh):
    cfg = example_cfg(fresh)
    del cfg["services"]["tts"]
    services, held, agents = launch.start_plan(cfg, None)
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert got["TTS (voice)"][0] == "warn" and "her voice" not in got


def test_check_on_a_fresh_copy_prints_one_line_per_check_and_a_result(tmp_path):
    """The real command, on a copy of the checkout with settings.json and nothing installed."""
    for rel in ("launch.py", "brain/iot_disclaimer.py", "brain/embed_server.py", "config/build_f.json",
                "config/llm_roles.json", "config/word_replace.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    cfg = example_cfg(tmp_path)
    for key, holder in (("brain_port", cfg), ("chat_port", cfg)):
        holder[key] = free_port()
    cfg["embedder"]["port"] = free_port()
    cfg["vision"]["port"] = free_port()
    cfg["messengers_port"] = free_port()
    (tmp_path / "settings.json").write_text(json.dumps(cfg), "utf-8")
    p = subprocess.run([sys.executable, "-B", "-X", "utf8", str(tmp_path / "launch.py"), "--check"], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    out = p.stdout
    assert p.returncode == 1, out + p.stderr
    assert "could not start" not in out + p.stderr
    assert "  !! brain python" in out and "  !! her voice" in out and "  !! Live2D model" in out
    assert "  -- API keys" in out
    assert out.rstrip().endswith("result: fix the !! lines above (-- lines are informational)")
    assert (tmp_path / "logs" / "preflight.json").is_file()


def test_her_roles_on_ollama_need_it_running_with_their_models_pulled(fresh, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", fresh)
    (fresh / "config").mkdir()
    picks = {role: f"{role}:ollama:gemma3:27b" for role in ("surface", "deep", "reflection")}
    picks["memory_write"] = "memory_write:ollama:qwen3:8b"
    (fresh / "config" / "llm_roles.json").write_text(json.dumps({"selections": picks}), "utf-8")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    (fresh / "config" / "ollama.json").write_text(json.dumps({"base_url": base}), "utf-8")
    cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor" / "tts-engine")

    def run():
        return checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))

    got = run()
    assert got["API keys"][0] == "ok"                                     # no key needed ...
    assert got["Ollama"][0] == "err" and f"not answering at {base} — start Ollama" in got["Ollama"][1]   # ... but Ollama
    with serve(port, {"/api/version": {"version": "0.12.6"}, "/api/tags": {"models": [{"name": "gemma3:27b"}]}}):
        got = run()
    assert got["Ollama"] == ("ok", f"0.12.6 at {base}")
    assert got["Ollama model gemma3:27b"] == ("ok", "pulled · surface, deep, reflection")
    assert got["Ollama model qwen3:8b"] == ("err", "not pulled — ollama pull qwen3:8b (memory_write runs on it)")


def test_no_ollama_rows_while_her_roles_run_on_openrouter(fresh):
    cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor" / "tts-engine")
    got = checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))
    assert not [name for name in got if name.startswith("Ollama")]


def test_a_voice_server_without_her_voice_on_her_port_is_a_problem(fresh):
    port = free_port()
    cfg = example_cfg(fresh)
    cfg["tts_url"] = f"http://127.0.0.1:{port}"
    cfg, (services, held, agents) = services_for(cfg, fresh / "vendor" / "tts-engine")

    def run():
        return checks(launch.preflight(cfg, None, None, None, None, services, agents, [], held_iot=held))

    sakura = {"/ping": "ok", "/api/characters": {"characters": {"MekaHime_Sakura": {}}, "current": "MekaHime_Sakura"}}
    with serve(port, sakura):                                         # another voice server answers on her port
        level, detail = run()["TTS (voice)"]
    assert level == "err" and f"port {port} answers but has no voice MekaHime_Amika" in detail and "OpenSakura" in detail
    with serve(port, {"/ping": "ok"}):                                # not a voice server OpenAmika can use at all
        assert run()["TTS (voice)"][0] == "err"
    with serve(port, {"/ping": "ok", "/api/characters": {"characters": {"MekaHime_Amika": {}}}}):
        assert run()["TTS (voice)"] == ("ok", f"http://127.0.0.1:{port} answers · voice MekaHime_Amika")


def test_ports_held_by_this_openamika_are_fine_and_others_are_named(fresh, monkeypatch):
    cfg = example_cfg(fresh)
    monkeypatch.setattr(launch, "listeners", lambda: {cfg["brain_port"]: 101, cfg["chat_port"]: 202, 5215: 303})
    procs = [{"pid": 101, "ppid": 1, "name": "python.exe", "exe": "python.exe",
              "cmd": f'python.exe -u "{launch.ROOT / "brain" / "webui.py"}"'},              # this folder's brain
             {"pid": 202, "ppid": 1, "name": "python.exe", "exe": "python.exe",
              "cmd": "python.exe -u D:\\Other\\OpenAmika\\speech\\api.py"},                 # another checkout
             {"pid": 303, "ppid": 1, "name": "python.exe", "exe": "", "cmd": ""}]         # an administrator window
    got = checks(launch.preflight(cfg, None, None, None, None, [], [], procs))
    assert got["brain port"] == ("ok", f"{cfg['brain_port']} held by this OpenAmika (running)")
    assert got["web/speech port"][0] == "err" and "in use by pid 202" in got["web/speech port"][1]
    assert got["messengers port"][0] == "err" and "administrator" in got["messengers port"][1]


def test_the_messengers_port_comes_from_settings(fresh):
    cfg = example_cfg(fresh)
    assert launch.messengers_port(cfg) == 5215
    cfg["messengers_port"] = 5299
    cores = {s.name: s for s in launch.core_services(cfg, None, None)}
    assert cores["messengers"].health == "http://127.0.0.1:5299/health"
    assert cores["messengers"].env["AMIKA_MESSENGERS_PORT"] == "5299"
    assert cores["speech"].env["AMIKA_MESSENGERS_URL"] == "http://127.0.0.1:5299"


def test_her_clock_without_a_time_zone_database_is_a_problem(fresh, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", fresh)
    (fresh / "config").mkdir()
    (fresh / "config" / "build_f.json").write_text(json.dumps({"timezone": "America/Los_Angeles"}), "utf-8")

    def no_database(key):                        # a python.org Python on Windows without the tzdata package
        raise KeyError(f"No time zone found with key {key}")

    level, detail = launch.her_clock(no_database)
    assert level == "err" and "tzdata" in detail and "brain cannot start" in detail
    tokyo = []

    def only_tokyo(key):                         # the database is there; the name is not a zone
        if key != "Asia/Tokyo":
            raise KeyError(f"No time zone found with key {key}")
        tokyo.append(key)
        return launch.datetime.now().astimezone().tzinfo

    (fresh / "config" / "build_f.json").write_text(json.dumps({"timezone": "Mars/Base"}), "utf-8")
    level, detail = launch.her_clock(only_tokyo)
    assert level == "warn" and "'Mars/Base'" in detail and "falls back to Asia/Tokyo" in detail and tokyo


# ── her helpers: the agents' start commands, Hermes's key, the paper doll ──

def tools_with(name, entry, enabled=True):
    return {"enabled": enabled, "mcpServers": {name: {"command": "python", "cwd": "../servers/openamika-agents",
                                                       "enabled": True, "kind": "agent", **entry}}}


def test_agent_start_commands_keep_quoted_paths_expand_variables_and_resolve_from_this_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("OA_TEST_APPDATA", str(tmp_path / "App Data"))
    port = free_port()
    spaced = launch.agent_services(tools_with("hermes", {"autostart": f'"{tmp_path / "My Apps" / "hermes.exe"}" gateway',
                                                         "env": {"HERMES_URL": f"http://127.0.0.1:{port}"}}))[0]
    assert spaced.cmd == [str(tmp_path / "My Apps" / "hermes.exe"), "gateway"]            # one path, spaces and all
    assert spaced.health == f"http://127.0.0.1:{port}/health" and spaced.log.name == "agent-hermes.log"
    var = launch.agent_services(tools_with("hermes", {"autostart": {
        "command": "%OA_TEST_APPDATA%/hermes/bin/hermes.exe", "args": ["gateway"], "cwd": "%OA_TEST_APPDATA%/hermes"}}))[0]
    assert var.cmd[0] == str(tmp_path / "App Data") + "/hermes/bin/hermes.exe"
    assert var.cwd == str(tmp_path / "App Data") + "/hermes"
    rel = launch.agent_services(tools_with("muse", {
        "args": ["-m", "openamika_agents", "nanomuse"], "env": {"NANOMUSE_URL": f"http://127.0.0.1:{port}"},
        "autostart": {"command": "Nanomuse/venv/Scripts/nanomuse.exe", "args": ["serve"], "cwd": "Nanomuse/data"}}))[0]
    assert rel.cmd == [str(launch.ROOT / "Nanomuse/venv/Scripts/nanomuse.exe"), "serve"]
    assert rel.cwd == str(launch.ROOT / "Nanomuse/data")
    assert rel.health == f"http://127.0.0.1:{port}/api/health"           # named "muse", but it runs the nanomuse bridge
    bare = launch.agent_services(tools_with("hermes", {"autostart": "hermes gateway"}))[0]
    assert bare.cmd == ["hermes", "gateway"]                              # a bare name is left to the PATH


def hermes_home_with(path, key):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text("# Hermes\nAPI_SERVER_ENABLED=true\n" + (f"API_SERVER_KEY={key}\n" if key is not None else ""),
                               "utf-8")
    return path


def test_hermes_key_is_checked_in_the_hermes_home_and_never_shown(fresh, monkeypatch):
    home = fresh / "hermes"
    good = "0123456789abcdef0123456789abcdef"
    port = free_port()

    def got(key, ours, autostart=None):
        hermes_home_with(home, key)
        entry = {"env": {"HERMES_URL": f"http://127.0.0.1:{port}", "HERMES_API_KEY": ours},
                 "autostart": autostart or {"command": str(home / "bin" / "hermes.exe"), "args": ["gateway"], "cwd": str(home)}}
        tools = tools_with("hermes", entry)
        cfg, (services, held, agents) = services_for(example_cfg(fresh), fresh / "vendor" / "tts-engine")
        agents = launch.agent_services(tools)
        report = checks(launch.preflight(cfg, None, None, tools, None, services, agents, [], held_iot=held))
        assert good not in json.dumps(report)                            # the key itself never appears
        return report

    rows = got(good, good)
    assert rows["hermes key"][0] == "ok" and str(home) in rows["hermes key"][1]
    assert rows["agent hermes"][0] == "err" and "start command does not exist" in rows["agent hermes"][1]
    assert got(good, "0123456789abcdef-something-else")["hermes key"][0] == "warn"
    assert "differs" in got(good, "0123456789abcdef-something-else")["hermes key"][1]
    assert "not set yet" in got(good, "PASTE-THE-API_SERVER_KEY-FROM-HERMES-.env")["hermes key"][1]
    assert "not set yet" in got(good, "<the API_SERVER_KEY from Hermes's .env>")["hermes key"][1]   # a doc's placeholder
    short = got("tooshort", "tooshort")["hermes key"]
    assert short[0] == "warn" and "shorter than 16 characters" in short[1]
    assert got(None, good)["hermes key"][0] == "warn"                   # no key: Hermes never opens its API server
    assert got("your_api_key_here_please_x_here", good)["hermes key"][0] == "warn"
    # a string autostart (no cwd): found by the folder above bin\hermes.exe
    assert got(good, good, autostart=f'"{home / "bin" / "hermes.exe"}" gateway')["hermes key"][0] == "ok"
    monkeypatch.setenv("API_SERVER_KEY", good)                          # set for the whole session, not in its .env
    assert got(None, good)["hermes key"][0] == "ok"


def test_hermes_key_check_finds_the_default_home_without_an_autostart(fresh, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(fresh / "Local"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    hermes_home_with(fresh / "Local" / "hermes", "0123456789abcdef0123456789abcdef")
    tools = tools_with("hermes", {"env": {"HERMES_URL": f"http://127.0.0.1:{free_port()}",
                                          "HERMES_API_KEY": "fedcba9876543210fedcba9876543210"}})
    assert launch.hermes_checks(tools["mcpServers"]["hermes"], None)[0][:2] == ("warn", "hermes key")
    monkeypatch.setenv("LOCALAPPDATA", str(fresh / "Elsewhere"))         # no Hermes home anywhere: nothing to compare
    assert launch.hermes_checks(tools["mcpServers"]["hermes"], None) == []


def test_the_paper_doll_row(fresh):
    cfg = example_cfg(fresh)
    models = fresh / "avatar" / "models"
    install_model(models)                                                # her own model
    hermes = tools_with("hermes", {"env": {"HERMES_URL": f"http://127.0.0.1:{free_port()}"}})
    assert launch.paper_doll(cfg, None) == (None, None)                  # no agent, no doll: nothing to say
    level, detail = launch.paper_doll(cfg, hermes)
    assert level == "warn" and '"role": "agent"' in detail and "setup_tts.py" in detail
    assert launch.paper_doll(cfg, {**hermes, "enabled": False}) == (None, None)    # the tool layer is off
    install_model(models, "Host", role="agent")
    assert launch.paper_doll(cfg, hermes) == ("ok", "Host · 1 motions — stands in for her helpers on the Use page")
    assert launch.paper_doll(cfg, None)[1].endswith("shown beside her while a task agent is switched on (Tools page)")
    services, held, agents = launch.start_plan(cfg, hermes)
    got = checks(launch.preflight(cfg, None, None, hermes, None, services, agents, [], held_iot=held))
    assert got["paper doll"][0] == "ok" and got["Live2D model"][1].startswith("Amika (")      # the doll is not her body


def test_the_shipped_tools_example_starts_the_helpers_from_their_documented_homes(fresh, monkeypatch):
    """config/tools.example.json, copied as it is and switched on: Hermes from its default home (%LOCALAPPDATA%/hermes,
    no path to edit), nanoMuse from the Nanomuse folder of an OpenAmika in C:/OpenAmika (README, docs/TOOLS.md), both
    through the bridge in servers/openamika-agents; the key placeholder is named as not set yet."""
    example = json.loads((ROOT / "config" / "tools.example.json").read_text("utf-8"))
    assert example["enabled"] is False and not any(s.get("enabled") for s in example["mcpServers"].values())
    for name in ("hermes", "nanomuse"):
        entry = example["mcpServers"][name]
        assert entry["kind"] == "agent" and entry["args"] == ["-m", "openamika_agents", name]
        assert (ROOT / "brain" / entry["cwd"] / "openamika_agents" / "adapters" / f"{name}.py").is_file()
        entry["enabled"] = True
    example["enabled"] = True
    monkeypatch.setenv("LOCALAPPDATA", str(fresh / "Local"))
    agents = {a.name: a for a in launch.agent_services(example)}
    assert agents["hermes"].cmd == [str(fresh / "Local") + "/hermes/bin/hermes.exe", "gateway"]
    assert agents["hermes"].cwd == str(fresh / "Local") + "/hermes"
    assert agents["hermes"].health == "http://127.0.0.1:8642/health"
    assert agents["nanomuse"].cmd[:4] == ["C:/OpenAmika/Nanomuse/venv/Scripts/nanomuse.exe", "serve", "--config",
                                          "C:/OpenAmika/Nanomuse/data/config.toml"]
    assert agents["nanomuse"].health == "http://127.0.0.1:8787/api/health"
    hermes_home_with(fresh / "Local" / "hermes", "0123456789abcdef0123456789abcdef")
    rows = launch.hermes_checks(example["mcpServers"]["hermes"], agents["hermes"])
    assert rows[0][0] == "warn" and "not set yet" in rows[0][2]
    assert launch.agents_switched_on(example) and not launch.agents_switched_on({**example, "enabled": False})
