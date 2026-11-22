"""Flask front end for the X-band radar timing engine.

Start it with:   python app.py        then open http://127.0.0.1:5050

Flask only serves the page and starts runner.py as a subprocess. The simulation never
runs inside the web server, so MyHDL's global state cannot be corrupted by two requests.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_file, stream_with_context

import ai_engine
from radar_params import ParamError, resolve

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
RUNS.mkdir(exist_ok=True)

# Flat layout: every file lives in this one folder. Only the two files below are served
# to the browser, so app.py and the rest are never exposed.
app = Flask(__name__, template_folder=str(BASE), static_folder=None)
PUBLIC = {"console.css": "text/css", "console.js": "text/javascript"}


@app.get("/<name>")
def public_file(name):
    if name not in PUBLIC:
        abort(404)
    return send_file(BASE / name, mimetype=PUBLIC[name], max_age=0)

# Values shown on first load. Later the AI engine will overwrite these.
DEFAULTS = {
    "resolution_m": 12,
    "range_km": 50,
    "min_range_m": 25,
    "rcs_m2": 1,
    "sea_state": 2,
    "rain_mmh": 0,
    "n_vessels": 10,
    "rpm": 24,
    "beamwidth_deg": 1.2,
    "pw_ns": 80,
    "pri_us": 333.33,
    "pulses": 3,
    "guard_cycles": 5,
    "fc_mhz": 9410,
}

RUN_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


# ---------------------------------------------------------------- Surfer launcher
SETTINGS = BASE / "settings.json"
SKIP_DIRS = {"venv", ".venv", ".git", "runs", "__pycache__", "node_modules", "site-packages"}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    SETTINGS.write_text(json.dumps(data, indent=2))


def _scan_for_surfer(roots, depth=3):
    """Look for surfer.exe (or surfer) a few folders deep. Cheap and skips big folders."""
    wanted = {"surfer.exe", "surfer"}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for cur, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
            if len(Path(cur).parts) - base_depth >= depth:
                dirs[:] = []
            for f in files:
                if f.lower() in wanted:
                    return Path(cur) / f
    return None


def _resolve_user_path(text: str):
    """Accept a file, a folder that contains surfer.exe, or a macOS .app bundle."""
    p = Path(os.path.expandvars(os.path.expanduser(text.strip().strip('"')))).resolve()
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        if platform.system() == "Darwin" and p.suffix == ".app":
            return ["open", "-a", str(p)]
        for name in ("surfer.exe", "surfer"):
            if (p / name).is_file():
                return [str(p / name)]
    return None


def find_surfer():
    """Return the command prefix that starts Surfer, or None if it cannot be found."""
    saved = load_settings().get("surfer_path")
    for candidate in (saved, os.environ.get("SURFER_PATH")):
        if candidate:
            cmd = _resolve_user_path(candidate)
            if cmd:
                return cmd

    exe = shutil.which("surfer")
    if exe:
        return [exe]

    system = platform.system()
    home = Path.home()
    if system == "Windows":
        roots = [home / ".cargo" / "bin", BASE, BASE.parent,
                 home / "Downloads", home / "Desktop", home / "Documents",
                 Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
                 Path(os.environ.get("ProgramFiles", "C:/Program Files")),
                 Path("D:/")]
    elif system == "Darwin":
        for bundle in (Path("/Applications/Surfer.app"), home / "Applications/Surfer.app"):
            if bundle.exists():
                return ["open", "-a", str(bundle)]
        roots = [home / ".cargo" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin"),
                 home / "Downloads", BASE.parent]
    else:
        roots = [home / ".cargo" / "bin", Path("/usr/local/bin"), home / "Downloads", BASE.parent]
    found = _scan_for_surfer(roots)
    return [str(found)] if found else None


def open_in_surfer(vcd: Path):
    cmd = find_surfer()
    if not cmd:
        return False, ("Surfer was not found. Paste the full path to surfer.exe into the "
                       "Surfer location box and press Save, then press Open in Surfer.")
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if platform.system() == "Windows":
        kwargs["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        args = [str(vcd)]
        state = vcd.with_suffix(".surf.ron")
        sucl = vcd.with_suffix(".sucl")
        if state.exists():                      # saved-state: zoom, analog rf, heights
            args += ["-s", str(state)]
        elif sucl.exists():
            args += ["-c", str(sucl)]
        if cmd[0] == "open":                    # macOS app bundle
            cmd = cmd + ["--args"]
        subprocess.Popen(cmd + args, **kwargs)
    except OSError as e:
        return False, f"Could not start Surfer: {e}"
    return True, f"Opened {vcd.name} in Surfer."


def run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        abort(404)
    d = RUNS / run_id
    if not (d / "radar.vcd").exists():
        abort(404)
    return d


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html", d=DEFAULTS)


@app.get("/api/status")
def status():
    cmd = find_surfer()
    ai_ok, ai_msg = ai_engine.available()
    return jsonify(surfer=cmd is not None, path=cmd[-1] if cmd else "", ai=ai_ok, ai_message=ai_msg)


@app.post("/api/synthesize")
def synthesize():
    try:
        return jsonify(ai_engine.synthesize(request.get_json(silent=True) or {}))
    except ai_engine.EngineUnavailable as e:
        return jsonify(error=str(e)), 503
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/surfer-path")
def set_surfer_path():
    text = (request.get_json(silent=True) or {}).get("path", "").strip()
    if not text:
        data = load_settings()
        data.pop("surfer_path", None)
        save_settings(data)
        return jsonify(ok=True, message="Saved path cleared. Searching automatically.")
    cmd = _resolve_user_path(text)
    if not cmd:
        return jsonify(ok=False, message="That path does not exist. Point to surfer.exe itself."), 400
    data = load_settings()
    data["surfer_path"] = text.strip().strip('"')
    save_settings(data)
    return jsonify(ok=True, message=f"Surfer location saved: {cmd[-1]}")


def ndjson(obj) -> str:
    return json.dumps(obj) + "\n"


@app.post("/api/run")
def run():
    body = request.get_json(silent=True) or {}
    try:
        p = resolve(body.get("pw_ns"), body.get("pri_us"), body.get("pulses", 3),
                    body.get("guard_cycles", 5), body.get("fc_mhz", 9410))
    except ParamError as e:
        return jsonify(error=str(e)), 400

    auto_open = bool(body.get("open_surfer", True))
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out = RUNS / run_id / "radar.vcd"

    cmd = [sys.executable, "-u", str(BASE / "runner.py"),
           "--pw-ns", str(p["pw_ns"]), "--pri-us", str(p["pri_us"]),
           "--pulses", str(p["pulses"]), "--guard", str(p["guard_cycles"]),
           "--fc-mhz", str(p["fc_hz"] / 1e6), "--out", str(out)]

    @stream_with_context
    def generate():
        yield ndjson({"type": "log", "text": f"$ python runner.py --pw-ns {p['pw_ns']} "
                                            f"--pri-us {p['pri_us']}"})
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        result = None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if line.startswith("@@RESULT "):
                    result = json.loads(line[len("@@RESULT "):])
                else:
                    yield ndjson({"type": "log", "text": line})
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()

        if code != 0 or result is None:
            yield ndjson({"type": "error", "text": f"Testbench exited with code {code}. No VCD produced."
                          if not out.exists() else f"Testbench exited with code {code}."})
            return

        size_kb = out.stat().st_size / 1024
        done = {"type": "done", "run_id": run_id, "size_kb": round(size_kb),
                "failed": result["failed"], "params": result["params"],
                "checks": result.get("checks", []), "echo": result.get("echo"),
                "download_url": f"/download/{run_id}"}
        yield ndjson({"type": "log", "text": f"VCD written: {out.name} ({size_kb:,.0f} KB)"})
        if auto_open:
            ok, msg = open_in_surfer(out)
            done["surfer"] = {"ok": ok, "message": msg}
            yield ndjson({"type": "log", "text": msg if ok else f"WARNING: {msg}"})
        yield ndjson(done)

    return Response(generate(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/download/<run_id>")
def download(run_id):
    d = run_dir(run_id)
    return send_file(d / "radar.vcd", as_attachment=True, download_name=f"radar_{run_id}.vcd",
                     mimetype="text/plain")


@app.post("/api/open/<run_id>")
def open_run(run_id):
    d = run_dir(run_id)
    ok, msg = open_in_surfer(d / "radar.vcd")
    return jsonify(ok=ok, message=msg), (200 if ok else 500)


if __name__ == "__main__":
    # 127.0.0.1 only: this server starts programs on your computer, so never expose it.
    # Port 5050 because macOS uses 5000 for AirPlay Receiver.
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
