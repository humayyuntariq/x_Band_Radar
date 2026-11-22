"""Builds a Surfer state file (.surf.ron, preferred) and command file (.sucl, fallback) for a radar VCD.

Surfer can run a command file right after it loads a waveform (surfer FILE.vcd -c FILE.sucl).
This module writes that file next to the VCD, so the view is configured automatically:

  - the five signals are added in a fixed order
  - digital signals are made taller, the rf carrier is drawn as an analog trace
  - the time axis is zoomed from the TX trigger rising edge to the TX gate falling edge

It reads the VCD itself (header for signal paths, body for the first pulse), so it keeps
working if MyHDL changes the scope names.
"""
from pathlib import Path

# name in the VCD, how to draw it
# (name in the VCD, scope it lives in or None = shallowest copy, how to draw it, row height)
# Same list and order as the saved Surfer view. Signals that are not in the VCD (e.g. the echo_sim ones
# when no echo simulation ran) are skipped and reported.
SIGNALS = [
    ("clk",       None,       "digital", 4),
    ("mode",      None,       "digital", 2),
    ("txTrigger", None,       "digital", 2),
    ("txGate",    None,         "digital", 2),
    ("rf",        "rf_carrier", "analog",  8),
    ("rxBlank",   None,       "digital", 2),
    ("rxEnable",  None,       "digital", 2),
    ("receiver",  "echo_sim",   "analog",  6),
    ("echoTrue",  "echo_sim",   "digital", 2),
]
ZOOM_TRIGGER = ("txTrigger", None)
ZOOM_GATE = [("txGate", None), ("txGate2", None)]   # first one present is used

DIGITAL_HEIGHT = 2        # Surfer default is 1
ANALOG_HEIGHT = 8         # rows of height for the carrier trace
ANALOG_STYLE = "interpolated"   # or "step"
PAD_FRACTION = 0.04       # empty space either side of the zoom window
ITEM_INDEX_BASE = 0       # if item_focus counts from 1 in your Surfer build, set this to 1


def _read_header(f):
    """Return (variables, timescale_text). variables = {name: (full_path, id_code)}."""
    chunks = []
    for line in f:
        chunks.append(line)
        if "$enddefinitions" in line:
            break
    tok = " ".join(chunks).split()

    scope, scope_ids, variables, timescale, i = [], [], {}, "", 0
    n_scope = n_var = 0
    while i < len(tok):
        t = tok[i]
        if t == "$scope":
            scope.append(tok[i + 2])
            n_scope += 1
            scope_ids.append(n_scope)
        elif t == "$upscope":
            if scope:
                scope.pop()
                scope_ids.pop()
        elif t == "$timescale":
            j = tok.index("$end", i)
            timescale = "".join(tok[i + 1:j])
        elif t == "$var":
            j = tok.index("$end", i)
            code, name = tok[i + 3], tok[i + 4]
            n_var += 1
            path = ".".join(scope + [name])
            variables[path] = (path, code, list(scope), scope_ids[-1] if scope_ids else 0, n_var, name)
        i += 1
    return variables, timescale


def _find(variables, name, scope):
    """Entry for `name` inside a scope called `scope` (any depth); else the shallowest copy."""
    hits = [v for v in variables.values() if v[5] == name]
    if scope:
        inside = [v for v in hits if scope in v[2]]
        if inside:
            hits = inside
    return min(hits, key=lambda v: v[0].count("."), default=None)


def _first_pulse(f, trig_code, gate_code):
    """((trigger rise, gate fall) of the first pulse or None, last timestamp), raw VCD ticks."""
    t, trig, gate_up, found = 0, None, False, None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            t = int(line[1:])
        elif found:
            continue
        elif trig_code and line[0] == "1" and line[1:] == trig_code and trig is None:
            trig = t
        elif gate_code and line[1:] == gate_code:
            if line[0] == "1" and trig is not None:
                gate_up = True
            elif line[0] == "0" and gate_up:
                found = (trig, t)
    return found, t


# ---------------------------------------------------------------- Surfer state file (.surf.ron)
# Same format Surfer's "Save state" writes. Launch with:  surfer FILE.vcd -s FILE.surf.ron
# ASSUMPTION: how the VCD is referenced. Native Surfer writes File("<path>"); the VS Code extension
# writes Url("..."). If Surfer ignores the state, change SOURCE_STYLE or send me View > Logs.
SOURCE_STYLE = "File"     # "File" or "Url"
ANALOG_BLOCK = """Some((
                    settings: (
                        render_style: Interpolated,
                        y_axis_scale: Viewport,
                    ),
                ))"""


def _ron_item(ref, name, path, scope_id, var_id, height, analog):
    strs = "\n".join(f'                            "{p}",' for p in path)
    return f"""            ({ref}): Variable((
                variable_ref: (
                    path: (
                        strs: [
{strs}
                        ],
                        id: Wellen(({scope_id})),
                    ),
                    name: "{name}",
                    id: Wellen(({var_id})),
                    index: None,
                ),
                color: None,
                background_color: None,
                display_name: "{name}",
                display_name_type: Unique,
                manual_name: None,
                format: None,
                field_formats: [],
                height_scaling_factor: Some({float(height)}),
                analog: {ANALOG_BLOCK if analog else "None"},
                unfolded_fields: [],
            )),"""


def _ron_state(vcd, shown, variables, window, t_end):
    items, tree = [], []
    for n, (name, v, kind, height) in enumerate(shown, 1):
        _path, _code, scope, scope_id, var_id, _n = v
        items.append(_ron_item(n, name, scope, scope_id, var_id, height, kind == "analog"))
        tree.append(f"                (item_ref: ({n}), level: 0, unfolded: true, selected: false),")
    if window and t_end > 0:
        t0, t1 = window
        pad = max(int((t1 - t0) * PAD_FRACTION), 1)
        left, right = max(t0 - pad, 0) / t_end, min(t1 + pad, t_end) / t_end
    else:
        left, right = 0.0, 1.0
    src = str(Path(vcd).resolve()).replace("\\", "/")
    source = f'File("{src}")' if SOURCE_STYLE == "File" else f'Url("file://{src}")'
    return f"""(
    waves: Some((
        source: {source},
        format: Vcd,
        active_scope: None,
        items_tree: (
            items: [
{chr(10).join(tree)}
            ],
        ),
        displayed_items: {{
{chr(10).join(items)}
        }},
        display_item_ref_counter: {len(shown)},
        viewports: [
            (
                curr_left: ({left!r}),
                curr_right: ({right!r}),
                target_left: ({left!r}),
                target_right: ({right!r}),
                move_start_left: ({left!r}),
                move_start_right: ({right!r}),
                move_duration: None,
                move_strategy: Instant,
            ),
        ],
        cursor: None,
        markers: {{}},
        selected_annotation: None,
        annotations: [],
        annotation_groups: [(name: "Ungrouped", cycle_counter: 0, annotations: [])],
        annotation_list_visible: false,
        annotation_counter: 0,
        last_active_viewport_idx: 0,
        focused_item: None,
        focused_transaction: (None, None),
        default_variable_name_type: Unique,
        scroll_offset: 0.0,
        display_variable_indices: true,
        graphics: {{}},
    )),
    wanted_timeunit: FemtoSeconds,
)
"""


def write_view(vcd_path):
    """Write <vcd>.sucl. Returns a dict describing what was done (for the terminal log)."""
    vcd = Path(vcd_path)
    with open(vcd) as f:
        variables, timescale = _read_header(f)
        trig = _find(variables, *ZOOM_TRIGGER)
        gate = next((g for g in (_find(variables, *x) for x in ZOOM_GATE) if g), None)
        window, t_end = _first_pulse(f, trig[1] if trig else None, gate[1] if gate else None)

    lines = [f"# Surfer view for {vcd.name} (generated, do not edit)"]
    shown, missing = [], []
    for name, scope, kind, height in SIGNALS:
        v = _find(variables, name, scope)
        if v is None:
            missing.append(name)
            continue
        lines.append(f"variable_add {v[0]}")
        shown.append((name, v, kind, height))

    lines.append("variable_force_name_type Local")
    for idx, (name, _v, kind, height) in enumerate(shown):
        lines.append(f"item_focus {idx + ITEM_INDEX_BASE}")
        if kind == "analog":
            lines.append(f"item_set_analog {ANALOG_STYLE}")
            lines.append(f"item_set_height {height}")
        else:
            lines.append(f"item_set_height {height}")
    lines.append("item_unfocus")

    info = {"signals": [x[0] for x in shown], "missing": missing, "window": None, "timescale": timescale}
    if window:
        t0, t1 = window
        pad = max(int((t1 - t0) * PAD_FRACTION), 1)
        start = max(t0 - pad, 0)
        end = t1 + pad
        lines.append(f"zoom_to {start} {end}")           # raw ticks, whatever the timescale is
        info["window"] = (t0, t1)
    else:
        lines.append("zoom_fit")

    state = vcd.with_suffix(".surf.ron")
    state.write_text(_ron_state(vcd, shown, variables, window, t_end))
    info["state"] = state.name
    sucl = vcd.with_suffix(".sucl")
    sucl.write_text("\n".join(lines) + "\n")
    info["file"] = sucl.name
    return info
