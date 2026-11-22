/* Timing and Waveform Generator Module of X Band Navigational Radar - Front-End Logic */
(() => {
  const C = 299792458;          // speed of light, m/s
  const CLK_NS = 10;            // 100 MHz clock (10 ns period)
  const FIXED_FC_MHZ = 9410.0;  // Fixed 9.41 GHz X-band carrier
  const DEFAULT_RPM = 24.0;     // Standard marine antenna rotation rate (rpm)
  const DEFAULT_BW_DEG = 1.2;   // Standard horizontal beamwidth (deg)
  const DEFAULT_NVESSELS = 5;   // Standard vessel count for ML recommender feature vector

  const $ = (id) => document.getElementById(id);

  const el = {
    // Section 1: ML Mode Recommender
    mlForm: $("ml-form"),
    inRange: $("in-range"),
    inRes: $("in-res"),
    inMinRange: $("in-minrange"),
    inRcs: $("in-rcs"),
    inSea: $("in-sea"),
    inRain: $("in-rain"),
    boxRange: $("box-range"),
    boxRes: $("box-res"),
    tabRangeMode: $("tab-range-mode"),
    tabResMode: $("tab-res-mode"),
    aiBtn: $("btn-ai"),

    // Section 2: Waveform & Derived Parameters
    pw: $("pw"),
    pri: $("pri"),
    dDr: $("d-dr"),
    dBlind: $("d-blind"),
    dRmax: $("d-rmax"),
    dRmaxNm: $("d-rmax-nm"),
    dPrf: $("d-prf"),
    dDuty: $("d-duty"),
    dHits: $("d-hits"),
    dFc: $("d-fc"),

    // Section 3: Simulation & Dual Panels
    simForm: $("sim-form"),
    pulses: $("pulses"),
    guard: $("guard"),
    run: $("btn-run"),
    dl: $("btn-dl"),
    surfer: $("btn-surfer"),
    sPath: $("surfer-path"),
    sSave: $("btn-surfer-save"),
    tDownload: $("t-download"),
    tSurfer: $("t-surfer"),

    // Left: Terminal
    term: $("term"),
    termState: $("term-state"),
    clear: $("btn-clear"),

    // Right: Testbench Results Table
    verifyHeadline: $("verify-headline"),
    verifyList: $("verify-list"),

    // Background Radar Animation Canvas
    radarBgCanvas: $("radar-bg"),
  };

  const fmt = (n, d = 2) => (Number.isFinite(n) ? String(+n.toFixed(d)) : "");
  const num = (input) => parseFloat(input.value);

  /* ============================================================================
   * 1. Preset Test Cases (Mode 1 to Mode 5) & Range / Resolution Switch
   * ============================================================================ */
  const PRESET_CASES = {
    1: { name: "Mode 1 (000)", range_km: 15,  resolution_m: 15,  min_range_m: 25,  rcs_m2: 50,  sea_state: 1, rain_mmh: 0 },
    2: { name: "Mode 2 (001)", range_km: 200, resolution_m: 130, min_range_m: 150, rcs_m2: 500, sea_state: 2, rain_mmh: 0 },
    3: { name: "Mode 3 (010)", range_km: 30,  resolution_m: 35,  min_range_m: 50,  rcs_m2: 50,  sea_state: 2, rain_mmh: 0 },
    4: { name: "Mode 4 (011)", range_km: 45,  resolution_m: 65,  min_range_m: 80,  rcs_m2: 100, sea_state: 2, rain_mmh: 0 },
    5: { name: "Mode 5 (100)", range_km: 160, resolution_m: 95,  min_range_m: 110, rcs_m2: 100, sea_state: 2, rain_mmh: 0 },
  };

  let rangeDrivesRes = true;

  function calcResFromRange(rKm) {
    if (!Number.isFinite(rKm) || rKm <= 0) return 15;
    if (rKm <= 20) return 15;
    if (rKm <= 38) return 35;
    if (rKm <= 75) return 65;
    if (rKm <= 187.5) return 95;
    return 130;
  }

  function calcRangeFromRes(resM) {
    if (!Number.isFinite(resM) || resM <= 0) return 15;
    if (resM < 30) return 15;
    if (resM < 60) return 30;
    if (resM < 90) return 45;
    if (resM < 120) return 160;
    return 200;
  }

  function calcMinRangeFromRes(resM) {
    if (resM < 30) return 25;
    if (resM < 60) return 50;
    if (resM < 90) return 80;
    if (resM < 120) return 110;
    return 150;
  }

  function syncRangeResolutionPair() {
    if (rangeDrivesRes) {
      const rKm = num(el.inRange);
      if (Number.isFinite(rKm) && rKm > 0) {
        const autoRes = calcResFromRange(rKm);
        el.inRes.value = fmt(autoRes, 1);
        const minNeeded = calcMinRangeFromRes(autoRes);
        if (num(el.inMinRange) < minNeeded) el.inMinRange.value = minNeeded;
      }
    } else {
      const resM = num(el.inRes);
      if (Number.isFinite(resM) && resM > 0) {
        const autoRange = calcRangeFromRes(resM);
        el.inRange.value = fmt(autoRange, 1);
        const minNeeded = calcMinRangeFromRes(resM);
        if (num(el.inMinRange) < minNeeded) el.inMinRange.value = minNeeded;
      }
    }
  }

  function setInputMode(useRangeFirst) {
    rangeDrivesRes = useRangeFirst;
    el.tabRangeMode.setAttribute("aria-selected", String(rangeDrivesRes));
    el.tabResMode.setAttribute("aria-selected", String(!rangeDrivesRes));

    if (rangeDrivesRes) {
      el.inRange.readOnly = false;
      el.inRes.readOnly = true;
      el.boxRange.classList.remove("is-readonly");
      el.boxRes.classList.add("is-readonly");
    } else {
      el.inRange.readOnly = true;
      el.inRes.readOnly = false;
      el.boxRange.classList.add("is-readonly");
      el.boxRes.classList.remove("is-readonly");
    }
    syncRangeResolutionPair();
  }

  el.tabRangeMode.addEventListener("click", () => setInputMode(true));
  el.tabResMode.addEventListener("click", () => setInputMode(false));

  el.inRange.addEventListener("input", () => {
    if (rangeDrivesRes) syncRangeResolutionPair();
  });

  el.inRes.addEventListener("input", () => {
    if (!rangeDrivesRes) syncRangeResolutionPair();
  });

  document.querySelectorAll(".preset-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-card").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      const p = PRESET_CASES[btn.dataset.preset];
      if (!p) return;
      el.inRange.value = p.range_km;
      el.inRes.value = p.resolution_m;
      el.inMinRange.value = p.min_range_m;
      el.inRcs.value = p.rcs_m2;
      el.inSea.value = String(p.sea_state);
      el.inRain.value = p.rain_mmh;
      runMlSynthesize();
    });
  });

  /* ============================================================================
   * 2. Waveform Section (Only PW and PRI editable; all others derived)
   * ============================================================================ */
  function updateWaveformDerived() {
    const pw = num(el.pw);
    const pri = num(el.pri);
    const rawGuard = num(el.guard);
    const guard = Number.isFinite(rawGuard) && rawGuard >= 0 ? rawGuard : 5;
    const okPw = Number.isFinite(pw) && pw > 0;
    const okPri = Number.isFinite(pri) && pri > 0;

    const dr = okPw ? (C * pw * 1e-9) / 2 : NaN;
    const blind = okPw ? (C * (pw + guard * CLK_NS) * 1e-9) / 2 : NaN;
    const rmaxKm = okPri ? (C * pri * 1e-6) / 2 / 1000 : NaN;
    const rmaxNm = okPri ? (rmaxKm * 1000) / 1852 : NaN;
    const prf = okPri ? 1e6 / pri : NaN;
    const duty = okPw && okPri ? (pw / (pri * 1000)) * 100 : NaN;
    const hits = okPri ? (prf * DEFAULT_BW_DEG) / (6 * DEFAULT_RPM) : NaN;

    el.dDr.textContent = okPw ? fmt(dr, 2) : "-";
    el.dBlind.textContent = okPw ? fmt(blind, 1) : "-";
    el.dRmax.textContent = okPri ? fmt(rmaxKm, 2) : "-";
    el.dRmaxNm.textContent = okPri ? `(c * PRI / 2, ${fmt(rmaxNm, 1)} NM)` : "(c * PRI / 2)";
    el.dPrf.textContent = okPri ? fmt(prf, 2) : "-";
    el.dDuty.textContent = okPw && okPri ? fmt(duty, 4) : "-";
    el.dHits.textContent = okPri ? fmt(hits, 1) : "-";
    el.dFc.textContent = "9410.0";
  }

  el.pw.addEventListener("input", updateWaveformDerived);
  el.pri.addEventListener("input", updateWaveformDerived);
  el.guard.addEventListener("input", updateWaveformDerived);
  updateWaveformDerived();

  /* ============================================================================
   * 3. Input Validation (:user-invalid sync on blur/input)
   * ============================================================================ */
  document.querySelectorAll("input, select").forEach((ctrl) => {
    ctrl.setAttribute("aria-invalid", "false");
    ctrl.addEventListener("blur", () => {
      if (ctrl.checkValidity) {
        const valid = ctrl.validity.valid;
        ctrl.setAttribute("aria-invalid", valid ? "false" : "true");
        ctrl.classList.toggle("user-invalid-fallback", !valid);
      }
    });
    ctrl.addEventListener("input", () => {
      if (ctrl.checkValidity && ctrl.validity.valid) {
        ctrl.setAttribute("aria-invalid", "false");
        ctrl.classList.remove("user-invalid-fallback");
        const b = ctrl.closest(".box");
        if (b) b.classList.remove("invalid");
      }
    });
  });

  /* ============================================================================
   * 4. Testbench Results Table (Right) & Terminal (Left)
   * ============================================================================ */
  const TEST_CATALOG = [
    { key: "Pulses captured",                   name: "Pulse Count" },
    { key: "PW matches setting",                name: "Pulse Width" },
    { key: "PRI matches setting",               name: "PRI" },
    { key: "RX blank starts with/before gate",  name: "RX Blanking Lead Time" },
    { key: "RX guard >= configured",            name: "RX Guard After Transmit" },
    { key: "Cycles = fc x PW (+/-1)",           name: "Cycles of RF" },
    { key: "Samples = fs x PW",                 name: "Samples of RF" },
  ];

  function displayTitleFor(rawLabel) {
    const found = TEST_CATALOG.find((t) => t.key === rawLabel);
    return found ? found.name : rawLabel;
  }

  function renderVerificationChecks(checks) {
    el.verifyList.textContent = "";
    let passed = 0;
    checks.forEach((c) => {
      if (c.ok === true) passed += 1;
      const tr = document.createElement("tr");

      const th = document.createElement("th");
      th.scope = "row";
      th.textContent = displayTitleFor(c.label);

      const tdResult = document.createElement("td");
      tdResult.textContent = c.detail || "-";

      const tdStatus = document.createElement("td");
      if (c.ok === null) {
        tdStatus.textContent = "-";
        tdStatus.className = "status-wait";
      } else if (c.ok) {
        tdStatus.textContent = "PASS";
        tdStatus.className = "status-pass";
      } else {
        tdStatus.textContent = "FAIL";
        tdStatus.className = "status-fail";
      }

      tr.append(th, tdResult, tdStatus);
      el.verifyList.appendChild(tr);
    });

    const total = checks.length;
    const doneChecks = checks.filter((c) => c.ok !== null);
    if (doneChecks.length === 0) {
      el.verifyHeadline.textContent = "No tests run yet";
      el.verifyHeadline.className = "results-summary";
      return;
    }

    const allPassed = passed === total && !doneChecks.some((c) => c.ok === false);
    el.verifyHeadline.textContent = `${passed} of ${total} tests passed`;
    el.verifyHeadline.className = "results-summary " + (allPassed ? "pass" : "fail");
  }

  let liveStreamChecks = TEST_CATALOG.map((t) => ({ label: t.key, ok: null, detail: "-" }));
  renderVerificationChecks(liveStreamChecks);

  function log(text, kind) {
    if (!kind) {
      if (/\bFAIL\b|^ERROR|Traceback|Error:/.test(text)) kind = "err";
      else if (/\bPASS\b/.test(text)) kind = "ok";
      else if (/^\s*WARNING/.test(text)) kind = "warn";
      else if (/^\$ /.test(text)) kind = "cmd";
      else kind = "sys";
    }
    const line = document.createElement("span");
    line.className = "ln " + kind;
    line.textContent = text || " ";
    el.term.appendChild(line);
    el.term.scrollTop = el.term.scrollHeight;

    const m = /^\s*(PASS|FAIL)\s+(.+?)\s{2,}(.+)$/.exec(text);
    if (m) {
      const isOk = m[1] === "PASS";
      const label = m[2].trim();
      const detail = m[3].trim();
      const idx = liveStreamChecks.findIndex((c) => c.label === label);
      if (idx >= 0) {
        liveStreamChecks[idx] = { label, ok: isOk, detail };
      } else {
        liveStreamChecks.push({ label, ok: isOk, detail });
      }
      renderVerificationChecks(liveStreamChecks);
    }
  }

  const setState = (label) => {
    if (el.termState) el.termState.textContent = label;
  };
  setState("Idle");

  el.clear.addEventListener("click", () => {
    el.term.textContent = "";
  });

  log("Timing and Waveform Generator ready.");
  log("Select a test case or configure PW and PRI, then press Run Testbench.");

  /* ============================================================================
   * 5. Surfer Path Configuration
   * ============================================================================ */
  fetch("/api/status")
    .then((r) => r.json())
    .then((s) => {
      if (s.path) el.sPath.value = s.path;
    })
    .catch(() => {});

  el.sSave.addEventListener("click", async () => {
    const res = await fetch("/api/surfer-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: el.sPath.value }),
    });
    const j = await res.json().catch(() => ({ message: "Request failed" }));
    log(res.ok ? j.message : "WARNING: " + j.message, res.ok ? "ok" : "warn");
  });

  /* ============================================================================
   * 6. ML Mode Recommender Execution
   * ============================================================================ */
  const missionBody = () => ({
    resolution_m: num(el.inRes),
    range_km: num(el.inRange),
    min_range_m: num(el.inMinRange),
    rcs_m2: num(el.inRcs),
    sea_state: num(el.inSea),
    rain_mmh: num(el.inRain),
    n_vessels: DEFAULT_NVESSELS,
  });

  function renderProbabilities(proba) {
    const container = $("r-proba");
    container.textContent = "";
    if (!proba) return;
    Object.entries(proba).forEach(([modeName, p]) => {
      const pct = Math.round(p * 100);
      const item = document.createElement("span");
      item.textContent = `${modeName}: ${pct}%`;
      container.appendChild(item);
    });
  }

  function showResult(r) {
    const card = $("ai-result");
    card.hidden = false;
    card.classList.remove("error");
    $("r-mode").textContent = `Recommended: ${r.mode_name}`;
    $("r-code").textContent = `code ${r.code}`;
    $("r-conf").textContent = `${(r.confidence * 100).toFixed(1)}%`;
    $("r-line").textContent = `PW = ${r.pw_ns} ns, PRF = ${r.prf_hz} Hz, PRI = ${r.pri_us} us. Applied to waveform parameters below.`;
    renderProbabilities(r.probabilities);
    const ul = $("r-warn");
    ul.textContent = "";
    (r.warnings || []).forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      ul.appendChild(li);
    });
  }

  function showInfeasible(r) {
    const card = $("ai-result");
    card.hidden = false;
    card.classList.add("error");
    $("r-mode").textContent = "No Feasible Mode";
    $("r-code").textContent = "none";
    $("r-conf").textContent = "0%";
    $("r-line").textContent = "Requirements exceed physical mode limits. Adjust range or resolution.";
    renderProbabilities(r.probabilities);
    const ul = $("r-warn");
    ul.textContent = "";
    (r.warnings || []).forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      ul.appendChild(li);
    });
  }

  function showError(msg) {
    const card = $("ai-result");
    card.hidden = false;
    card.classList.add("error");
    $("r-mode").textContent = "Error";
    $("r-code").textContent = "error";
    $("r-conf").textContent = "-";
    $("r-line").textContent = msg;
    $("r-proba").textContent = "";
    $("r-warn").textContent = "";
  }

  window.applyParameters = ({ pw_ns, pri_us }) => {
    el.pw.value = fmt(pw_ns, 2);
    el.pri.value = fmt(pri_us, 2);
    updateWaveformDerived();
  };

  async function runMlSynthesize() {
    if (el.aiBtn.classList.contains("busy")) return;
    el.aiBtn.classList.add("busy");
    const body = missionBody();
    log(`[ML] Evaluating inputs: Range=${body.range_km} km, Resolution=${body.resolution_m} m, MinRange=${body.min_range_m} m, RCS=${body.rcs_m2} m^2`, "sys");
    try {
      const res = await fetch("/api/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = j.error || "Request failed";
        log("ERROR: " + msg, "err");
        showError(msg);
        return;
      }
      if (!j.feasible) {
        showInfeasible(j);
        log("WARNING: No feasible mode satisfies these constraints.", "warn");
        (j.warnings || []).forEach((w) => log("WARNING: " + w, "warn"));
        return;
      }
      window.applyParameters(j);
      showResult(j);
      log(`[ML] Recommended ${j.mode_name} (code ${j.code}): PW=${j.pw_ns} ns, PRI=${j.pri_us} us`, "ok");
      (j.warnings || []).forEach((w) => log("WARNING: " + w, "warn"));
    } catch (err) {
      log("ERROR: " + err.message, "err");
      showError(err.message);
    } finally {
      el.aiBtn.classList.remove("busy");
    }
  }

  el.mlForm.addEventListener("submit", (e) => {
    e.preventDefault();
    runMlSynthesize();
  });

  /* ============================================================================
   * 7. Testbench Execution & Live Streaming
   * ============================================================================ */
  let lastRun = null;
  const box = (input) => input.closest(".box");
  const invalid = (input, bad) => {
    const b = box(input);
    if (b) b.classList.toggle("invalid", bad);
    input.setAttribute("aria-invalid", bad ? "true" : "false");
    return bad;
  };

  function checkInputs() {
    const pw = num(el.pw);
    const pri = num(el.pri);
    const pulses = num(el.pulses);
    const guard = num(el.guard);
    const bad = [
      invalid(el.pw, !(pw >= 10 && pw <= 2000)),
      invalid(el.pri, !(pri > 0 && pri <= 2000)),
      invalid(el.pulses, !(pulses >= 1 && pulses <= 10)),
      invalid(el.guard, !(Number.isFinite(guard) && guard >= 0 && guard <= 100)),
    ];
    return !bad.some(Boolean);
  }

  function enableResults(run) {
    lastRun = run;
    el.dl.href = run.download_url;
    el.dl.classList.remove("is-disabled");
    el.dl.removeAttribute("aria-disabled");
    el.dl.removeAttribute("tabindex");
    el.surfer.disabled = false;
  }

  async function runTestbench() {
    if (!checkInputs()) {
      log("ERROR: PW must be 10 to 2000 ns, PRI above 0 to 2000 us, Pulses 1 to 10, and RX Guard 0 to 100 clk.", "err");
      return;
    }
    const payload = {
      pw_ns: num(el.pw),
      pri_us: num(el.pri),
      pulses: num(el.pulses),
      guard_cycles: num(el.guard),
      fc_mhz: FIXED_FC_MHZ,
      open_surfer: el.tSurfer.checked,
    };

    liveStreamChecks = TEST_CATALOG.map((t) => ({ label: t.key, ok: null, detail: "-" }));
    renderVerificationChecks(liveStreamChecks);

    el.run.disabled = true;
    el.run.classList.add("busy");
    setState("Running");
    let finished = null;
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        log("ERROR: " + (e.error || "request failed"), "err");
        setState("Error");
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const msg = JSON.parse(buf.slice(0, i));
          buf = buf.slice(i + 1);
          if (msg.type === "log") log(msg.text);
          else if (msg.type === "error") {
            log("ERROR: " + msg.text, "err");
            finished = false;
          } else if (msg.type === "done") {
            finished = msg;
          }
        }
      }
      if (finished && finished.type === "done") {
        enableResults(finished);
        if (Array.isArray(finished.checks) && finished.checks.length > 0) {
          renderVerificationChecks(finished.checks);
        }
        const ok = finished.failed === 0;
        setState(ok ? "Done" : "Failed");
        if (el.tDownload.checked) triggerDownload(finished.download_url);
      } else {
        setState("Error");
      }
    } catch (err) {
      log("ERROR: " + err.message, "err");
      setState("Error");
    } finally {
      el.run.disabled = false;
      el.run.classList.remove("busy");
    }
  }

  function triggerDownload(url) {
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  el.simForm.addEventListener("submit", (e) => {
    e.preventDefault();
    runTestbench();
  });

  el.surfer.addEventListener("click", async () => {
    if (!lastRun) return;
    const res = await fetch("/api/open/" + lastRun.run_id, { method: "POST" });
    const j = await res.json().catch(() => ({ message: "Request failed" }));
    log(res.ok ? j.message : "WARNING: " + j.message, res.ok ? "ok" : "warn");
  });

  /* ============================================================================
   * 8. Green Background Radar Animation
   * ============================================================================ */
  function initBackgroundRadar() {
    const canvas = el.radarBgCanvas;
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(cx, cy) - 10;
    let sweepAngle = -Math.PI / 2;

    const blips = [
      { angle: 0.75, dist: 0.52, alpha: 0.5 },
      { angle: 2.35, dist: 0.72, alpha: 0.4 },
      { angle: -1.1, dist: 0.36, alpha: 0.4 },
    ];

    function drawFrame() {
      ctx.clearRect(0, 0, W, H);

      // Green radar rings
      ctx.strokeStyle = "rgba(46, 204, 113, 0.38)";
      ctx.lineWidth = 1;
      for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, (R * i) / 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Crosshairs
      ctx.strokeStyle = "rgba(46, 204, 113, 0.24)";
      ctx.beginPath();
      ctx.moveTo(cx - R, cy);
      ctx.lineTo(cx + R, cy);
      ctx.moveTo(cx, cy - R);
      ctx.lineTo(cx, cy + R);
      ctx.stroke();

      // Faint green fade trail
      const trailArc = 0.6;
      const steps = 18;
      for (let s = 0; s < steps; s++) {
        const a0 = sweepAngle - (s / steps) * trailArc;
        const a1 = sweepAngle - ((s + 1) / steps) * trailArc;
        const alpha = 0.22 * (1 - s / steps);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, R, a1, a0);
        ctx.closePath();
        ctx.fillStyle = `rgba(46, 204, 113, ${alpha.toFixed(3)})`;
        ctx.fill();
      }

      // Green sweeping line
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * Math.cos(sweepAngle), cy + R * Math.sin(sweepAngle));
      ctx.strokeStyle = "#2ecc71";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Target blips
      blips.forEach((b) => {
        const diff = Math.atan2(Math.sin(sweepAngle - b.angle), Math.cos(sweepAngle - b.angle));
        if (Math.abs(diff) < 0.08) b.alpha = 0.95;
        else b.alpha = Math.max(0.18, b.alpha * 0.992);

        const bx = cx + R * b.dist * Math.cos(b.angle);
        const by = cy + R * b.dist * Math.sin(b.angle);
        ctx.beginPath();
        ctx.arc(bx, by, 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(46, 204, 113, ${b.alpha.toFixed(2)})`;
        ctx.fill();
      });

      sweepAngle = (sweepAngle + 0.018) % (Math.PI * 2);
      requestAnimationFrame(drawFrame);
    }

    requestAnimationFrame(drawFrame);
  }

  initBackgroundRadar();
})();
