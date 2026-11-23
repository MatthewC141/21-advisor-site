'use strict';
// Screen-capture card reader. Watches a box the user draws on a shared window (the phone-mirroring app),
// waits for the box to settle after it changes, and matches it against cards the user has taught it.
// Needs Chrome or Edge (getDisplayMedia + MediaStreamTrackProcessor). Frames are read in a worker, so the
// reader keeps running while the mirroring window covers this tab.

// Worker: keeps only the newest frame and every `interval` ms posts the box scaled to w x h grayscale.
// Sampling on a timer (not per frame) keeps samples coming when the captured window stops repainting.
function visionWorker(scope) {
  let latest = null;
  let roi = null;
  let g2 = null;
  let W = 0;
  let H = 0;
  function sample() {
    if (!latest || !roi) return;
    const fw = latest.displayWidth;
    const fh = latest.displayHeight;
    g2.drawImage(latest, roi.x * fw, roi.y * fh, Math.max(1, roi.w * fw), Math.max(1, roi.h * fh), 0, 0, W, H);
    const d = g2.getImageData(0, 0, W, H).data;
    const g = new Uint8Array(W * H);
    for (let i = 0; i < g.length; i++) g[i] = (d[4 * i] * 77 + d[4 * i + 1] * 150 + d[4 * i + 2] * 29) >> 8;
    scope.postMessage({ gray: g.buffer }, [g.buffer]);
  }
  async function pump(reader) {
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (latest) latest.close();
        latest = value;
      }
    } catch {}
    scope.postMessage({ ended: true });
  }
  scope.onmessage = (e) => {
    const m = e.data;
    if (m.cmd === 'start') {
      W = m.w;
      H = m.h;
      g2 = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
      g2.imageSmoothingQuality = 'high';
      roi = m.roi;
      pump(m.readable.getReader());
      setInterval(sample, m.interval);
    } else if (m.cmd === 'roi') roi = m.roi;
  };
}

const Vision = (function () {
  const $ = (id) => document.getElementById(id);
  const W = 24; // sample grid
  const H = 32;
  const N = W * H;
  const INTERVAL = 100; // ms between samples
  const STABLE = 3; // motion-free samples before the box counts as settled
  const MOTION = 4; // mean gray-level change between samples that counts as motion
  const CHANGE = 10; // mean change from the last settled image that means a different card (or none) showed
  const BLANK = 8; // gray-level spread below which the box is an empty slot
  const KEEP = 6; // templates kept per label (oldest dropped)
  const EMPTY = '-'; // label for "no card in the box"
  const LS = 'tw21.vision.v1';

  let opts = null;
  let saved = { roi: null, thr: 0.8, items: [] };
  let tpl = []; // {f, g: Uint8Array, z: Float32Array}
  let stream = null;
  let clone = null;
  let worker = null;
  let workerUrl = null;

  // detector
  let prev = null;
  let stableN = 0;
  let accepted = null; // last settled image that was acted on
  let changed = true; // box moved away from `accepted` since
  let initial = true; // next event is the first since start/rearm
  let last = null; // newest sample
  let eventGray = null; // image of the last card event (what learn() stores)

  const b64 = (g) => btoa(String.fromCharCode.apply(null, g));
  const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  function zscore(g) {
    let m = 0;
    for (let i = 0; i < N; i++) m += g[i];
    m /= N;
    let v = 0;
    for (let i = 0; i < N; i++) v += (g[i] - m) * (g[i] - m);
    const sd = Math.sqrt(v / N);
    const z = new Float32Array(N);
    if (sd > 0) for (let i = 0; i < N; i++) z[i] = (g[i] - m) / sd;
    return { z, sd };
  }
  function mad(a, b) {
    let s = 0;
    for (let i = 0; i < N; i++) s += Math.abs(a[i] - b[i]);
    return s / N;
  }
  function makeTpl(f, g) {
    return { f, g, z: zscore(g).z };
  }

  function loadSaved() {
    try {
      const s = JSON.parse(localStorage.getItem(LS));
      if (s) {
        saved = Object.assign(saved, s);
        tpl = (s.items || []).map((t) => makeTpl(t.f, unb64(t.g)));
      }
    } catch {}
  }
  function persist() {
    saved.items = tpl.map((t) => ({ f: t.f, g: b64(t.g) }));
    try {
      localStorage.setItem(LS, JSON.stringify(saved));
    } catch {}
  }

  // Pearson correlation against every template; best and runner-up by label.
  function classify(g) {
    const { z, sd } = zscore(g);
    if (sd < BLANK) return { blank: true };
    const byFace = new Map();
    for (const t of tpl) {
      let c = 0;
      for (let i = 0; i < N; i++) c += z[i] * t.z[i];
      c /= N;
      if (!(byFace.get(t.f) >= c)) byFace.set(t.f, c);
    }
    const ranked = [...byFace].map(([f, c]) => ({ f, c })).sort((a, b) => b.c - a.c);
    return { best: ranked[0], second: ranked[1] };
  }

  function onGray(g) {
    last = g;
    drawSig(g);
    const motion = prev ? mad(g, prev) : 255;
    prev = g;
    if (accepted && mad(g, accepted) > CHANGE) changed = true;
    stableN = motion < MOTION ? stableN + 1 : 0;
    const m = classify(g);
    show(m, stableN >= STABLE);
    if (stableN < STABLE || !changed) return;
    accepted = g;
    changed = false;
    const first = initial;
    initial = false;
    const known = m.best && m.best.c >= saved.thr;
    if (m.blank || (known && m.best.f === EMPTY)) return;
    eventGray = g;
    opts.onCard(known ? m.best.f : null, first);
  }

  // ---------- capture ----------
  async function start() {
    if (typeof MediaStreamTrackProcessor === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      setState('needs Chrome or Edge', 'error');
      return;
    }
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'window', frameRate: 30, cursor: 'never' },
        audio: false,
        selfBrowserSurface: 'exclude',
      });
    } catch {
      return; // picker cancelled
    }
    const track = stream.getVideoTracks()[0];
    track.addEventListener('ended', stop);
    const video = $('capVideo');
    video.srcObject = stream;
    video.play().catch(() => {});
    clone = track.clone();
    const proc = new MediaStreamTrackProcessor({ track: clone });
    if (!workerUrl) workerUrl = URL.createObjectURL(new Blob(['(' + visionWorker.toString() + ')(self);'], { type: 'text/javascript' }));
    worker = new Worker(workerUrl);
    worker.onmessage = (e) => (e.data.ended ? stop() : onGray(new Uint8Array(e.data.gray)));
    worker.postMessage({ cmd: 'start', readable: proc.readable, roi: saved.roi, w: W, h: H, interval: INTERVAL }, [proc.readable]);
    rearm();
    renderPanel();
  }

  function stop() {
    if (worker) worker.terminate();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (clone) clone.stop();
    worker = stream = clone = null;
    $('capVideo').srcObject = null;
    renderPanel();
    opts.onStatus('');
  }

  function rearm() {
    prev = accepted = null;
    stableN = 0;
    changed = initial = true;
  }

  // Stores the image of the last card event (or `g`) as a template for `face`.
  function learn(face, g) {
    g = g || eventGray;
    if (!g || !saved.roi) return false;
    tpl.push(makeTpl(face, g));
    const mine = tpl.filter((t) => t.f === face);
    if (mine.length > KEEP) tpl.splice(tpl.indexOf(mine[0]), 1);
    persist();
    renderLearned();
    return true;
  }

  // ---------- panel ----------
  function setState(text, tone) {
    const el = $('capState');
    el.textContent = text;
    el.className = 'n' + (tone ? ' s-' + tone : '');
  }

  function renderPanel() {
    const on = !!stream;
    $('btnCap').textContent = on ? 'Stop capture' : 'Capture mirror window';
    $('capView').hidden = !on;
    if (!on) setState('off');
    else if (!saved.roi) setState('draw a box on the preview', 'warning');
    else setState('watching', 'success');
    $('capThr').value = saved.thr;
    renderLearned();
    if (!saved.roi) $('capRead').textContent = 'Drag a box around the rank in the corner of the card you are dealt.';
    drawOverlay();
  }

  function renderLearned() {
    const count = (f) => tpl.filter((t) => t.f === f).length;
    $('capLearned').innerHTML = opts.faces
      .concat([EMPTY])
      .map((f) => {
        const n = count(f);
        return `<span class="${n ? 'on' : ''}" title="${n} sample${n === 1 ? '' : 's'}">${f === EMPTY ? 'empty' : f}</span>`;
      })
      .join('');
  }

  function show(m, settled) {
    let t;
    if (m.blank) t = 'Empty box';
    else if (!m.best) t = 'Unknown card · nothing learned yet';
    else {
      const name = (f) => (f === EMPTY ? 'empty' : f);
      t = (m.best.c >= saved.thr ? `Sees ${name(m.best.f)}` : `Unknown (closest ${name(m.best.f)})`) + ` ${m.best.c.toFixed(2)}`;
      if (m.second) t += ` · runner-up ${name(m.second.f)} ${m.second.c.toFixed(2)}`;
    }
    $('capRead').textContent = t + (settled ? ' · settled' : ' · moving');
    opts.onStatus('Reader: ' + t);
  }

  function drawSig(g) {
    const c = $('capSig').getContext('2d');
    const img = c.createImageData(W, H);
    for (let i = 0; i < N; i++) {
      img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = g[i];
      img.data[4 * i + 3] = 255;
    }
    c.putImageData(img, 0, 0);
  }

  function drawOverlay(r) {
    const video = $('capVideo');
    const ov = $('capRoi');
    const w = video.clientWidth;
    const h = video.clientHeight;
    if (ov.width !== w) ov.width = w;
    if (ov.height !== h) ov.height = h;
    const c = ov.getContext('2d');
    c.clearRect(0, 0, w, h);
    r = r || saved.roi;
    if (!r) return;
    c.fillStyle = 'rgba(0,0,0,.45)';
    c.fillRect(0, 0, w, h);
    c.clearRect(r.x * w, r.y * h, r.w * w, r.h * h);
    c.strokeStyle = '#f7c548';
    c.lineWidth = 2;
    c.strokeRect(r.x * w, r.y * h, r.w * w, r.h * h);
  }

  function setRoi(r) {
    if (tpl.length && !confirm(`Moving the box forgets the ${tpl.length} learned card images. Continue?`)) return drawOverlay();
    tpl = [];
    saved.roi = r;
    persist();
    if (worker) worker.postMessage({ cmd: 'roi', roi: r });
    rearm();
    renderPanel();
  }

  function bindRoiDrawing() {
    const ov = $('capRoi');
    const clamp = (v) => Math.min(1, Math.max(0, v));
    const pt = (e) => {
      const b = ov.getBoundingClientRect();
      return { x: clamp((e.clientX - b.left) / b.width), y: clamp((e.clientY - b.top) / b.height) };
    };
    let drag = null;
    ov.addEventListener('pointerdown', (e) => {
      drag = { p: pt(e), r: null };
      ov.setPointerCapture(e.pointerId);
    });
    ov.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const q = pt(e);
      drag.r = { x: Math.min(drag.p.x, q.x), y: Math.min(drag.p.y, q.y), w: Math.abs(q.x - drag.p.x), h: Math.abs(q.y - drag.p.y) };
      drawOverlay(drag.r);
    });
    ov.addEventListener('pointerup', () => {
      const r = drag && drag.r;
      drag = null;
      if (r && r.w * ov.width >= 6 && r.h * ov.height >= 6) setRoi(r);
      else drawOverlay();
    });
    new ResizeObserver(() => drawOverlay()).observe($('capVideo'));
  }

  function init(o) {
    opts = o;
    loadSaved();
    bindRoiDrawing();
    $('btnCap').onclick = () => (stream ? stop() : start());
    $('btnCapEmpty').onclick = () => {
      if (!last || !learn(EMPTY, last)) return;
      accepted = last;
      changed = false;
      opts.onEmpty();
    };
    $('btnCapForget').onclick = () => {
      if (!tpl.length || !confirm(`Forget all ${tpl.length} learned card images?`)) return;
      tpl = [];
      persist();
      renderLearned();
    };
    $('capThr').addEventListener('change', () => {
      const v = Number($('capThr').value);
      if (v > 0 && v < 1) saved.thr = v;
      persist();
      $('capThr').value = saved.thr;
    });
    renderPanel();
  }

  return { init, learn, rearm, active: () => !!stream };
})();
