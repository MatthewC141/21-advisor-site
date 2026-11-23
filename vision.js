'use strict';
// Card reader. Watches a box the user draws on the phone's screen, waits for the box to settle after it
// changes, and matches it against cards the user has taught it. Two frame sources:
//  - 'phone': the bundled AirPlay receiver (receiver/serve.py) when the page is served by it; the iPhone
//    mirrors straight to the advisor.
//  - 'window': a shared window (any mirroring app) via getDisplayMedia + MediaStreamTrackProcessor.
// Frames are read in a worker, so the reader keeps running while the tab is hidden.

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
    const fw = latest.displayWidth || latest.width; // VideoFrame or ImageBitmap
    const fh = latest.displayHeight || latest.height;
    g2.drawImage(latest, roi.x * fw, roi.y * fh, Math.max(1, roi.w * fw), Math.max(1, roi.h * fh), 0, 0, W, H);
    const d = g2.getImageData(0, 0, W, H).data;
    const g = new Uint8Array(W * H);
    for (let i = 0; i < g.length; i++) g[i] = (d[4 * i] * 77 + d[4 * i + 1] * 150 + d[4 * i + 2] * 29) >> 8;
    scope.postMessage({ gray: g.buffer }, [g.buffer]);
  }
  // Board spots: each roi {x, y, w, h} (fractions of the frame) scaled to ow x oh grayscale.
  const g3 = new OffscreenCanvas(1, 1).getContext('2d', { willReadFrequently: true });
  function grabAll(rois) {
    if (!latest) return null;
    const fw = latest.displayWidth || latest.width;
    const fh = latest.displayHeight || latest.height;
    return rois.map((r) => {
      g3.canvas.width = r.ow;
      g3.canvas.height = r.oh;
      g3.imageSmoothingQuality = 'high';
      g3.drawImage(latest, r.x * fw, r.y * fh, r.w * fw, r.h * fh, 0, 0, r.ow, r.oh);
      const d = g3.getImageData(0, 0, r.ow, r.oh).data;
      const g = new Uint8Array(r.ow * r.oh);
      for (let i = 0; i < g.length; i++) g[i] = (d[4 * i] * 77 + d[4 * i + 1] * 150 + d[4 * i + 2] * 29) >> 8;
      return g;
    });
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
  // Receiver frames: fetch the newest JPEG each interval; report when the iPhone starts/stops mirroring.
  async function poll(url, interval) {
    let live = null;
    for (;;) {
      const t = Date.now();
      let ok = false;
      try {
        const r = await fetch(url, { cache: 'no-store' });
        if (r.status === 200) {
          const bmp = await createImageBitmap(await r.blob());
          if (latest) latest.close();
          latest = bmp;
          ok = true;
        }
      } catch {}
      if (ok !== live) scope.postMessage({ live: (live = ok) });
      if (ok) sample();
      await new Promise((res) => setTimeout(res, Math.max(0, interval - (Date.now() - t))));
    }
  }
  scope.onmessage = (e) => {
    const m = e.data;
    if (m.cmd === 'start') {
      W = m.w;
      H = m.h;
      g2 = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
      g2.imageSmoothingQuality = 'high';
      roi = m.roi;
      if (m.url) poll(m.url, m.interval);
      else {
        pump(m.readable.getReader());
        setInterval(sample, m.interval);
      }
    } else if (m.cmd === 'roi') roi = m.roi;
    else if (m.cmd === 'grab')
      // Two grabs `gap` ms apart, so the page can tell a settled board from one still animating.
      setTimeout(() => {
        const a = grabAll(m.rois);
        setTimeout(() => scope.postMessage({ grab: m.id, a, b: grabAll(m.rois) }), m.gap);
      }, m.delay);
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
  let saved = { roi: null, thr: 0.8, items: [], nums: [], board: true };
  let tpl = []; // {f, g: Uint8Array, z: Float32Array}
  let numTpl = []; // board numbers: {k: 'total' | 'deck', v, g, z}
  let stream = null;
  let clone = null;
  let worker = null;
  let workerUrl = null;
  let phone = null; // AirPlay receiver name when the page is served by receiver/serve.py
  let live = false; // phone mode: the iPhone is mirroring
  const view = () => (phone ? $('capImg') : $('capVideo'));

  // detector
  let prev = null;
  let stableN = 0;
  let accepted = null; // last settled image that was acted on
  let acceptedFace = null; // card recognised in `accepted` (null: unknown or no card)
  let changed = true; // box pixels moved away from `accepted` since
  let left = true; // a sample since `accepted` did not show acceptedFace (the card really left the box)
  let initial = true; // next event is the first since start/rearm
  let last = null; // newest sample
  let eventGray = null; // image of the last card event (what learn() stores)
  const events = []; // reader log, newest first

  const b64 = (g) => btoa(String.fromCharCode.apply(null, g));
  const unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  function zscore(g) {
    const n = g.length;
    let m = 0;
    for (let i = 0; i < n; i++) m += g[i];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (g[i] - m) * (g[i] - m);
    const sd = Math.sqrt(v / n);
    const z = new Float32Array(n);
    if (sd > 0) for (let i = 0; i < n; i++) z[i] = (g[i] - m) / sd;
    return { z, sd };
  }
  function mad(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s / a.length;
  }
  function makeTpl(f, g) {
    return { f, g, z: zscore(g).z };
  }
  // Pearson correlation of z against each template's z; best and runner-up per label.
  function rank(z, list, label) {
    const best = new Map();
    for (const t of list) {
      let c = 0;
      for (let i = 0; i < z.length; i++) c += z[i] * t.z[i];
      c /= z.length;
      if (!(best.get(label(t)) >= c)) best.set(label(t), c);
    }
    const ranked = [...best].map(([f, c]) => ({ f, c })).sort((a, b) => b.c - a.c);
    return { best: ranked[0], second: ranked[1] };
  }

  function loadSaved() {
    try {
      const s = JSON.parse(localStorage.getItem(LS));
      if (s) {
        saved = Object.assign(saved, s);
        tpl = (s.items || []).map((t) => makeTpl(t.f, unb64(t.g)));
        numTpl = (s.nums || []).map((t) => Object.assign(makeTpl(t.v, unb64(t.g)), { k: t.k, v: t.v }));
      }
    } catch {}
  }
  function persist() {
    saved.items = tpl.map((t) => ({ f: t.f, g: b64(t.g) }));
    saved.nums = numTpl.map((t) => ({ k: t.k, v: t.v, g: b64(t.g) }));
    try {
      localStorage.setItem(LS, JSON.stringify(saved));
    } catch {}
  }

  // Pearson correlation against every template; best and runner-up by label.
  function classify(g) {
    const { z, sd } = zscore(g);
    if (sd < BLANK) return { blank: true };
    return rank(z, tpl, (t) => t.f);
  }

  // ---------- board numbers: column totals and cards left, read after each move ----------
  // Fixed spots on the 21 game screen as fractions of the phone frame (measured on the iPhone mirror, 498 x 1080):
  // the total in the black circle above each column, and the cards-left count under the deck. Each spot is
  // grabbed at about native size, split into bright digit glyphs, and each glyph matched against learned digits
  // (10 per kind, so the reader soon knows every number).
  const GW = 10; // digit glyph grid
  const GH = 16;
  const NUM_THR = 0.85; // correlation needed to read a digit
  const NUM_KEEP = 3; // templates kept per digit
  const BOARD = [66, 187, 310, 434]
    .map((cx) => ({ x: (cx - 28) / 498, y: 256 / 1080, w: 56 / 498, h: 36 / 1080, ow: 56, oh: 36 }))
    .concat([{ x: 165 / 498, y: 998 / 1080, w: 44 / 498, h: 28 / 1080, ow: 44, oh: 28 }]);
  let grabSeq = 0;
  const grabs = new Map();

  // Bright-on-dark digits in a w x h gray image, left to right, each scaled to GH rows (keeping its shape, so a
  // narrow 1 stays narrow) and centred in GW columns. [] when nothing bright.
  function glyphs(g, w, h) {
    let lo = 255;
    let hi = 0;
    for (const v of g) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi - lo < 50) return [];
    const t = (lo + hi) / 2;
    const on = (x, y) => g[y * w + x] > t;
    const runs = [];
    for (let x = 0; x < w; x++) {
      let any = false;
      for (let y = 0; y < h && !any; y++) any = on(x, y);
      if (!any) continue;
      const r = runs[runs.length - 1];
      if (r && r[1] === x - 1) r[1] = x;
      else runs.push([x, x]);
    }
    const out = [];
    for (const [x0, x1] of runs) {
      let y0 = h;
      let y1 = -1;
      for (let y = 0; y < h; y++)
        for (let x = x0; x <= x1; x++)
          if (on(x, y)) {
            y0 = Math.min(y0, y);
            y1 = Math.max(y1, y);
          }
      if (x1 - x0 < 1 || y1 - y0 < h * 0.3) continue; // speck, not a digit
      const q = new Uint8Array(GW * GH).fill(lo);
      const rw = x1 - x0 + 1;
      const rh = y1 - y0 + 1;
      const gw = Math.min(GW, Math.max(1, Math.round((rw * GH) / rh)));
      const off = (GW - gw) >> 1;
      for (let gy = 0; gy < GH; gy++)
        for (let gx = 0; gx < gw; gx++) {
          const sx = Math.min(x1, Math.floor(x0 + ((gx + 0.5) * rw) / gw));
          const sy = Math.min(y1, Math.floor(y0 + ((gy + 0.5) * rh) / GH));
          q[gy * GW + off + gx] = g[sy * w + sx];
        }
      out.push(q);
    }
    return out;
  }

  // {pat, gl}: pat is the number as a digit string with '?' for digits not learned yet; null when no 1-2 digits.
  function readNumber(k, g, roi) {
    const gl = glyphs(g, roi.ow, roi.oh);
    if (!gl.length || gl.length > 2) return { pat: null, gl };
    const mine = numTpl.filter((t) => t.k === k);
    const pat = gl
      .map((q) => {
        const r = rank(zscore(q).z, mine, (t) => t.v);
        return r.best && r.best.c >= NUM_THR ? String(r.best.f) : '?';
      })
      .join('');
    return { pat, gl };
  }

  // Reads the board `delay` ms from now: {stable, totals: [{pat, gl}] x4, deck: {pat, gl}}. stable = the spots did
  // not change over 300 ms (no animation running). Phone mode only: a shared window has its own borders, so the
  // fixed spots would miss. null without frames.
  function readBoard(delay) {
    if (!boardOn()) return Promise.resolve(null);
    const id = ++grabSeq;
    return new Promise((res) => {
      grabs.set(id, (a, b) => {
        if (!a || !b) return res(null);
        const stable = a.every((g, i) => mad(g, b[i]) < MOTION);
        const read = b.map((g, i) => readNumber(i < 4 ? 'total' : 'deck', g, BOARD[i]));
        res({ stable, totals: read.slice(0, 4), deck: read[4] });
      });
      worker.postMessage({ cmd: 'grab', id, rois: BOARD, delay, gap: 300 });
    });
  }

  // Learns the unread ('?') digits of reading `rd` as the digits of `value`.
  function learnNumber(k, value, rd) {
    const ds = String(value);
    if (!rd.pat || ds.length !== rd.gl.length) return;
    rd.gl.forEach((q, i) => {
      if (rd.pat[i] !== '?') return;
      const v = Number(ds[i]);
      numTpl.push(Object.assign(makeTpl(v, q), { k, v }));
      const mine = numTpl.filter((t) => t.k === k && t.v === v);
      if (mine.length > NUM_KEEP) numTpl.splice(numTpl.indexOf(mine[0]), 1);
    });
    persist();
  }

  function note(text) {
    events.unshift(new Date().toLocaleTimeString() + '  ' + text);
    events.length = Math.min(events.length, 40);
    $('capLog').textContent = events.join('\n');
  }

  function onGray(g) {
    last = g;
    drawSig(g);
    const motion = prev ? mad(g, prev) : 255;
    prev = g;
    const m = classify(g);
    const face = m.best && m.best.c >= saved.thr ? m.best.f : null;
    if (accepted && mad(g, accepted) > CHANGE) changed = true;
    if (face !== acceptedFace) left = true;
    stableN = motion < MOTION ? stableN + 1 : 0;
    show(m, stableN >= STABLE);
    if (stableN < STABLE || !changed) return;
    const score = m.best ? ` (${m.best.c.toFixed(2)})` : '';
    if (acceptedFace && face === acceptedFace && !left) {
      // Same card never left the box: its pixels drifted (video compression, glow, a finger) but no new card came.
      accepted = g;
      changed = false;
      note(`${face}${score} still showing, ignored`);
      return;
    }
    accepted = g;
    acceptedFace = face;
    changed = left = false;
    const first = initial;
    initial = false;
    if (m.blank || face === EMPTY) return note('no card in the box');
    eventGray = g;
    note(face ? `new card ${face}${score}` : `unknown image${score}`);
    opts.onCard(face, first);
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
    spawn({ readable: proc.readable }, [proc.readable]);
  }

  // Page served by the bundled AirPlay receiver: read the iPhone's frames straight from it.
  function startPhone(name) {
    phone = name;
    const panel = $('capPanel');
    panel.parentNode.insertBefore(panel, panel.parentNode.children[1]); // right under the top bar
    $('capTitle').textContent = 'iPhone';
    $('capIntro').textContent = `On your iPhone: Control Center → Screen Mirroring → "${name}". The advisor reads each card you're dealt, takes it that you played the advised move, and calls out the next one. Played something else? Undo twice and fix it by hand.`;
    $('btnCap').hidden = true;
    $('capVideo').hidden = true;
    $('capImg').hidden = false;
    $('capImg').src = 'receiver/stream.mjpg';
    spawn({ url: new URL('receiver/frame.jpg', location.href).href }, []);
  }

  function spawn(src, transfer) {
    if (!workerUrl) workerUrl = URL.createObjectURL(new Blob(['(' + visionWorker.toString() + ')(self);'], { type: 'text/javascript' }));
    worker = new Worker(workerUrl);
    worker.onmessage = (e) => {
      const d = e.data;
      if (d.ended) stop();
      else if (d.grab) {
        const done = grabs.get(d.grab);
        grabs.delete(d.grab);
        if (done) done(d.a, d.b);
      }
      else if ('live' in d) {
        live = d.live;
        if (live) rearm(); // mirroring (re)started: the card showing may already be entered
        renderPanel();
        if (!live) opts.onStatus(`Waiting for iPhone · Screen Mirroring → ${phone}`);
      } else onGray(new Uint8Array(d.gray));
    };
    worker.postMessage(Object.assign({ cmd: 'start', roi: saved.roi, w: W, h: H, interval: INTERVAL }, src), transfer);
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
    prev = accepted = acceptedFace = null;
    stableN = 0;
    changed = left = initial = true;
  }

  // Stores the image of the last card event (or `g`) as a template for `face`.
  function learn(face, g) {
    g = g || eventGray;
    if (!g || !saved.roi) return false;
    tpl.push(makeTpl(face, g));
    if (g === accepted) {
      // The box still holds the image just taught (or corrected): it is now this card, not a new one.
      acceptedFace = face === EMPTY ? null : face;
      left = false;
    }
    note(`learned ${face === EMPTY ? 'empty box' : face}`);
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
    const on = !!worker;
    $('btnCap').textContent = on ? 'Stop capture' : 'Capture mirror window';
    $('capView').hidden = !on;
    if (!on) setState('off');
    else if (phone && !live) setState('waiting for iPhone', 'warning');
    else if (!saved.roi) setState('draw a box on the preview', 'warning');
    else setState(phone ? 'iPhone connected' : 'watching', 'success');
    $('capThr').value = saved.thr;
    $('capBoard').checked = saved.board;
    $('capBoardRow').hidden = !phone;
    renderLearned();
    if (phone && !live) $('capRead').textContent = `Waiting for the iPhone to mirror to "${phone}".`;
    else if (!saved.roi) $('capRead').textContent = 'Drag a box around the rank in the corner of the card you are dealt.';
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

  // Every card face has a template: from then on, a box without a known card means no game is on screen.
  const learnedAll = () => opts.faces.every((f) => tpl.some((t) => t.f === f));

  function show(m, settled) {
    const name = (f) => (f === EMPTY ? 'empty' : f);
    const known = m.best && m.best.c >= saved.thr && m.best.f !== EMPTY;
    let t;
    if (m.blank) t = 'Empty box';
    else if (!m.best) t = 'Unknown card · nothing learned yet';
    else {
      t = (m.best.c >= saved.thr ? `Sees ${name(m.best.f)}` : `Unknown (closest ${name(m.best.f)})`) + ` ${m.best.c.toFixed(2)}`;
      if (m.second) t += ` · runner-up ${name(m.second.f)} ${m.second.c.toFixed(2)}`;
    }
    if (known) t = 'In game · ' + t;
    else if (learnedAll()) t = 'Not in a game · ' + t;
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
    const el = view();
    const ov = $('capRoi');
    const w = el.clientWidth;
    const h = el.clientHeight;
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
    if (!phone || !saved.board) return;
    c.strokeStyle = '#6fb3ff'; // board numbers checked after each move
    c.setLineDash([4, 3]);
    for (const b of BOARD) c.strokeRect(b.x * w, b.y * h, b.w * w, b.h * h);
    c.setLineDash([]);
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
    const ro = new ResizeObserver(() => drawOverlay());
    ro.observe($('capVideo'));
    ro.observe($('capImg'));
  }

  function init(o) {
    opts = o;
    loadSaved();
    bindRoiDrawing();
    $('btnCap').onclick = () => (stream ? stop() : start());
    $('btnCapEmpty').onclick = () => {
      if (!last || !learn(EMPTY, last)) return;
      accepted = last;
      acceptedFace = null;
      changed = left = false;
      opts.onEmpty();
    };
    $('btnCapForget').onclick = () => {
      const n = tpl.length + numTpl.length;
      if (!n || !confirm(`Forget all ${tpl.length} learned card images and ${numTpl.length} board numbers?`)) return;
      tpl = [];
      numTpl = [];
      persist();
      renderLearned();
    };
    $('capThr').addEventListener('change', () => {
      const v = Number($('capThr').value);
      if (v > 0 && v < 1) saved.thr = v;
      persist();
      $('capThr').value = saved.thr;
    });
    $('capBoard').onchange = () => {
      saved.board = $('capBoard').checked;
      persist();
      drawOverlay();
    };
    renderPanel();
    if (location.protocol.startsWith('http'))
      fetch('receiver/status', { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : null))
        .then((s) => s && s.name && startPhone(s.name))
        .catch(() => {});
  }

  const boardOn = () => !!(worker && phone && live && saved.board);

  return { init, learn, rearm, learnedAll, readBoard, learnNumber, boardOn, active: () => !!worker };
})();
