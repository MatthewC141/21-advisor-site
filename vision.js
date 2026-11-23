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
      if (ok) roi ? sample() : scope.postMessage({ tick: true }); // no card box yet: still drive the payout check
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
  let saved = { roi: null, thr: 0.8, items: [], nums: [], slots: [], bustCal: null, board: true };
  let tpl = []; // {f, g: Uint8Array, z: Float32Array}
  let numTpl = []; // board numbers: {k: 'total' | 'deck' | 'score', v, g, z}
  let slotTpl = []; // column card corners: {f: label, g, z}
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
  let boxFace = null; // card recognised in the box on the latest sample

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
        slotTpl = (s.slots || []).map((t) => makeTpl(t.f, unb64(t.g)));
      }
    } catch {}
  }
  function persist() {
    saved.items = tpl.map((t) => ({ f: t.f, g: b64(t.g) }));
    saved.nums = numTpl.map((t) => ({ k: t.k, v: t.v, g: b64(t.g) }));
    saved.slots = slotTpl.map((t) => ({ f: t.f, g: b64(t.g) }));
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

  // ---------- board: column totals, cards left, every card in the columns, score and busts ----------
  // Fixed spots on the 21 game screen as fractions of the phone frame (measured on the iPhone mirror, 498 x 1080).
  // Numbers (totals in the black circles, cards left under the deck, score top right) are split into bright digit
  // glyphs matched against learned digits. Each column's cards overlap 49 px apart with the rank in the top-left
  // corner; the stack's bottom edge gives how many there are. All of it is learned from the advisor's own board
  // while it agrees with the screen.
  const GW = 10; // digit glyph grid
  const GH = 16;
  const NUM_THR = 0.85; // correlation needed to read a digit
  const NUM_KEEP = 3; // templates kept per digit
  const SLOT_KEEP = 4; // templates kept per card label
  const SLOT_THR = 0.9; // correlation needed to read a column card (its totals must agree before it's used anyway)
  const CARD_LEFT = [17, 139.5, 262, 384.5]; // left edge of each column's cards
  const px = (x, y, w, h, ow = w, oh = h) => ({ x: x / 498, y: y / 1080, w: w / 498, h: h / 1080, ow, oh });
  const B_TOTAL = 0; // 4 column totals
  const B_DECK = 4; // cards left
  const B_SLOT = 5; // 16 rank corners: column c, card k at B_SLOT + 4c + k
  const B_STACK = 21; // 4 strips down each column, to find where its cards end
  const B_SCORE = 25;
  const B_BUST = 26; // 3 bust markers at the top
  const BOARD = [66, 187, 310, 434]
    .map((cx) => px(cx - 28, 256, 56, 36))
    .concat([px(165, 998, 44, 28)])
    .concat(CARD_LEFT.flatMap((l) => [0, 1, 2, 3].map((k) => px(l + 2, 324 + 49 * k, 40, 40, 24, 24))))
    .concat(CARD_LEFT.map((l) => px(l + 20, 315, 55, 405)))
    .concat([px(345, 84, 130, 32)])
    .concat([204, 251, 298].map((cx) => px(cx - 7, 80, 14, 14)));
  const SHOWN = [0, 1, 2, 3, 4, B_SCORE]; // spots outlined on the preview
  let grabSeq = 0;
  const grabs = new Map();

  // Bright-on-dark digits in a w x h gray image, left to right, each scaled to gh rows (keeping its shape, so a
  // narrow 1 stays narrow) and centred in gw columns; avg = area-average each cell (for upscaled grabs) instead of
  // sampling its centre. [] when nothing bright.
  function glyphs(g, w, h, gw0 = GW, gh0 = GH, avg = false) {
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
      const q = new Uint8Array(gw0 * gh0).fill(lo);
      const rw = x1 - x0 + 1;
      const rh = y1 - y0 + 1;
      const gw = Math.min(gw0, Math.max(1, Math.round((rw * gh0) / rh)));
      const off = (gw0 - gw) >> 1;
      for (let gy = 0; gy < gh0; gy++)
        for (let gx = 0; gx < gw; gx++) {
          if (!avg) {
            const sx = Math.min(x1, Math.floor(x0 + ((gx + 0.5) * rw) / gw));
            const sy = Math.min(y1, Math.floor(y0 + ((gy + 0.5) * rh) / gh0));
            q[gy * gw0 + off + gx] = g[sy * w + sx];
            continue;
          }
          const sx0 = x0 + Math.floor((gx * rw) / gw);
          const sx1 = Math.max(x0 + Math.floor(((gx + 1) * rw) / gw), sx0 + 1);
          const sy0 = y0 + Math.floor((gy * rh) / gh0);
          const sy1 = Math.max(y0 + Math.floor(((gy + 1) * rh) / gh0), sy0 + 1);
          let s = 0;
          for (let y = sy0; y < sy1; y++) for (let x = sx0; x < sx1; x++) s += g[y * w + x];
          q[gy * gw0 + off + gx] = Math.round(s / ((sy1 - sy0) * (sx1 - sx0)));
        }
      out.push(q);
    }
    return out;
  }

  // {pat, gl}: pat is the number as a digit string with '?' for digits not learned yet; null when it isn't
  // 1..maxLen digits.
  function readNumber(k, g, roi, maxLen = 2) {
    const gl = glyphs(g, roi.ow, roi.oh);
    if (!gl.length || gl.length > maxLen) return { pat: null, gl };
    const mine = numTpl.filter((t) => t.k === k);
    const pat = gl
      .map((q) => {
        const r = rank(zscore(q).z, mine, (t) => t.v);
        return r.best && r.best.c >= NUM_THR ? String(r.best.f) : '?';
      })
      .join('');
    return { pat, gl };
  }

  // Rows down a column strip (from y 315) to the last bright card pixel: 1 card ends at y 463, each more adds 49.
  function stackCount(g, roi) {
    for (let y = roi.oh - 1; y >= 0; y--)
      for (let x = 0; x < roi.ow; x++)
        if (g[y * roi.ow + x] > 170) {
          const bottom = 315 + y;
          return bottom < 430 ? 0 : Math.min(4, Math.max(1, Math.round((bottom - 463) / 49) + 1));
        }
    return 0;
  }
  function readSlot(g) {
    const { z, sd } = zscore(g);
    if (sd < BLANK) return { label: null, g };
    const r = rank(z, slotTpl, (t) => t.f);
    return { label: r.best && r.best.c >= SLOT_THR ? r.best.f : null, g };
  }

  // Reads the board `delay` ms from now: {stable, totals: [{pat, gl}] x4, deck, score: {pat, gl},
  // cols: [{n, slots: [{label, g}]}] x4, busts: count or null}. stable = nothing changed over 300 ms (no animation).
  // Phone mode only: a shared window has its own borders, so the fixed spots would miss. null without frames.
  function readBoard(delay) {
    if (!boardOn()) return Promise.resolve(null);
    const id = ++grabSeq;
    return new Promise((res) => {
      grabs.set(id, (a, b) => {
        if (!a || !b) return res(null);
        const stable = a.every((g, i) => mad(g, b[i]) < MOTION);
        const bustMeans = [0, 1, 2].map((i) => b[B_BUST + i].reduce((s, v) => s + v, 0) / b[B_BUST + i].length);
        res({
          stable,
          totals: [0, 1, 2, 3].map((c) => readNumber('total', b[B_TOTAL + c], BOARD[B_TOTAL + c])),
          deck: readNumber('deck', b[B_DECK], BOARD[B_DECK]),
          score: readNumber('score', b[B_SCORE], BOARD[B_SCORE], 6),
          cols: [0, 1, 2, 3].map((c) => {
            const n = stackCount(b[B_STACK + c], BOARD[B_STACK + c]);
            return { n, slots: [0, 1, 2, 3].slice(0, n).map((k) => readSlot(b[B_SLOT + 4 * c + k])) };
          }),
          bustMeans,
          busts: readBusts(bustMeans),
        });
      });
      worker.postMessage({ cmd: 'grab', id, rois: BOARD, delay, gap: 300 });
    });
  }

  // Bust markers: their brightness is learned per marker while the advisor's bust count agrees with the screen.
  function readBusts(means) {
    const cal = saved.bustCal;
    if (!cal || cal.empty.some((v) => v == null)) return null;
    let n = 0;
    means.forEach((m, i) => {
      const filled = cal.filled[i] != null ? Math.abs(m - cal.filled[i]) < Math.abs(m - cal.empty[i]) : m > cal.empty[i] + 25;
      if (filled) n++;
    });
    return n;
  }
  function calibrateBusts(busts, means) {
    const cal = (saved.bustCal = saved.bustCal || { empty: [null, null, null], filled: [null, null, null] });
    means.forEach((m, i) => {
      const side = i < busts ? cal.filled : cal.empty;
      side[i] = side[i] == null ? m : 0.7 * side[i] + 0.3 * m;
    });
    persist();
  }

  // Board templates added by the last board check. If the next check finds a board it can't explain, they were
  // probably learned from a wrong advisor board (a misread card) and are dropped.
  let recent = [];
  let pending = [];
  function settleLearning(ok) {
    if (!ok) {
      const drop = new Set(recent);
      numTpl = numTpl.filter((t) => !drop.has(t));
      slotTpl = slotTpl.filter((t) => !drop.has(t));
      persist();
    }
    recent = pending;
    pending = [];
  }

  // Learns a column card's corner as `label` when it can't be read, or when it reads as a different rank (adding the
  // correct example of this exact image outranks the wrong template next time).
  const rankOf = (f) => Math.min(opts.faces.indexOf(f), 9);
  function learnSlot(label, rd) {
    if (rd.label !== null && rankOf(rd.label) === rankOf(label)) return;
    const t = makeTpl(label, rd.g);
    slotTpl.push(t);
    pending.push(t);
    const mine = slotTpl.filter((x) => x.f === label);
    if (mine.length > SLOT_KEEP) slotTpl.splice(slotTpl.indexOf(mine[0]), 1);
    persist();
  }

  // Learns the unread ('?') digits of reading `rd` as the digits of `value`.
  function learnNumber(k, value, rd) {
    const ds = String(value);
    if (!rd.pat || ds.length !== rd.gl.length) return;
    rd.gl.forEach((q, i) => {
      if (rd.pat[i] !== '?') return;
      const v = Number(ds[i]);
      const t = Object.assign(makeTpl(v, q), { k, v });
      numTpl.push(t);
      pending.push(t);
      const mine = numTpl.filter((x) => x.k === k && x.v === v);
      if (mine.length > NUM_KEEP) numTpl.splice(numTpl.indexOf(mine[0]), 1);
    });
    persist();
  }

  // ---------- payout screen: read the game's payout graph before a cash game ----------
  // The graph's right edge lists 10 scores on evenly spaced gridlines (iPhone mirror, 498 x 1080: rows 43 px apart
  // from y 455). Gridlines are evenly spaced in payout: Break Even (the lowest white label; lower ones are red) is 1x,
  // the row with the "2.2x" box is 2.2x, so each row is a fixed step; the top row is the 2.7x maximum. Labels are
  // read at 3x with built-in digit templates (payout-digits.js), so it works without any teaching.
  const PAY_ROWS = 10;
  const PAY_ROIS = [];
  for (let k = 0; k < PAY_ROWS; k++) {
    const y = 455 + 43 * k;
    PAY_ROIS.push({ x: 398 / 498, y: (y - 12) / 1080, w: 54 / 498, h: 24 / 1080, ow: 162, oh: 72 }); // score label
  }
  for (let k = 0; k < PAY_ROWS; k++) {
    const y = 455 + 43 * k;
    PAY_ROIS.push({ x: 140 / 498, y: (y - 1) / 1080, w: 230 / 498, h: 3 / 1080, ow: 230, oh: 3 }); // label box strip
  }
  const payTpl =
    typeof PAYOUT_DIGITS === 'undefined'
      ? []
      : Object.entries(PAYOUT_DIGITS).flatMap(([v, list]) => list.map((s) => Object.assign(makeTpl(Number(v), unb64(s)), { v: Number(v) })));
  const PAY_EVERY = 15; // samples between payout-screen checks (about 1.5 s)
  let payTick = 0;
  let payBusy = false;
  let lastPay = '';

  // {be, s22, s27, pts: [[score, multiplier]] ascending} from grabbed label/strip images, or null if this isn't the
  // payout screen (every label must read, scores must fall row by row, boxes must sit on the 2.7x, 2.2x, 1x rows).
  function parsePayout(gr) {
    const rows = [];
    for (let k = 0; k < PAY_ROWS; k++) {
      const g = gr[k];
      const gl = glyphs(g, 162, 72, 12, 20, true);
      if (gl.length < 3 || gl.length > 5) return null;
      let digits = '';
      for (const q of gl) {
        const r = rank(zscore(q).z, payTpl, (t) => t.v);
        if (!r.best || r.best.c < 0.8) return null;
        digits += r.best.f;
      }
      let hi = 0;
      for (const v of g) if (v > hi) hi = v;
      let box = 0;
      for (const v of gr[PAY_ROWS + k]) if (v > 35 && v < 95) box++;
      rows.push({ score: Number(digits), hi, box: box >= 60 });
    }
    // Red labels (below break even) are darker in gray than white ones (~125 vs ~160): split at the midpoint
    // when the two brightness groups are clearly apart.
    const his = rows.map((r) => r.hi);
    const mx = Math.max(...his);
    const mn = Math.min(...his);
    for (const r of rows) r.red = mx - mn >= 20 && r.hi < (mx + mn) / 2;
    for (let k = 1; k < PAY_ROWS; k++) if (rows[k].score >= rows[k - 1].score) return null;
    const be = rows.findIndex((r) => r.red) - 1; // lowest white row
    if (be < 2 || rows.slice(be + 1).some((r) => !r.red)) return null;
    const mids = rows.map((r, k) => (r.box && k > 0 && k < be ? k : -1)).filter((k) => k >= 0);
    if (!rows[0].box || !rows[be].box || mids.length !== 1) return null;
    const k22 = mids[0];
    const step = 1.2 / (be - k22);
    const mult = (k) => (k === 0 ? 2.7 : Math.max(0, Math.min(2.7, 1 + (be - k) * step)));
    const pts = rows.map((r, k) => [r.score, Math.round(mult(k) * 1000) / 1000]).reverse();
    return { be: rows[be].score, s22: rows[k22].score, s27: rows[0].score, pts };
  }

  // One look at the payout screen: the parsed curve, or null (not on screen, or the screen was still moving).
  function readPayout() {
    if (!worker || !phone || !live) return Promise.resolve(null);
    const id = ++grabSeq;
    return new Promise((res) => {
      grabs.set(id, (a, b) => res(a && b && a.every((g, i) => mad(g, b[i]) < MOTION) ? parsePayout(b) : null));
      worker.postMessage({ cmd: 'grab', id, rois: PAY_ROIS, delay: 0, gap: 300 });
    });
  }

  function maybeReadPayout() {
    if (payBusy || !phone || !live || !opts.wantPayout() || ++payTick % PAY_EVERY) return;
    payBusy = true;
    readPayout().then((curve) => {
      payBusy = false;
      if (!curve) return;
      const key = JSON.stringify(curve.pts);
      if (key === lastPay) return;
      lastPay = key;
      note(`payout screen: 1x ${curve.be} · 2.2x ${curve.s22} · 2.7x ${curve.s27}`);
      opts.onPayout(curve);
    });
  }

  // The "Scan payout graph" button: a few looks over about 2 s, so a screen that is still settling is read too.
  // {curve} on success, else {reason}.
  async function scanPayout() {
    if (!worker || !phone) return { reason: 'receiver' };
    if (!live) return { reason: 'phone' };
    for (let i = 0; i < 4; i++) {
      const curve = await readPayout();
      if (curve) {
        lastPay = JSON.stringify(curve.pts);
        note(`payout screen (scan): 1x ${curve.be} · 2.2x ${curve.s22} · 2.7x ${curve.s27}`);
        return { curve };
      }
    }
    return { reason: 'screen' };
  }

  function note(text) {
    events.unshift(new Date().toLocaleTimeString() + '  ' + text);
    events.length = Math.min(events.length, 40);
    $('capLog').textContent = events.join('\n');
    opts.onLog(text);
  }

  function onGray(g) {
    last = g;
    drawSig(g);
    maybeReadPayout();
    opts.onTick();
    const motion = prev ? mad(g, prev) : 255;
    prev = g;
    const m = classify(g);
    // Until every card is learned, an unknown card can look like a known one: demand a closer match then.
    const thr = learnedAll() ? saved.thr : Math.max(saved.thr, 0.9);
    const face = m.best && m.best.c >= thr ? m.best.f : null;
    boxFace = face === EMPTY ? null : face;
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

  // The bundled AirPlay receiver: the page is served by it (base ''), or the public site reaches it on this PC
  // (base RX_URL). Reads the iPhone's frames straight from it.
  function startPhone(name, base) {
    if (worker) stop(); // a shared-window capture gives way
    phone = name;
    const panel = $('capPanel');
    panel.parentNode.insertBefore(panel, panel.parentNode.children[1]); // right under the top bar
    $('capTitle').textContent = 'iPhone';
    $('capIntro').textContent = `On your iPhone: Control Center → Screen Mirroring → "${name}". The advisor reads each card you're dealt, takes it that you played the advised move, and calls out the next one. Played something else? Undo twice and fix it by hand.`;
    $('btnCap').hidden = true;
    $('btnRx').hidden = true;
    $('capVideo').hidden = true;
    $('capImg').hidden = false;
    $('capImg').src = base + 'receiver/stream.mjpg';
    spawn({ url: new URL(base + 'receiver/frame.jpg', location.href).href }, []);
    opts.onReceiver(base);
  }

  // Public site: the receiver runs on this PC at RX_URL (?rx= overrides). Connecting is a click the first time
  // (Chrome may ask to allow local network access); after that it reconnects on load.
  const RX_URL = new URLSearchParams(location.search).get('rx') || 'http://127.0.0.1:8721/';
  const RX_AUTO = 'tw21.rx.auto';
  async function probe(base) {
    try {
      const r = await fetch(base + 'receiver/status', { cache: 'no-store' });
      const s = r.ok ? await r.json() : null;
      return s && s.name ? s : null;
    } catch {
      return null;
    }
  }
  async function connectReceiver(manual) {
    const s = await probe(RX_URL);
    if (s) {
      localStorage.setItem(RX_AUTO, '1');
      return startPhone(s.name, RX_URL);
    }
    if (manual) setState('receiver not found: start 21 Advisor Receiver on this PC', 'error');
  }
  function offerReceiver() {
    $('btnRx').hidden = false;
    if (localStorage.getItem(RX_AUTO)) connectReceiver(false);
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
      } else if (d.tick) {
        maybeReadPayout();
        opts.onTick();
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
    for (const i of SHOWN) c.strokeRect(BOARD[i].x * w, BOARD[i].y * h, BOARD[i].w * w, BOARD[i].h * h);
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
      const n = tpl.length + numTpl.length + slotTpl.length;
      if (!n || !confirm(`Forget all ${tpl.length} learned card images and everything learned about the board?`)) return;
      tpl = [];
      numTpl = [];
      slotTpl = [];
      saved.bustCal = null;
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
    $('btnRx').onclick = () => connectReceiver(true);
    if (location.protocol === 'http:') probe('').then((s) => (s ? startPhone(s.name, '') : offerReceiver()));
    else if (location.protocol === 'https:') offerReceiver();
  }

  const boardOn = () => !!(worker && phone && live && saved.board);

  return {
    init, learn, rearm, learnedAll, readBoard, learnNumber, learnSlot, settleLearning, calibrateBusts, scanPayout, boardOn,
    active: () => !!worker,
    boxFace: () => boxFace, // card recognised in the box right now (null: none, unknown or not learned)
  };
})();
