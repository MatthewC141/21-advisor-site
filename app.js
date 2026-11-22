'use strict';

// Runs inside a Web Worker (or inline fallback). `scope` is the worker global or a fake.
function workerMain(scope) {
  let params = null;
  let ctx = null;
  let ctxKey = '';
  let job = null;
  function loop() {
    const j = job;
    if (!j) return;
    const t = performance.now();
    const out = Engine.rolloutBatch(ctx, j.S, j.cnt, j.r, j.actions, j.batch, j.rng);
    const dt = performance.now() - t;
    if (job !== j) return;
    scope.postMessage({ id: j.id, n: j.batch, scores: out.buffer }, [out.buffer]);
    if (dt < 40) j.batch = Math.min(j.batch * 2, 8192);
    else if (dt > 150 && j.batch > 1) j.batch >>= 1;
    setTimeout(loop, 0);
  }
  scope.onmessage = function (e) {
    const m = e.data;
    if (m.cmd === 'init') params = m.params;
    else if (m.cmd === 'start') {
      const key = JSON.stringify(m.rules);
      if (key !== ctxKey) {
        ctx = Engine.makeContext(m.rules, params);
        ctxKey = key;
      }
      job = {
        id: m.id,
        S: Int32Array.from(m.S),
        cnt: Int32Array.from(m.cnt),
        r: m.r,
        actions: Int32Array.from(m.actions),
        rng: Engine.mulberry32(m.seed),
        batch: 4,
      };
      setTimeout(loop, 0);
    } else if (m.cmd === 'stop') job = null;
  };
}

(function () {
  const E = Engine;
  const $ = (id) => document.getElementById(id);
  const LS_SET = 'tw21.settings.v1';
  const LS_GAME = 'tw21.game.v1';
  const LS_HIST = 'tw21.hist.v1';
  const FACES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
  const faceRank = (f) => Math.min(FACES.indexOf(f), 9);
  const fmt = (x) => Math.round(x).toLocaleString();
  const pct = (x) => (100 * x).toFixed(1) + '%';

  function defaultSettings() {
    return {
      objective: 'payout',
      curve: Object.assign({}, E.DEFAULT_CURVE),
      rules: Object.assign({}, E.DEFAULT_RULES),
      thinkSec: 3,
      workers: Math.max(1, Math.min(12, (navigator.hardwareConcurrency || 4) - 1)),
    };
  }
  function load(k) {
    try {
      return JSON.parse(localStorage.getItem(k));
    } catch {
      return null;
    }
  }
  function save(k, v) {
    try {
      localStorage.setItem(k, JSON.stringify(v));
    } catch {}
  }

  let settings = Object.assign(defaultSettings(), load(LS_SET) || {});
  settings.rules = Object.assign({}, E.DEFAULT_RULES, settings.rules);
  if (!settings.curve || !(settings.curve.be > 0)) settings.curve = Object.assign({}, E.DEFAULT_CURVE); // pre-exponential saves
  let ctx = E.makeContext(settings.rules, ENGINE_PARAMS);

  // ---------- payout lookup ----------
  let payTab = null;
  let BE = 4700;
  let MAXS = 7900;
  function rebuildPayout() {
    const c = settings.curve;
    payTab = new Float64Array(30001);
    for (let s = 0; s <= 30000; s++) payTab[s] = E.payout(s, c);
    BE = c.be;
    MAXS = c.s27;
  }
  rebuildPayout();
  const utility = (s) => (settings.objective === 'score' ? s : payTab[s < 0 ? 0 : s > 30000 ? 30000 : s]);

  // ---------- game state ----------
  function newGame() {
    return { cols: [[], [], [], []], busts: 0, staysLeft: settings.rules.stays, streak: 0, score: 0, deck: E.FULL_DECK.slice(), current: null, turn: 0 };
  }
  let G = load(LS_GAME) || newGame();
  let history = load(LS_HIST) || [];

  const colCode = (cards) => E.codeFromCards(cards.map((c) => c.r));
  const deckTotal = () => G.deck.reduce((a, b) => a + b, 0);
  const stateS = () => E.makeState(G.cols.map(colCode), G.busts, G.staysLeft, G.streak, G.score);
  const isOver = () => G.busts >= settings.rules.maxBusts || (deckTotal() === 0 && !G.current);

  function snapshot() {
    history.push(JSON.stringify(G));
    if (history.length > 150) history.shift();
  }
  function persist() {
    save(LS_GAME, G);
    save(LS_HIST, history);
  }
  function commit() {
    persist();
    render();
    analyze();
  }

  let toastTimer = 0;
  function toast(big, small, bad) {
    const t = $('toast');
    t.querySelector('.t1').textContent = big;
    t.querySelector('.t2').textContent = small || '';
    t.classList.toggle('bad', !!bad);
    t.classList.remove('show');
    void t.offsetWidth;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 1300);
  }

  function enterCard(face) {
    if (isOver() && !G.current) return;
    const r = faceRank(face);
    snapshot();
    if (G.current) G.deck[G.current.r]++;
    if (G.deck[r] <= 0) toast('Check deck', 'Tracker shows no ' + (r === 9 ? '10-value' : FACES[r]) + ' left', true);
    else G.deck[r]--;
    G.current = { r, label: face };
    commit();
  }

  function clearCurrent() {
    if (!G.current) return;
    snapshot();
    G.deck[G.current.r]++;
    G.current = null;
    commit();
  }

  function doStay(c) {
    G.cols[c] = [];
    G.staysLeft--;
  }

  function stay(c) {
    if (G.staysLeft <= 0 || !G.cols[c].length) return;
    snapshot();
    doStay(c);
    toast('Stay', 'column ' + (c + 1) + ' cleared');
    commit();
  }

  function place(c, stayCol) {
    if (!G.current || isOver()) return;
    snapshot();
    if (stayCol >= 0) doStay(stayCol);
    const S = stateS();
    const res = E.step(ctx, S, c, G.current.r);
    G.busts = S[4];
    G.streak = S[6];
    G.score = S[7];
    if (res.type === 0) G.cols[c].push(G.current);
    else G.cols[c] = [];
    G.current = null;
    G.turn++;
    if (res.type === 1) toast(`${outcomeName(res.pts)}! +${res.pts}`, res.bonus ? `On a roll! +${res.bonus}` : '');
    else if (res.type === 2) {
      const lost = G.busts >= settings.rules.maxBusts && settings.rules.bustOutLoses;
      toast('Bust!', lost ? 'Game lost' : `${G.busts} of ${settings.rules.maxBusts}`, true);
    }
    commit();
  }

  function outcomeName(pts) {
    const R = settings.rules;
    if (pts === R.pts5x21) return '5 cards = 21';
    if (pts === R.pts5) return '5 cards';
    if (pts === R.pts777) return 'Three 7s';
    return '21';
  }

  function undo() {
    if (!history.length) return;
    G = JSON.parse(history.pop());
    commit();
  }

  // ---------- analysis (parallel rollouts) ----------
  const workerSrc = EngineFactory.toString() + '\nself.Engine = EngineFactory();\n(' + workerMain.toString() + ')(self);';
  let workerUrl = null;
  let workers = [];
  let cur = null;
  let jobId = 0;

  function makeWorker() {
    try {
      if (!workerUrl) workerUrl = URL.createObjectURL(new Blob([workerSrc], { type: 'text/javascript' }));
      return new Worker(workerUrl);
    } catch {
      // Inline fallback: same code on the main thread.
      const fake = { onmessage: null };
      const scope = {
        postMessage: (data) => setTimeout(() => fake.onmessage && fake.onmessage({ data }), 0),
        onmessage: null,
      };
      workerMain(scope);
      fake.postMessage = (data) => scope.onmessage({ data });
      fake.terminate = () => scope.onmessage({ data: { cmd: 'stop' } });
      return fake;
    }
  }
  function ensureWorkers() {
    const n = Math.max(1, settings.workers | 0);
    while (workers.length > n) workers.pop().terminate();
    while (workers.length < n) {
      const w = makeWorker();
      w.onmessage = onWorkerMsg;
      w.postMessage({ cmd: 'init', params: ENGINE_PARAMS });
      workers.push(w);
    }
  }
  function onWorkerMsg(e) {
    const m = e.data;
    if (!cur || m.id !== cur.id || !cur.running) return;
    cur.chunks.push(new Int32Array(m.scores));
    cur.n += m.n;
  }
  function launch() {
    workers.forEach((w, i) =>
      w.postMessage({
        cmd: 'start',
        id: cur.id,
        rules: settings.rules,
        S: Array.from(cur.S),
        cnt: Array.from(cur.cnt),
        r: cur.r,
        actions: cur.actions.map((x) => x.a),
        seed: ((Math.random() * 2 ** 31) | 0) ^ (i * 7919),
      }),
    );
    cur.running = true;
    clearInterval(cur.timer);
    cur.timer = setInterval(tick, 200);
  }
  function stopWorkers() {
    workers.forEach((w) => w.postMessage({ cmd: 'stop' }));
    if (cur) {
      cur.running = false;
      clearInterval(cur.timer);
    }
  }

  function analyze() {
    stopWorkers();
    cur = null;
    if (isOver()) {
      renderRec();
      return;
    }
    ensureWorkers();
    const S = stateS();
    const cnt = Int32Array.from(G.deck);
    const r = G.current ? G.current.r : -1;
    const actions = r >= 0 ? E.enumActions(ctx, S, r) : [{ a: 0 }];
    if (r < 0 && deckTotal() === 0) {
      renderRec();
      return;
    }
    const budget = r >= 0 && actions.length > 1 ? settings.thinkSec * 1000 : 900;
    cur = { id: ++jobId, S, cnt, r, actions, chunks: [], n: 0, t0: performance.now(), budget, running: false, timer: 0, stats: null };
    launch();
    renderRec();
  }

  function extend() {
    if (!cur) return;
    cur.budget = performance.now() - cur.t0 + settings.thinkSec * 1000;
    launch();
    renderRec();
  }

  function computeStats() {
    const A = cur.actions.length;
    const n = cur.n;
    const sumU = new Float64Array(A);
    const sumS = new Float64Array(A);
    const be = new Float64Array(A);
    const mx = new Float64Array(A);
    for (const ch of cur.chunks) {
      for (let i = 0; i < ch.length; i++) {
        const j = i % A;
        const s = ch[i];
        sumU[j] += utility(s);
        sumS[j] += s;
        if (s >= BE) be[j]++;
        if (s >= MAXS) mx[j]++;
      }
    }
    let best = 0;
    for (let j = 1; j < A; j++) if (sumU[j] > sumU[best]) best = j;
    // paired difference vs best (same futures)
    const sd = new Float64Array(A);
    const sd2 = new Float64Array(A);
    for (const ch of cur.chunks) {
      for (let k = 0; k < ch.length; k += A) {
        const ub = utility(ch[k + best]);
        for (let j = 0; j < A; j++) {
          const d = ub - utility(ch[k + j]);
          sd[j] += d;
          sd2[j] += d * d;
        }
      }
    }
    const rows = cur.actions.map((act, j) => {
      const md = n ? sd[j] / n : 0;
      const varD = n > 1 ? Math.max(0, sd2[j] / n - md * md) / (n - 1) : 0;
      return Object.assign({}, act, {
        ev: n ? sumU[j] / n : 0,
        diff: md,
        se: Math.sqrt(varD),
        meanScore: n ? sumS[j] / n : 0,
        pBE: n ? be[j] / n : 0,
        pMax: n ? mx[j] / n : 0,
      });
    });
    return { n, best, rows };
  }

  function tick() {
    if (!cur) return;
    const st = computeStats();
    cur.stats = st;
    const el = performance.now() - cur.t0;
    let decided = st.n >= 800 && st.rows.length > 1;
    if (decided) for (let j = 0; j < st.rows.length; j++) if (j !== st.best && !(st.rows[j].diff > 4 * st.rows[j].se && st.rows[j].se > 0)) decided = false;
    if (el >= cur.budget || decided) stopWorkers();
    renderRec();
  }

  // ---------- rendering ----------
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const totalText = (code) => {
    const sum = code & 31;
    return (code >> 5) & 1 && sum + 10 <= 21 ? `${sum}/${sum + 10}` : `${sum}`;
  };

  function describeMove(row) {
    const label = G.current ? G.current.label : '';
    const S = stateS();
    const equiv = [];
    if (row.stay !== row.col) for (let i = 0; i < 4; i++) if (i !== row.col && i !== row.stay && S[i] === S[row.col]) equiv.push(i + 1);
    const alt = equiv.length ? ` (or ${equiv.join('/')})` : '';
    if (row.stay >= 0) {
      if (row.stay === row.col) return `Stay column ${row.col + 1}, then place ${label} in it`;
      return `Stay column ${row.stay + 1}, then place ${label} on column ${row.col + 1}${alt}`;
    }
    return `Place ${label} on column ${row.col + 1}${alt}`;
  }

  // [text, tone] of what the move does to its column
  function resultOf(row) {
    if (row.type === 1) return [`Scores ${outcomeName(row.pts)} +${row.pts}` + (row.bonus ? ` · streak +${row.bonus}` : ''), 'success'];
    if (row.type === 2) return [`Busts (${G.busts + 1}/${settings.rules.maxBusts})`, 'error'];
    const S = stateS();
    if (row.stay >= 0) S[row.stay] = 0;
    const nc = ctx.T.nextCode[S[row.col] * 10 + G.current.r];
    const n = E.cardCount(nc);
    return [`Column becomes ${totalText(nc)} · ${n} card${n === 1 ? '' : 's'}`, ''];
  }

  const evText = (x) => (settings.objective === 'score' ? fmt(x) : x.toFixed(3) + 'x');
  const diffText = (x) => (settings.objective === 'score' ? fmt(x) : x.toFixed(3));

  function verdict(st) {
    if (!st || !st.n) return null;
    if (st.rows.length === 1) return ['Only one distinct move', 'muted'];
    const rows = st.rows.slice().sort((a, b) => b.ev - a.ev);
    return rows[1].diff < 2 * rows[1].se ? ['Close call', 'warning'] : ['Clear best', 'success'];
  }

  // Everything that depends on the running analysis.
  function renderRec() {
    renderBadges();
    renderDetail();
    renderMoves();
    renderBoard();
  }

  function renderBadges() {
    const b = [];
    if (isOver()) b.push(['Game over', 'muted', false]);
    else if (cur && cur.running) b.push(['Thinking', 'accent', true]);
    else if (!G.current) b.push(['Waiting for card', 'pending', false]);
    else {
      const v = cur && verdict(cur.stats);
      if (v) b.push([v[0], v[1], false]);
    }
    $('badges').innerHTML = b.map(([t, tone, p]) => `<span class="badge s-${tone}"><span class="dot${p ? ' blink' : ''}"></span>${t}</span>`).join('');
  }

  function meterHtml() {
    const prog = cur ? (cur.running ? Math.min(1, (performance.now() - cur.t0) / cur.budget) : 1) : 0;
    return `<div class="meter"><div style="width:${(prog * 100).toFixed(1)}%"></div></div>`;
  }

  function kv(rows) {
    return `<div class="kv">${rows.map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join('')}</div>`;
  }

  function renderDetail() {
    const p = $('recPanel');
    const R = settings.rules;
    p.className = 'panel advice';
    $('btnApplyMain').disabled = !recommended();
    if (isOver()) {
      const lost = G.busts >= R.maxBusts && R.bustOutLoses;
      const m = lost ? 0 : E.payout(G.score, settings.curve);
      p.innerHTML = `<div class="ph">Game over</div>
        <div class="move">${lost ? 'Lost on busts' : fmt(G.score) + ' points'}</div>
        <div class="res ${m >= 1 ? 's-success' : 's-error'}">${m.toFixed(2)}x your entry</div>
        ${kv([['Ended by', G.busts >= R.maxBusts ? `${R.maxBusts} busts` : 'all 52 cards played']])}
        <div class="row" style="margin-top:12px"><button class="btn gold sm" id="btnNew2">New game</button></div>`;
      $('btnNew2').onclick = newGameClick;
      return;
    }
    const st = cur && cur.stats;
    const sims = st ? `<span class="mono">${fmt(st.n)}</span> <span class="muted">per move</span>` : '<span class="muted">starting…</span>';
    if (!G.current) {
      const row = st && st.n ? st.rows[0] : null;
      p.innerHTML = `<div class="ph">Next card <span class="n">turn ${G.turn + 1}</span></div>
        <div class="move">Tap the card you were dealt</div>
        ${
          row
            ? kv([
                ['Game value now', `<span class="mono">${evText(row.ev)}</span>`],
                ['Avg final score', `<span class="mono">${fmt(row.meanScore)}</span>`],
                [`Reach ${fmt(BE)} (1x)`, `<span class="mono">${pct(row.pBE)}</span>`],
                [`Reach ${fmt(MAXS)} (max)`, `<span class="mono">${pct(row.pMax)}</span>`],
                ['Simulations', sims],
              ])
            : '<span class="muted">Estimating…</span>'
        }
        ${meterHtml()}
        <div class="hint">Values assume the fast policy plays every later move; following the advisor each turn does better, so treat them as floors.</div>`;
      return;
    }
    if (!st || !st.n) {
      p.innerHTML = `<div class="ph">Best move</div><div class="move">Thinking…</div>${meterHtml()}`;
      return;
    }
    const best = st.rows[st.best];
    const sorted = st.rows.slice().sort((a, b) => b.ev - a.ev);
    const second = sorted[1];
    const [rt, rtone] = resultOf(best);
    const v = verdict(st);
    if (best.stay >= 0) p.classList.add('staying');
    p.innerHTML = `<div class="ph">Best move <span class="badge s-${v[1]}"><span class="dot"></span>${v[0]}</span></div>
      <div class="move">${esc(describeMove(best))}</div>
      <div class="res ${rtone ? 's-' + rtone : 'muted'}">${esc(rt)}</div>
      ${kv([
        [settings.objective === 'score' ? 'Avg score' : 'Expected payout', `<span class="mono">${evText(best.ev)}</span>`],
        ['Margin', second ? `<span class="mono">+${diffText(second.diff)}</span> <span class="muted">± ${diffText(second.se)} over next best</span>` : '<span class="muted">no alternative</span>'],
        ['Avg final score', `<span class="mono">${fmt(best.meanScore)}</span>`],
        [`Reach ${fmt(BE)} (1x)`, `<span class="mono">${pct(best.pBE)}</span>`],
        [`Reach ${fmt(MAXS)} (max)`, `<span class="mono">${pct(best.pMax)}</span>`],
        ['Simulations', sims],
      ])}
      ${meterHtml()}
      <div class="row"><button class="btn gold sm" id="btnApply">Play it <kbd>↵</kbd></button>
        <button class="btn dark" id="btnMore" ${cur.running ? 'disabled' : ''}>Think longer <kbd>L</kbd></button></div>
      <div class="hint" style="margin-top:8px">Numbers assume the fast policy plays the rest of the game, so they are floors; the ranking between moves is what matters.</div>`;
    $('btnApply').onclick = applyRec;
    $('btnMore').onclick = extend;
  }

  function renderMoves() {
    const el = $('moves');
    $('movesNote').textContent = settings.objective === 'score' ? 'All moves · by expected score' : 'All moves · by expected payout';
    const st = cur && cur.stats;
    if (!G.current || isOver() || !st || !st.n) {
      el.innerHTML = `<div class="empty"><b>${isOver() ? 'Game over' : G.current ? 'Simulating…' : 'No card yet'}</b>
        ${isOver() ? 'Start a new game to keep going.' : G.current ? 'Moves appear as simulations come in.' : 'Tap the dealt card on the table, or type it.'}</div>`;
      return;
    }
    const rows = st.rows.map((r, j) => Object.assign({ j }, r)).sort((a, b) => b.ev - a.ev);
    el.innerHTML = `<table><thead><tr><th>Move</th><th class="e">${settings.objective === 'score' ? 'Avg' : 'EV'}</th>
      <th class="e">vs best</th><th class="e">Avg score</th><th class="e">≥ ${fmt(BE)}</th><th class="e">≥ ${fmt(MAXS)}</th></tr></thead><tbody>
      ${rows
        .slice(0, 8)
        .map((r) => {
          const [rt, tone] = resultOf(r);
          const isBest = r.j === st.best;
          return `<tr class="${isBest ? 'sel' : ''}"><td>${esc(describeMove(r))}<br><span class="${tone ? 's-' + tone : 'muted'}" style="font-size:12px">${esc(rt)}</span></td>
            <td class="e">${evText(r.ev)}</td><td class="e">${isBest ? '<span class="dim">—</span>' : `−${diffText(r.diff)}`}</td>
            <td class="e">${fmt(r.meanScore)}</td><td class="e">${pct(r.pBE)}</td><td class="e">${pct(r.pMax)}</td></tr>`;
        })
        .join('')}</tbody></table>${rows.length > 8 ? `<div class="hint" style="text-align:center">+${rows.length - 8} weaker moves</div>` : ''}`;
  }

  function recommended() {
    if (!cur || cur.r < 0 || !cur.stats || !cur.stats.n || !G.current) return null;
    return cur.stats.rows[cur.stats.best];
  }
  function applyRec() {
    const r = recommended();
    if (r) place(r.col, r.stay);
  }

  function renderStatus() {
    const R = settings.rules;
    const m = E.payout(G.score, settings.curve);
    let hex = '';
    for (let i = 0; i < R.maxBusts; i++) hex += `<span class="hex${i < G.busts ? ' on' : ''}"></span>`;
    let stays = '';
    for (let i = 0; i < R.stays; i++) stays += `<span class="staytok${i < G.staysLeft ? '' : ' used'}">STAY</span>`;
    const next = R.streakBase + R.streakInc * Math.max(0, G.streak - 1);
    $('status').innerHTML = `<div class="pill" title="Busts">${hex}</div>
      <div class="pill" title="Stays left">${stays || '<small>no stays</small>'}</div>
      <div class="pill" title="Score (payout on this curve)"><span class="chip"></span>${fmt(G.score)}<small>${m.toFixed(2)}x</small></div>`;
    $('meta').innerHTML = G.streak
      ? `<span class="s-accent">On a roll ×${G.streak}</span> · score this turn for +${next} · turn ${G.turn + 1}`
      : `Turn ${G.turn + 1} · 1x at ${fmt(BE)} · max at ${fmt(MAXS)}`;
  }

  const cardHtml = (label, extra) => `<div class="pc${extra || ''}"><span class="c">${label}</span><span class="b">${label}</span></div>`;

  function renderCards() {
    const can = !isOver();
    $('cardBtns').innerHTML =
      `<div class="lbl">${G.current ? 'Wrong card? Tap the right one' : 'Tap the card you were dealt'}</div>` +
      FACES.map((f) => {
        const r = faceRank(f);
        const sel = G.current && G.current.label === f;
        return `<button class="mini${sel ? ' on' : ''}" data-face="${f}" ${(G.deck[r] <= 0 && !sel) || !can ? 'disabled' : ''}>${f}<small>${G.deck[r]}</small></button>`;
      }).join('');
    for (const b of $('cardBtns').querySelectorAll('button')) b.onclick = () => enterCard(b.dataset.face);
    const cc = $('curCard');
    if (G.current) {
      cc.className = 'slot has';
      cc.innerHTML = cardHtml(G.current.label, ' new');
      cc.title = 'Tap to take back';
      cc.onclick = clearCurrent;
    } else {
      cc.className = 'slot';
      cc.innerHTML = isOver() ? 'Game over' : 'Next card';
      cc.onclick = null;
    }
    $('deckTotal').textContent = deckTotal();
  }

  let lastCount = [0, 0, 0, 0];
  function renderBoard() {
    const rec = recommended();
    const done = !!(rec && cur && !cur.running);
    const S = stateS();
    const can = G.current && !isOver();
    renderTableStatus(rec, done);
    $('board').innerHTML = G.cols
      .map((cards, c) => {
        let preview = '';
        let pcls = '';
        if (G.current) {
          const idx = S[c] * 10 + G.current.r;
          const ty = ctx.T.outType[idx];
          if (ty === 1) {
            preview = `${outcomeName(ctx.T.outPts[idx])} +${ctx.T.outPts[idx]}`;
            pcls = ' good';
          } else if (ty === 2) {
            preview = 'BUST';
            pcls = ' bad';
          } else preview = '→ ' + totalText(ctx.T.nextCode[idx]);
        }
        const isRec = rec && rec.col === c;
        const isStay = done && rec.stay === c;
        const code = colCode(cards);
        const soft = cards.length && (code >> 5) & 1 && (code & 31) + 10 <= 21;
        const fresh = cards.length > lastCount[c];
        const stack = cards.map((x, i) => cardHtml(x.label, fresh && i === cards.length - 1 ? ' new' : '')).join('');
        return `<div class="col${isRec ? (done ? ' go' : ' lean') : ''}">
          <div class="tot">${cards.length ? E.bestTotal(code) : 0}${soft ? '<small>SOFT</small>' : ''}</div>
          <div class="lane${can ? ' can' : ''}" data-place="${c}" title="${can ? 'Place here (Shift+' + (c + 1) + ')' : ''}">${stack}
            ${isRec && done ? '<div class="arrow"></div>' : ''}<div class="preview${pcls}">${preview}</div></div>
          <button class="stay${isStay ? ' hot' : ''}" data-stay="${c}" ${G.staysLeft > 0 && cards.length && !isOver() ? '' : 'disabled'}>STAY</button>
        </div>`;
      })
      .join('');
    lastCount = G.cols.map((x) => x.length);
    for (const b of $('board').querySelectorAll('[data-place]')) b.onclick = () => place(+b.dataset.place, -1);
    for (const b of $('board').querySelectorAll('[data-stay]')) b.onclick = () => stay(+b.dataset.stay);
  }

  // Big status pill on the table: thinking progress, then the move in green once simulations stop.
  function renderTableStatus(rec, done) {
    const el = $('tableStatus');
    if (isOver()) {
      el.className = 'tstatus';
      el.innerHTML = '<span>Game over</span>';
    } else if (!G.current) {
      el.className = 'tstatus';
      el.innerHTML = '<span>Waiting for next card</span>';
    } else if (!done) {
      const prog = cur ? Math.min(1, (performance.now() - cur.t0) / cur.budget) : 0;
      el.className = 'tstatus think';
      el.innerHTML = `<div class="fill" style="width:${(prog * 100).toFixed(1)}%"></div><span>Thinking… ${Math.round(prog * 100)}%</span>`;
    } else if (rec.stay >= 0) {
      el.className = 'tstatus stay';
      el.innerHTML = `<span>✓ Stay ${rec.stay + 1}, then place on ${rec.col + 1}</span>`;
    } else {
      el.className = 'tstatus go';
      el.innerHTML = `<span>✓ Place on column ${rec.col + 1}</span>`;
    }
  }

  function renderDeck() {
    $('deckLeft').textContent = `${deckTotal()} of 52 unseen`;
    $('deck').innerHTML = E.RANK_LABELS.map(
      (l, r) => `<div><span class="r">${r === 9 ? '10·K' : l}</span><span class="n${G.deck[r] ? '' : ' dim'}">${G.deck[r]}</span>
        <div class="meter"><div style="width:${(100 * G.deck[r]) / E.FULL_DECK[r]}%"></div></div>
        <span class="adj"><button data-dm="${r}" title="One fewer">−</button><button data-dp="${r}" title="One more">+</button></span></div>`,
    ).join('');
    for (const b of $('deck').querySelectorAll('[data-dm]'))
      b.onclick = () => {
        const r = +b.dataset.dm;
        if (G.deck[r] > 0) {
          snapshot();
          G.deck[r]--;
          commit();
        }
      };
    for (const b of $('deck').querySelectorAll('[data-dp]'))
      b.onclick = () => {
        const r = +b.dataset.dp;
        if (G.deck[r] < E.FULL_DECK[r]) {
          snapshot();
          G.deck[r]++;
          commit();
        }
      };
  }

  function render() {
    renderStatus();
    renderCards();
    renderDeck();
    renderRec();
    $('btnUndo').disabled = !history.length;
  }
  $('btnApplyMain').onclick = () => applyRec();

  // ---------- payout inputs (change every game) ----------
  function renderPayoutInputs() {
    $('pBE').value = settings.curve.be;
    $('p22').value = settings.curve.s22;
    $('p27').value = settings.curve.s27;
    $('pEntry').value = settings.entry || 10;
    $('payErr').textContent = '';
    renderEV();
  }
  function readCurveInputs() {
    const be = Number($('pBE').value);
    const s22 = Number($('p22').value);
    const s27 = Number($('p27').value);
    return be > 0 && s22 > be && s27 > s22 ? Object.assign({}, settings.curve, { be, s22, s27 }) : null;
  }
  // Expected payout of a whole game on the entered curve, from games played start to finish with the advisor.
  function renderEV() {
    const c = readCurveInputs();
    const entry = Number($('pEntry').value) || 0;
    if (!c || typeof ADVISOR_SCORES === 'undefined') {
      $('evLine').innerHTML = '';
      return;
    }
    const xs = ADVISOR_SCORES;
    let ev = 0;
    let be = 0;
    let m22 = 0;
    let lost = 0;
    for (const s of xs) {
      if (s < 0) {
        lost++;
        continue;
      }
      ev += E.payout(s, c);
      if (s >= c.be) be++;
      if (s >= c.s22) m22++;
    }
    ev /= xs.length;
    const money = (x) => '$' + x.toFixed(2);
    $('evLine').innerHTML = `<div class="big${ev < 1 ? ' neg' : ''}"><small>Your EV this game</small><b class="${ev < 1 ? 's-error' : 's-success'}">${ev.toFixed(2)}x</b>
        <small>${entry ? `${money(entry * ev)} back on ${money(entry)} · ${ev >= 1 ? '+' : '−'}${money(Math.abs(entry * (ev - 1)))} per game` : ''}</small></div>
      ${kv([
        [`Reach ${fmt(c.be)} (1x)`, `<span class="mono">${pct(be / xs.length)}</span>`],
        [`Reach ${fmt(c.s22)} (2.2x)`, `<span class="mono">${pct(m22 / xs.length)}</span>`],
        ['Lose on 3 busts', `<span class="mono">${pct(lost / xs.length)}</span>`],
        ['Based on', `<span class="muted">${xs.length} full games played with the advisor</span>`],
      ])}`;
  }
  function applyPayoutInputs() {
    const entry = Number($('pEntry').value);
    if (entry >= 0 && entry !== settings.entry) {
      settings.entry = entry;
      save(LS_SET, settings);
    }
    const c = readCurveInputs();
    if (!c) {
      $('payErr').textContent = 'Needs break even < 2.2x score < 2.7x score';
      return;
    }
    $('payErr').textContent = '';
    renderEV();
    if (c.be === settings.curve.be && c.s22 === settings.curve.s22 && c.s27 === settings.curve.s27) return;
    settings.curve = c;
    save(LS_SET, settings);
    rebuildPayout();
    render();
    analyze();
  }
  for (const id of ['pBE', 'p22', 'p27', 'pEntry']) {
    $(id).addEventListener('input', renderEV);
    $(id).addEventListener('change', applyPayoutInputs);
    $(id).addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        applyPayoutInputs();
        e.target.blur();
      }
    });
  }

  // ---------- dialogs ----------
  function openSettings() {
    const f = $('formSettings');
    f.objective.value = settings.objective;
    for (const k of Object.keys(E.DEFAULT_RULES)) f[k].value = settings.rules[k];
    f.thinkSec.value = settings.thinkSec;
    f.workers.value = settings.workers;
    $('dlgSettings').showModal();
  }
  $('btnDefaults').onclick = () => {
    const d = defaultSettings();
    const f = $('formSettings');
    f.objective.value = d.objective;
    for (const k of Object.keys(E.DEFAULT_RULES)) f[k].value = d.rules[k];
    f.thinkSec.value = d.thinkSec;
    f.workers.value = d.workers;
  };
  $('dlgSettings').addEventListener('close', () => {
    if ($('dlgSettings').returnValue !== 'ok') return;
    const f = $('formSettings');
    settings.objective = f.objective.value;
    for (const k of Object.keys(E.DEFAULT_RULES)) {
      const v = Number(f[k].value);
      if (Number.isFinite(v)) settings.rules[k] = v;
    }
    settings.thinkSec = Math.max(0.5, Number(f.thinkSec.value) || 3);
    settings.workers = Math.max(1, Math.min(32, Number(f.workers.value) | 0 || 1));
    save(LS_SET, settings);
    ctx = E.makeContext(settings.rules, ENGINE_PARAMS);
    rebuildPayout();
    render();
    analyze();
  });

  function cardsToText(cards) {
    return cards.map((c) => c.label).join(' ');
  }
  function parseCards(text) {
    const out = [];
    for (let tok of text.toUpperCase().split(/[\s,]+/).filter(Boolean)) {
      if (tok === '1') tok = 'A';
      if (tok === 'T') tok = '10';
      if (!FACES.includes(tok)) throw new Error(`Unknown card "${tok}"`);
      out.push({ r: faceRank(tok), label: tok });
    }
    return out;
  }
  function openEdit() {
    const f = $('formEdit');
    for (let c = 0; c < 4; c++) f['c' + c].value = cardsToText(G.cols[c]);
    f.score.value = G.score;
    f.busts.value = G.busts;
    f.staysLeft.value = G.staysLeft;
    f.streak.value = G.streak;
    f.deck.value = G.deck.join(' ');
    f.recalc.checked = false;
    $('editErr').textContent = '';
    $('dlgEdit').showModal();
  }
  $('formEdit').addEventListener('submit', (e) => {
    if (e.submitter && e.submitter.value !== 'ok') return;
    const f = $('formEdit');
    try {
      const cols = [];
      for (let c = 0; c < 4; c++) {
        const cards = parseCards(f['c' + c].value);
        if (cards.length > 4 || colCode(cards) < 0) throw new Error(`Column ${c + 1} would already have scored or busted`);
        cols.push(cards);
      }
      let deck;
      if (f.recalc.checked) {
        deck = E.FULL_DECK.slice();
        for (const col of cols) for (const x of col) deck[x.r]--;
        if (G.current) deck[G.current.r]--;
      } else {
        deck = f.deck.value.trim().split(/[\s,]+/).map(Number);
        if (deck.length !== 10 || deck.some((x, i) => !Number.isInteger(x) || x < 0 || x > E.FULL_DECK[i])) throw new Error('Unseen cards must be 10 whole numbers (A..9 max 4, 10-value max 16)');
      }
      if (deck.some((x) => x < 0)) throw new Error('More cards entered than exist in one deck');
      snapshot();
      G.cols = cols;
      G.score = Number(f.score.value) | 0;
      G.busts = Number(f.busts.value) | 0;
      G.staysLeft = Number(f.staysLeft.value) | 0;
      G.streak = Number(f.streak.value) | 0;
      G.deck = deck;
      commit();
    } catch (err) {
      e.preventDefault();
      $('editErr').textContent = err.message;
    }
  });

  function newGameClick() {
    if (G.turn > 0 && !isOver() && !confirm('Start a new game?')) return;
    snapshot();
    G = newGame();
    commit();
  }

  $('btnNew').onclick = newGameClick;
  $('btnUndo').onclick = undo;
  $('btnEdit').onclick = openEdit;
  $('btnSettings').onclick = openSettings;

  document.addEventListener('keydown', (e) => {
    if (document.querySelector('dialog[open]') || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.shiftKey && /^Digit[1-4]$/.test(e.code)) {
      place(+e.code.slice(5) - 1, -1);
      e.preventDefault();
      return;
    }
    const k = e.key.toUpperCase();
    const map = { A: 'A', 1: 'A', 0: '10', T: '10', J: 'J', Q: 'Q', K: 'K' };
    if (map[k]) enterCard(map[k]);
    else if (/^[2-9]$/.test(k)) enterCard(k);
    else if (e.key === 'Enter') applyRec();
    else if (e.key === 'Backspace') undo();
    else if (k === 'L') extend();
    else if (k === 'N') newGameClick();
    else if (k === 'E') openEdit();
    else if (e.key === ',') openSettings();
    else return;
    e.preventDefault();
  });

  renderPayoutInputs();
  render();
  analyze();
})();
