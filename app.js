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
      mode: 'cash', // 'cash': payout curve; 'streak': pass every game's target to win streakGame.payout; 'h2h': 1v1
      streakGame: { targets: [6350, 7751], payout: 3, game: 0 },
      h2h: { win: 0.4, opp: 0 }, // win: earned on top of the entry back; opp: typical opponent score (0 = advisor median)
      boost: 0, // cash: payout booster, % added to the whole payout (score thresholds unchanged)
      rules: Object.assign({}, E.DEFAULT_RULES),
      thinkSec: 3,
      workers: Math.max(1, Math.min(12, (navigator.hardwareConcurrency || 4) - 1)),
      speak: true,
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
  settings.streakGame = Object.assign(defaultSettings().streakGame, settings.streakGame);
  settings.h2h = Object.assign(defaultSettings().h2h, settings.h2h);
  const isStreak = () => settings.mode === 'streak';
  const isH2H = () => settings.mode === 'h2h';
  const target = () => settings.streakGame.targets[Math.min(settings.streakGame.game, settings.streakGame.targets.length - 1)];
  let ctx = E.makeContext(settings.rules, ENGINE_PARAMS);

  // ---------- 1v1 opponent model ----------
  // The opponent's final score is unknown until both games end. Model it as the advisor's own full-game score
  // sample, shifted so its median sits at the typical opponent score; opponent bust-outs score 0. Each sample is
  // blurred by OPP_SD so win chance rises smoothly with score (rollouts need a gradient); equal scores count half.
  const OPP_SD = 250;
  function normCdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const p = 0.3989423 * Math.exp((-x * x) / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }
  const sampleMedian = () => {
    const ok = ADVISOR_SCORES.filter((s) => s >= 0).sort((a, b) => a - b);
    return ok[ok.length >> 1];
  };
  const typicalOpp = (v) => (v > 0 ? v : sampleMedian());
  const oppSample = (typical) => ADVISOR_SCORES.map((s) => (s < 0 ? 0 : s + typical - sampleMedian()));
  function winChance(s, opp) {
    let w = 0;
    for (const o of opp) w += normCdf((s - o) / OPP_SD);
    return w / opp.length;
  }

  // ---------- payout lookup ----------
  let payTab = null;
  let BE = 4700;
  let MAXS = 7900;
  // Cash payout booster: the whole payout times (1 + boost%); the score thresholds stay where they are.
  const boostFactor = (pct) => 1 + (pct > 0 ? pct : 0) / 100;
  const cashPayout = (score) => E.payout(score, settings.curve) * boostFactor(settings.boost);
  // Utility per final score: the cash curve, a shaped curve around this game's target in streak mode, or the
  // chance to beat the opponent in 1v1.
  function rebuildPayout() {
    const c = settings.curve;
    payTab = new Float64Array(30001);
    if (isH2H()) {
      const opp = oppSample(typicalOpp(settings.h2h.opp));
      for (let s = 0; s <= 30000; s++) payTab[s] = s % 10 ? payTab[s - (s % 10)] : winChance(s, opp);
      BE = MAXS = typicalOpp(settings.h2h.opp);
      return;
    }
    if (isStreak()) {
      // Rank moves by a smooth, mostly linear score curve whose 1x point sits 1,250 below the target, not by
      // pass/fail. 240-game tests at a 7,751 target: pass/fail utility beat it 35%, a curve anchored at the
      // target 47%, this shape 51% (rollouts need a score gradient; pushing score is what passes targets).
      const t = target();
      const sc = Object.assign({}, c, { be: t - 1250, s22: t + 1383, s27: t + 4000 });
      for (let s = 0; s <= 30000; s++) payTab[s] = E.payout(s, sc);
      BE = MAXS = t;
      return;
    }
    for (let s = 0; s <= 30000; s++) payTab[s] = E.payout(s, c) * boostFactor(settings.boost);
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
  // Ticks are driven by worker results rather than a timer: timers in a hidden tab are throttled, and the
  // tab is usually hidden behind the mirroring window while the screen reader plays along.
  function onWorkerMsg(e) {
    const m = e.data;
    if (!cur || m.id !== cur.id || !cur.running) return;
    cur.chunks.push(new Int32Array(m.scores));
    cur.n += m.n;
    if (performance.now() - cur.lastTick >= 200) tick();
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
  }
  function stopWorkers() {
    workers.forEach((w) => w.postMessage({ cmd: 'stop' }));
    if (cur) cur.running = false;
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
    cur = { id: ++jobId, S, cnt, r, actions, chunks: [], n: 0, t0: performance.now(), budget, running: false, lastTick: 0, stats: null };
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
    cur.lastTick = performance.now();
    const el = performance.now() - cur.t0;
    let decided = st.n >= 800 && st.rows.length > 1;
    if (decided) for (let j = 0; j < st.rows.length; j++) if (j !== st.best && !(st.rows[j].diff > 4 * st.rows[j].se && st.rows[j].se > 0)) decided = false;
    if (el >= cur.budget || decided) {
      stopWorkers();
      announce();
    }
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

  const pctText = () => isStreak() || isH2H();
  const evText = (x) => (settings.objective === 'score' ? fmt(x) : pctText() ? pct(x) : x.toFixed(3) + 'x');
  const diffText = (x) => (settings.objective === 'score' ? fmt(x) : pctText() ? (100 * x).toFixed(1) + 'pt' : x.toFixed(3));
  const evLabel = () => (settings.objective === 'score' ? 'Avg score' : isStreak() ? 'Chance to beat target' : isH2H() ? 'Chance to win' : 'Expected payout');
  const reachRows = (row) =>
    isStreak()
      ? [[`Beat ${fmt(BE)}`, `<span class="mono">${pct(row.pBE)}</span>`]]
      : isH2H()
        ? [[`Beat ${fmt(BE)} (typical opponent)`, `<span class="mono">${pct(row.pBE)}</span>`]]
        : [
            [`Reach ${fmt(BE)} (1x)`, `<span class="mono">${pct(row.pBE)}</span>`],
            [`Reach ${fmt(MAXS)} (max)`, `<span class="mono">${pct(row.pMax)}</span>`],
          ];

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
      if (isStreak()) {
        const sg = settings.streakGame;
        const passed = !lost && G.score >= target();
        const more = passed && sg.game < sg.targets.length - 1;
        p.innerHTML = `<div class="ph">Game ${sg.game + 1} of ${sg.targets.length} over</div>
          <div class="move">${passed ? (more ? 'Target beaten!' : `Streak won! ${sg.payout}x`) : 'Streak lost'}</div>
          <div class="res ${passed ? 's-success' : 's-error'}">${lost ? 'Lost on busts' : fmt(G.score) + ' points'} vs target ${fmt(target())}</div>
          <div class="row" style="margin-top:12px"><button class="btn gold sm" id="btnNew2">${more ? `Start game ${sg.game + 2} (target ${fmt(sg.targets[sg.game + 1])})` : 'New streak'}</button></div>`;
        $('btnNew2').onclick = () => {
          nextGame();
          Vision.rearm();
        };
        return;
      }
      const ended = kv([['Ended by', G.busts >= R.maxBusts ? `${R.maxBusts} busts` : 'all 52 cards played']]);
      const btn = '<div class="row" style="margin-top:12px"><button class="btn gold sm" id="btnNew2">New game</button></div>';
      if (isH2H()) {
        const w = payTab[Math.min(30000, Math.max(0, G.score))];
        p.innerHTML = `<div class="ph">Game over</div>
          <div class="move">${lost ? 'Lost on busts' : fmt(G.score) + ' points'}</div>
          <div class="res ${w >= 0.5 ? 's-success' : 's-error'}">About ${pct(w)} chance you beat a typical opponent</div>${ended}${matchFormHtml()}${btn}`;
        $('btnNew2').onclick = newGameClick;
        bindMatchForm(lost);
        return;
      }
      const m = lost ? 0 : cashPayout(G.score);
      p.innerHTML = `<div class="ph">Game over</div>
        <div class="move">${lost ? 'Lost on busts' : fmt(G.score) + ' points'}</div>
        <div class="res ${m >= 1 ? 's-success' : 's-error'}">${m.toFixed(2)}x your entry</div>${ended}${btn}`;
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
                [isStreak() ? 'Chance to beat target' : isH2H() ? 'Chance to win' : 'Game value now', `<span class="mono">${isStreak() ? pct(row.pBE) : evText(row.ev)}</span>`],
                ['Avg final score', `<span class="mono">${fmt(row.meanScore)}</span>`],
                ...reachRows(row),
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
      ${best.stay >= 0 ? `<div class="stayflag">STAY FIRST — column ${best.stay + 1}</div>` : ''}
      <div class="move">${esc(describeMove(best))}</div>
      <div class="res ${rtone ? 's-' + rtone : 'muted'}">${esc(rt)}</div>
      ${kv([
        [evLabel(), `<span class="mono">${isStreak() ? pct(best.pBE) : evText(best.ev)}</span>`],
        [
          'Margin',
          !second
            ? '<span class="muted">no alternative</span>'
            : isStreak()
              ? `<span class="mono">${((best.pBE - second.pBE) * 100).toFixed(1)}pt</span> <span class="muted">beat-chance over next best</span>`
              : `<span class="mono">+${diffText(second.diff)}</span> <span class="muted">± ${diffText(second.se)} over next best</span>`,
        ],
        ['Avg final score', `<span class="mono">${fmt(best.meanScore)}</span>`],
        ...reachRows(best),
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
    $('movesNote').textContent =
      settings.objective === 'score'
        ? 'All moves · by expected score'
        : isStreak()
          ? 'All moves · by chance to beat target'
          : isH2H()
            ? 'All moves · by chance to win'
            : 'All moves · by expected payout';
    const st = cur && cur.stats;
    if (!G.current || isOver() || !st || !st.n) {
      el.innerHTML = `<div class="empty"><b>${isOver() ? 'Game over' : G.current ? 'Simulating…' : 'No card yet'}</b>
        ${isOver() ? 'Start a new game to keep going.' : G.current ? 'Moves appear as simulations come in.' : 'Tap the dealt card on the table, or type it.'}</div>`;
      return;
    }
    const rows = st.rows.map((r, j) => Object.assign({ j }, r)).sort((a, b) => b.ev - a.ev);
    const two = !isStreak() && !isH2H();
    el.innerHTML = `<table><thead><tr><th>Move</th><th class="e">${settings.objective === 'score' ? 'Avg' : isStreak() ? 'Beat' : isH2H() ? 'Win' : 'EV'}</th>
      <th class="e">vs best</th><th class="e">Avg score</th>${two ? `<th class="e">≥ ${fmt(BE)}</th><th class="e">≥ ${fmt(MAXS)}</th>` : ''}</tr></thead><tbody>
      ${rows
        .slice(0, 8)
        .map((r) => {
          const [rt, tone] = resultOf(r);
          const isBest = r.j === st.best;
          return `<tr class="${isBest ? 'sel' : ''}"><td>${r.stay >= 0 ? '<b class="staytag">STAY</b> ' : ''}${esc(describeMove(r))}<br><span class="${tone ? 's-' + tone : 'muted'}" style="font-size:12px">${esc(rt)}</span></td>
            <td class="e">${isStreak() ? pct(r.pBE) : evText(r.ev)}</td><td class="e">${isBest ? '<span class="dim">—</span>' : isStreak() ? `${((r.pBE - st.rows[st.best].pBE) * 100).toFixed(1)}pt` : `−${diffText(r.diff)}`}</td>
            <td class="e">${fmt(r.meanScore)}</td>${two ? `<td class="e">${pct(r.pBE)}</td><td class="e">${pct(r.pMax)}</td>` : ''}</tr>`;
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

  // ---------- screen capture: read the dealt card, assume the advised move was played ----------
  let autoCard = null; // G.current entered by the reader; picking a different card corrects it and teaches the reader
  let askCard = false; // the reader saw a new card it doesn't know yet
  let quietUnknown = false; // all cards learned and the box shows something else: no game on screen (a pick still counts)
  let spokenId = 0;
  let readerText = '';

  // The player made the advised move on the phone before the next card showed.
  function advance() {
    lastPre = null;
    if (!G.current || isOver()) return;
    lastPre = JSON.stringify(G);
    if (cur && cur.chunks.length) cur.stats = computeStats();
    let r = recommended();
    if (!r) {
      const a = E.greedy(ctx, stateS(), Int32Array.from(G.deck), deckTotal(), G.current.r);
      r = { col: E.actCol(a), stay: E.actStay(a) };
    }
    place(r.col, r.stay);
  }

  // Starts the game after one that just ended (streak: the next target if this one was beaten).
  function nextGame() {
    if (isStreak()) {
      const sg = settings.streakGame;
      const lost = G.busts >= settings.rules.maxBusts && settings.rules.bustOutLoses;
      sg.game = !lost && G.score >= target() && sg.game < sg.targets.length - 1 ? sg.game + 1 : 0;
      save(LS_SET, settings);
      rebuildPayout();
      renderPayoutInputs();
    }
    snapshot();
    G = newGame();
    commit();
  }

  // In a game = the box shows a known card. Once every card is learned, anything else is a menu, lobby or
  // results screen and is ignored; until then unknown images are asked about so they can be taught.
  function onSeen(face, first) {
    if (first && G.current && G.current.label === face) return; // capture started on a card already entered
    if (!face) {
      if (Vision.learnedAll()) {
        quietUnknown = true;
        return;
      }
      if (askCard) toast('Missed a card', 'Fix the board with Edit state', true);
      askCard = true;
      renderRec();
      say('Which card?');
      return;
    }
    if (askCard) toast('Missed a card', 'Fix the board with Edit state', true);
    askCard = quietUnknown = false;
    if (isOver()) {
      // A card after the game ended: a new game is on screen. In 1v1, the last match must be logged first.
      if (isH2H() && matchLog !== null && !G.logged) {
        toast('Log the last match', 'then press New game', true);
        return;
      }
      nextGame();
      toast('New game', 'detected on screen');
    }
    advance();
    enterCard(face);
    autoCard = G.current;
    checkBoard();
  }

  // Card picked by hand (picker, keyboard, pop-out).
  function pickCard(face) {
    if (askCard || quietUnknown) {
      askCard = quietUnknown = false;
      Vision.learn(face);
      advance();
      enterCard(face);
      checkBoard();
      return;
    }
    if (autoCard && G.current === autoCard && face !== autoCard.label) Vision.learn(face);
    enterCard(face);
  }

  // ---------- board check: after each move, read the column totals and cards-left count off the screen ----------
  // Numbers are learned from the advisor's own board while it agrees with the screen. When the totals disagree:
  // columns in a different order are re-aligned to the screen (an equivalent column was used); otherwise the move
  // actually made is looked for among the legal moves from the previous board and replayed; failing that, it asks
  // for a check. A wrong cards-left count means a card was missed or counted twice.
  let lastPre = null; // game JSON just before the reader's assumed move
  let boardSeq = 0;
  const PERMS = (() => {
    const out = [];
    const build = (p, rest) => (rest.length ? rest.forEach((x, i) => build([...p, x], rest.filter((_, j) => j !== i))) : out.push(p));
    build([], [0, 1, 2, 3]);
    const moved = (p) => p.filter((x, i) => x !== i).length;
    return out.sort((a, b) => moved(a) - moved(b)); // identity first
  })();
  const colTotals = (cols) => cols.map((cards) => (cards.length ? E.bestTotal(colCode(cards)) : 0));
  // Does number n fit reading pattern pat ('?' = digit not learned yet; null = nothing readable)?
  const fits = (n, pat) => pat === null || (String(n).length === pat.length && [...pat].every((ch, j) => ch === '?' || ch === String(n)[j]));
  // Column order p (screen column i shows advisor column p[i]) under which exp fits the readings obs; or null.
  const alignment = (exp, obs) => PERMS.find((p) => p.every((src, i) => fits(exp[src], obs[i]))) || null;
  const isIdentity = (p) => p.every((x, i) => x === i);

  function setBoard(text, tone) {
    const el = $('capBoardState');
    el.textContent = text;
    el.className = 'hint' + (tone ? ' s-' + tone : '');
  }

  async function checkBoard() {
    if (!Vision.boardOn()) return;
    const seq = ++boardSeq;
    const pre = lastPre;
    for (let tries = 0; tries < 4; tries++) {
      const r = await Vision.readBoard(tries ? 500 : 1200); // let the placed card and totals finish animating
      if (seq !== boardSeq || !r) return; // a newer card arrived, or no frames
      if (r.stable) return judgeBoard(r, pre);
    }
    setBoard('Board: still moving, not checked', 'muted');
  }

  function judgeBoard(r, pre) {
    if (isOver() || !G.current) return;
    const obs = r.totals.map((t) => t.pat);
    let p = alignment(colTotals(G.cols), obs);
    let note = '';
    if (!p && pre) {
      // The board differs from the advised move: find what actually happened from the previous board. First the
      // legal moves with the card as read, then (a misread card) the same with every other rank still unseen.
      const P = JSON.parse(pre);
      const S = E.makeState(P.cols.map(colCode), P.busts, P.staysLeft, P.streak, P.score);
      const fitting = (r) =>
        E.enumActions(ctx, S, r)
          .filter((a) => {
            const T = S.slice();
            E.step(ctx, T, a.a, r);
            return alignment(Array.from(T.subarray(0, 4), (code) => (code ? E.bestTotal(code) : 0)), obs);
          })
          .map((a) => Object.assign(a, { r }));
      let found = fitting(P.current.r);
      if (!found.length)
        found = E.RANK_LABELS.flatMap((_, r) => (r !== P.current.r && P.deck[r] > 0 ? fitting(r) : []));
      // Only a unique explanation is used; between several, the one without a STAY (a stay is never assumed).
      if (found.length > 1) found = found.filter((a) => a.stay < 0);
      if (found.length === 1) {
        const label = G.current.label;
        const a = found[0];
        if (a.r !== P.current.r) {
          // The previous card was misread: swap it in the unseen counts.
          P.deck[P.current.r]++;
          P.deck[a.r]--;
          note = `that card was a ${E.RANK_LABELS[a.r]}, not ${P.current.label} · `;
          P.current = { r: a.r, label: E.RANK_LABELS[a.r] };
        }
        G = P;
        place(a.col, a.stay);
        enterCard(label);
        autoCard = G.current;
        p = alignment(colTotals(G.cols), obs);
        note += a.stay >= 0 ? `you stayed ${a.stay + 1} and played ${a.col + 1}` : `played on column ${a.col + 1}`;
      }
    }
    if (!p) {
      setBoard(`Board: screen ${obs.map((v) => v ?? '?').join(' ')} · advisor ${colTotals(G.cols).join(' ')}`, 'error');
      toast('Board mismatch', 'Fix it with Undo or Edit state', true);
      say('Check the board');
      return;
    }
    if (!isIdentity(p)) {
      snapshot();
      G.cols = p.map((src) => G.cols[src]);
      commit();
      if (!note) note = 'columns re-aligned';
    }
    if (note) {
      toast('Board corrected', note);
      say('Board corrected');
    }
    // Board agrees: learn digits not read yet, then check the cards-left count.
    const exp = colTotals(G.cols);
    r.totals.forEach((t, i) => Vision.learnNumber('total', exp[i], t));
    const left = deckTotal();
    if (!fits(left, r.deck.pat)) {
      setBoard(`Board: screen has ${r.deck.pat} cards left, advisor ${left}`, 'error');
      toast('Card count is off', `Screen ${r.deck.pat} left, advisor ${left}: a card was missed or counted twice`, true);
      say('Card count is off');
      return;
    }
    Vision.learnNumber('deck', left, r.deck);
    const unread = [...obs, r.deck.pat].filter((v) => v === null || v.includes('?')).length;
    setBoard(note ? `Board ✓ (${note})` : unread ? `Board ✓ (still learning ${unread} number${unread === 1 ? '' : 's'})` : 'Board ✓', 'success');
  }

  // Speech: one phrase at a time, each finishing before the next starts. Guards against Chrome's
  // speechSynthesis dropping phrases: cancel()+speak() races (no cancel here), flaky network voices (a local
  // voice is preferred), utterances garbage-collected before onend, onend never firing (deadline), and
  // "not-allowed" until the page has had a click (the phrase waits for the next click or key).
  const speech = { queue: [], busy: false, blocked: false, voice: null, keep: null, finish: null, deadline: 0, timer: 0 };
  function pickVoice() {
    const en = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang));
    speech.voice = en.find((v) => v.localService && v.default) || en.find((v) => v.localService) || en[0] || null;
  }
  if (window.speechSynthesis) {
    pickVoice();
    speechSynthesis.addEventListener('voiceschanged', pickVoice);
  }
  function unblockSpeech() {
    if (!speech.blocked) return;
    speech.blocked = false;
    speakNext();
  }
  for (const ev of ['pointerdown', 'keydown']) document.addEventListener(ev, unblockSpeech, true);
  function say(text) {
    if (!settings.speak || !Vision.active() || !window.speechSynthesis) return;
    if (speech.queue[speech.queue.length - 1] === text) return;
    speech.queue.push(text);
    if (speech.busy && performance.now() > speech.deadline) {
      speechSynthesis.cancel(); // stuck: onend never came
      speech.finish();
    } else if (!speech.busy) speakNext();
  }
  function speakNext() {
    if (speech.busy || speech.blocked || !speech.queue.length) return;
    if (navigator.userActivation && !navigator.userActivation.hasBeenActive) {
      speech.blocked = true;
      toast('Voice is off', 'Click the advisor page once to turn it on', true);
      return;
    }
    const text = speech.queue.shift();
    const u = new SpeechSynthesisUtterance(text);
    if (speech.voice) u.voice = speech.voice;
    u.rate = 1.1;
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(speech.timer);
      speech.busy = false;
      speech.keep = null;
      speakNext();
    };
    u.onend = finish;
    u.onerror = (e) => {
      if (e.error === 'not-allowed' && !done) {
        speech.queue.unshift(text);
        speech.blocked = true;
      }
      finish();
    };
    speech.busy = true;
    speech.keep = u;
    speech.finish = finish;
    const ms = 3000 + 150 * text.length; // generous: only for an engine that never reports the end
    speech.deadline = performance.now() + ms;
    speech.timer = setTimeout(() => {
      speechSynthesis.cancel();
      finish();
    }, ms);
    speechSynthesis.resume(); // clears a stuck paused state
    speechSynthesis.speak(u);
  }
  function announce() {
    const r = recommended();
    if (!r || spokenId === cur.id) return;
    spokenId = cur.id;
    say(r.stay < 0 ? `Column ${r.col + 1}` : `Stay! Stay column ${r.stay + 1}, then place in column ${r.col + 1}`);
  }

  // Always-on-top advice window to keep over the mirroring app.
  let pip = null;
  async function openPip() {
    if (pip) return pip.focus();
    if (!window.documentPictureInPicture) return toast('Not supported', 'Pop-out needs Chrome or Edge 116+', true);
    pip = await documentPictureInPicture.requestWindow({ width: 420, height: 250 });
    const d = pip.document;
    for (const s of document.styleSheets) {
      try {
        const st = d.createElement('style');
        st.textContent = Array.from(s.cssRules, (r) => r.cssText).join('\n');
        d.head.append(st);
      } catch {
        const l = d.createElement('link');
        l.rel = 'stylesheet';
        l.href = s.href;
        d.head.append(l);
      }
    }
    d.body.className = 'pipbody';
    d.body.innerHTML = `<div class="pip"><div class="pipcard" id="pipCard"></div><div class="tstatus" id="pipStatus"></div></div>
      <div class="pipread" id="pipRead"></div>
      <div class="picker pippick">${FACES.map((f) => `<button class="mini" data-face="${f}">${f}</button>`).join('')}<button class="mini undo" id="pipUndo">Undo</button></div>`;
    for (const b of d.querySelectorAll('[data-face]')) b.onclick = () => pickCard(b.dataset.face);
    d.getElementById('pipUndo').onclick = undo;
    d.addEventListener('keydown', onKey);
    for (const ev of ['pointerdown', 'keydown']) d.addEventListener(ev, unblockSpeech, true);
    pip.addEventListener('pagehide', () => (pip = null));
    renderPip();
  }
  function renderPip() {
    if (!pip) return;
    const d = pip.document;
    const ts = $('tableStatus');
    const st = d.getElementById('pipStatus');
    st.className = ts.className;
    st.innerHTML = ts.innerHTML;
    d.getElementById('pipCard').innerHTML = G.current ? cardHtml(G.current.label) : '';
    d.getElementById('pipRead').textContent = readerText;
  }

  $('btnPip').onclick = openPip;
  $('capSpeak').checked = settings.speak;
  $('capSpeak').onchange = () => {
    settings.speak = $('capSpeak').checked;
    save(LS_SET, settings);
  };
  Vision.init({
    faces: FACES,
    onCard: onSeen,
    onEmpty: () => {
      askCard = false;
      renderRec();
    },
    onStatus: (t) => {
      readerText = t;
      if (pip) pip.document.getElementById('pipRead').textContent = t;
    },
  });

  function renderStatus() {
    const R = settings.rules;
    const m = cashPayout(G.score);
    let hex = '';
    for (let i = 0; i < R.maxBusts; i++) hex += `<span class="hex${i < G.busts ? ' on' : ''}"></span>`;
    let stays = '';
    for (let i = 0; i < R.stays; i++) stays += `<span class="staytok${i < G.staysLeft ? '' : ' used'}">STAY</span>`;
    const next = R.streakBase + R.streakInc * Math.max(0, G.streak - 1);
    const sg = settings.streakGame;
    const sub = isStreak()
      ? G.score >= target()
        ? '<small class="s-success">✓ target</small>'
        : `<small>need ${fmt(target() - G.score)}</small>`
      : isH2H()
        ? `<small>vs ~${fmt(BE)}</small>`
        : `<small>${m.toFixed(2)}x</small>`;
    $('status').innerHTML = `<div class="pill" title="Busts">${hex}</div>
      <div class="pill" title="Stays left">${stays || '<small>no stays</small>'}</div>
      <div class="pill" title="Score"><span class="chip"></span>${fmt(G.score)}${sub}</div>`;
    const where = isStreak()
      ? `Streak game ${sg.game + 1} of ${sg.targets.length} · target ${fmt(target())}`
      : isH2H()
        ? `1v1 · typical opponent ${fmt(BE)}`
        : `1x at ${fmt(BE)} · max at ${fmt(MAXS)}`;
    $('meta').innerHTML = G.streak
      ? `<span class="s-accent">On a roll ×${G.streak}</span> · score this turn for +${next} · ${where}`
      : `Turn ${G.turn + 1} · ${where}`;
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
    for (const b of $('cardBtns').querySelectorAll('button')) b.onclick = () => pickCard(b.dataset.face);
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
            ${isRec && done ? '<div class="arrow"></div>' : ''}${isStay ? '<div class="stayover">STAY<small>first</small></div>' : ''}<div class="preview${pcls}">${preview}</div></div>
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
    } else if (askCard) {
      el.className = 'tstatus stay';
      el.innerHTML = '<span>New card — tap which one it is</span>';
    } else if (!G.current) {
      el.className = 'tstatus';
      el.innerHTML = '<span>Waiting for next card</span>';
    } else if (!done) {
      const prog = cur ? Math.min(1, (performance.now() - cur.t0) / cur.budget) : 0;
      el.className = 'tstatus think';
      el.innerHTML = `<div class="fill" style="width:${(prog * 100).toFixed(1)}%"></div><span>Thinking… ${Math.round(prog * 100)}%</span>`;
    } else if (rec.stay >= 0) {
      el.className = 'tstatus stay';
      el.innerHTML = `<span><b>STAY column ${rec.stay + 1} first</b> → then place on ${rec.col + 1}</span>`;
    } else {
      el.className = 'tstatus go';
      el.innerHTML = `<span>✓ Place on column ${rec.col + 1}</span>`;
    }
    renderPip();
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
    const sg = settings.streakGame;
    $('pBE').value = settings.curve.be;
    $('p22').value = settings.curve.s22;
    $('p27').value = settings.curve.s27;
    $('pBoost').value = settings.boost || 0;
    $('pEntry').value = settings.entry || 10;
    if (document.activeElement !== $('sTargets')) $('sTargets').value = sg.targets.join(', ');
    $('sPayout').value = sg.payout;
    for (const b of $('modeSeg').querySelectorAll('button')) b.classList.toggle('on', b.dataset.mode === settings.mode);
    $('cashBox').style.display = settings.mode === 'cash' ? '' : 'none';
    $('streakBox').style.display = isStreak() ? '' : 'none';
    $('h2hBox').style.display = isH2H() ? '' : 'none';
    $('hWin').value = settings.h2h.win;
    $('hOpp').value = settings.h2h.opp || '';
    $('hOpp').placeholder = fmt(sampleMedian());
    $('hOppHint').textContent = `Typical opponent = the score a usual opponent finishes on. Blank assumes they play as well as the advisor (about ${fmt(sampleMedian())}); set it to what your opponents actually score once you've seen a few results.`;
    $('sSteps').innerHTML = sg.targets
      .map(
        (t, i) => `<button data-game="${i}" class="${i === sg.game ? 'on' : i < sg.game ? 'done' : ''}" title="Set the game you are on">
          <small>${i < sg.game ? '✓ ' : ''}Game ${i + 1}${i === sg.game ? ' · now' : ''}</small><b>${fmt(t)}</b></button>`,
      )
      .join('');
    for (const b of $('sSteps').querySelectorAll('button'))
      b.onclick = () => {
        sg.game = +b.dataset.game;
        save(LS_SET, settings);
        rebuildPayout();
        renderPayoutInputs();
        render();
        analyze();
      };
    $('payErr').textContent = '';
    renderEV();
  }
  function readCurveInputs() {
    const be = Number($('pBE').value);
    const s22 = Number($('p22').value);
    const s27 = Number($('p27').value);
    return be > 0 && s22 > be && s27 > s22 ? Object.assign({}, settings.curve, { be, s22, s27 }) : null;
  }
  function readStreakInputs() {
    const targets = $('sTargets').value.split(/[\s,]+/).filter(Boolean).map(Number);
    const payout = Number($('sPayout').value);
    return targets.length && targets.every((t) => t > 0) && payout > 0 ? { targets, payout } : null;
  }
  function readH2HInputs() {
    const win = Number($('hWin').value);
    const opp = $('hOpp').value.trim() === '' ? 0 : Number($('hOpp').value);
    return win > 0 && opp >= 0 ? { win, opp } : null;
  }
  // Expected payout from games played start to finish with the advisor (sample of final scores).
  function renderEV() {
    const entry = Number($('pEntry').value) || 0;
    if (typeof ADVISOR_SCORES === 'undefined') return;
    const xs = ADVISOR_SCORES;
    const n = xs.length;
    const passRate = (t) => xs.filter((s) => s >= t).length / n;
    const lostRate = xs.filter((s) => s < 0).length / n;
    let ev;
    let rows;
    let label = 'Your EV this game';
    const money = (x) => '$' + x.toFixed(2);
    if (isStreak()) {
      const s = readStreakInputs();
      if (!s) return void ($('evLine').innerHTML = '');
      const from = Math.min(settings.streakGame.game, s.targets.length - 1);
      let p = 1;
      rows = s.targets.map((t, i) => {
        const r = passRate(t);
        if (i >= from) p *= r;
        return [`Game ${i + 1} · beat ${fmt(t)}`, i < from ? '<span class="s-success">✓ done</span>' : `<span class="mono">${pct(r)}</span>`];
      });
      rows.push(['Win the streak', `<span class="mono">${pct(p)}</span> <span class="muted">× ${s.payout}x</span>`]);
      ev = p * s.payout;
      label = from > 0 ? `Your EV from game ${from + 1}` : 'Your EV this streak';
    } else if (isH2H()) {
      const h = readH2HInputs();
      if (!h || !entry) return void ($('evLine').innerHTML = '');
      const opp = oppSample(typicalOpp(h.opp));
      const pWin = xs.reduce((a, s) => a + winChance(Math.max(0, s), opp), 0) / n;
      const back = entry + h.win;
      ev = (pWin * back) / entry;
      rows = [
        [`Win vs typical opponent (${fmt(typicalOpp(h.opp))})`, `<span class="mono">${pct(pWin)}</span>`],
        ['A win returns', `<span class="mono">${money(back)}</span> <span class="muted">${money(entry)} entry back + ${money(h.win)}</span>`],
        ['Break-even win rate', `<span class="mono">${pct(entry / back)}</span>`],
      ];
    } else {
      const c = readCurveInputs();
      if (!c) return void ($('evLine').innerHTML = '');
      ev = (xs.reduce((a, s) => a + (s < 0 ? 0 : E.payout(s, c)), 0) / n) * boostFactor(Number($('pBoost').value));
      rows = [
        [`Reach ${fmt(c.be)} (1x)`, `<span class="mono">${pct(passRate(c.be))}</span>`],
        [`Reach ${fmt(c.s22)} (2.2x)`, `<span class="mono">${pct(passRate(c.s22))}</span>`],
      ];
    }
    rows.push(['Lose on 3 busts', `<span class="mono">${pct(lostRate)}</span> <span class="muted">per game</span>`]);
    rows.push(['Based on', `<span class="muted">${n} full games played with the advisor</span>`]);
    $('evLine').innerHTML = `<div class="big${ev < 1 ? ' neg' : ''}"><small>${label}</small><b class="${ev < 1 ? 's-error' : 's-success'}">${ev.toFixed(2)}x</b>
        <small>${entry ? `${money(entry * ev)} back on ${money(entry)} · ${ev >= 1 ? '+' : '−'}${money(Math.abs(entry * (ev - 1)))}` : ''}</small></div>
      ${kv(rows)}`;
  }
  function applyPayoutInputs() {
    const entry = Number($('pEntry').value);
    if (entry >= 0 && entry !== settings.entry) {
      settings.entry = entry;
      save(LS_SET, settings);
    }
    $('payErr').textContent = '';
    if (isStreak()) {
      const s = readStreakInputs();
      if (!s) {
        $('payErr').textContent = 'Enter targets like 6350, 7751 and a payout above 0';
        return;
      }
      const sg = settings.streakGame;
      if (s.targets.join() === sg.targets.join() && s.payout === sg.payout) return renderEV();
      sg.targets = s.targets;
      sg.payout = s.payout;
      sg.game = Math.min(sg.game, s.targets.length - 1);
    } else if (isH2H()) {
      const h = readH2HInputs();
      if (!h) {
        $('payErr').textContent = 'Enter what a win earns (above 0); leave typical opponent blank or enter a score';
        return;
      }
      if (h.win === settings.h2h.win && h.opp === settings.h2h.opp) return renderEV();
      settings.h2h = h;
    } else {
      const c = readCurveInputs();
      if (!c) {
        $('payErr').textContent = 'Needs break even < 2.2x score < 2.7x score';
        return;
      }
      const boost = Math.max(0, Number($('pBoost').value) || 0);
      if (c.be === settings.curve.be && c.s22 === settings.curve.s22 && c.s27 === settings.curve.s27 && boost === settings.boost) return renderEV();
      settings.curve = c;
      settings.boost = boost;
    }
    save(LS_SET, settings);
    rebuildPayout();
    renderPayoutInputs();
    render();
    analyze();
  }
  for (const id of ['pBE', 'p22', 'p27', 'pBoost', 'pEntry', 'sTargets', 'sPayout', 'hWin', 'hOpp']) {
    $(id).addEventListener('input', renderEV);
    $(id).addEventListener('change', applyPayoutInputs);
    $(id).addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        applyPayoutInputs();
        e.target.blur();
      }
    });
  }
  for (const b of $('modeSeg').querySelectorAll('button'))
    b.onclick = () => {
      if (settings.mode === b.dataset.mode) return;
      settings.mode = b.dataset.mode;
      save(LS_SET, settings);
      rebuildPayout();
      renderPayoutInputs();
      render();
      analyze();
    };

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
    Vision.rearm();
    commit();
  }

  $('btnNew').onclick = newGameClick;
  $('btnUndo').onclick = undo;
  $('btnEdit').onclick = openEdit;
  $('btnSettings').onclick = openSettings;

  function onKey(e) {
    if (document.querySelector('dialog[open]') || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.shiftKey && /^Digit[1-4]$/.test(e.code)) {
      place(+e.code.slice(5) - 1, -1);
      e.preventDefault();
      return;
    }
    const k = e.key.toUpperCase();
    const map = { A: 'A', 1: 'A', 0: '10', T: '10', J: 'J', Q: 'Q', K: 'K' };
    if (map[k]) pickCard(map[k]);
    else if (/^[2-9]$/.test(k)) pickCard(k);
    else if (e.key === 'Enter') applyRec();
    else if (e.key === 'Backspace') undo();
    else if (k === 'L') extend();
    else if (k === 'N') newGameClick();
    else if (k === 'E') openEdit();
    else if (e.key === ',') openSettings();
    else return;
    e.preventDefault();
  }
  document.addEventListener('keydown', onKey);

  // ---------- 1v1 match log: opponents, their scores and the bet, kept in data/advisor.db by the receiver ----------
  let matchLog = null; // newest first; null when the page isn't served by the receiver
  async function sendMatch(method, url, body) {
    const r = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  }
  function matchFormHtml() {
    if (matchLog === null) return '<div class="hint" style="margin-top:8px">Start 21 Advisor Receiver to log 1v1 matches.</div>';
    if (G.logged) return `<div class="hint s-success" style="margin-top:8px">Logged vs ${esc(G.logged)}</div>`;
    return `<form id="formMatch" class="payrow" style="margin-top:10px">
        <label class="pin wide"><span>Opponent</span><input name="opponent" list="oppNames" required autocomplete="off"></label>
        <label class="pin"><span>Their score</span><input name="opponent_score" type="number" min="0" step="50" placeholder="later"></label>
        <label class="pin"><span>Bet $</span><input name="bet" type="number" min="0" step="0.01" value="${settings.entry || ''}"></label>
        <button class="btn gold sm">Log match</button>
      </form><div class="err" id="matchErr"></div>`;
  }
  function bindMatchForm(lost) {
    const f = $('formMatch');
    if (!f) return;
    f.onsubmit = async (e) => {
      e.preventDefault();
      try {
        const row = await sendMatch('POST', 'receiver/matches', {
          opponent: f.opponent.value,
          opponent_score: f.opponent_score.value === '' ? null : Number(f.opponent_score.value),
          my_score: G.score,
          my_busted_out: lost ? 1 : 0,
          bet: f.bet.value === '' ? null : Number(f.bet.value),
          win: settings.h2h.win,
          typical_opp: Math.round(BE),
        });
        matchLog.unshift(row);
        G.logged = row.opponent;
        persist();
        renderMatches();
        renderRec();
      } catch (err) {
        $('matchErr').textContent = 'Not saved: ' + err.message;
      }
    };
  }
  function renderMatches() {
    $('matchPanel').hidden = false;
    $('matchCount').textContent = `${matchLog.length} logged · data/advisor.db`;
    $('oppNames').innerHTML = [...new Set(matchLog.map((m) => m.opponent))].map((n) => `<option value="${esc(n)}">`).join('');
    if (!matchLog.length) {
      $('matches').innerHTML = '<div class="empty"><b>No matches yet</b>Log one from the game-over panel in 1v1 mode.</div>';
      return;
    }
    const tone = { win: 'success', loss: 'error', tie: 'muted' };
    $('matches').innerHTML = `<table><thead><tr><th>When</th><th>Opponent</th><th class="e">Bet</th><th class="e">You</th><th class="e">Them</th><th>Result</th></tr></thead><tbody>
      ${matchLog
        .slice(0, 15)
        .map(
          (m) => `<tr><td>${esc(m.played_at.slice(5, 16))}</td><td>${esc(m.opponent)}</td><td class="e">${m.bet == null ? '' : '$' + m.bet.toFixed(2)}</td>
            <td class="e">${fmt(m.my_score)}</td>
            <td class="e">${m.opponent_score == null ? `<input class="oppfill" data-id="${m.id}" type="number" min="0" step="50" placeholder="add">` : fmt(m.opponent_score)}</td>
            <td>${m.result ? `<span class="s-${tone[m.result]}">${m.result}</span>` : '<span class="dim">pending</span>'}</td></tr>`,
        )
        .join('')}</tbody></table>`;
    for (const inp of $('matches').querySelectorAll('.oppfill'))
      inp.onchange = async () => {
        if (inp.value === '') return;
        try {
          const row = await sendMatch('PATCH', 'receiver/matches/' + inp.dataset.id, { opponent_score: Number(inp.value) });
          matchLog[matchLog.findIndex((m) => m.id === row.id)] = row;
          renderMatches();
        } catch (err) {
          toast('Not saved', err.message, true);
        }
      };
  }
  if (location.protocol.startsWith('http'))
    fetch('receiver/matches', { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((rows) => {
        if (!Array.isArray(rows)) return;
        matchLog = rows;
        renderMatches();
        renderRec();
      })
      .catch(() => {});

  renderPayoutInputs();
  render();
  analyze();
})();
