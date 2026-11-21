// "21" column-blackjack solitaire engine: rules, base policy, Monte Carlo rollouts.
// Wrapped in a factory so the browser can ship its source into a Web Worker via toString().
function EngineFactory() {
  'use strict';

  const NB = 3; // phase buckets (cards left) for the column value table
  const NCODE = 1024; // column state code space
  const RANK_LABELS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10'];
  const FULL_DECK = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16];
  const MAX_STREAK = 64;

  const DEFAULT_RULES = {
    pts21: 400, // column totals exactly 21
    pts777: 600, // three 7s
    pts5: 800, // five cards without busting
    pts5x21: 1000, // five cards totalling exactly 21
    streakBase: 250, // bonus on the 2nd consecutive scoring turn
    streakInc: 0, // extra bonus per further consecutive scoring turn
    maxBusts: 3,
    stays: 2,
  };

  function bucketOf(left) {
    return left <= 5 ? 0 : left <= 15 ? 1 : 2;
  }

  // Column code: hard sum (aces = 1) bits 0-4, has-ace bit 5, card count bits 6-8, all-sevens bit 9.
  function enc(sum, soft, n, all7) {
    return sum | (soft << 5) | (n << 6) | (all7 << 9);
  }
  function dec(code) {
    return { sum: code & 31, soft: (code >> 5) & 1, n: (code >> 6) & 7, all7: (code >> 9) & 1 };
  }
  function bestTotal(code) {
    const s = code & 31;
    return (code >> 5) & 1 && s + 10 <= 21 ? s + 10 : s;
  }
  function cardCount(code) {
    return (code >> 6) & 7;
  }

  function buildTables(rules) {
    const nextCode = new Int16Array(NCODE * 10);
    const outType = new Int8Array(NCODE * 10); // 0 continue, 1 scored (clears), 2 bust (clears)
    const outPts = new Int16Array(NCODE * 10);
    const compMask = new Int16Array(NCODE); // ranks that score this column on the next card
    for (let code = 0; code < NCODE; code++) {
      const { sum, soft, n, all7 } = dec(code);
      for (let r = 0; r < 10; r++) {
        const i = code * 10 + r;
        const ns = sum + r + 1;
        const nsoft = soft | (r === 0 ? 1 : 0);
        const nn = n + 1;
        const n7 = r === 6 && (n === 0 || all7) ? 1 : 0;
        if (ns > 21) {
          outType[i] = 2;
          continue;
        }
        const best = nsoft && ns + 10 <= 21 ? ns + 10 : ns;
        if (best === 21) {
          outType[i] = 1;
          outPts[i] = nn >= 5 ? rules.pts5x21 : nn === 3 && n7 ? rules.pts777 : rules.pts21;
          compMask[code] |= 1 << r;
        } else if (nn >= 5) {
          outType[i] = 1;
          outPts[i] = rules.pts5;
          compMask[code] |= 1 << r;
        } else {
          nextCode[i] = enc(ns, nsoft, nn, n7);
        }
      }
    }
    return { nextCode, outType, outPts, compMask };
  }

  function makeContext(rules, params) {
    rules = Object.assign({}, DEFAULT_RULES, rules || {});
    const bonus = new Float64Array(MAX_STREAK + 2);
    for (let k = 2; k < bonus.length; k++) bonus[k] = rules.streakBase + rules.streakInc * (k - 2);
    const F = new Float64Array(NCODE * NB);
    if (params.F) for (let i = 0; i < F.length && i < params.F.length; i++) F[i] = params.F[i];
    return { rules, P: params, F, T: buildTables(rules), bonus, maxBusts: rules.maxBusts };
  }

  // ---- state helpers (S = Int32Array [c0,c1,c2,c3,busts,staysLeft,streak,score]) ----
  function makeState(cols, busts, staysLeft, streak, score) {
    const S = new Int32Array(8);
    for (let i = 0; i < 4; i++) S[i] = cols[i];
    S[4] = busts;
    S[5] = staysLeft;
    S[6] = streak;
    S[7] = score;
    return S;
  }

  function codeFromCards(ranks) {
    // Returns {code, cleared?} folding ranks into a column; used for manual state entry.
    let code = 0;
    for (const r of ranks) {
      const i = code * 10 + r;
      if (TABLE_DEFAULT.outType[i] !== 0) return -1; // would have cleared: invalid live column
      code = TABLE_DEFAULT.nextCode[i];
    }
    return code;
  }
  const TABLE_DEFAULT = buildTables(DEFAULT_RULES);

  // Action a: stay column s (-1 = none) then place on column c. a = (s + 1) * 4 + c.
  function actStay(a) {
    return (a >> 2) - 1;
  }
  function actCol(a) {
    return a & 3;
  }

  // Applies action with card r. Returns {type, pts, bonus}. Mutates S.
  function step(ctx, S, a, r) {
    const s = (a >> 2) - 1;
    const c = a & 3;
    if (s >= 0) {
      S[s] = 0;
      S[5]--;
    }
    const idx = S[c] * 10 + r;
    const t = ctx.T.outType[idx];
    if (t === 1) {
      S[6] = Math.min(S[6] + 1, MAX_STREAK);
      const b = ctx.bonus[S[6]];
      const p = ctx.T.outPts[idx];
      S[7] += p + b;
      S[c] = 0;
      return { type: 1, pts: p, bonus: b };
    }
    S[6] = 0;
    if (t === 2) {
      S[4]++;
      S[c] = 0;
      return { type: 2, pts: 0, bonus: 0 };
    }
    S[c] = ctx.T.nextCode[idx];
    return { type: 0, pts: 0, bonus: 0 };
  }

  // Fast in-place step for simulations (no allocation).
  function stepFast(ctx, S, a, r) {
    const s = (a >> 2) - 1;
    const c = a & 3;
    if (s >= 0) {
      S[s] = 0;
      S[5]--;
    }
    const idx = S[c] * 10 + r;
    const t = ctx.T.outType[idx];
    if (t === 1) {
      const k = S[6] < MAX_STREAK ? S[6] + 1 : MAX_STREAK;
      S[6] = k;
      S[7] += ctx.T.outPts[idx] + ctx.bonus[k];
      S[c] = 0;
    } else {
      S[6] = 0;
      if (t === 2) {
        S[4]++;
        S[c] = 0;
      } else S[c] = ctx.T.nextCode[idx];
    }
    return t;
  }

  // Greedy base policy: immediate points + change in learned column values + bust/stay costs
  // + bonus for leaving columns that the next card can complete (streak potential).
  // cnt = unseen cards after r was drawn; left = sum(cnt).
  function greedy(ctx, S, cnt, left, r) {
    const T = ctx.T;
    const F = ctx.F;
    const P = ctx.P;
    const b = bucketOf(left);
    const busts = S[4];
    const staysLeft = S[5];
    const lives = ctx.maxBusts - busts - 1;
    const li = lives >= 2 ? 2 : lives < 0 ? 0 : lives;
    const bustPen = P.Ba[li] * left + P.Bb[li];
    const stayPen = P.Sa * left + P.Sb;
    const nextBonus = ctx.bonus[S[6] < MAX_STREAK ? S[6] + 1 : MAX_STREAK];
    const covS = P.wCovS;
    const covN = P.wCovN;
    const outType = T.outType;
    const nextCode = T.nextCode;
    const compMask = T.compMask;
    let best = -1e18;
    let bestA = 0;
    for (let s = -1; s < 4; s++) {
      if (s >= 0) {
        if (staysLeft <= 0) break;
        const cs = S[s];
        if (cs === 0) continue;
        let dup = false;
        for (let k = 0; k < s; k++) if (S[k] === cs) dup = true;
        if (dup) continue;
      }
      for (let c = 0; c < 4; c++) {
        const before = S[c];
        if (c !== s) {
          let dup = false;
          for (let k = 0; k < c; k++) if (k !== s && S[k] === before) dup = true;
          if (dup) continue;
        }
        const old = c === s ? 0 : before;
        const idx = old * 10 + r;
        const t = outType[idx];
        const nc = nextCode[idx];
        let v = F[nc * NB + b] - F[before * NB + b];
        if (t === 1) v += T.outPts[idx] + nextBonus;
        else if (t === 2) v -= bustPen;
        if (s >= 0) {
          v -= stayPen;
          if (s !== c) v += F[b] - F[S[s] * NB + b];
        }
        if (left > 0 && !(t === 2 && lives <= 0)) {
          let mask = compMask[nc];
          for (let i = 0; i < 4; i++) if (i !== c && i !== s) mask |= compMask[S[i]];
          let hit = 0;
          for (let rr = 0; mask; rr++, mask >>= 1) if (mask & 1) hit += cnt[rr];
          v += (hit / left) * (t === 1 ? covS : covN);
        }
        if (v > best) {
          best = v;
          bestA = (s + 1) * 4 + c;
        }
      }
    }
    return bestA;
  }

  // Plays out order[from..to) with the greedy policy. S and cnt are mutated. Returns final score.
  function simulate(ctx, S, cnt, order, from, to) {
    for (let i = from; i < to; i++) {
      const r = order[i];
      cnt[r]--;
      const a = greedy(ctx, S, cnt, to - i - 1, r);
      stepFast(ctx, S, a, r);
      if (S[4] >= ctx.maxBusts) break;
    }
    return S[7];
  }

  // ---- RNG ----
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function deckArray(cnt) {
    let n = 0;
    for (let r = 0; r < 10; r++) n += cnt[r];
    const arr = new Int8Array(n);
    let k = 0;
    for (let r = 0; r < 10; r++) for (let j = 0; j < cnt[r]; j++) arr[k++] = r;
    return arr;
  }

  function shuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = (rng() * (i + 1)) | 0;
      const t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }

  // Distinct legal actions for card r (deduplicated by resulting position; column order is irrelevant).
  function enumActions(ctx, S, r) {
    const out = [];
    const seen = new Set();
    for (let s = -1; s < 4; s++) {
      if (s >= 0 && (S[5] <= 0 || S[s] === 0)) continue;
      for (let c = 0; c < 4; c++) {
        const a = (s + 1) * 4 + c;
        const T = S.slice();
        const res = step(ctx, T, a, r);
        const key = Array.from(T.subarray(0, 4)).sort((x, y) => x - y).join(',') + '|' + Array.from(T.subarray(4)).join(',');
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ a, stay: s, col: c, type: res.type, pts: res.pts, bonus: res.bonus });
      }
    }
    return out;
  }

  // Runs n rollouts. For each rollout the same shuffled future is used for every action
  // (common random numbers). Returns Int32Array scores[rollout * A + action].
  // r = current card (cnt must already exclude it), or -1 to value the position before the draw.
  function rolloutBatch(ctx, S0, cnt0, r, actions, n, rng) {
    const A = r < 0 ? 1 : actions.length;
    const out = new Int32Array(n * A);
    const deck = deckArray(cnt0);
    const len = deck.length;
    const S = new Int32Array(8);
    const cnt = new Int32Array(10);
    for (let k = 0; k < n; k++) {
      shuffle(deck, rng);
      for (let j = 0; j < A; j++) {
        S.set(S0);
        for (let q = 0; q < 10; q++) cnt[q] = cnt0[q];
        if (r >= 0) {
          stepFast(ctx, S, actions[j], r);
          if (S[4] >= ctx.maxBusts) {
            out[k * A + j] = S[7];
            continue;
          }
        }
        out[k * A + j] = simulate(ctx, S, cnt, deck, 0, len);
      }
    }
    return out;
  }

  // ---- payout curve ----
  // The game states three scores: break even (1x), the 2.2x score and the max (2.7x) score.
  // Exponential through (be, 1) and (s22, 2.2); exponential again from (s22, 2.2) to (s27, 2.7);
  // flat at the max beyond. Below break even it decays more gently, at `low` per (s22 - be) points:
  // fitted to an observed game (be 6500, s22 10717: score 6350 paid 0.988x).
  const LOW_RATE = -Math.log(0.988) / (150 / (10717 - 6500));
  const DEFAULT_CURVE = { be: 4700, s22: 7823, s27: 7900, m22: 2.2, m27: 2.7, low: LOW_RATE };
  function payout(score, c) {
    if (score >= c.s27) return c.m27;
    if (score >= c.s22) return c.m22 * Math.pow(c.m27 / c.m22, (score - c.s22) / (c.s27 - c.s22));
    if (score >= c.be) return Math.pow(c.m22, (score - c.be) / (c.s22 - c.be));
    return Math.exp((-(c.low || LOW_RATE) * (c.be - score)) / (c.s22 - c.be));
  }

  return {
    NB,
    NCODE,
    RANK_LABELS,
    FULL_DECK,
    DEFAULT_RULES,
    DEFAULT_CURVE,
    bucketOf,
    enc,
    dec,
    bestTotal,
    cardCount,
    buildTables,
    makeContext,
    makeState,
    codeFromCards,
    actStay,
    actCol,
    step,
    stepFast,
    greedy,
    simulate,
    mulberry32,
    deckArray,
    shuffle,
    enumActions,
    rolloutBatch,
    payout,
  };
}

if (typeof module !== 'undefined' && module.exports) module.exports = EngineFactory();
else globalThis.Engine = EngineFactory();
