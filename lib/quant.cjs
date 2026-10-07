'use strict';
// Quantization parsing + "which file fits this machine" selection.

// Approximate bits-per-weight, used as a quality ranking.
const BPW = {
  IQ1_S: 1.56, IQ1_M: 1.75, IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ2_S: 2.5, IQ2_M: 2.7,
  Q2_K_S: 2.8, Q2_K: 2.96, Q2_K_L: 3.1, Q2_K_XL: 3.1,
  IQ3_XXS: 3.06, IQ3_XS: 3.3, IQ3_S: 3.44, Q3_K_S: 3.5, IQ3_M: 3.66, Q3_K_M: 3.91, Q3_K_L: 4.27, Q3_K_XL: 4.3,
  IQ4_XS: 4.25, MXFP4: 4.25, MXFP4_MOE: 4.25, IQ4_NL: 4.5, Q4_0: 4.55, Q4_K_S: 4.58, Q4_K_M: 4.89, Q4_1: 5.0, Q4_K_L: 5.0, Q4_K_XL: 5.0,
  Q5_0: 5.5, Q5_K_S: 5.54, Q5_K_M: 5.7, Q5_K_L: 5.8, Q5_K_XL: 5.8, Q5_1: 6.0,
  Q6_K: 6.56, Q6_K_L: 6.6, Q6_K_XL: 6.7, Q8_0: 8.5, Q8_K_XL: 8.6, BF16: 16, F16: 16, F32: 32,
};

// Highest quality we will pick automatically, per preference.
const CAPS = { speed: BPW.Q4_K_M, balanced: BPW.Q6_K, quality: BPW.Q8_0 };
const MIN_GOOD_BPW = 3.4; // below this, quality drops noticeably
const OVERHEAD_GB = 0.6;  // llama.cpp compute buffers etc.

const QUANT_RE = /(?<![A-Za-z0-9])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|MXFP4(?:_MOE)?|BF16|FP16|F16|F32))(?![A-Za-z0-9])/gi;

/** "Qwen3-8B-Q4_K_M-00001-of-00002.gguf" -> "Q4_K_M" (last match wins). */
function parseQuant(name) {
  const m = [...String(name || '').matchAll(QUANT_RE)].pop();
  return m ? m[1].toUpperCase().replace(/^FP16$/, 'F16') : null;
}

function bpw(q) {
  return q ? BPW[q.replace(/^UD-/, '')] ?? null : null;
}

/** Pre-download KV-cache guess; refined later from the real GGUF header. */
function heuristicKvGB(sizeGB, ctx) {
  return (ctx / 8192) * Math.min(4, Math.max(0.25, sizeGB * 0.1));
}

/** Annotate each candidate with memory needs and where it can run. */
function evaluate(cands, budget, ctx) {
  return cands.map((c) => {
    const kvGB = heuristicKvGB(c.sizeGB, ctx);
    const needGB = c.sizeGB + kvGB + OVERHEAD_GB;
    let mode = 'no';
    if (budget.gpu > 0 && needGB <= budget.gpu) mode = 'gpu';
    else if (budget.gpu <= 0 && needGB <= budget.ram) mode = 'cpu';
    else if (budget.gpu > 0 && budget.ram > 0 && needGB <= budget.gpu + budget.ram) mode = 'hybrid';
    return { ...c, bpw: bpw(c.quant), kvGB, needGB, mode };
  }).sort((a, b) => a.sizeGB - b.sizeGB);
}

function pickFrom(rows, prefer) {
  const cap = CAPS[prefer] ?? CAPS.balanced;
  const capped = rows.filter((r) => r.bpw == null || r.bpw <= cap);
  const pool = capped.length ? capped : rows;
  const largest = (list) => (list.length ? list[list.length - 1] : null);

  // Best file that runs fully on GPU (or fully on CPU when there is no GPU).
  const full = largest(pool.filter((r) => r.mode === 'gpu' || r.mode === 'cpu'));
  // Best GPU+CPU split, capped at Q4_K_M to keep it usable.
  const hybrid = largest(pool.filter((r) => r.mode === 'hybrid' && (r.bpw == null || r.bpw <= Math.min(cap, CAPS.speed))));

  if (full && (prefer === 'speed' || !hybrid || (full.bpw ?? 99) >= MIN_GOOD_BPW)) return full;
  return hybrid || full || null;
}

/**
 * Choose quant + context size.
 * @param cands  [{ key, quant, sizeGB, parts }]
 * @param budget { gpu, ram } in GB
 * @param opts   { prefer: 'speed'|'balanced'|'quality', ctx, quant }  (quant may also be a file key)
 */
function choosePlan(cands, budget, { prefer = 'balanced', ctx = null, quant = null } = {}) {
  if (!cands.length) throw new Error('No GGUF files found for this model.');
  const ctxList = ctx ? [ctx] : [8192, 4096];
  let first = null;
  for (const c of ctxList) {
    const rows = evaluate(cands, budget, c);
    let pick;
    if (quant) {
      const want = String(quant).toUpperCase();
      pick = rows.find((r) => r.quant === want || r.key === quant) || null;
      if (!pick) throw new Error(`"${quant}" not found. Available: ${rows.map((r) => r.quant || r.key).join(', ')}`);
    } else {
      pick = pickFrom(rows, prefer);
    }
    const plan = { ctx: c, rows, pick };
    first ??= plan;
    if (quant || (pick && pick.mode !== 'hybrid')) return plan;
  }
  return first; // nothing fully fits at any context: keep 8K plan (hybrid or none)
}

module.exports = { BPW, CAPS, OVERHEAD_GB, MIN_GOOD_BPW, parseQuant, bpw, heuristicKvGB, evaluate, choosePlan };
