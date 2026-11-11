'use strict';
// Run-time planner.
//  1. plan():   model specs (GGUF header) + computer specs  →  abstract settings, each with a reason
//  2. render(): abstract settings  →  flags this llama.cpp build understands (config/llama-flags.json + --help)
// New llama.cpp flags only need a catalog update; the planner keeps working with older builds.
const fs = require('fs');
const path = require('path');
const gguf = require('./gguf.cjs');
const { OVERHEAD_GB } = require('./quant.cjs');

const KV_BYTES = { f32: 4, f16: 2, bf16: 2, q8_0: 1.0625, q5_1: 0.75, q5_0: 0.6875, q4_1: 0.625, q4_0: 0.5625, iq4_nl: 0.5625 };
const VISION_EXTRA_GB = 0.4;   // image encoder compute buffers on top of the mmproj weights
const HYBRID_STATE_GB = 0.25;  // linear-attention / recurrent state
const OS_RESERVE_GB = 4;       // RAM left for macOS/Windows/Linux + apps
const MIN_CTX = 2048;
const F16_KV_MIN_CTX = 32768;  // use the (faster) f16 KV cache whenever it still allows this much context
const BATCH_SIZE = 2048;       // batch = ubatch on GPU: much faster prompt processing
const BATCH_EXTRA_GB = 0.5;    // larger compute buffers for that batch size
const CATALOG_PATH = path.join(__dirname, '..', 'config', 'llama-flags.json');

/** Largest power of two ≤ n (2048, 4096, 8192 … 65536, 131072), never below MIN_CTX. */
function pow2Floor(n) {
  return Math.max(MIN_CTX, 2 ** Math.floor(Math.log2(Math.max(1, n))));
}

function loadCatalog(extra = {}) {
  const cat = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));
  // User-supplied variants are tried first (lets users adapt to a newer llama.cpp before a release).
  for (const [k, variants] of Object.entries(extra || {})) {
    cat.flags[k] = [...variants, ...(cat.flags[k] || [])];
    if (!cat.order.includes(k)) cat.order.splice(cat.order.length - 2, 0, k);
  }
  return cat;
}

/** Can this build express `key` (optionally with this value)? */
function variantFor(cat, help, key, value) {
  const v = value === undefined ? undefined : String(value);
  for (const variant of cat.flags[key] || []) {
    if (!new RegExp(variant.detect).test(help)) continue;
    if (variant.only && v !== undefined && !variant.only.includes(v)) continue;
    const mapped = variant.map?.[v] ?? v;
    if (variant.valueInHelp && mapped !== undefined && !help.includes(mapped)) continue;
    return { variant, mapped };
  }
  return null;
}

function caps(cat, help) {
  return { has: (key, value) => Boolean(variantFor(cat, help, key, value)) };
}

/**
 * @param o.cfg      model.json (sizeGB, ctx, ngl, mode, model, mmproj)
 * @param o.info     gguf.modelInfo() or null
 * @param o.hw       hardware.detect()
 * @param o.budget   hardware.budgets()
 * @param o.vision   use the mmproj
 * @param o.mmprojGB size of the mmproj
 * @param o.can      caps(): which settings this build can express
 * @param o.want     user intent: { think, spec, safe }
 * @param o.override settings forced by config file / CLI (win over the plan)
 */
function plan({ cfg, info, hw, budget, vision = false, mmprojGB = 0, can, want = {}, override = {} }) {
  const s = {}, why = {};
  const set = (k, v, reason) => { s[k] = v; why[k] = reason; };
  const mode = cfg.mode || 'gpu';
  const safe = Boolean(want.safe);
  let tight = false;

  set('model', cfg.model, 'installed GGUF');

  // --- Embedding models: different shape entirely
  if (info?.embedding) {
    const ctx = Math.min(info.ctxTrain || 8192, 8192);
    set('ctx', ctx, 'embedding model: full input length');
    set('batch', ctx, 'embeddings need the whole input in one batch');
    set('ubatch', ctx, 'embeddings need the whole input in one batch');
    set('ngl', mode === 'cpu' ? 0 : 999, mode === 'cpu' ? 'no GPU' : 'all layers on GPU');
    set('embeddings', true, `${info.arch} is an embedding model`);
    return finish();
  }

  // --- Context / KV cache type
  // A quantized KV cache costs generation speed (dequantize on every token; measured ~2x slower
  // on Metal for gpt-oss-20b), so f16 is preferred and q8_0 is only used when f16 would squeeze the context.
  const pool = mode === 'cpu' ? budget.ram : budget.gpu;
  const batchGB = mode === 'gpu' && !safe && !info?.embedding && can.has('batch') && can.has('ubatch') ? BATCH_EXTRA_GB : 0;
  const reserve = (vision ? mmprojGB + VISION_EXTRA_GB : 0) + (info?.hybrid ? HYBRID_STATE_GB : 0) + batchGB;
  const maxCtxFor = (kvType) => {
    if (!info || mode === 'hybrid') return null;
    const freeGB = pool - cfg.sizeGB - OVERHEAD_GB - reserve;
    return Math.floor(freeGB / gguf.kvCacheGB(info, 1, KV_BYTES[kvType] || 2) / 1024) * 1024;
  };
  const quantOk = can.has('flashAttn', 'on') && can.has('kvK');
  let kv, kvReason;
  if (override.kv) { kv = override.kv; kvReason = 'your setting'; }
  else if (vision || safe) { kv = 'f16'; kvReason = vision ? 'vision: full-precision KV cache' : 'safe mode'; }
  else if (!quantOk) { kv = 'f16'; kvReason = 'this build has no flash attention (needed for a quantized KV cache)'; }
  else {
    const f16Max = maxCtxFor('f16');
    const goal = Math.min(info?.ctxTrain || F16_KV_MIN_CTX, F16_KV_MIN_CTX);
    if (f16Max == null || f16Max >= goal) { kv = 'f16'; kvReason = 'f16 KV cache: fastest generation, and it still leaves room for a large context'; }
    else { kv = 'q8_0'; kvReason = 'memory is tight: 8-bit KV cache ≈ 2× context, tiny quality cost (slower generation than f16)'; }
  }

  let ctx = cfg.ctx, ctxReason = 'installer default';
  if (info && mode !== 'hybrid') {
    let max = maxCtxFor(kv);
    tight = max < MIN_CTX;
    const capped = info.ctxTrain && max > info.ctxTrain;
    if (capped) max = info.ctxTrain;
    ctx = pow2Floor(vision ? Math.min(cfg.ctx, max) : max);
    const exact = vision ? Math.min(cfg.ctx, max) : max;
    ctxReason = vision ? 'vision: installer context, limited by memory left after the mmproj'
      : capped ? 'model maximum (memory allows more)' : `largest that fits ${pool.toFixed(1)} GB ${mode === 'cpu' ? 'RAM' : 'GPU'} budget`;
    if (ctx < exact) ctxReason += ` (rounded down to a power of two from ${exact})`;
    if (info.hybrid) ctxReason += `; hybrid attention: KV on ${info.kvLayers}/${info.nLayer} layers`;
  }
  set('ctx', ctx, ctxReason);

  // --- GPU layers
  if (mode === 'cpu') set('ngl', 0, 'no supported GPU');
  else if (mode === 'hybrid' && can.has('nCpuMoe') && info?.moe) {
    // MoE on a small GPU: keep attention + shared layers on GPU, push expert weights to CPU (llama.cpp --fit tunes the rest)
    set('ngl', 999, 'MoE: all layers on GPU except expert weights');
    const perLayer = cfg.sizeGB / (info.nLayer || 1);
    const gpuLayersFit = Math.max(0, Math.floor((budget.gpu - OVERHEAD_GB - gguf.kvCacheGB(info, ctx, KV_BYTES[kv] || 2)) / perLayer));
    set('nCpuMoe', Math.max(0, info.nLayer - gpuLayersFit), 'expert weights of these layers stay in RAM');
  } else if (mode === 'hybrid') set('ngl', cfg.ngl, 'GPU+CPU split from the installer');
  else set('ngl', 999, 'whole model fits on the GPU');

  // --- Attention + KV
  if (can.has('flashAttn', 'on') && !safe) set('flashAttn', 'on', 'faster attention, less memory; required for a quantized KV cache');
  if (kv !== 'f16') { set('kvK', kv, kvReason); set('kvV', kv, kvReason); }
  if (batchGB) { set('batch', BATCH_SIZE, 'GPU: faster prompt processing'); set('ubatch', BATCH_SIZE, 'GPU: faster prompt processing'); }
  set('parallel', 1, 'one chat slot gets the whole context (auto would split it)');

  // --- CPU threads (only matter when layers run on the CPU)
  if (mode !== 'gpu') {
    const t = hw.perfCores || hw.physicalCores || hw.cpuThreads;
    set('threads', t, hw.perfCores ? 'performance cores' : 'physical cores (hyper-threads slow generation down)');
    set('threadsBatch', hw.cpuThreads, 'all logical cores for prompt processing');
  }

  // --- Host prompt cache: default 8 GiB can push a small machine into swap
  const kvGB = info ? gguf.kvCacheGB(info, ctx, KV_BYTES[kv] || 2) : 0;
  const residentGB = hw.accel === 'metal' || mode === 'cpu' ? cfg.sizeGB + kvGB + OVERHEAD_GB + reserve : 0;
  const spareRamGB = hw.totalGB - residentGB - OS_RESERVE_GB;
  const cacheMiB = Math.max(0, Math.min(8192, Math.floor((spareRamGB * 1024) / 512) * 512));
  if (cacheMiB < 8192 && can.has('cacheRam')) {
    set('cacheRam', cacheMiB < 1024 ? 0 : cacheMiB, cacheMiB < 1024 ? 'no spare RAM: prompt cache off to avoid swapping' : `prompt cache limited to spare RAM (${(cacheMiB / 1024).toFixed(1)} GB)`);
  }

  // --- Thinking
  if (want.think === false) {
    if (can.has('reasoning', 'off')) set('reasoning', 'off', '--no-think: skip the hidden thinking phase');
  }

  // --- Speculative decoding (faster generation on a single chat)
  const specWanted = want.spec ?? 'auto';
  if (!safe && specWanted !== 'off' && specWanted !== 'none') {
    if ((specWanted === 'auto' || specWanted === 'mtp') && info?.mtpLayers > 0 && can.has('specType', 'draft-mtp')) {
      set('specType', 'draft-mtp', `model has ${info.mtpLayers} built-in draft (MTP) layer(s)`);
    } else if (specWanted === 'ngram' && can.has('specType', 'ngram')) {
      set('specType', 'ngram', 'n-gram drafting: fast on repetitive text / code edits, no extra model');
    }
  }

  // --- Vision
  if (vision) set('mmproj', cfg.mmproj, 'image input enabled');

  return finish();

  function finish() {
    if (can.has('jinja', true)) set('jinja', true, "use the model's own chat template");
    // Config file / CLI overrides win.
    for (const [k, v] of Object.entries(override)) {
      if (k === 'kv') continue;
      if (v === null || v === 'default') { delete s[k]; why[k] = 'removed by your setting'; continue; }
      set(k, v, 'your setting');
    }
    if (override.kv && override.kv !== 'f16') { set('kvK', override.kv, 'your setting'); set('kvV', override.kv, 'your setting'); }
    if (s.specType === 'none') delete s.specType;
    const kvType = s.kvK || 'f16';
    return {
      settings: s, why, tight,
      kvType, kvGB: info ? gguf.kvCacheGB(info, Number(s.ctx) || 0, KV_BYTES[kvType] || 2) : null,
    };
  }
}

/** Abstract settings → argv for this build. Settings the build can't express are reported, not sent. */
function render(cat, help, settings) {
  const args = [], skipped = [], used = {};
  const keys = [...cat.order, ...Object.keys(settings).filter((k) => !cat.order.includes(k))];
  for (const key of keys) {
    if (!(key in settings)) continue;
    const value = settings[key];
    if (value === false || value === undefined) continue;
    const hit = variantFor(cat, help, key, value);
    if (!hit) { skipped.push({ key, value }); continue; }
    const a = hit.variant.args.map((x) => x.replace('{v}', String(hit.mapped)));
    args.push(...a); used[key] = a;
  }
  return { args, skipped, used };
}

/** Drop flags (and their value) the user asked to remove, then append extras. */
function adjust(args, removeArgs = [], extraArgs = []) {
  const out = [];
  for (let i = 0; i < args.length; i++) {
    if (removeArgs.includes(args[i])) { if (args[i + 1] != null && !String(args[i + 1]).startsWith('-')) i++; continue; }
    out.push(args[i]);
  }
  return [...out, ...extraArgs];
}

/** Flag-level diff between two argv lists, for "changes since last run". */
function diffArgs(prev, next) {
  const toMap = (a) => { const m = new Map(); for (let i = 0; i < a.length; i++) if (String(a[i]).startsWith('-')) { const v = a[i + 1] != null && !String(a[i + 1]).startsWith('-') ? String(a[++i]) : ''; m.set(a[i - (v ? 1 : 0)], v); } return m; };
  const a = toMap(prev), b = toMap(next), out = [];
  for (const [k, v] of b) if (!a.has(k)) out.push(`+ ${k} ${v}`.trim()); else if (a.get(k) !== v) out.push(`~ ${k} ${a.get(k)} → ${v}`);
  for (const [k, v] of a) if (!b.has(k)) out.push(`- ${k} ${v}`.trim());
  return out;
}

module.exports = { pow2Floor, plan, render, adjust, diffArgs, caps, loadCatalog, variantFor, KV_BYTES, CATALOG_PATH };
