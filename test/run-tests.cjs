'use strict';
// Offline tests: no network, no downloads.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const quant = require('../lib/quant.cjs');
const hf = require('../lib/sources/huggingface.cjs');
const ollama = require('../lib/sources/ollama.cjs');
const { resolve } = require('../lib/sources/index.cjs');
const gguf = require('../lib/gguf.cjs');
const hardware = require('../lib/hardware.cjs');

let n = 0;
const t = (name, fn) => { fn(); n++; console.log(`  ✔ ${name}`); };
const GiB = 1024 ** 3;

t('parseQuant', () => {
  assert.strictEqual(quant.parseQuant('Qwen3-8B-Q4_K_M.gguf'), 'Q4_K_M');
  assert.strictEqual(quant.parseQuant('model-Q8_0-00001-of-00003.gguf'), 'Q8_0');
  assert.strictEqual(quant.parseQuant('gemma-3-27b-it-UD-Q4_K_XL.gguf'), 'UD-Q4_K_XL');
  assert.strictEqual(quant.parseQuant('Llama-3.2-3B-Instruct-IQ4_XS.gguf'), 'IQ4_XS');
  assert.strictEqual(quant.parseQuant('model.f16.gguf'), 'F16');
  assert.strictEqual(quant.parseQuant('3b-instruct-q4_K_M'), 'Q4_K_M');
  assert.strictEqual(quant.parseQuant('8b-fp16'), 'F16');
  assert.strictEqual(quant.parseQuant('Phi-3-mini-4k.gguf'), null);
  assert.strictEqual(quant.bpw('UD-Q4_K_XL'), 5.0);
});

t('Hugging Face link parsing', () => {
  assert.deepStrictEqual(hf.parse('https://huggingface.co/unsloth/Qwen3-8B-GGUF'),
    { repo: 'unsloth/Qwen3-8B-GGUF', quant: null, revision: 'main', file: null });
  assert.deepStrictEqual(hf.parse('hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M'),
    { repo: 'bartowski/Llama-3.2-3B-Instruct-GGUF', quant: 'Q5_K_M', revision: 'main', file: null });
  assert.strictEqual(hf.parse('https://huggingface.co/a/b/blob/main/sub/m-Q4_K_M.gguf?download=true').file, 'sub/m-Q4_K_M.gguf');
  assert.strictEqual(hf.parse('https://huggingface.co/a/b/tree/main').revision, 'main');
});

t('Ollama link parsing', () => {
  assert.deepStrictEqual(ollama.parse('https://ollama.com/library/llama3.2:3b'), { ns: 'library', model: 'llama3.2', tag: '3b' });
  assert.deepStrictEqual(ollama.parse('ollama.com/someone/cool-model'), { ns: 'someone', model: 'cool-model', tag: 'latest' });
  assert.deepStrictEqual(ollama.parse('qwen3:8b-q4_K_M'), { ns: 'library', model: 'qwen3', tag: '8b-q4_K_M' });
});

t('source routing', () => {
  assert.strictEqual(resolve('https://huggingface.co/unsloth/Qwen3-8B-GGUF').kind, 'huggingface');
  assert.strictEqual(resolve('unsloth/Qwen3-8B-GGUF').kind, 'huggingface');
  assert.strictEqual(resolve('https://ollama.com/library/gemma3:4b').kind, 'ollama');
  assert.strictEqual(resolve('gemma3:4b').kind, 'ollama');
  assert.strictEqual(resolve('not a link!'), null);
});

t('HF grouping: split files, mmproj, imatrix', () => {
  const ref = { repo: 'o/r', revision: 'main' };
  const files = [
    { type: 'file', path: 'README.md', size: 10 },
    { type: 'file', path: 'm-Q4_K_M.gguf', size: 5 * GiB, lfs: { size: 5 * GiB, oid: 'aa' } },
    { type: 'file', path: 'Q8_0/m-Q8_0-00001-of-00002.gguf', size: 5 * GiB, lfs: { size: 5 * GiB, oid: 'b1' } },
    { type: 'file', path: 'Q8_0/m-Q8_0-00002-of-00002.gguf', size: 4 * GiB, lfs: { size: 4 * GiB, oid: 'b2' } },
    { type: 'file', path: 'mmproj-F16.gguf', size: 0.8 * GiB },
    { type: 'file', path: 'm.imatrix.gguf', size: 1 },
  ];
  const { candidates, mmproj } = hf.groupGguf(files, ref);
  assert.strictEqual(candidates.length, 2);
  const q8 = candidates.find((c) => c.quant === 'Q8_0');
  assert.strictEqual(q8.parts.length, 2);
  assert.strictEqual(Math.round(q8.sizeGB), 9);
  assert.ok(q8.parts[0].url.endsWith('/o/r/resolve/main/Q8_0/m-Q8_0-00001-of-00002.gguf'));
  assert.ok(mmproj.path.includes('mmproj'));
});

// A typical 27B and 8B quant ladder (sizes in GB).
const ladder = (sizes) => Object.entries(sizes).map(([q, s]) => ({ key: q, quant: q, label: q, sizeGB: s, parts: [] }));
const m27 = ladder({ Q2_K: 10.5, Q3_K_S: 12.1, Q3_K_M: 13.5, IQ4_XS: 15.3, Q4_K_M: 16.8, Q5_K_M: 19.5, Q6_K: 22.4, Q8_0: 29 });
const m8 = ladder({ Q3_K_M: 4.1, Q4_K_M: 5.0, Q5_K_M: 5.9, Q6_K: 6.7, Q8_0: 8.7, F16: 16.4 });

const mac24 = hardware.budgets({ accel: 'metal', gpuGB: 24 * 0.67, totalGB: 24 });
const rtx24 = hardware.budgets({ accel: 'cuda', gpuGB: 24, totalGB: 64 });
const rtx8 = hardware.budgets({ accel: 'cuda', gpuGB: 8, totalGB: 32 });
const cpu16 = hardware.budgets({ accel: 'cpu', gpuGB: 0, totalGB: 16 });

t('Mac M4 24GB + 27B → Q3_K_M on GPU', () => {
  const p = quant.choosePlan(m27, mac24);
  assert.strictEqual(p.pick.quant, 'Q3_K_M');
  assert.strictEqual(p.pick.mode, 'gpu');
});
t('Mac M4 24GB + 8B → Q6_K (balanced cap)', () => {
  assert.strictEqual(quant.choosePlan(m8, mac24).pick.quant, 'Q6_K');
  assert.strictEqual(quant.choosePlan(m8, mac24, { prefer: 'quality' }).pick.quant, 'Q8_0');
  assert.strictEqual(quant.choosePlan(m8, mac24, { prefer: 'speed' }).pick.quant, 'Q4_K_M');
});
t('RTX 24GB + 27B → Q4_K_M or better on GPU', () => {
  const p = quant.choosePlan(m27, rtx24);
  assert.strictEqual(p.pick.mode, 'gpu');
  assert.ok(p.pick.bpw >= quant.BPW.Q4_K_M);
});
t('RTX 8GB + 27B → hybrid GPU+CPU', () => {
  const p = quant.choosePlan(m27, rtx8);
  assert.strictEqual(p.pick.mode, 'hybrid');
  assert.ok(p.pick.bpw <= quant.BPW.Q4_K_M);
});
t('CPU-only 16GB + 8B → CPU', () => {
  const p = quant.choosePlan(m8, cpu16);
  assert.strictEqual(p.pick.mode, 'cpu');
});
t('nothing fits → pick is null', () => {
  const p = quant.choosePlan(m27, hardware.budgets({ accel: 'metal', gpuGB: 8 * 0.67, totalGB: 8 }));
  assert.strictEqual(p.pick, null);
});
t('pinned quant from link', () => {
  assert.strictEqual(quant.choosePlan(m8, mac24, { quant: 'q8_0' }).pick.quant, 'Q8_0');
  assert.throws(() => quant.choosePlan(m8, mac24, { quant: 'Q2_K' }), /not found/);
});

t('GGUF header reader + KV cache size', () => {
  const parts = [];
  const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); parts.push(b); };
  const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); parts.push(b); };
  const str = (s) => { u64(Buffer.byteLength(s)); parts.push(Buffer.from(s)); };
  const kvU32 = (k, v) => { str(k); u32(4); u32(v); };
  parts.push(Buffer.from('GGUF')); u32(3); u64(0); u64(8);
  str('general.architecture'); u32(8); str('llama');
  str('tokenizer.ggml.tokens'); u32(9); u32(8); u64(3); str('a'); str('bb'); str('ccc'); // skipped array
  str('tokenizer.ggml.scores'); u32(9); u32(6); u64(3); parts.push(Buffer.alloc(12));     // skipped float array
  kvU32('llama.block_count', 32);
  kvU32('llama.attention.head_count', 32);
  kvU32('llama.attention.head_count_kv', 8);
  kvU32('llama.embedding_length', 4096);
  kvU32('llama.context_length', 131072);
  const f = path.join(os.tmpdir(), `t-${process.pid}.gguf`);
  fs.writeFileSync(f, Buffer.concat(parts));
  const info = gguf.modelInfo(gguf.readMetadata(f));
  fs.unlinkSync(f);
  assert.strictEqual(info.nLayer, 32);
  assert.strictEqual(info.keyLen, 128);
  assert.strictEqual(gguf.kvCacheGB(info, 8192), 1); // Llama-3-8B @ 8K = 1 GiB f16
});

// ── Run-time planner + flag catalog ──────────────────────────────────────────
const tuning = require('../lib/tuning.cjs');
const userconfig = require('../lib/userconfig.cjs');
const cat = tuning.loadCatalog();
// Excerpts of `llama-server --help` from a current build and an older one.
const NEW_HELP = `-m, --model FNAME\n-c, --ctx-size N\n-ngl, --n-gpu-layers N\n-ncmoe, --n-cpu-moe N\n-fa, --flash-attn [on|off|auto]
-ctk, --cache-type-k TYPE\n-ctv, --cache-type-v TYPE\n-np, --parallel N\n-t, --threads N\n-tb, --threads-batch N\n-b, --batch-size N\n-ub, --ubatch-size N
-cram, --cache-ram N\n-lm, --load-mode MODE\n-rea, --reasoning [on|off|auto]\n--reasoning-budget N\n--spec-default
--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,ngram-mod\n--spec-draft-n-max N\n-mm, --mmproj FILE\n--embedding, --embeddings\n--jinja, --no-jinja\n--metrics\n--host HOST\n--port PORT`;
const OLD_HELP = `-m, --model FNAME\n-c, --ctx-size N\n-ngl, --n-gpu-layers N\n-fa, --flash-attn\n-ctk, --cache-type-k TYPE\n-ctv, --cache-type-v TYPE
-np, --parallel N\n-t, --threads N\n--mlock\n--no-mmap\n--reasoning-budget N\n--mmproj FILE\n--embedding\n--jinja\n--host HOST\n--port PORT`;
const can = (h) => tuning.caps(cat, h);
const llama8 = { arch: 'llama', nLayer: 32, kvLayers: 32, nHeadKv: 8, keyLen: 128, valLen: 128, ctxTrain: 131072 }; // 128 KiB/token f16
const qwen27 = { arch: 'qwen35', nLayer: 64, kvLayers: 16, hybrid: true, nHeadKv: 4, keyLen: 256, valLen: 256, ctxTrain: 262144, mtpLayers: 1, thinking: true };
const m4 = { accel: 'metal', totalGB: 24, gpuGB: 24 * 0.67, cpuThreads: 12, physicalCores: 12, perfCores: 8, cpuModel: 'Apple M4 Pro' };
const cfg8 = { mode: 'gpu', sizeGB: 6.3, ctx: 8192, ngl: 999, model: '/m/8b.gguf' };
const cfg27 = { mode: 'gpu', sizeGB: 13.56, ctx: 4096, ngl: 999, model: '/m/27b.gguf', mmproj: '/m/mmproj.gguf' };
const planFor = (o) => tuning.plan({ hw: m4, budget: mac24, can: can(NEW_HELP), ...o });
const argsFor = (p, help = NEW_HELP) => tuning.render(cat, help, p.settings).args.join(' ');

t('text-only: q8_0 KV + flash attn + 1 slot, context fills the budget', () => {
  const p = planFor({ cfg: cfg8, info: llama8 });
  assert.strictEqual(p.kvType, 'q8_0');
  assert.ok(p.settings.ctx > 50000 && p.settings.ctx % 1024 === 0, `ctx ${p.settings.ctx}`);
  assert.ok(p.kvGB + cfg8.sizeGB + quant.OVERHEAD_GB <= mac24.gpu, 'fits the budget');
  assert.match(argsFor(p), /-fa on -ctk q8_0 -ctv q8_0 -np 1/);
});
t('vision: f16 KV, installer context, mmproj passed', () => {
  const p = planFor({ cfg: { ...cfg8, mmproj: '/m/mm.gguf' }, info: llama8, vision: true, mmprojGB: 0.9 });
  assert.strictEqual(p.kvType, 'f16');
  assert.strictEqual(p.settings.ctx, 8192);
  assert.match(argsFor(p), /--mmproj \/m\/mm\.gguf/);
});
t('context capped at the model maximum', () => {
  assert.strictEqual(planFor({ cfg: { ...cfg8, sizeGB: 2 }, info: { ...llama8, ctxTrain: 32768 } }).settings.ctx, 32768);
});
t('Qwen3.8 27B on M4 24GB: hybrid ≥3× context, MTP speculative, vision flagged tight', () => {
  const hyb = planFor({ cfg: cfg27, info: qwen27 });
  const dense = planFor({ cfg: cfg27, info: { ...qwen27, kvLayers: 64, hybrid: false } });
  assert.ok(hyb.settings.ctx >= dense.settings.ctx * 3, `hybrid ${hyb.settings.ctx} vs dense ${dense.settings.ctx}`);
  assert.strictEqual(hyb.settings.specType, 'draft-mtp');
  assert.match(argsFor(hyb), /--spec-type draft-mtp/);
  assert.strictEqual(planFor({ cfg: cfg27, info: qwen27, vision: true, mmprojGB: 0.9 }).tight, true);
});
t('host prompt cache limited to spare RAM on a 24 GB Mac', () => {
  const p = planFor({ cfg: cfg27, info: qwen27 });
  assert.ok(p.settings.cacheRam != null && p.settings.cacheRam < 8192, `cacheRam ${p.settings.cacheRam}`);
});
t('--no-think: new build → --reasoning off, old build → --reasoning-budget 0', () => {
  const pNew = planFor({ cfg: cfg8, info: llama8, want: { think: false } });
  assert.match(argsFor(pNew), /--reasoning off/);
  const pOld = tuning.plan({ cfg: cfg8, info: llama8, hw: m4, budget: mac24, can: can(OLD_HELP), want: { think: false } });
  assert.match(argsFor(pOld, OLD_HELP), /--reasoning-budget 0/);
});
t('old build: legacy -fa switch, --mlock spelling, no MTP, unknown flags skipped', () => {
  const p = tuning.plan({ cfg: cfg27, info: qwen27, hw: m4, budget: mac24, can: can(OLD_HELP), override: { mlock: true, ubatch: 1024 } });
  const r = tuning.render(cat, OLD_HELP, p.settings);
  const a = r.args.join(' ');
  assert.match(a, /-fa -ctk q8_0/);
  assert.match(a, /--mlock/);
  assert.ok(!a.includes('--spec-type') && !a.includes('--load-mode'));
  assert.ok(r.skipped.some((x) => x.key === 'ubatch'), 'ubatch reported as unsupported');
  const pn = planFor({ cfg: cfg8, info: llama8, override: { mlock: true } });
  assert.match(argsFor(pn), /--load-mode mmap\+mlock/);
});
t('build without flash attention keeps f16 KV', () => {
  const p = tuning.plan({ cfg: cfg8, info: llama8, hw: m4, budget: mac24, can: can(OLD_HELP.replace('-fa, --flash-attn\n', '')) });
  assert.strictEqual(p.kvType, 'f16');
});
t('CPU-only: threads = physical cores, batch threads = logical', () => {
  const pc = { accel: 'cpu', totalGB: 16, cpuThreads: 16, physicalCores: 8, perfCores: null };
  const p = tuning.plan({ cfg: { ...cfg8, mode: 'cpu' }, info: llama8, hw: pc, budget: cpu16, can: can(NEW_HELP) });
  assert.strictEqual(p.settings.ngl, 0);
  assert.strictEqual(p.settings.threads, 8);
  assert.strictEqual(p.settings.threadsBatch, 16);
});
t('MoE on a small NVIDIA GPU: experts pushed to CPU with --n-cpu-moe', () => {
  const moe = { ...llama8, nLayer: 48, kvLayers: 48, moe: true, experts: 128, expertsUsed: 8 };
  const p = tuning.plan({ cfg: { ...cfg8, mode: 'hybrid', sizeGB: 18, ngl: 20 }, info: moe, hw: { accel: 'cuda', totalGB: 64, cpuThreads: 16, physicalCores: 8 }, budget: { gpu: 7, ram: 38 }, can: can(NEW_HELP) });
  assert.strictEqual(p.settings.ngl, 999);
  assert.ok(p.settings.nCpuMoe > 0 && p.settings.nCpuMoe <= 48);
});
t('embedding model: --embeddings, batch = ctx, no chat flags', () => {
  const p = planFor({ cfg: cfg8, info: { ...llama8, arch: 'bert', embedding: true, ctxTrain: 512 } });
  assert.strictEqual(p.settings.embeddings, true);
  assert.strictEqual(p.settings.ubatch, 512);
  assert.ok(!argsFor(p).includes('-ctk'));
});
t('overrides: config/CLI settings win, null removes, removeArgs + extraArgs', () => {
  const p = planFor({ cfg: cfg27, info: qwen27, override: { kv: 'f16', specType: 'none', ctx: 16384, metrics: true } });
  assert.strictEqual(p.settings.ctx, 16384);
  assert.strictEqual(p.kvType, 'f16');
  assert.ok(!('specType' in p.settings));
  const a = tuning.adjust(tuning.render(cat, NEW_HELP, p.settings).args, ['-np'], ['--temp', '0.7']).join(' ');
  assert.ok(!a.includes('-np') && a.endsWith('--temp 0.7') && a.includes('--metrics'));
});
t('user flag variants (newer llama.cpp) are tried before the catalog', () => {
  const c2 = tuning.loadCatalog({ flashAttn: [{ detect: '--flash-attention-v2', args: ['--flash-attention-v2', '{v}'] }] });
  assert.match(tuning.render(c2, NEW_HELP + '\n--flash-attention-v2 MODE', { flashAttn: 'on' }).args.join(' '), /--flash-attention-v2 on/);
});
t('user config: global + per-model merge', () => {
  const m = userconfig.forModel({ settings: { kv: 'q8_0', _x: 1 }, extraArgs: ['--metrics'], models: { a__b: { settings: { kv: 'f16' }, extraArgs: ['--temp', '0.6'] } } }, 'a__b');
  assert.deepStrictEqual(m.settings, { kv: 'f16' });
  assert.deepStrictEqual(m.extraArgs, ['--metrics', '--temp', '0.6']);
});
t('changes since last run (after an update)', () => {
  const d = tuning.diffArgs(['-c', '8192', '-fa', 'on', '--mlock'], ['-c', '16384', '-fa', 'on', '--spec-type', 'draft-mtp']);
  assert.deepStrictEqual(d.sort(), ['+ --spec-type draft-mtp', '- --mlock', '~ -c 8192 → 16384'].sort());
});
t('GGUF reader: hybrid, MoE, MTP, thinking template', () => {
  const parts = [];
  const u32 = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); parts.push(b); };
  const u64 = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); parts.push(b); };
  const str = (s) => { u64(Buffer.byteLength(s)); parts.push(Buffer.from(s)); };
  const kv = (k, v) => { str(k); u32(4); u32(v); };
  parts.push(Buffer.from('GGUF')); u32(3); u64(0); u64(11);
  str('general.architecture'); u32(8); str('qwen35');
  str('tokenizer.chat_template'); u32(8); str('{% if enable_thinking %}<think>{% endif %}');
  kv('qwen35.block_count', 64); kv('qwen35.attention.head_count', 24); kv('qwen35.attention.head_count_kv', 4);
  kv('qwen35.embedding_length', 5120); kv('qwen35.attention.key_length', 256); kv('qwen35.context_length', 262144);
  kv('qwen35.full_attention_interval', 4); kv('qwen35.expert_count', 1); kv('qwen35.nextn_predict_layers', 1);
  const f = path.join(os.tmpdir(), `h-${process.pid}.gguf`);
  fs.writeFileSync(f, Buffer.concat(parts));
  const info = gguf.modelInfo(gguf.readMetadata(f));
  fs.unlinkSync(f);
  assert.strictEqual(info.kvLayers, 16);
  assert.strictEqual(info.mtpLayers, 1);
  assert.strictEqual(info.thinking, true);
  assert.strictEqual(info.moe, false);
  assert.strictEqual(info.embedding, false);
});

console.log(`\n  ${n} tests passed\n`);
