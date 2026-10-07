#!/usr/bin/env node
'use strict';
// Interactive GGUF installer for llama.cpp.
//
//   node bin/install.cjs [model-link] [--quant=Q4_K_M] [--ctx=8192]
//                        [--prefer=speed|balanced|quality] [--yes] [--dry-run] [--no-verify] [--connections=16]
const fs = require('fs');
const path = require('path');
const hardware = require('../lib/hardware.cjs');
const quant = require('../lib/quant.cjs');
const sources = require('../lib/sources/index.cjs');
const gguf = require('../lib/gguf.cjs');
const llamacpp = require('../lib/llamacpp.cjs');
const { download, freeDiskGB } = require('../lib/download.cjs');
const { ask, close } = require('../lib/prompt.cjs');
const { fmtGB, table } = require('../lib/util.cjs');

const ROOT = path.join(__dirname, '..');
const MODELS = path.join(ROOT, 'models');

function parseArgs(argv) {
  const a = { url: null, quant: null, ctx: null, prefer: 'balanced', yes: false, dryRun: false, verify: true, connections: undefined };
  for (const x of argv) {
    if (x.startsWith('--quant=')) a.quant = x.slice(8);
    else if (x.startsWith('--ctx=')) a.ctx = Number(x.slice(6)) || null;
    else if (x.startsWith('--prefer=')) a.prefer = x.slice(9);
    else if (x === '--yes' || x === '-y') a.yes = true;
    else if (x === '--dry-run') a.dryRun = true;
    else if (x === '--no-verify') a.verify = false;
    else if (x.startsWith('--connections=')) a.connections = Number(x.slice(14));
    else if (x === '--help' || x === '-h') a.help = true;
    else if (!x.startsWith('-')) a.url = x;
  }
  if (!quant.CAPS[a.prefer]) throw new Error('--prefer must be speed, balanced or quality');
  return a;
}

const MODE = { gpu: '✔ GPU', cpu: '✔ CPU', hybrid: '~ GPU+CPU', no: '✘ too big' };
const ACCEL = { metal: 'Metal (Apple Silicon, unified memory)', cuda: 'CUDA (NVIDIA)', cpu: 'CPU only' };

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(fs.readFileSync(__filename, 'utf8').split('\n').slice(2, 6).join('\n').replace(/\/\/ ?/g, ''));
    return;
  }

  console.log('\n  GGUF installer for llama.cpp\n');

  // 1. Hardware
  const hw = hardware.detect();
  const budget = hardware.budgets(hw);
  console.log('  Hardware');
  console.log(`    OS / arch     ${hw.platform} ${hw.arch}`);
  console.log(`    CPU           ${hw.cpuModel} (${hw.cpuThreads} threads)`);
  console.log(`    RAM           ${fmtGB(hw.totalGB)}`);
  console.log(`    Accelerator   ${ACCEL[hw.accel]}${hw.gpuName && hw.accel === 'cuda' ? ` — ${hw.gpuName}, ${fmtGB(hw.gpuGB)} VRAM` : ''}`);
  console.log(`    Model budget  ${budget.gpu > 0 ? `${fmtGB(budget.gpu)} GPU` : ''}${budget.gpu > 0 && budget.ram > 0 ? ' + ' : ''}${budget.ram > 0 ? `${fmtGB(budget.ram)} RAM` : ''}`);

  // 2. llama.cpp
  const lc = llamacpp.detect();
  console.log(`    llama.cpp     ${lc.server ? `found (build ${lc.server.version})` : 'NOT FOUND'}\n`);
  if (!lc.server) console.log(`  ⚠ Install llama.cpp before running the model:\n    ${llamacpp.installHint(hw)}\n`);

  // 3. Model link
  let input = args.url;
  while (!input) {
    input = await ask('  Paste a Hugging Face or Ollama model link: ');
    if (!input && !process.stdin.isTTY) throw new Error('No model link given. Usage: node bin/install.cjs <link>');
  }
  const src = sources.resolve(input);
  if (!src) throw new Error(`Couldn't understand "${input}". Use a huggingface.co/owner/repo or ollama.com/library/model:tag link.`);

  console.log(`\n  Looking up ${src.kind === 'huggingface' ? 'Hugging Face' : 'Ollama'} model…`);
  const listing = await src.list();
  if (listing.note) console.log(`  ${listing.note}`);

  // 4. Choose
  const plan = quant.choosePlan(listing.candidates, budget, { prefer: args.prefer, ctx: args.ctx, quant: args.quant || listing.pinned });
  console.log(`\n  ${listing.title} — ${plan.rows.length} file(s), estimates at ${plan.ctx.toLocaleString()} context:\n`);
  table(['#', 'Quant', 'Size', 'Needs', 'Runs on', ''],
    plan.rows.map((r, i) => [i + 1, r.label, fmtGB(r.sizeGB), `~${fmtGB(r.needGB)}`, MODE[r.mode], r === plan.pick ? '★ recommended' : '']));
  console.log('');

  let pick = plan.pick;
  if (!pick) {
    const smallest = plan.rows[0];
    console.log(`  ✘ Nothing fits: the smallest file needs ~${fmtGB(smallest.needGB)}, your budget is ${fmtGB(budget.gpu + budget.ram)}.`);
    console.log('    Try a smaller model (fewer parameters).\n');
    if (args.yes) process.exit(1);
  } else {
    if (pick.mode === 'hybrid') console.log('  Note: this will split layers between GPU and CPU — expect it to be slower.');
    if (pick.bpw != null && pick.bpw < quant.MIN_GOOD_BPW) console.log('  Note: below ~3.4 bits/weight, answer quality drops noticeably. Consider a smaller model.');
  }

  if (!args.yes) {
    const q = pick ? `  Download ★ ${pick.label} (${fmtGB(pick.sizeGB)})? [Y/n, or a # to choose another] ` : '  Enter a # to download anyway, or press Enter to quit: ';
    const ans = (await ask(q)).toLowerCase();
    if (/^\d+$/.test(ans)) {
      pick = plan.rows[Number(ans) - 1];
      if (!pick) throw new Error(`No row #${ans}.`);
    } else if (!pick || ans === 'n' || ans === 'no') {
      console.log('  Cancelled.'); return;
    }
  }

  // 5. Vision projector (multimodal models)
  let mmproj = listing.mmproj;
  if (mmproj && !args.yes) {
    console.log(`\n  This model can read images with a vision projector (mmproj, ${fmtGB(mmproj.bytes / 1024 ** 3)}).`);
    console.log('  If you download it, run.cjs will ask each time whether to enable image input.');
    const a = (await ask('  Download the vision projector? [Y/n] ')).toLowerCase();
    if (a === 'n' || a === 'no') mmproj = null;
  }

  const dir = path.join(MODELS, listing.slug);
  const totalGB = pick.sizeGB + (mmproj ? mmproj.bytes / 1024 ** 3 : 0);
  fs.mkdirSync(dir, { recursive: true });
  const free = freeDiskGB(dir);
  if (free != null && free < totalGB + 1) throw new Error(`Not enough disk space: need ${fmtGB(totalGB)}, ${fmtGB(free)} free.`);

  if (args.dryRun) { console.log(`\n  Dry run: would download ${fmtGB(totalGB)} into ${dir}\n`); return; }

  // 6. Download
  console.log(`\n  Downloading to ${dir}`);
  const local = [];
  for (const p of pick.parts) {
    local.push(await download(p.url, path.join(dir, path.basename(p.path)), { bytes: p.bytes, sha256: p.sha256, headers: listing.headers, verify: args.verify, connections: args.connections }));
  }
  let mmprojPath = null;
  if (mmproj) mmprojPath = await download(mmproj.url, path.join(dir, path.basename(mmproj.path)), { bytes: mmproj.bytes, sha256: mmproj.sha256, headers: listing.headers, verify: args.verify, connections: args.connections });

  // 7. Refine settings from the real GGUF header
  let ctx = plan.ctx;
  let ngl = pick.mode === 'cpu' ? 0 : 999;
  let info = null;
  try { info = gguf.modelInfo(gguf.readMetadata(local[0])); } catch (e) { console.log(`  (could not read GGUF header: ${e.message})`); }
  if (info) {
    if (info.ctxTrain) ctx = Math.min(ctx, info.ctxTrain);
    const need = (c) => pick.sizeGB + gguf.kvCacheGB(info, c) + quant.OVERHEAD_GB;
    if (pick.mode === 'gpu') while (ctx > 2048 && need(ctx) > budget.gpu) ctx /= 2;
    if (pick.mode === 'hybrid' || (pick.mode === 'gpu' && need(ctx) > budget.gpu)) {
      const perLayer = pick.sizeGB / info.nLayer;
      ngl = Math.max(0, Math.min(info.nLayer, Math.floor((budget.gpu - quant.OVERHEAD_GB - gguf.kvCacheGB(info, ctx)) / perLayer)));
    }
  }

  const config = {
    source: input, title: listing.title, quant: pick.label, sizeGB: +pick.sizeGB.toFixed(2),
    model: local[0], mmproj: mmprojPath, ctx, ngl, mode: pick.mode,
    arch: info?.arch || null, layers: info?.nLayer || null, created: new Date().toISOString(),
  };
  const cfgPath = path.join(dir, 'model.json');
  fs.writeFileSync(cfgPath, JSON.stringify(config, null, 2));
  fs.writeFileSync(path.join(MODELS, 'latest.json'), JSON.stringify({ config: cfgPath }, null, 2));

  console.log(`\n  ✔ Ready: ${listing.title} ${pick.label}`);
  console.log(`    context ${ctx.toLocaleString()} · GPU layers ${ngl === 999 ? 'all' : ngl}${info ? ` / ${info.nLayer}` : ''}`);
  // Preview the tuned llama.cpp command (the same planner run.cjs uses; re-checked on every run).
  if (lc.server) {
    try {
      const b = require('../lib/runplan.cjs').build({ cfgPath, cfg: config, bin: 'llama-server', vision: false, want: { think: true } });
      const st = b.plan.settings;
      console.log(`    tuned for this computer: context ${Number(st.ctx).toLocaleString()} · KV ${b.plan.kvType}${st.flashAttn ? ' · flash attn' : ''}${st.specType ? ` · speculative ${st.specType}` : ''}${st.cacheRam != null ? ` · prompt cache ${st.cacheRam} MiB` : ''}`);
      console.log('    see every setting and why: node bin/run.cjs --explain');
    } catch (e) { console.log(`    (could not preview run settings: ${e.message})`); }
  }
  console.log('\n  Start it:');
  console.log('    node bin/run.cjs            # OpenAI-compatible server on http://127.0.0.1:8033');
  if (mmprojPath) console.log('    (asks whether to enable image input; --vision / --no-vision skip the question)');
  console.log('    node bin/run.cjs --cli      # chat in the terminal\n');
}

main().catch((e) => { console.error(`\n  ✘ ${e.message}\n`); process.exitCode = 1; }).finally(close);
