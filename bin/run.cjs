#!/usr/bin/env node
'use strict';
// Start llama.cpp with the best settings for this model on this computer.
//
//   node bin/run.cjs [model.json] [--cli] [--port=N] [--vision|--no-vision] [--no-think]
//                    [--spec=auto|mtp|ngram|off] [--safe] [--explain] [--print]
//                    [--ctx=N --ngl=N --kv=f16|q8_0|q4_0 --threads=N --ubatch=N --cache-ram=MiB --mlock --set key=value]
//                    [--init-config] [-- extra llama.cpp args]
//
// Every run re-reads the model header, the hardware and `llama-server --help`, so the command
// follows new llama.cpp releases (flag spellings live in config/llama-flags.json).
const fs = require('fs');
const path = require('path');
const net = require('net');
const { spawn } = require('child_process');
const runplan = require('../lib/runplan.cjs');
const userconfig = require('../lib/userconfig.cjs');
const { ask, close } = require('../lib/prompt.cjs');

const MODELS = path.join(__dirname, '..', 'models');
const DEFAULT_PORT = 8033;
const argv = process.argv.slice(2);
const dd = argv.indexOf('--');
const extra = dd >= 0 ? argv.slice(dd + 1) : [];
const own = dd >= 0 ? argv.slice(0, dd) : argv;
const opt = (k) => own.find((a) => a.startsWith(`--${k}=`))?.split('=').slice(1).join('=');
const has = (f) => own.includes(f);

async function main() {
  if (has('--init-config')) {
    const r = userconfig.init();
    console.log(`\n  ${r.created ? 'Created' : 'Already exists'}: ${r.path}\n  Edit it to override any setting for all models or per model (see README "Tuning").\n`);
    return;
  }
  let cfgPath = own.find((a) => a.endsWith('.json'));
  if (!cfgPath) {
    const latest = path.join(MODELS, 'latest.json');
    if (!fs.existsSync(latest)) throw new Error('No model installed yet. Run: node bin/install.cjs');
    cfgPath = JSON.parse(fs.readFileSync(latest, 'utf8')).config;
  }
  const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
  const cli = has('--cli');

  // 1. Vision or text-only?
  const hasMmproj = Boolean(cfg.mmproj && fs.existsSync(cfg.mmproj));
  let vision = false;
  if (hasMmproj) {
    if (has('--vision')) vision = true;
    else if (has('--no-vision')) vision = false;
    else if (!process.stdin.isTTY) vision = true;
    else {
      console.log('\n  This model has a vision projector (mmproj) downloaded.');
      console.log('    y = read images too (uses extra memory, shorter context)');
      console.log('    n = text only: longest context and fastest answers');
      const a = (await ask('  Enable image input? [y/N] ')).toLowerCase();
      vision = a === 'y' || a === 'yes';
    }
  }
  close();

  // 2. Port (server only)
  const bin = cli ? (vision ? 'llama-mtmd-cli' : 'llama-cli') : 'llama-server';
  let port = null;
  if (!cli) {
    const asked = opt('port');
    if (asked) {
      if (!(await portFree(asked))) {
        throw new Error(`Port ${asked} is already in use by ${await whoIsOn(asked)}.\n` +
          `    Stop it (find it with: lsof -i :${asked}) or use a free port: node bin/run.cjs --port=${await findFreePort(Number(asked) + 1)}`);
      }
      port = Number(asked);
    } else {
      port = await findFreePort(DEFAULT_PORT);
      if (port !== DEFAULT_PORT) console.log(`\n  Port ${DEFAULT_PORT} is busy (${await whoIsOn(DEFAULT_PORT)}), using free port ${port}.`);
    }
  }

  const want = { think: !has('--no-think'), spec: has('--no-spec') ? 'off' : opt('spec') || 'auto', safe: has('--safe') };
  const plan = (w) => runplan.build({ cfgPath, cfg, bin, vision, want: w, cli: runplan.cliSettings(own), port, extra });
  let b = plan(want);
  if (!b.ll) fail(bin, { code: 'ENOENT' });
  report(b, cfg, bin, vision);

  const cmp = runplan.compareWithLast(cfgPath, b);
  if (cmp.changes.length) {
    const from = cmp.prev ? `gguf-autopilot ${cmp.prev.autopilot} / llama.cpp ${cmp.prev.llamacpp}` : '';
    console.log(`\n  Changes since last run (${from} → ${b.autopilot} / ${b.ll.version}):`);
    cmp.changes.forEach((c) => console.log(`    ${c}`));
  }
  printCommand(bin, b.args);
  if (has('--print') || has('--explain')) return;

  if (cli) {
    cmp.save();
    const child = spawn(bin, b.args, { stdio: 'inherit' });
    child.on('error', (e) => fail(bin, e));
    child.on('exit', (code) => process.exit(code ?? 0));
    return;
  }

  // 3. Start. If it dies before it is ready, back off step by step:
  //    1) drop speculative decoding (newest, most build-dependent)  2) safe mode (f16 KV, default attention).
  let r = await serve(bin, b.args, cfgPath, port, b.info);
  const ladder = [];
  if (b.plan.settings.specType) ladder.push({ label: 'without speculative decoding', w: { ...want, spec: 'off' } });
  if (!want.safe && (b.plan.settings.flashAttn || (b.plan.settings.kvK && b.plan.settings.kvK !== 'f16'))) ladder.push({ label: 'in safe mode (f16 KV cache, default attention)', w: { ...want, spec: 'off', safe: true } });
  for (const step of ladder) {
    if (r.ready || r.stoppedByUser) break;
    console.log(`\n  ↻ Retrying ${step.label}…`);
    b = plan(step.w);
    printCommand(bin, b.args);
    r = await serve(bin, b.args, cfgPath, port, b.info);
    if (r.ready || r.stoppedByUser) {
      const flag = step.w.safe ? '--safe' : '--spec=off';
      console.log(`  Tip: this worked ${step.label}. Use ${flag} next time, or add it to your config (node bin/run.cjs --init-config).`);
    }
  }
  if (r.ready) runplan.compareWithLast(cfgPath, b).save(); // remember the command that worked
  if (!r.ready && !r.stoppedByUser) process.exit(1);
}

function report(b, cfg, bin, vision) {
  const s = b.plan.settings, i = b.info, why = b.plan.why;
  console.log(`\n  ${cfg.title} ${cfg.quant} · ${bin} (llama.cpp ${b.ll.version}) · gguf-autopilot ${b.autopilot}`);
  if (i) {
    const bits = [`${i.arch}`, i.sizeLabel, `${i.nLayer} layers`, i.hybrid ? `hybrid attention (KV on ${i.kvLayers})` : null,
      i.moe ? `MoE ${i.expertsUsed}/${i.experts} experts` : null, i.mtpLayers ? `${i.mtpLayers} MTP draft layer(s)` : null,
      i.slidingWindow ? `sliding window ${i.slidingWindow}` : null, i.thinking ? 'thinking' : null, i.embedding ? 'embedding' : null,
      i.ctxTrain ? `max ctx ${i.ctxTrain.toLocaleString()}` : null].filter(Boolean);
    console.log(`    model       ${bits.join(' · ')}`);
  }
  const h = b.hw;
  console.log(`    computer    ${h.cpuModel} · ${h.perfCores ? `${h.perfCores}P/` : ''}${h.physicalCores} cores · ${h.totalGB.toFixed(0)} GB RAM · ${h.accel === 'metal' ? 'Metal' : h.accel === 'cuda' ? `CUDA ${h.gpuName}` : 'CPU'} · budget ${(b.budget.gpu || b.budget.ram).toFixed(1)} GB`);
  console.log(`    mode        ${i?.embedding ? 'embeddings' : vision ? 'vision + text' : 'text only (max context)'}`);
  console.log(`    context     ${Number(s.ctx).toLocaleString()} tokens · KV ${b.plan.kvType}${b.plan.kvGB != null ? ` ~${b.plan.kvGB.toFixed(1)} GB` : ''}`);
  console.log(`    speed       ${[s.flashAttn ? 'flash attn' : null, s.specType ? `speculative (${s.specType})` : null, s.reasoning === 'off' ? 'thinking off' : null, s.threads ? `${s.threads} threads` : null].filter(Boolean).join(' · ') || 'defaults'}`);
  if (b.user.extraArgs.length || b.user.removeArgs.length || Object.keys(b.user.settings).length) console.log(`    your config ${userconfig.configPath()}`);
  if (b.rendered.skipped.length) console.log(`    not in this llama.cpp build: ${b.rendered.skipped.map((x) => `${x.key}=${x.value}`).join(', ')}`);
  if (b.plan.tight) console.log(`\n  ⚠ Not enough memory for ${vision ? 'vision with this quant' : 'this quant'}: it may swap or fail to load.${vision ? ' Try --no-vision or a smaller quant.' : ' Try a smaller quant.'}`);
  if (has('--explain')) {
    console.log('\n  Setting          Value               Why                                                   Flag');
    console.log(`  ${'─'.repeat(15)}  ${'─'.repeat(18)}  ${'─'.repeat(52)}  ${'─'.repeat(20)}`);
    for (const [k, v] of Object.entries(s)) {
      if (k === 'model' || k === 'mmproj') continue;
      const flag = b.rendered.used[k] ? b.rendered.used[k].join(' ') : '(not supported: skipped)';
      console.log(`  ${k.padEnd(15)}  ${String(v).slice(0, 18).padEnd(18)}  ${String(why[k] || '').slice(0, 52).padEnd(52)}  ${flag}`);
    }
  }
}

function printCommand(bin, args) {
  console.log(`\n  Command:\n    ${bin} ${args.map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(' ')}\n`);
}

/** Run llama-server, stream logs, wait for /health, then send a test request. Resolves when ready or when it exits. */
function serve(bin, args, cfgPath, port, info) {
  return new Promise((resolve) => {
    const logPath = path.resolve(path.dirname(cfgPath), 'server.log');
    const log = fs.createWriteStream(logPath, { flags: 'w' });
    const tail = [];
    const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const pass = (out) => (d) => {
      out.write(d); log.write(d);
      tail.push(...String(d).split('\n').filter(Boolean)); tail.splice(0, Math.max(0, tail.length - 30));
    };
    child.stdout.on('data', pass(process.stdout));
    child.stderr.on('data', pass(process.stderr));
    child.on('error', (e) => fail(bin, e));
    let ready = false, userStop = false, exited = false;
    const onInt = () => { userStop = true; child.kill('SIGINT'); };
    process.on('SIGINT', onInt);

    child.on('exit', (code, signal) => {
      exited = true; log.end(); process.removeListener('SIGINT', onInt);
      if (userStop || signal === 'SIGINT' || code === 130) return resolve({ ready, stoppedByUser: true });
      if (code === 0) return resolve({ ready, stoppedByUser: true });
      console.log(`\n  ✘ ${bin} stopped (${signal || `exit code ${code}`})${ready ? '' : ' before the model was ready'}.`);
      console.log(`    Last log lines:\n${tail.slice(-15).map((l) => `      ${l}`).join('\n')}`);
      console.log(`    Full log: ${logPath}\n`);
      resolve({ ready: false, stoppedByUser: false, code });
    });

    (async () => {
      const base = `http://127.0.0.1:${port}`;
      const t0 = Date.now();
      while (!ready && !exited) {
        await new Promise((r) => setTimeout(r, 1000));
        try {
          const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(2000) });
          const ct = res.headers.get('content-type') || '';
          if (exited) break;
          if (res.ok && ct.includes('json')) ready = true;
          else if (!ct.includes('json') && res.status !== 503) {
            console.log(`\n  ✘ Something else is answering on ${base} (got ${ct || 'no content-type'}, HTTP ${res.status}).`);
            child.kill();
          }
        } catch {}
      }
      if (!ready) return;
      const secs = Math.round((Date.now() - t0) / 1000);
      const test = info?.embedding
        ? { url: '/v1/embeddings', body: { input: 'hello' } }
        : { url: '/v1/chat/completions', body: { messages: [{ role: 'user', content: 'Reply with OK.' }], max_tokens: 8 } };
      try {
        const res = await fetch(`${base}${test.url}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(test.body), signal: AbortSignal.timeout(120000),
        });
        const text = await res.text();
        if (!res.ok || !text.trim().startsWith('{')) {
          console.log(`\n  ⚠ Model loaded in ${secs}s, but a test request failed: HTTP ${res.status} ${text.replace(/\s+/g, ' ').slice(0, 200)}`);
          console.log(`    Full log: ${logPath}\n`);
        } else {
          console.log(`\n  ✔ Ready in ${secs}s — test ${info?.embedding ? 'embedding' : 'chat'} request OK.`);
          console.log(`    Web UI + OpenAI API: ${base}   (log: ${logPath}, Ctrl+C to stop)\n`);
        }
      } catch (e) {
        console.log(`\n  ⚠ Model loaded in ${secs}s, but the test request failed: ${e.message}\n`);
      }
    })();
  });
}

function fail(bin, e) {
  console.error(e.code === 'ENOENT' ? `\n  ${bin} not found on PATH. Install llama.cpp first (see README).` : e.message);
  process.exit(1);
}

/** Is anything accepting connections on this port (IPv4 or IPv6)? Catches Docker / wildcard listeners. */
function answers(host, port) {
  return new Promise((resolve) => {
    const sock = net.connect({ host, port: Number(port) });
    const done = (v) => { sock.destroy(); resolve(v); };
    sock.setTimeout(400, () => done(false));
    sock.once('connect', () => done(true));
    sock.once('error', () => done(false));
  });
}

function canBind(host, port) {
  return new Promise((resolve) => {
    const s = net.createServer().once('error', () => resolve(false)).once('listening', () => s.close(() => resolve(true)));
    s.listen({ host, port: Number(port), exclusive: true });
  });
}

/** Free = nobody answers on 127.0.0.1 or ::1, and we can bind it ourselves. */
async function portFree(port) {
  const [v4, v6] = await Promise.all([answers('127.0.0.1', port), answers('::1', port)]);
  return !v4 && !v6 && (await canBind('127.0.0.1', port));
}

async function findFreePort(from) {
  for (let p = from; p < from + 200 && p <= 65535; p++) if (await portFree(p)) return p;
  throw new Error(`No free port found between ${from} and ${from + 199}.`);
}

async function whoIsOn(port) {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/props`, { signal: AbortSignal.timeout(2000) });
    const ct = res.headers.get('content-type') || '';
    if (ct.includes('json')) {
      const j = await res.json();
      return `another llama.cpp server (model: ${path.basename(j.model_path || j.model_alias || 'unknown')})`;
    }
    return `another web server (HTTP ${res.status}, ${ct || 'no content-type'})`;
  } catch { return 'another process'; }
}

main().catch((e) => { close(); console.error(`\n  ✘ ${e.message}\n`); process.exitCode = 1; });
