#!/usr/bin/env node
'use strict';
// Download speed diagnostic. Measures each layer separately so we can see where the time goes.
//
//   node bin/diagnose-download.cjs <model-link with :QUANT> [--mb=512]
//   e.g. node bin/diagnose-download.cjs orcarouter/Qwen3.8-27B-Uncensored-GGUF:Q3_K_L
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const sources = require('../lib/sources/index.cjs');

const args = process.argv.slice(2);
const link = args.find((a) => !a.startsWith('-'));
const MB = 1024 * 1024;
const WINDOW = (Number((args.find((a) => a.startsWith('--mb=')) || '').slice(5)) || 512) * MB;
if (!link) { console.log('Usage: node bin/diagnose-download.cjs <owner/repo:QUANT> [--mb=512]'); process.exit(1); }

const results = [];
const rate = (bytes, ms) => (bytes / MB / (ms / 1000));
const fmt = (x) => x.toFixed(1);
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)] || 0; };
function record(test, mbps, note = '') { results.push({ test, mbps, note }); console.log(`  → ${test}: ${fmt(mbps)} MB/s ${note}`); }

/** Fetch one byte range. sink: null = discard, or { fh } for positional writes. */
async function getRange(url, headers, from, to, sink) {
  const t0 = performance.now();
  const res = await fetch(url, { headers: { ...headers, Range: `bytes=${from}-${to}` } });
  const ttfb = performance.now() - t0;
  if (res.status !== 206) throw new Error(`expected 206, got ${res.status} from ${new URL(res.url).host}`);
  let got = 0, pos = from, chunks = 0;
  for await (const buf of res.body) {
    if (sink) await sink.fh.write(buf, 0, buf.length, pos);
    pos += buf.length; got += buf.length; chunks++;
  }
  return { got, ttfb, ms: performance.now() - t0, avgChunkKB: got / chunks / 1024 };
}

async function parallelTest(label, url, headers, chunkSize, conns, sink) {
  const n = Math.floor(WINDOW / chunkSize);
  const queue = Array.from({ length: n }, (_, i) => i);
  const stats = [];
  const t0 = performance.now();
  await Promise.all(Array.from({ length: Math.min(conns, n) }, async () => {
    while (queue.length) { const i = queue.shift(); stats.push(await getRange(url, headers, i * chunkSize, (i + 1) * chunkSize - 1, sink)); }
  }));
  const ms = performance.now() - t0;
  const total = stats.reduce((s, x) => s + x.got, 0);
  const ttfbs = stats.map((s) => s.ttfb);
  const perReq = stats.map((s) => rate(s.got, s.ms - s.ttfb));
  record(label, rate(total, ms),
    `(${n} requests; TTFB median ${Math.round(median(ttfbs))} ms, max ${Math.round(Math.max(...ttfbs))} ms; per-request ${fmt(median(perReq))} MB/s; avg read ${Math.round(median(stats.map((s) => s.avgChunkKB)))} KB)`);
}

(async () => {
  console.log(`\n  Download diagnostic — node ${process.version}, ${os.platform()} ${os.arch()}, window ${WINDOW / MB} MB\n`);
  const src = sources.resolve(link);
  if (!src) throw new Error('Could not parse link');
  const listing = await src.list();
  const want = (listing.pinned || '').toUpperCase();
  const cand = listing.candidates.find((c) => want && (c.label.toUpperCase() === want || c.key.toUpperCase().includes(want))) || listing.candidates[0];
  const part = cand.parts[0];
  console.log(`  File: ${part.path} (${fmt(part.bytes / 1024 ** 3)} GB)  token: ${listing.headers.Authorization ? 'yes' : 'no'}`);

  // 1. Redirect chain
  console.log('\n  [1] Redirect chain');
  let url = part.url, hops = 0;
  const t0 = performance.now();
  for (;;) {
    const sameHost = new URL(url).host === new URL(part.url).host;
    const res = await fetch(url, { headers: { ...(sameHost ? listing.headers : {}), Range: 'bytes=0-0' }, redirect: 'manual' });
    console.log(`      ${res.status}  ${new URL(url).host}  ${res.headers.get('server') || ''} ${res.headers.get('x-cache') || ''} ${res.headers.get('content-range') || ''}`);
    try { await res.body?.cancel(); } catch {}
    const loc = res.headers.get('location');
    if (res.status >= 300 && res.status < 400 && loc && hops++ < 5) { url = new URL(loc, url).toString(); continue; }
    break;
  }
  console.log(`      resolved in ${Math.round(performance.now() - t0)} ms → final host ${new URL(url).host}`);
  const finalHeaders = new URL(url).host === new URL(part.url).host ? listing.headers : {};

  // 2. curl single connection (reference)
  console.log('\n  [2] curl, 1 connection, 256 MB');
  try {
    const out = execFileSync('curl', ['-s', '-o', '/dev/null', '-r', `0-${256 * MB - 1}`, '-w', '%{speed_download} %{time_starttransfer}', url], { encoding: 'utf8', timeout: 300000 });
    const [bps, ttfb] = out.trim().split(' ').map(Number);
    record('curl 1 conn', bps / MB, `(TTFB ${Math.round(ttfb * 1000)} ms)`);
  } catch (e) { console.log(`      curl failed: ${e.message.split('\n')[0]}`); }

  // 3. Node fetch, 1 connection, discard
  console.log('\n  [3] Node fetch, 1 connection, 256 MB, discard');
  { const r = await getRange(url, finalHeaders, 0, 256 * MB - 1, null); record('node 1 conn', rate(r.got, r.ms), `(TTFB ${Math.round(r.ttfb)} ms, avg read ${Math.round(r.avgChunkKB)} KB)`); }

  // 4. Node parallel, discard (network only)
  console.log('\n  [4] Node fetch, parallel, discard');
  await parallelTest('node 4 conn × 32 MB', url, finalHeaders, 32 * MB, 4, null);
  await parallelTest('node 16 conn × 32 MB', url, finalHeaders, 32 * MB, 16, null);
  await parallelTest('node 16 conn × 8 MB', url, finalHeaders, 8 * MB, 16, null);

  // 5. Node parallel + positional disk writes (what the installer does)
  console.log('\n  [5] Node fetch, 16 conn × 32 MB, writing to disk like the installer');
  const tmp = path.join(path.dirname(__dirname), 'models', '.diag.tmp');
  fs.mkdirSync(path.dirname(tmp), { recursive: true });
  const fh = await fs.promises.open(tmp, 'w+');
  await fh.truncate(WINDOW);
  await parallelTest('node 16 conn + disk', url, finalHeaders, 32 * MB, 16, { fh });
  await fh.close();

  // 6. Disk only
  console.log('\n  [6] Disk only: 1 GB of 64 KB positional writes');
  { const fh2 = await fs.promises.open(tmp, 'w+'); const buf = Buffer.alloc(64 * 1024, 7); const t = performance.now();
    for (let p = 0; p < 1024 * MB; p += buf.length) await fh2.write(buf, 0, buf.length, p);
    await fh2.sync(); await fh2.close(); record('disk write', rate(1024 * MB, performance.now() - t)); }
  try { fs.unlinkSync(tmp); } catch {}

  // Summary
  console.log('\n  Summary\n  ───────');
  for (const r of results) console.log(`  ${r.test.padEnd(24)} ${fmt(r.mbps).padStart(7)} MB/s   ≈ ${fmt(part.bytes / MB / r.mbps / 60)} min for this file`);
  console.log('\n  Paste this whole output back to Claude.\n');
})().catch((e) => { console.error(`\n  ✘ ${e.message}\n`); process.exit(1); });
