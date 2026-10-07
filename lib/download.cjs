'use strict';
// Resumable download, live progress, sha256 check.
// Uses the system curl when present (much faster than Node's fetch on large files),
// otherwise parallel ranged Node fetch. Force Node with GGUF_DOWNLOADER=node.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Readable, Transform } = require('stream');
const { pipeline } = require('stream/promises');
const { spawn, spawnSync } = require('child_process');
const { GiB, fmtDur } = require('./util.cjs');

function hashFile(file) {
  return new Promise((resolve, reject) => {
    const h = crypto.createHash('sha256');
    fs.createReadStream(file).on('data', (d) => h.update(d)).on('end', () => resolve(h.digest('hex'))).on('error', reject);
  });
}

function freeDiskGB(dir) {
  try { const s = fs.statfsSync(dir); return (s.bavail * s.bsize) / GiB; } catch { return null; }
}

/** Turn a failed download response into a message that says why and what to do. */
async function explainFailure(res, name, headers = {}) {
  let host = '';
  try { host = new URL(res.url).host; } catch {}
  const code = res.headers.get('x-error-code') || '';
  let detail = res.headers.get('x-error-message') || '';
  if (!detail) {
    try { detail = (await res.text()).replace(/\s+/g, ' ').trim().slice(0, 200); } catch {}
  }
  const hasToken = Boolean(headers.Authorization);
  const lines = [`Download failed (HTTP ${res.status}${host ? ` from ${host}` : ''}) for ${name}`];
  if (code) lines.push(`  Hugging Face error: ${code}`);
  if (detail) lines.push(`  ${detail}`);
  if ((res.status === 401 || res.status === 403) && /huggingface\.co|hf\.co/.test(host)) {
    lines.push(hasToken
      ? '  Your HF_TOKEN was sent but has no access. Open the model page, accept its licence/request access, and check the token is "read" scope.'
      : '  This repo is probably gated. Accept its licence on the model page, then:  export HF_TOKEN=hf_...  and re-run.');
  } else if (res.status === 403) {
    lines.push('  The file host refused the request. Re-run to retry (downloads resume); if it persists, try a different repo for this model.');
  }
  return lines.join('\n');
}

const CHUNK = 32 * 1024 * 1024;
// Default to the maximum. Above ~16, Hugging Face starts rate-limiting (HTTP 429) and speed stops improving.
const MAX_CONNECTIONS = 16;
const clampConn = (n) => Math.min(MAX_CONNECTIONS, Math.max(1, Math.floor(Number(n)) || MAX_CONNECTIONS));
const DEFAULT_CONNECTIONS = clampConn(process.env.GGUF_CONNECTIONS);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function progress(name, total, start) {
  let done = start, last = 0;
  const t0 = Date.now();
  const samples = [[t0, start]]; // [time, bytes] for the live speed (last ~3 s)
  const MBs = (b) => (b / 1048576).toFixed(1).padStart(5);
  const bar = (f) => { const w = 24, n = Math.round(f * w); return `[${'█'.repeat(n)}${'░'.repeat(w - n)}]`; };
  const show = (final) => {
    const now = Date.now();
    samples.push([now, done]);
    while (samples.length > 2 && now - samples[0][0] > 3000) samples.shift();
    const [ts, bs] = samples[0];
    const live = now > ts ? ((done - bs) / (now - ts)) * 1000 : 0;
    const avg = ((done - start) / Math.max(now - t0, 1)) * 1000;
    const frac = total ? Math.min(done / total, 1) : 0;
    const gb = `${(done / GiB).toFixed(2)}/${(total / GiB).toFixed(2)} GB`;
    const line = final
      ? `  ✔ ${name}  ${bar(1)} 100%  ${(total / GiB).toFixed(2)} GB in ${fmtDur((now - t0) / 1000)}  avg ${MBs(avg).trim()} MB/s`
      : `  ${name}  ${bar(frac)} ${(frac * 100).toFixed(1).padStart(5)}%  ${gb}  ${MBs(live)} MB/s  ETA ${live > 0 ? fmtDur((total - done) / (live * 0.5 + avg * 0.5)) : '--'}`;
    process.stdout.write(`\r${line}\x1b[K${final ? '\n' : ''}`);
  };
  return {
    add(n) { done += n; const now = Date.now(); if (now - last > 400) { last = now; show(false); } },
    sub(n) { done -= n; },
    end() { show(true); },
  };
}

/**
 * Follow redirects once (HF → signed CDN URL) with a 1-byte Range probe.
 * Returns { url, headers, total, ranges } for the final location.
 */
async function probe(url, headers) {
  const res = await fetch(url, { headers: { ...headers, Range: 'bytes=0-0' }, redirect: 'follow' });
  if (!res.ok) return { error: res };
  const final = res.url || url;
  let total = 0;
  const cr = res.headers.get('content-range');
  if (res.status === 206 && cr) total = Number(cr.split('/')[1]) || 0;
  try { await res.body?.cancel(); } catch {}
  // Never send the HF token to a different host (the CDN rejects it anyway).
  const sameHost = new URL(final).host === new URL(url).host;
  return { url: final, headers: sameHost ? headers : {}, total, ranges: res.status === 206 && total > 0 };
}

/** Old behaviour: one connection, appended to .part. Used when the server has no Range support. */
async function singleStream(url, headers, part, name, bytes) {
  let start = fs.existsSync(part) ? fs.statSync(part).size : 0;
  if (bytes && start > bytes) start = 0;
  if (bytes && start === bytes) return;
  const res = await fetch(url, { headers: { ...headers, ...(start ? { Range: `bytes=${start}-` } : {}) }, redirect: 'follow' });
  if (!res.ok) throw new Error(await explainFailure(res, name, headers));
  if (start && res.status !== 206) start = 0;
  const total = bytes || start + Number(res.headers.get('content-length') || 0);
  const pg = progress(name, total, start);
  const counter = new Transform({ transform(chunk, _e, cb) { pg.add(chunk.length); cb(null, chunk); } });
  await pipeline(Readable.fromWeb(res.body), counter, fs.createWriteStream(part, { flags: start ? 'a' : 'w' }));
  pg.end();
}

/** Parallel ranged download into a pre-sized .part, with a .part.json map of finished chunks for resume. */
async function parallel(src, refresh, part, name, total, connections) {
  const statePath = `${part}.json`;
  const n = Math.ceil(total / CHUNK);
  let doneSet = new Set();
  let resumedBytes = 0;

  if (fs.existsSync(statePath)) {
    try {
      const st = JSON.parse(fs.readFileSync(statePath, 'utf8'));
      if (st.total === total && st.chunk === CHUNK) doneSet = new Set(st.done);
    } catch {}
  } else if (fs.existsSync(part)) {
    // .part from the old single-stream downloader: its prefix is valid, keep whole chunks of it.
    const have = Math.min(fs.statSync(part).size, total);
    for (let i = 0; (i + 1) * CHUNK <= have; i++) doneSet.add(i);
  }
  if (!fs.existsSync(part)) fs.writeFileSync(part, '');
  const fh = await fs.promises.open(part, 'r+');
  await fh.truncate(total);

  const len = (i) => Math.min(CHUNK, total - i * CHUNK);
  for (const i of doneSet) resumedBytes += len(i);
  if (resumedBytes) console.log(`  ↻ resuming ${name} from ${(resumedBytes / GiB).toFixed(2)} GB`);

  let saveTimer = null;
  const save = () => fs.writeFileSync(statePath, JSON.stringify({ total, chunk: CHUNK, done: [...doneSet] }));
  const saveSoon = () => { if (!saveTimer) saveTimer = setTimeout(() => { saveTimer = null; save(); }, 1000); };

  const queue = [];
  for (let i = 0; i < n; i++) if (!doneSet.has(i)) queue.push(i);
  const pg = progress(name, total, resumedBytes);
  let failure = null;

  async function fetchChunk(i) {
    const from = i * CHUNK, to = from + len(i) - 1;
    for (let attempt = 1; ; attempt++) {
      let got = 0;
      try {
        const res = await fetch(src.url, { headers: { ...src.headers, Range: `bytes=${from}-${to}` }, redirect: 'follow' });
        if ((res.status === 403 || res.status === 410 || res.status === 401) && attempt <= 2) {
          try { await res.body?.cancel(); } catch {}
          await refresh(); // signed CDN URL expired: resolve a fresh one
          continue;
        }
        if (res.status !== 206) throw Object.assign(new Error(await explainFailure(res, name, src.headers)), { fatal: res.status !== 200 && res.status < 500 && res.status !== 429 });
        let pos = from;
        for await (const buf of res.body) {
          await fh.write(buf, 0, buf.length, pos);
          pos += buf.length; got += buf.length; pg.add(buf.length);
        }
        if (got !== len(i)) throw new Error(`short read on chunk ${i}`);
        doneSet.add(i); saveSoon();
        return;
      } catch (e) {
        pg.sub(got);
        if (e.fatal || attempt >= 6 || failure) throw e;
        await sleep(Math.min(1000 * 2 ** (attempt - 1), 15000));
      }
    }
  }

  const worker = async () => {
    while (queue.length && !failure) {
      const i = queue.shift();
      try { await fetchChunk(i); } catch (e) { failure = failure || e; }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(connections, queue.length)) }, worker));
  clearTimeout(saveTimer); save();
  await fh.close();
  if (failure) { process.stdout.write('\n'); throw failure; }
  pg.end();
  fs.unlinkSync(statePath);
}

let curlChecked = null;
function hasCurl() {
  if (curlChecked == null) {
    if ((process.env.GGUF_DOWNLOADER || '').toLowerCase() === 'node') curlChecked = false;
    else { try { curlChecked = spawnSync('curl', ['--version'], { stdio: 'ignore' }).status === 0; } catch { curlChecked = false; } }
  }
  return curlChecked;
}

/** A .part left by the parallel Node downloader is pre-sized; keep only its contiguous finished prefix. */
function trimParallelPart(part) {
  const statePath = `${part}.json`;
  if (!fs.existsSync(statePath)) return;
  let keep = 0;
  try {
    const st = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    const done = new Set(st.done);
    let i = 0; while (done.has(i)) i++;
    keep = Math.min(i * st.chunk, st.total);
  } catch {}
  if (fs.existsSync(part)) fs.truncateSync(part, keep);
  fs.unlinkSync(statePath);
}

/** Download with the system curl, resuming from the .part file. Our own progress line is drawn from the file size. */
function curlOnce(url, headers, part, name, total) {
  const start = fs.existsSync(part) ? fs.statSync(part).size : 0;
  if (total && start === total) return Promise.resolve();
  if (start) console.log(`  ↻ resuming ${name} from ${(start / GiB).toFixed(2)} GB`);
  const args = ['-fL', '-sS', '--retry', '5', '--retry-delay', '2', '--connect-timeout', '30', '-C', '-', '-o', part];
  for (const [k, v] of Object.entries(headers)) args.push('-H', `${k}: ${v}`);
  args.push(url);
  const pg = progress(name, total, start);
  let seen = start;
  const tick = setInterval(() => {
    try { const now = fs.statSync(part).size; pg.add(now - seen); seen = now; } catch {}
  }, 400);
  return new Promise((resolve, reject) => {
    let err = '';
    const child = spawn('curl', args, { stdio: ['ignore', 'ignore', 'pipe'] });
    child.stderr.on('data', (d) => { err += d; });
    const stop = () => child.kill('SIGINT');
    process.once('SIGINT', stop);
    child.on('error', (e) => { clearInterval(tick); reject(e); });
    child.on('close', (code) => {
      clearInterval(tick); process.removeListener('SIGINT', stop);
      try { const now = fs.statSync(part).size; pg.add(now - seen); seen = now; } catch {}
      if (code === 0) { pg.end(); resolve(); }
      else { process.stdout.write('\n'); reject(Object.assign(new Error(`curl exited with code ${code}: ${err.trim().split('\n').pop() || ''}`), { curlCode: code })); }
    });
  });
}

async function curlDownload(src, refresh, part, name, total) {
  trimParallelPart(part);
  for (let attempt = 1; ; attempt++) {
    try { return await curlOnce(src.url, src.headers, part, name, total); }
    catch (e) {
      if (e.curlCode === 130 || e.curlCode == null) throw e;          // Ctrl+C or curl missing
      if (e.curlCode === 33 && fs.existsSync(part)) fs.truncateSync(part, 0); // server refused resume: restart
      if (attempt >= 3) throw e;
      await refresh();                                                   // signed CDN URL may have expired
    }
  }
}

async function download(url, dest, { bytes = 0, sha256 = null, headers = {}, verify = true, connections = DEFAULT_CONNECTIONS } = {}) {
  connections = connections == null ? DEFAULT_CONNECTIONS : clampConn(connections);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const name = path.basename(dest);
  if (fs.existsSync(dest) && (!bytes || fs.statSync(dest).size === bytes)) {
    console.log(`  ✔ ${name} already downloaded`);
    return dest;
  }
  const part = `${dest}.part`;
  const partDone = bytes && fs.existsSync(part) && !fs.existsSync(`${part}.json`) && fs.statSync(part).size === bytes;

  if (!partDone) {
    const src = await probe(url, headers);
    if (src.error) throw new Error(await explainFailure(src.error, name, headers));
    const refresh = async () => { const s = await probe(url, headers); if (!s.error) Object.assign(src, s); };
    const total = src.total || bytes;
    if (hasCurl()) await curlDownload(src, refresh, part, name, total);
    else if (src.ranges && connections > 1 && total > CHUNK) await parallel(src, refresh, part, name, total, connections);
    else await singleStream(src.url, src.headers, part, name, bytes);
  }

  if (bytes && fs.statSync(part).size !== bytes) {
    throw new Error(`${name}: size mismatch (expected ${bytes} bytes). Re-run to resume.`);
  }
  if (verify && sha256) {
    process.stdout.write(`  verifying sha256 of ${name}… `);
    const got = await hashFile(part);
    if (got !== sha256) { fs.unlinkSync(part); throw new Error(`${name}: checksum mismatch, file removed. Re-run to download again.`); }
    console.log('ok');
  }
  fs.renameSync(part, dest);
  return dest;
}

module.exports = { download, freeDiskGB, hashFile, MAX_CONNECTIONS };
