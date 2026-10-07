'use strict';
const GiB = 1024 ** 3;

const fmtGB = (gb) => (gb == null || Number.isNaN(gb) ? '?' : `${gb.toFixed(1)} GB`);

function fmtDur(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '--';
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
  return h ? `${h}h${String(m).padStart(2, '0')}m` : m ? `${m}m${String(s).padStart(2, '0')}s` : `${s}s`;
}

/** Run async fn over items with limited concurrency, preserving order. */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0;
  const worker = async () => { while (i < items.length) { const k = i++; out[k] = await fn(items[k], k); } };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

/** Print a simple aligned table. */
function table(headers, rows) {
  const w = headers.map((h, c) => Math.max(h.length, ...rows.map((r) => String(r[c]).length)));
  const line = (r) => '  ' + r.map((v, c) => String(v).padEnd(w[c])).join('  ');
  console.log(line(headers));
  console.log('  ' + w.map((n) => '─'.repeat(n)).join('  '));
  rows.forEach((r) => console.log(line(r)));
}

module.exports = { GiB, fmtGB, fmtDur, mapLimit, table };
