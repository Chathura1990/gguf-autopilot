'use strict';
// Fetch a web page and return clean, markdown-like text. Zero dependencies (Node 18.17+ fetch).
//   fetchPageContent(url, { maxLength = 20000 })
//   → { title, final_url, status_code, content, truncated }   or   { error, url, status_code }
const TIMEOUT_MS = 10000;          // one deadline for connect + redirects + body
const MAX_BYTES = 5 * 1024 * 1024; // stop reading after 5 MB
const MAX_REDIRECTS = 5;
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 gguf-autopilot/1.0';
const DROP = ['script', 'style', 'noscript', 'nav', 'footer', 'header', 'svg', 'head', 'title', 'template'];
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—', hellip: '…', copy: '©', rsquo: '’', lsquo: '‘', rdquo: '”', ldquo: '“' };

const err = (message, url, status = null) => ({ error: message, url, status_code: status });

function decodeEntities(s) {
  return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => {
    if (e[0] === '#') {
      const code = e[1].toLowerCase() === 'x' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      try { return String.fromCodePoint(code); } catch { return m; }
    }
    return ENTITIES[e.toLowerCase()] ?? m;
  });
}

const stripTags = (s) => s.replace(/<[^>]+>/g, '');

function pageTitle(html) {
  for (const m of html.match(/<meta\b[^>]*>/gi) || []) {
    if (/(?:property|name)\s*=\s*["']og:title["']/i.test(m)) {
      const c = m.match(/content\s*=\s*(?:"([^"]*)"|'([^']*)')/i);
      if (c && (c[1] || c[2])) return decodeEntities(c[1] || c[2]).trim();
    }
  }
  const t = html.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  return t ? decodeEntities(stripTags(t[1])).replace(/\s+/g, ' ').trim() || null : null;
}

/** HTML → markdown-like plain text (headings, lists, code blocks kept). Returns { title, text }. */
function htmlToText(html) {
  const title = pageTitle(html);
  let h = html.replace(/<!--[\s\S]*?-->/g, '');
  for (const tag of DROP) h = h.replace(new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?<\\/${tag}>`, 'gi'), '');
  const pre = [];
  h = h.replace(/<pre\b[^>]*>([\s\S]*?)<\/pre>/gi, (_, inner) => { pre.push(decodeEntities(stripTags(inner)).replace(/^\n+|\s+$/g, '')); return `\n\u0000${pre.length - 1}\u0000\n`; });
  h = h.replace(/<code\b[^>]*>([\s\S]*?)<\/code>/gi, (_, c) => '`' + stripTags(c) + '`');
  h = h.replace(/<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/gi, (_, n, c) => `\n\n${'#'.repeat(n)} ${stripTags(c).replace(/\s+/g, ' ').trim()}\n\n`);
  h = h.replace(/<li\b[^>]*>/gi, '\n- ')
    .replace(/<\/(p|div|section|article|tr|ul|ol|table|blockquote|figure|dl|dd|dt)>/gi, '\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/t[dh]>/gi, ' ');
  let text = decodeEntities(stripTags(h))
    .replace(/[ \t\f\v ]+/g, ' ')
    .replace(/ ?\n ?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  text = text.replace(/\u0000(\d+)\u0000/g, (_, i) => '```\n' + pre[i] + '\n```');
  return { title, text };
}

function decode(buf, contentType) {
  const cs = (/charset\s*=\s*["']?([\w-]+)/i.exec(contentType) || [])[1];
  try { return new TextDecoder(cs || 'utf-8').decode(buf); } catch { return new TextDecoder('utf-8').decode(buf); }
}

async function fetchPageContent(url, { maxLength = 20000 } = {}) {
  let cur;
  try { cur = new URL(url); } catch { return err('Invalid URL', url); }
  if (cur.protocol !== 'http:' && cur.protocol !== 'https:') return err('Invalid URL: only http:// and https:// are supported', url);
  maxLength = Math.max(1, Number(maxLength) || 20000);
  const signal = AbortSignal.timeout(TIMEOUT_MS);
  const headers = { 'User-Agent': UA, Accept: 'text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.5', 'Accept-Language': 'en' };

  try {
    let res;
    for (let hop = 0; ; hop++) {
      res = await fetch(cur, { redirect: 'manual', headers, signal });
      if (![301, 302, 303, 307, 308].includes(res.status) || !res.headers.get('location')) break;
      if (hop >= MAX_REDIRECTS) return err(`Too many redirects (max ${MAX_REDIRECTS})`, url);
      await res.body?.cancel();
      cur = new URL(res.headers.get('location'), cur);
    }
    const finalUrl = cur.href;
    if (res.status >= 400) { await res.body?.cancel(); return err(`${res.status} ${res.statusText || 'Error'}`, finalUrl, res.status); }

    const ct = (res.headers.get('content-type') || '').toLowerCase();
    if (ct && !/^text\/|json|xml|html/.test(ct)) { await res.body?.cancel(); return err(`Unsupported content type: ${ct}`, finalUrl, res.status); }

    const chunks = []; let size = 0, cut = false;
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length; chunks.push(value);
      if (size > MAX_BYTES) { cut = true; await reader.cancel(); break; }
    }
    let raw = decode(Buffer.concat(chunks).subarray(0, MAX_BYTES), ct);
    let title = null;
    if (/html/.test(ct) || (!ct && /^\s*<(!doctype|html)/i.test(raw))) ({ title, text: raw } = htmlToText(raw));
    const truncated = cut || raw.length > maxLength;
    return { title, final_url: finalUrl, status_code: res.status, content: raw.length > maxLength ? raw.slice(0, maxLength).trimEnd() + '…' : raw, truncated };
  } catch (e) {
    if (e.name === 'TimeoutError' || e.name === 'AbortError') return err(`Timeout after ${TIMEOUT_MS / 1000}s`, url);
    return err(`Connection failed: ${e.cause?.code || e.message}`, url);
  }
}

module.exports = { fetchPageContent, htmlToText, MAX_REDIRECTS, TIMEOUT_MS };
