'use strict';
// Hugging Face: list every .gguf in a repo and group split files per quantization.
const { GiB } = require('../util.cjs');
const { parseQuant } = require('../quant.cjs');

const HF = 'https://huggingface.co';
const SPLIT_RE = /-(\d{5})-of-(\d{5})\.gguf$/i;

function authHeaders() {
  const t = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/**
 * Accepts:
 *   https://huggingface.co/owner/repo
 *   https://huggingface.co/owner/repo/blob/main/file-Q4_K_M.gguf  (pins that file)
 *   hf.co/owner/repo:Q4_K_M                                      (pins that quant)
 *   owner/repo
 */
function parse(input) {
  const s = String(input).trim()
    .replace(/^hf:/i, '')
    .replace(/^https?:\/\//i, '')
    .replace(/^(www\.)?(huggingface\.co|hf\.co)\//i, '')
    .replace(/[?#].*$/, '');
  const m = s.match(/^([\w.-]+)\/([\w.-]+?)(?::([\w.-]+))?(?:\/(?:tree|blob|resolve)\/([^/]+)(?:\/(.+?))?)?\/?$/);
  if (!m) return null;
  return {
    repo: `${m[1]}/${m[2]}`,
    quant: m[3] ? m[3].toUpperCase() : null,
    revision: m[4] ? decodeURIComponent(m[4]) : 'main',
    file: m[5] && /\.gguf$/i.test(m[5]) ? decodeURIComponent(m[5]) : null,
  };
}

const fileUrl = (ref, p) =>
  `${HF}/${ref.repo}/resolve/${encodeURIComponent(ref.revision)}/${p.split('/').map(encodeURIComponent).join('/')}`;

/** Pure: turn the HF tree listing into download candidates. */
function groupGguf(files, ref) {
  const groups = new Map();
  const mmproj = [];
  for (const f of files) {
    if (f.type && f.type !== 'file') continue;
    if (!/\.gguf$/i.test(f.path)) continue;
    const base = f.path.split('/').pop();
    const part = { path: f.path, bytes: f.lfs?.size ?? f.size, sha256: f.lfs?.oid || null, url: fileUrl(ref, f.path) };
    if (/mmproj/i.test(base)) { mmproj.push(part); continue; }
    if (/imatrix/i.test(base)) continue;
    const key = f.path.replace(SPLIT_RE, '.gguf');
    if (!groups.has(key)) groups.set(key, { key, quant: parseQuant(base) || parseQuant(f.path), parts: [] });
    groups.get(key).parts.push(part);
  }
  const candidates = [...groups.values()].map((g) => {
    g.parts.sort((a, b) => a.path.localeCompare(b.path));
    g.sizeGB = g.parts.reduce((s, p) => s + (p.bytes || 0), 0) / GiB;
    g.label = g.quant || g.key.split('/').pop();
    return g;
  });
  // Vision projector: prefer F16, else the smallest.
  const proj = mmproj.find((p) => /f16/i.test(p.path)) || mmproj.sort((a, b) => a.bytes - b.bytes)[0] || null;
  return { candidates, mmproj: proj };
}

async function list(ref) {
  const files = [];
  let url = `${HF}/api/models/${ref.repo}/tree/${encodeURIComponent(ref.revision)}?recursive=true`;
  while (url) {
    const res = await fetch(url, { headers: authHeaders() });
    if (res.status === 401 || res.status === 403) {
      throw new Error(`Access denied to ${ref.repo}. If it is gated/private, accept its licence on Hugging Face and set HF_TOKEN.`);
    }
    if (res.status === 404) throw new Error(`Hugging Face repo not found: ${ref.repo} (revision "${ref.revision}").`);
    if (!res.ok) throw new Error(`Hugging Face API error ${res.status}.`);
    files.push(...(await res.json()));
    const next = (res.headers.get('link') || '').match(/<([^>]+)>;\s*rel="next"/);
    url = next ? next[1] : null;
  }
  const out = groupGguf(files, ref);
  if (!out.candidates.length) {
    throw new Error(`${ref.repo} has no .gguf files. Search Hugging Face for "${ref.repo.split('/')[1]} GGUF" (e.g. from unsloth, bartowski or lmstudio-community).`);
  }
  // Pin from the link, if the user gave a file or :QUANT.
  let pinned = null;
  if (ref.file) pinned = ref.file.replace(SPLIT_RE, '.gguf');
  else if (ref.quant) pinned = ref.quant;
  return { ...out, pinned, headers: authHeaders(), slug: ref.repo.replace('/', '__'), title: ref.repo };
}

module.exports = { parse, list, groupGguf };
