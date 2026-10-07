'use strict';
// Ollama registry: model layers ARE GGUF files, so llama.cpp can load them directly.
// Ollama has no public "list tags" API, so we probe common quant tag names.
const { GiB, mapLimit } = require('../util.cjs');
const { parseQuant } = require('../quant.cjs');

const REG = 'https://registry.ollama.ai/v2';
const ACCEPT = 'application/vnd.docker.distribution.manifest.v2+json';
const PROBE = ['q2_K', 'q3_K_S', 'q3_K_M', 'q3_K_L', 'q4_0', 'q4_1', 'q4_K_S', 'q4_K_M',
  'q5_0', 'q5_1', 'q5_K_S', 'q5_K_M', 'q6_K', 'q8_0', 'fp16', 'bf16'];

/**
 * Accepts:
 *   https://ollama.com/library/llama3.2:3b
 *   https://ollama.com/someuser/somemodel:tag
 *   llama3.2:3b   |   ollama:llama3.2
 */
function parse(input) {
  const s = String(input).trim()
    .replace(/^ollama:/i, '')
    .replace(/^https?:\/\//i, '')
    .replace(/^(www\.)?(ollama\.com|registry\.ollama\.ai)\//i, '')
    .replace(/^library\//, '')
    .replace(/[?#].*$/, '');
  const m = s.match(/^(?:([\w.-]+)\/)?([\w.-]+)(?::([\w.-]+))?\/?$/);
  if (!m) return null;
  return { ns: m[1] || 'library', model: m[2], tag: m[3] || 'latest' };
}

const blobUrl = (ref, digest) => `${REG}/${ref.ns}/${ref.model}/blobs/${digest}`;

async function candidateFromTag(ref, tag) {
  const res = await fetch(`${REG}/${ref.ns}/${ref.model}/manifests/${tag}`, { headers: { Accept: ACCEPT } });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Ollama registry error ${res.status} for ${ref.model}:${tag}`);
  const m = await res.json();
  const layer = (m.layers || []).find((l) => l.mediaType === 'application/vnd.ollama.image.model');
  if (!layer) return null;
  const proj = (m.layers || []).find((l) => l.mediaType === 'application/vnd.ollama.image.projector');

  let quant = parseQuant(tag);
  if (!quant && m.config?.digest) {
    try {
      const cfg = await (await fetch(blobUrl(ref, m.config.digest))).json();
      quant = cfg.file_type ? parseQuant(cfg.file_type) || String(cfg.file_type).toUpperCase() : null;
    } catch { /* ignore */ }
  }
  const part = (digest, bytes, name) => ({ path: name, bytes, sha256: digest.replace(/^sha256:/, ''), url: blobUrl(ref, digest) });
  return {
    key: `${ref.model}:${tag}`,
    tag,
    digest: layer.digest,
    quant,
    label: quant || tag,
    sizeGB: layer.size / GiB,
    parts: [part(layer.digest, layer.size, `${ref.model}-${tag}.gguf`.replace(/[:/]/g, '_'))],
    mmproj: proj ? part(proj.digest, proj.size, `${ref.model}-${tag}-mmproj.gguf`.replace(/[:/]/g, '_')) : null,
  };
}

async function list(ref) {
  const tags = [ref.tag];
  const canProbe = !parseQuant(ref.tag) && ref.tag !== 'latest';
  if (canProbe) for (const q of PROBE) tags.push(`${ref.tag}-${q}`, `${ref.tag}-instruct-${q}`);

  const found = (await mapLimit(tags, 8, (t) => candidateFromTag(ref, t).catch(() => null))).filter(Boolean);
  if (!found.length) throw new Error(`Ollama model not found: ${ref.ns}/${ref.model}:${ref.tag}`);

  const byDigest = new Map();
  for (const c of found) if (!byDigest.has(c.digest)) byDigest.set(c.digest, c); // same file under several tags
  const candidates = [...byDigest.values()];
  return {
    candidates,
    mmproj: candidates.find((c) => c.mmproj)?.mmproj || null,
    pinned: parseQuant(ref.tag) ? `${ref.model}:${ref.tag}` : null,
    headers: {},
    slug: `ollama__${ref.ns === 'library' ? '' : ref.ns + '__'}${ref.model}_${ref.tag}`.replace(/[:/]/g, '_'),
    title: `${ref.ns === 'library' ? '' : ref.ns + '/'}${ref.model}:${ref.tag}`,
    note: canProbe ? null : ref.tag === 'latest'
      ? 'Tip: give a size tag (e.g. llama3.2:3b) so other quantizations can be found.'
      : null,
  };
}

module.exports = { parse, list };
