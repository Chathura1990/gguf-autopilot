'use strict';
// Minimal GGUF metadata reader (spec: github.com/ggml-org/ggml/blob/master/docs/gguf.md).
// Reads only the key/value header to get layer count and attention shape,
// so we can size the KV cache and the GPU layer split exactly.
const fs = require('fs');
const { GiB } = require('./util.cjs');

const SIZES = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
const WANT = /^(general\.(architecture|name|size_label|basename)|tokenizer\.chat_template|[\w.]+\.(block_count|context_length|embedding_length|attention\.(head_count|head_count_kv|key_length|value_length|sliding_window)|full_attention_interval|expert_count|expert_used_count|nextn_predict_layers|pooling_type))$/;
const EMBEDDING_ARCHS = /^(bert|nomic-bert|nomic-bert-moe|jina-bert-v2|jina-bert-v3|modern-bert|neo-bert|t5encoder|gte)$/;

function readMetadata(file) {
  const fd = fs.openSync(file, 'r');
  let buf = Buffer.alloc(0), pos = 0, fileOff = 0;
  const ensure = (n) => {
    while (buf.length - pos < n) {
      const chunk = Buffer.alloc(Math.max(1 << 20, n));
      const r = fs.readSync(fd, chunk, 0, chunk.length, fileOff);
      if (!r) throw new Error('Unexpected end of GGUF header');
      fileOff += r;
      buf = Buffer.concat([buf.subarray(pos), chunk.subarray(0, r)]);
      pos = 0;
    }
  };
  const skip = (n) => {
    const avail = buf.length - pos;
    if (n <= avail) pos += n; else { fileOff += n - avail; buf = Buffer.alloc(0); pos = 0; }
  };
  const u32 = () => { ensure(4); const v = buf.readUInt32LE(pos); pos += 4; return v; };
  const u64 = () => { ensure(8); const v = Number(buf.readBigUInt64LE(pos)); pos += 8; return v; };
  const str = () => { const n = u64(); ensure(n); const s = buf.toString('utf8', pos, pos + n); pos += n; return s; };
  const scalar = (t) => {
    ensure(SIZES[t]);
    const f = {
      0: () => buf.readUInt8(pos), 1: () => buf.readInt8(pos), 2: () => buf.readUInt16LE(pos), 3: () => buf.readInt16LE(pos),
      4: () => buf.readUInt32LE(pos), 5: () => buf.readInt32LE(pos), 6: () => buf.readFloatLE(pos), 7: () => buf.readUInt8(pos) !== 0,
      10: () => Number(buf.readBigUInt64LE(pos)), 11: () => Number(buf.readBigInt64LE(pos)), 12: () => buf.readDoubleLE(pos),
    }[t];
    if (!f) throw new Error(`Unknown GGUF value type ${t}`);
    const v = f(); pos += SIZES[t]; return v;
  };
  const value = (t, keep) => {
    if (t === 8) return str();
    if (t !== 9) return scalar(t);
    const it = u32(), n = u64();
    if (!keep) {
      if (it === 8) { for (let i = 0; i < n; i++) str(); } else if (SIZES[it]) skip(SIZES[it] * n);
      else for (let i = 0; i < n; i++) value(it, false);
      return undefined;
    }
    const arr = [];
    for (let i = 0; i < n; i++) arr.push(value(it, true));
    return arr;
  };

  try {
    ensure(4);
    if (buf.toString('ascii', 0, 4) !== 'GGUF') throw new Error('Not a GGUF file');
    pos = 4;
    const version = u32();
    if (version < 2) throw new Error(`GGUF v${version} is too old`);
    u64(); // tensor count
    const kvCount = u64();
    const meta = { 'gguf.version': version };
    for (let i = 0; i < kvCount; i++) {
      const key = str();
      const keep = WANT.test(key);
      const v = value(u32(), keep);
      if (keep) meta[key] = v;
    }
    return meta;
  } finally {
    fs.closeSync(fd);
  }
}

function modelInfo(meta) {
  const arch = meta['general.architecture'];
  if (!arch) return null;
  const g = (k) => meta[`${arch}.${k}`];
  const num = (v) => (Array.isArray(v) ? Math.max(...v) : v);
  const nLayer = num(g('block_count'));
  const nHead = num(g('attention.head_count'));
  const nHeadKv = num(g('attention.head_count_kv')) ?? nHead;
  const nEmbd = g('embedding_length');
  if (!nLayer || !nHeadKv || !(g('attention.key_length') || (nEmbd && nHead))) return null;
  const headDim = nEmbd && nHead ? nEmbd / nHead : null;
  // Hybrid models (e.g. Qwen3-Next / Qwen3.8: linear attention + every Nth layer full attention):
  // only the full-attention layers keep a KV cache that grows with context.
  const interval = Number(g('full_attention_interval')) || 1;
  const kvLayers = interval > 1 ? Math.ceil(nLayer / interval) : nLayer;
  return {
    arch, name: meta['general.name'] || null, nLayer, kvLayers, hybrid: interval > 1, nHead, nHeadKv, nEmbd,
    keyLen: g('attention.key_length') || headDim,
    valLen: g('attention.value_length') || g('attention.key_length') || headDim,
    ctxTrain: g('context_length') || null,
    // Extra specs used by the run-time planner
    experts: Number(g('expert_count')) || 0,
    expertsUsed: Number(g('expert_used_count')) || 0,
    moe: Number(g('expert_count')) > 1,
    mtpLayers: Number(g('nextn_predict_layers')) || 0,      // built-in draft head → speculative decoding (draft-mtp)
    slidingWindow: Number(num(g('attention.sliding_window'))) || 0,
    embedding: EMBEDDING_ARCHS.test(arch) || (g('pooling_type') != null && !meta['tokenizer.chat_template']),
    chatTemplate: Boolean(meta['tokenizer.chat_template']),
    thinking: /<think>|enable_thinking|reasoning_content|thinking/i.test(meta['tokenizer.chat_template'] || ''),
    sizeLabel: meta['general.size_label'] || null,
  };
}

/** K+V cache size (default f16 = 2 bytes/element; q8_0 ≈ 1.0625). Conservative for sliding-window / hybrid-attention models. */
function kvCacheGB(info, ctx, bytesPerElem = 2) {
  return ((info.kvLayers ?? info.nLayer) * ctx * info.nHeadKv * (info.keyLen + info.valLen) * bytesPerElem) / GiB;
}

module.exports = { readMetadata, modelInfo, kvCacheGB };
