'use strict';
const hf = require('./huggingface.cjs');
const ollama = require('./ollama.cjs');

/** Decide whether a link/name points at Hugging Face or Ollama. */
function resolve(input) {
  const s = String(input || '').trim();
  if (/^hf:/i.test(s) || /^(https?:\/\/)?(www\.)?(huggingface\.co|hf\.co)\//i.test(s)) {
    const ref = hf.parse(s); return ref && { kind: 'huggingface', ref, list: () => hf.list(ref) };
  }
  if (/^ollama:/i.test(s) || /^(https?:\/\/)?(www\.)?(ollama\.com|registry\.ollama\.ai)\//i.test(s)) {
    const ref = ollama.parse(s); return ref && { kind: 'ollama', ref, list: () => ollama.list(ref) };
  }
  if (/^[\w.-]+\/[\w.-]+(:[\w.-]+)?$/.test(s) && !/:/.test(s.split('/')[0])) {
    const ref = hf.parse(s); return ref && { kind: 'huggingface', ref, list: () => hf.list(ref) }; // owner/repo
  }
  if (/^[\w.-]+(:[\w.-]+)?$/.test(s)) {
    const ref = ollama.parse(s); return ref && { kind: 'ollama', ref, list: () => ollama.list(ref) }; // llama3.2:3b
  }
  return null;
}

module.exports = { resolve };
