'use strict';
// User config: lives outside the repo/brew prefix, so it survives upgrades.
//   ~/.config/gguf-autopilot/config.json   (or $GGUF_AUTOPILOT_CONFIG)
const fs = require('fs');
const os = require('os');
const path = require('path');

const configPath = () => process.env.GGUF_AUTOPILOT_CONFIG || path.join(os.homedir(), '.config', 'gguf-autopilot', 'config.json');

const TEMPLATE = {
  _help: 'Overrides for gguf-autopilot. Precedence: auto plan < settings < models[<slug>].settings < command-line flags. Remove keys you do not need. Docs: README "Tuning".',
  settings: {
    _examples: 'ctx, ngl, kv ("f16"|"q8_0"|"q4_0"), flashAttn ("on"|"off"|"auto"), parallel, threads, threadsBatch, batch, ubatch, cacheRam (MiB), mlock, noMmap, reasoning ("on"|"off"|"auto"), specType ("draft-mtp"|"ngram"|"none"), specDraftMax, imageMaxTokens, metrics',
  },
  extraArgs: [],
  removeArgs: [],
  models: {
    '_example__owner__repo': { settings: { kv: 'f16' }, extraArgs: ['--temp', '0.7'] },
  },
  flags: {},
};

function clean(o) { return Object.fromEntries(Object.entries(o || {}).filter(([k]) => !k.startsWith('_'))); }

function load() {
  const p = configPath();
  if (!fs.existsSync(p)) return { path: p, exists: false, data: {} };
  try { return { path: p, exists: true, data: JSON.parse(fs.readFileSync(p, 'utf8')) }; }
  catch (e) { throw new Error(`Could not read ${p}: ${e.message}`); }
}

/** Global section merged with the model's own section. */
function forModel(data, slug) {
  const m = (data.models || {})[slug] || {};
  return {
    settings: { ...clean(data.settings), ...clean(m.settings) },
    extraArgs: [...(data.extraArgs || []), ...(m.extraArgs || [])].map(String),
    removeArgs: [...(data.removeArgs || []), ...(m.removeArgs || [])].map(String),
    flags: data.flags || {},
  };
}

function init() {
  const p = configPath();
  if (fs.existsSync(p)) return { path: p, created: false };
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, `${JSON.stringify(TEMPLATE, null, 2)}\n`);
  return { path: p, created: true };
}

module.exports = { configPath, load, forModel, init, TEMPLATE };
