'use strict';
// Builds the full llama.cpp command for an installed model: specs → plan → flags for this build → user overrides.
const fs = require('fs');
const path = require('path');
const hardware = require('./hardware.cjs');
const gguf = require('./gguf.cjs');
const tuning = require('./tuning.cjs');
const llamacpp = require('./llamacpp.cjs');
const userconfig = require('./userconfig.cjs');
const { GiB } = require('./util.cjs');
const PKG = require('../package.json');

const parseVal = (v) => (v === 'true' ? true : v === 'false' ? false : /^-?\d+(\.\d+)?$/.test(v) ? Number(v) : v);

/** Settings given on the command line: --ctx=, --ngl=, --kv=, --threads=, ... and the generic --set key=value. */
function cliSettings(own) {
  const out = {};
  const map = { ctx: 'ctx', ngl: 'ngl', kv: 'kv', threads: 'threads', 'threads-batch': 'threadsBatch', batch: 'batch', ubatch: 'ubatch', 'cache-ram': 'cacheRam', 'n-cpu-moe': 'nCpuMoe', fa: 'flashAttn', parallel: 'parallel', 'image-max-tokens': 'imageMaxTokens' };
  for (const a of own) {
    const m = /^--([\w-]+)=(.*)$/.exec(a);
    if (m && map[m[1]]) out[map[m[1]]] = parseVal(m[2]);
    if (m && m[1] === 'set') { const [k, ...v] = m[2].split('='); out[k] = parseVal(v.join('=')); }
  }
  if (own.includes('--mlock')) out.mlock = true;
  if (own.includes('--no-mmap')) out.noMmap = true;
  if (own.includes('--metrics')) out.metrics = true;
  return out;
}

function build({ cfgPath, cfg, bin, vision, want = {}, cli = {}, port = null, extra = [] }) {
  const hw = hardware.detect();
  const budget = hardware.budgets(hw);
  let info = null;
  try { info = gguf.modelInfo(gguf.readMetadata(cfg.model)); } catch {}
  const slug = path.basename(path.dirname(cfgPath));
  const user = userconfig.forModel(userconfig.load().data, slug);
  const cat = tuning.loadCatalog(user.flags);
  const ll = llamacpp.inspect(bin);
  const help = ll?.help || '';
  const can = tuning.caps(cat, help);
  const mmprojGB = cfg.mmproj && fs.existsSync(cfg.mmproj) ? fs.statSync(cfg.mmproj).size / GiB : 0;
  const override = { ...user.settings, ...cli };
  const p = tuning.plan({ cfg, info, hw, budget, vision, mmprojGB, can, want, override });
  if (port) { p.settings.host = '127.0.0.1'; p.settings.port = String(port); p.why.host = 'local only'; p.why.port = 'first free port'; }
  const r = tuning.render(cat, help, p.settings);
  const args = tuning.adjust(r.args, user.removeArgs, [...user.extraArgs, ...extra]);
  return { hw, budget, info, user, cat, ll, plan: p, rendered: r, args, slug, autopilot: PKG.version };
}

/** Remember the last command per model, to show what changed after a gguf-autopilot or llama.cpp update. */
function compareWithLast(cfgPath, b) {
  const file = path.join(path.dirname(cfgPath), 'last-run.json');
  let prev = null;
  try { prev = JSON.parse(fs.readFileSync(file, 'utf8')); } catch {}
  const now = { autopilot: b.autopilot, llamacpp: b.ll?.version || null, catalog: b.cat.version, args: b.args.filter((a, i, all) => a !== '--port' && all[i - 1] !== '--port') };
  const changes = prev ? tuning.diffArgs(prev.args || [], now.args) : [];
  return { prev, now, changes, save: () => fs.writeFileSync(file, JSON.stringify({ ...now, at: new Date().toISOString() }, null, 2)) };
}

module.exports = { build, cliSettings, compareWithLast };
