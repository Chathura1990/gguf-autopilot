#!/usr/bin/env node
'use strict';
// Maintainer tool: compare config/llama-flags.json with the installed llama.cpp build.
//   node bin/check-flags.cjs [llama-server]
// Run it after `brew upgrade llama.cpp`, update the catalog, bump its "version", then release.
const tuning = require('../lib/tuning.cjs');
const llamacpp = require('../lib/llamacpp.cjs');

const bin = process.argv[2] || 'llama-server';
const ll = llamacpp.inspect(bin);
if (!ll) { console.error(`\n  ✘ ${bin} not found on PATH.\n`); process.exit(1); }
const cat = tuning.loadCatalog();

console.log(`\n  ${bin} · llama.cpp ${ll.version}${ll.commit ? ` (${ll.commit})` : ''} · catalog v${cat.version} (${cat.updated})\n`);
console.log('  Catalog setting   Status');
console.log(`  ${'─'.repeat(16)}  ${'─'.repeat(50)}`);
for (const key of cat.order) {
  const hit = (cat.flags[key] || []).find((v) => new RegExp(v.detect).test(ll.help));
  console.log(`  ${key.padEnd(16)}  ${hit ? `✔ ${hit.args.join(' ')}` : '✘ no matching flag in this build'}`);
}

const covered = new Set();
for (const variants of Object.values(cat.flags)) for (const v of variants) (v.detect.match(/--[a-z0-9-]+/g) || []).forEach((f) => covered.add(f));
const all = [...new Set(ll.help.match(/(?<![\w-])--[a-z][a-z0-9-]+/g) || [])].sort();
const fresh = all.filter((f) => !covered.has(f));
console.log(`\n  ${fresh.length} of ${all.length} flags in this build are not used by the planner (expected: most are niche).`);
console.log('  Review for new performance options:\n');
for (let i = 0; i < fresh.length; i += 4) console.log(`    ${fresh.slice(i, i + 4).map((f) => f.padEnd(28)).join('')}`);
console.log('\n  To adopt a flag: add a variant to config/llama-flags.json (and a rule in lib/tuning.cjs if it needs a decision),');
console.log('  bump "version", add a test with a --help excerpt, release. Users can also add variants under "flags" in their config.json.\n');
