'use strict';
const readline = require('readline/promises');
let rl = null;

/** Ask a question on the terminal. Returns '' when not interactive. */
async function ask(question) {
  if (!process.stdin.isTTY) return '';
  rl ??= readline.createInterface({ input: process.stdin, output: process.stdout });
  return (await rl.question(question)).trim();
}

function close() { rl?.close(); rl = null; }

module.exports = { ask, close };
