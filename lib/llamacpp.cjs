'use strict';
const { spawnSync } = require('child_process');

function has(bin) {
  try {
    const r = spawnSync(bin, ['--version'], { encoding: 'utf8', windowsHide: true, timeout: 8000 });
    if (r.error) return null;
    const out = `${r.stdout || ''}${r.stderr || ''}`;
    const m = out.match(/version:\s*(\S+)/i);
    return { version: m ? m[1] : 'unknown' };
  } catch { return null; }
}

/** Version (build number) + full --help text of a llama.cpp binary, used to pick flag spellings. */
function inspect(bin) {
  const run = (a) => { try { const r = spawnSync(bin, a, { encoding: 'utf8', windowsHide: true, timeout: 15000 }); return r.error ? null : `${r.stdout || ''}${r.stderr || ''}`; } catch { return null; } };
  const help = run(['--help']);
  if (help == null) return null;
  const v = run(['--version']) || '';
  const m = v.match(/version:\s*(\S+)(?:\s*\(([0-9a-f]+)\))?/i);
  return { bin, help, version: m ? m[1] : 'unknown', commit: m?.[2] || null };
}

function detect() {
  return { server: has('llama-server'), cli: has('llama-cli'), mtmd: has('llama-mtmd-cli') };
}

function installHint(hw) {
  if (hw.platform === 'darwin') return 'brew install llama.cpp   (Metal support included)';
  if (hw.platform === 'win32') {
    return hw.accel === 'cuda'
      ? 'Download the CUDA build (llama-*-bin-win-cuda-*.zip) from https://github.com/ggml-org/llama.cpp/releases and add it to PATH'
      : 'winget install llama.cpp   (or download a build from https://github.com/ggml-org/llama.cpp/releases)';
  }
  return hw.accel === 'cuda'
    ? 'Build with CUDA: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda'
    : 'brew install llama.cpp   (or build: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)';
}

module.exports = { detect, inspect, installHint };
