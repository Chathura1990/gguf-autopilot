'use strict';
// Detects OS, CPU, RAM and GPU, and works out how much memory a model may use.
const os = require('os');
const { spawnSync } = require('child_process');
const { GiB } = require('./util.cjs');

function sh(cmd, args) {
  try {
    const r = spawnSync(cmd, args, { encoding: 'utf8', windowsHide: true, timeout: 8000 });
    return r.status === 0 ? r.stdout.trim() : null;
  } catch { return null; }
}

function detectNvidia() {
  const out = sh('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits']);
  if (!out) return null;
  const gpus = out.split(/\r?\n/).map((l) => {
    const [name, mib] = l.split(',').map((s) => s.trim());
    return { name, vramGB: Number(mib) / 1024 };
  }).filter((g) => g.vramGB > 0);
  return gpus.length ? gpus : null;
}

/** Physical cores (no hyper-threads): llama.cpp generation is fastest with -t = physical (or performance) cores. */
function physicalCores(platform, logical) {
  if (platform === 'darwin') return Number(sh('sysctl', ['-n', 'hw.physicalcpu'])) || logical;
  if (platform === 'linux') {
    try {
      const txt = require('fs').readFileSync('/proc/cpuinfo', 'utf8');
      const ids = new Set();
      for (const block of txt.split(/\n\n/)) {
        const p = /physical id\s*:\s*(\d+)/.exec(block), c = /core id\s*:\s*(\d+)/.exec(block);
        if (p && c) ids.add(`${p[1]}:${c[1]}`);
      }
      if (ids.size) return ids.size;
    } catch {}
    return logical;
  }
  return Math.max(1, Math.round(logical / 2)); // Windows: assume SMT
}

// Approximate memory bandwidth (GB/s). Token generation is memory-bound, so this sets the speed ceiling.
const APPLE_BW = [
  [/M1 Ultra/i, 800], [/M1 Max/i, 400], [/M1 Pro/i, 200], [/M1/i, 68],
  [/M2 Ultra/i, 800], [/M2 Max/i, 400], [/M2 Pro/i, 200], [/M2/i, 100],
  [/M3 Ultra/i, 800], [/M3 Max/i, 400], [/M3 Pro/i, 150], [/M3/i, 100],
  [/M4 Max/i, 410], [/M4 Pro/i, 273], [/M4/i, 120], // M4 Max: 410 (32-core GPU) to 546 (40-core): use the lower
];
const NVIDIA_BW = [
  [/5090/, 1792], [/5080/, 960], [/5070 Ti/, 896], [/4090/, 1008], [/4080 SUPER/i, 736], [/4080/, 717],
  [/4070 Ti SUPER/i, 672], [/4070 Ti/, 504], [/4070/, 504], [/3090/, 936], [/3080 Ti/, 912], [/3080/, 760], [/3070/, 448],
];
function lookupBandwidth(table, name) {
  const hit = table.find(([re]) => re.test(name || ''));
  return hit ? hit[1] : null;
}

function detect() {
  const platform = process.platform;
  const arch = os.arch();
  const cpus = os.cpus();
  const hw = {
    platform, arch,
    totalGB: os.totalmem() / GiB,
    cpuModel: cpus[0]?.model?.trim() || 'unknown',
    cpuThreads: cpus.length,
    accel: 'cpu', gpuName: null, gpuGB: 0, gpuCount: 0,
    physicalCores: physicalCores(platform, cpus.length),
    perfCores: null, // Apple Silicon performance cores (best -t value for CPU work)
  };

  if (platform === 'darwin' && arch === 'arm64') {
    // Apple Silicon = unified memory. By default macOS lets the GPU use ~2/3 of RAM
    // (machines with ≤36 GB) or ~3/4 (larger), unless raised via `sysctl iogpu.wired_limit_mb`.
    hw.cpuModel = sh('sysctl', ['-n', 'machdep.cpu.brand_string']) || hw.cpuModel;
    hw.perfCores = Number(sh('sysctl', ['-n', 'hw.perflevel0.physicalcpu'])) || null;
    const wiredMb = Number(sh('sysctl', ['-n', 'iogpu.wired_limit_mb']) || 0);
    hw.accel = 'metal';
    hw.gpuName = `${hw.cpuModel} GPU (unified memory)`;
    hw.gpuGB = wiredMb > 0 ? wiredMb / 1024 : hw.totalGB * (hw.totalGB > 36 ? 0.75 : 0.67);
    hw.gpuCount = 1;
    hw.bandwidthGBs = lookupBandwidth(APPLE_BW, hw.cpuModel);
  } else {
    const nv = detectNvidia();
    if (nv) {
      hw.accel = 'cuda';
      hw.gpuName = nv.map((g) => g.name).join(' + ');
      hw.gpuGB = nv.reduce((s, g) => s + g.vramGB, 0);
      hw.gpuCount = nv.length;
      hw.bandwidthGBs = nv.length === 1 ? lookupBandwidth(NVIDIA_BW, hw.gpuName) : null; // multi-GPU: unknown
    }
  }
  return hw;
}

/** Memory (GB) a model may use, leaving room for the OS, display and other apps. */
function budgets(hw) {
  if (hw.accel === 'metal') return { gpu: hw.gpuGB - 0.5, ram: 0 }; // one shared pool
  if (hw.accel === 'cuda') return { gpu: hw.gpuGB * 0.92 - 0.4, ram: hw.totalGB * 0.6 };
  return { gpu: 0, ram: hw.totalGB * 0.65 };
}

module.exports = { detect, budgets, lookupBandwidth, APPLE_BW, NVIDIA_BW };
