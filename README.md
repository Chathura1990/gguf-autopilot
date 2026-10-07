# gguf-autopilot

**Paste a model link. Get the GGUF that fits your machine, ready to run.**

Give it a **Hugging Face** or **Ollama** model link. It checks your hardware, picks the GGUF quantization that fits, downloads it quickly, and runs it with **llama.cpp**, tuned for your machine.

Zero npm dependencies · Node 18.17+ · macOS / Windows / Linux

## Quick start

```bash
brew install llama.cpp            # Windows: winget install llama.cpp (or a CUDA build from GitHub releases)
node bin/install.cjs              # asks for the model link, picks + downloads the right quant
node bin/run.cjs                  # server + web UI on the first free port from 8033 (address printed at start)
node bin/run.cjs --explain        # see every setting and why
node bin/run.cjs --no-think       # faster replies from reasoning models
node bin/run.cjs --cli            # or chat in the terminal
```

Open the address `run.cjs` prints (e.g. `http://127.0.0.1:8033`). Use `127.0.0.1`, not `localhost`: `localhost` may reach another app on the same port.

## Accepted links

| Source | Example | Notes |
|---|---|---|
| Hugging Face repo | `https://huggingface.co/unsloth/Qwen3-8B-GGUF` | Lists every `.gguf` and picks the best fit |
| HF with quant pinned | `hf.co/unsloth/Qwen3-8B-GGUF:Q5_K_M` | Skips auto-choice |
| HF specific file | `https://huggingface.co/owner/repo/blob/main/x-Q4_K_M.gguf` | Skips auto-choice |
| HF short form | `unsloth/Qwen3-8B-GGUF` | |
| Ollama library | `https://ollama.com/library/llama3.2:3b` or `llama3.2:3b` | Probes `3b-q4_K_M`, `3b-instruct-q8_0`, … for other quants |
| Ollama user model | `https://ollama.com/someuser/model:tag` | Use the full URL |

**Gated or private HF repos:** accept the licence on the model page, then `export HF_TOKEN=hf_...`. If access is missing, the download error says so (HF error code + what to do).

## 1. Install (`install.cjs`)

### How it chooses a quant

| Step | What happens |
|---|---|
| 1. Hardware | Apple Silicon: ~67% of RAM usable by the GPU (≤36 GB Macs), ~75% above. NVIDIA: VRAM via `nvidia-smi`. Otherwise: CPU with ~65% of RAM. |
| 2. Estimate | Each file needs: size + KV cache (context) + ~0.6 GB overhead. |
| 3. Pick | Largest quant that runs **fully on GPU**, capped by `--prefer`. If only a very low-bit quant fits, a GPU+CPU split at ≤ Q4_K_M is preferred (NVIDIA only). Tries 8K context, then 4K. |
| 4. Refine | After download, reads the GGUF header (layers, KV heads, hybrid attention) to set the exact context and GPU layer count. |

| `--prefer` | Max auto quant | Use when |
|---|---|---|
| `speed` | Q4_K_M | Fastest tokens/sec |
| `balanced` (default) | Q6_K | Good quality/speed trade-off |
| `quality` | Q8_0 | Plenty of memory |

### Vision models

If the repo has a vision projector (mmproj), the installer explains what it is and asks **"Download the vision projector? [Y/n]"**. If you download it, `run.cjs` asks each time whether to use it.

### Downloads

| Feature | Detail |
|---|---|
| Downloader | System `curl` (ships with macOS and Windows 10+). Node's `fetch` stalled on large files, curl did ~57 MB/s on one connection. |
| Fallback | No curl, or `GGUF_DOWNLOADER=node`: 16 parallel ranged Node connections (32 MB chunks), then a single stream if the server has no Range support |
| Progress | Bar + % + GB + **live speed (last 3 s)** + ETA; final line shows total time and average speed |
| Resume | Ctrl+C and re-run to continue (`.part` file). Works across both downloaders. |
| Reliability | Retries; expired signed CDN links are refreshed automatically; your HF token is never sent to the CDN |
| Integrity | sha256 checked against the HF LFS hash / Ollama digest (`--no-verify` to skip) |
| Other | Multi-part (`-00001-of-00003.gguf`) files grouped automatically; disk space checked first |

> Sizes are shown in GiB (1024³ bytes). Finder and the HF website use GB (1000³), so a "13.56 GB" file shows as 14.56 GB there. Same file.

### Install options

| Flag | Meaning |
|---|---|
| `--quant=Q4_K_M` | Force a quant |
| `--ctx=16384` | Force context size |
| `--prefer=speed\|balanced\|quality` | See above |
| `--yes` / `-y` | Non-interactive: accept the recommendation |
| `--dry-run` | Show the plan, don't download |
| `--no-verify` | Skip sha256 check (faster on huge files) |
| `--connections=N` | Node fallback downloader only: parallel connections, 1–16 (default 16, or `GGUF_CONNECTIONS`) |

## 2. Run (`run.cjs`)

Every run re-reads **the model** (GGUF header), **the computer** (CPU cores, RAM, GPU, memory budget) and **the installed llama.cpp** (`llama-server --help` and `--version`), then builds the fastest command that fits. Nothing is baked in at install time, so a new llama.cpp or gguf-autopilot release is picked up automatically.

```bash
node bin/run.cjs --explain     # every setting, its value, why, and the exact flag, without starting
node bin/run.cjs               # start (asks about vision if an mmproj is installed)
```

### What is checked

| Model (GGUF header) | Computer | llama.cpp build |
|---|---|---|
| Size, layers, KV heads, head size, max context | OS, CPU model, performance / physical / logical cores | Version + build commit |
| Hybrid attention (`full_attention_interval`) | RAM, GPU (Metal / CUDA / CPU), GPU memory budget | Which flags exist and their spelling (`--help`) |
| MoE experts (`expert_count`) | Free ports (IPv4 + IPv6) | Allowed values (e.g. `--spec-type draft-mtp`) |
| Built-in draft / MTP layers (`nextn_predict_layers`) | | |
| Sliding window, embedding model, thinking chat template | | |

### What it decides

| Setting | Rule | Flag (current build) |
|---|---|---|
| Context | Spare memory ÷ KV cost per token, capped at the model max. Hybrid models: KV only on full-attention layers (~4× more) | `-c` |
| GPU layers | All when it fits; MoE on a small NVIDIA GPU: all layers on GPU, expert weights of N layers on CPU | `-ngl`, `--n-cpu-moe` |
| Flash attention | On when supported | `-fa on` (old builds: `-fa`) |
| KV cache | Text-only: q8_0 (≈2× context). Vision / safe mode: f16 | `-ctk` / `-ctv` |
| Chat slots | 1, so one chat gets the whole context | `-np 1` |
| Threads | CPU / GPU+CPU runs only: performance or physical cores; all logical cores for prompt processing | `-t`, `-tb` |
| Prompt cache | Limited to spare RAM (default 8 GiB can push a small machine into swap); 0 when there is none | `--cache-ram` |
| Speculative decoding | Model has MTP draft layers and the build supports it → draft-mtp. `--spec=ngram` → n-gram drafting (code / repetitive text) | `--spec-type`, `--spec-default` |
| Thinking | `--no-think` → off | `--reasoning off` (old builds: `--reasoning-budget 0`) |
| Vision | `--mmproj` when enabled; context reduced to what fits | `--mmproj` |
| Embedding models | Embeddings endpoint, batch = context, embedding test request | `--embeddings`, `-b`, `-ub` |
| Chat template | Model's own | `--jinja` |
| Memory lock / mmap | Only if you ask (`--mlock`, `--no-mmap`) | `--load-mode …` (old builds: `--mlock`, `--no-mmap`) |

A setting the installed build can't express is **skipped and listed** ("not in this llama.cpp build"), never sent.

### Self-healing start-up

If llama-server exits before the model is ready, `run.cjs` retries automatically: first **without speculative decoding**, then in **safe mode** (f16 KV, default attention), and tells you which flag to keep (`--spec=off` or `--safe`). The command that worked is saved to `models/<model>/last-run.json`.

### After an update

When gguf-autopilot (e.g. `brew upgrade`) or llama.cpp changes the command, the next run shows the difference:

```
  Changes since last run (gguf-autopilot 0.1.0 / llama.cpp 6900 → 0.2.0 / 7100):
    + --spec-type draft-mtp
    ~ -c 18432 → 35840
    - --mlock
```

### Your settings (survive upgrades)

`node bin/run.cjs --init-config` creates `~/.config/gguf-autopilot/config.json` (or `$GGUF_AUTOPILOT_CONFIG`). Precedence: **auto plan < `settings` < `models.<slug>.settings` < command line**.

```json
{
  "settings": { "kv": "f16", "cacheRam": 2048 },
  "extraArgs": ["--metrics"],
  "removeArgs": ["--cache-ram"],
  "models": {
    "orcarouter__Qwen3.8-27B-Uncensored-GGUF": { "settings": { "specType": "none" }, "extraArgs": ["--temp", "0.7"] }
  },
  "flags": {
    "flashAttn": [{ "detect": "--some-new-flag", "args": ["--some-new-flag", "{v}"] }]
  }
}
```

| Key | Meaning |
|---|---|
| `settings` | Any planner setting: `ctx`, `ngl`, `kv`, `flashAttn`, `parallel`, `threads`, `threadsBatch`, `batch`, `ubatch`, `cacheRam`, `mlock`, `noMmap`, `reasoning`, `specType` (`draft-mtp` / `ngram` / `none`), `specDraftMax`, `nCpuMoe`, `imageMaxTokens`, `metrics`. `null` removes a setting. |
| `extraArgs` | Appended to the command as-is (e.g. sampling: `--temp`, `--top-p`, `--min-p`) |
| `removeArgs` | Flags to drop from the generated command |
| `models.<slug>` | Same keys for one model (`<slug>` = folder name under `models/`) |
| `flags` | Extra flag spellings, tried before the built-in catalog: adapt to a newer llama.cpp before a gguf-autopilot release |

### Run options

| Flag | Meaning |
|---|---|
| `--explain` | Show each setting, value, reason and flag; don't start |
| `--print` | Show summary + command; don't start |
| `--cli` | Terminal chat (`llama-cli`, or `llama-mtmd-cli` with vision) |
| `--port=N` | Fixed port (default: first free from 8033) |
| `--vision` / `--no-vision` | Skip the image-input question |
| `--no-think` | Thinking off for faster replies |
| `--spec=auto\|mtp\|ngram\|off`, `--no-spec` | Speculative decoding (default auto = MTP when the model has it) |
| `--safe` | No speculative decoding, f16 KV, default attention |
| `--ctx=` `--ngl=` `--kv=` `--threads=` `--threads-batch=` `--batch=` `--ubatch=` `--cache-ram=` `--n-cpu-moe=` `--fa=` `--parallel=` | Override one setting |
| `--mlock`, `--no-mmap`, `--metrics` | Switch these on |
| `--set key=value` | Override any setting by name (also future ones) |
| `--init-config` | Create your config file |
| `path/to/model.json` | Run a specific installed model (default: latest install) |
| `-- …` | Anything after `--` is passed straight to llama.cpp |

### Server start-up

| Step | What happens |
|---|---|
| Port | First free port from 8033 upward. "Free" = nothing answers on IPv4 or IPv6 **and** we can bind it (skips Docker etc.). A busy `--port=` is refused with a free alternative. |
| Logs | Streamed live and saved to `models/<model>/server.log` |
| Ready check | Waits for `/health`, sends a test chat (or embedding) request, prints **✔ Ready** with the URL |

### For maintainers: new llama.cpp release

```bash
brew upgrade llama.cpp
node bin/check-flags.cjs        # catalog vs installed build: ✔/✘ per setting + flags not used yet
```

Flag spellings live in `config/llama-flags.json` (variants tried in order, matched against `--help`). Add a variant for renamed flags, a rule in `lib/tuning.cjs` for new decisions, bump the catalog `version`, add a test with a `--help` excerpt, release. Old llama.cpp builds keep working because older spellings stay as fallback variants.

## 3. Download speed diagnostic

```bash
node bin/diagnose-download.cjs owner/repo:QUANT [--mb=512]
```

Measures each layer separately: redirect chain, curl vs Node (1 connection), Node parallel (4/16 connections, 32/8 MB chunks), Node + disk writes, disk only. The summary converts each speed into minutes for the full file.

## Layout

```
bin/install.cjs            interactive installer (+ preview of the tuned command)
bin/run.cjs                runs llama.cpp: vision prompt, planner, free port, logs, ready check, self-healing retry
bin/check-flags.cjs        maintainer: catalog vs installed llama.cpp
bin/diagnose-download.cjs  download speed diagnostic
config/llama-flags.json    flag catalog: abstract setting → flag spellings per llama.cpp version
lib/tuning.cjs             planner (model + computer → settings + reasons) and renderer (settings → flags)
lib/runplan.cjs            glue: specs + user config + catalog → command; last-run diff
lib/userconfig.cjs         ~/.config/gguf-autopilot/config.json
lib/hardware.cjs           OS / CPU cores / RAM / GPU detection + memory budget
lib/gguf.cjs               GGUF header reader (KV sizing, hybrid, MoE, MTP, embedding, thinking)
lib/quant.cjs              quant parsing + selection
lib/download.cjs           curl / parallel Node downloader, progress bar, resume, sha256, HF error messages
lib/llamacpp.cjs           finds llama.cpp binaries, version + --help, install hints
lib/sources/               huggingface.cjs, ollama.cjs, index.cjs (link routing)
test/run-tests.cjs         29 offline tests (npm test)
models/                    downloads, model.json, last-run.json, server.log per model (git-ignored)
```

## Known limits

- Ollama has no public tag-list API, so other quants are found by guessing tag names. Pin one if probing misses it.
- Some newer Ollama models use Ollama's own engine and may not load in upstream llama.cpp. Prefer the HF GGUF in that case.
- AMD (ROCm) and Intel GPUs are not detected yet, so they are treated as CPU-only.
- KV-cache sizing is still conservative for sliding-window models (hybrid linear/full attention is handled).
- Large models + vision may not fit together on 24 GB Macs; `run.cjs` warns and suggests `--no-vision`.
