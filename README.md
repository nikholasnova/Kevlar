# Kevlar

Anthropic Messages API compatibility layer for running Claude Code against local MLX models on Apple Silicon.

Handles the full API surface Claude Code depends on: tool calling, SSE streaming, thinking blocks, token counting, and the background Haiku requests used for compaction, codebase exploration, and title generation. Normalizes prompts to stabilize KV cache prefixes across turns, avoiding the full re-prefill that other local servers hit on every turn.

## Install

```bash
git clone https://github.com/nikholasnova/Kevlar.git
cd Kevlar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires macOS with Apple Silicon (`mlx`, `mlx-lm`).

## Models

Any MLX-compatible model works. Browse quantized models ready for MLX at [huggingface.co/mlx-community](https://huggingface.co/mlx-community). Models download automatically on first use and are cached locally by `huggingface-hub`.

Pick models based on your available memory. The main model should be the largest that fits alongside the haiku model and OS overhead. See [Memory budget](#memory-budget-128gb-example) below.

## Usage

```bash
kevlar                     # start both servers + launch Claude Code (default command)
kevlar run --model mlx-community/Qwen3-Coder-Next-8bit
kevlar run --haiku-model mlx-community/Qwen3-4B-4bit
kevlar run --no-haiku      # disable haiku subprocess
```

`kevlar run` starts the main model server (port 8080), the haiku model server (port 8081), waits for both health checks, and launches `claude` with `ANTHROPIC_BASE_URL` pointed at the main server.

```bash
kevlar serve               # server only (no Claude Code launch)
kevlar serve --haiku-port 8081
kevlar status
kevlar cache clear [-f]
```

### Menu bar app

Native SwiftUI app that lives in the menu bar. Start/stop servers, switch models, clear cache, launch Claude Code -- all without the terminal.

```bash
make install               # build and install to /Applications
```

Then open "Kevlar" from Spotlight. Requires Xcode Command Line Tools.

Manual setup without `kevlar run`:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
claude
```

## Architecture

Kevlar runs two processes:

- **Main model** (default port 8080) -- inference for coding tasks. Tool calls, generation, thinking, streaming. Default: Qwen 3.5 122B-A10B 4-bit.
- **Haiku model** (default port 8081) -- Claude Code background tasks: auto-compaction, Explore subagent, conversation titles, `/resume` summaries. Default: Qwen3-8B 4-bit.

Separate processes because MLX cannot do concurrent inference from separate threads in the same process ([ml-explore/mlx#3078](https://github.com/ml-explore/mlx/issues/3078)). Each process gets its own Metal command queue; the GPU interleaves them.

Routing: requests with "haiku" in the model name are proxied to the haiku subprocess via httpx. Everything else goes to the main engine. If the haiku subprocess is down, returns 503.

```
kevlar/
  api/
    app.py          FastAPI server, model loading, warmup
    routes.py       /v1/messages, /v1/messages/count_tokens, haiku routing
    models.py       Anthropic message models (Pydantic)
    sse.py          SSE event builders
    haiku_proxy.py  httpx proxy to haiku subprocess
  engine/           MLX model loading, generation loop, sampling
  preprocessing/    Prompt normalization
  cache/            LRU memory cache, prefix matching, SSD persistence
  cli/              Rich console output
  utils/            Chat template translation, thinking extraction, tool call parsing
KevlarApp/          Native SwiftUI menu bar app
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API (streaming + non-streaming) |
| `/v1/messages/count_tokens` | POST | Token counting |
| `/v1/model/load` | POST | Hot-swap model without restarting server |
| `/v1/model/unload` | POST | Unload model to free memory |
| `/v1/status` | GET | Model info, cache stats, uptime |
| `/health` | GET | Health check |

### Prompt normalization

Claude Code injects timestamps, file trees, `<system-reminder>` blocks, and working directory paths into every prompt. These change on every turn, which invalidates the token prefix and forces a full KV cache miss.

Kevlar extracts volatile content and relocates it after the stable portion of the system prompt. The model sees the same information in a different order. The first 95% of tokens stay identical across turns, so prefix matching finds the cached KV state and only the delta needs prefilling.

RoPE embeddings are baked into KV cache entries at insertion time -- you can't rearrange cached blocks without corrupting attention. This is why normalization happens upstream rather than at the cache level.

MoE models use a mixed cache (`ArraysCache` + `KVCache` per layer). Prefix trimming preserves the ArraysCache state while slicing KVCache layers.

### SSD persistence

KV caches checkpoint to `~/.kevlar/cache/<model-name>/` as safetensors. LRU eviction at a configurable cap (default 10GB per model). On M-series NVMe (~7GB/s read), a 3GB cache loads in under 500ms.

### Thinking

Models with `<think>` support (Qwen3.5, Qwen3-Coder-Next, DeepSeek, etc.) stream thinking traces as Anthropic thinking content blocks. Budget capped at 16k tokens locally. Thinking tokens have a separate budget from content tokens.

## Haiku model sizing

The haiku subprocess handles auto-compaction (summarizing 100k+ token conversations), the Explore subagent (code search and analysis), and conversation summarization for `--resume`. These tasks require reasonable instruction following and code comprehension.

| Size | Viable? |
|------|---------|
| < 2B | No -- compaction summaries degrade over successive rounds |
| 3-4B | Minimum viable |
| 7-8B | Recommended |

Default is `mlx-community/Qwen3-8B-4bit` (~5GB). Override with `--haiku-model`.

## Memory budget (128GB example)

| Component | Footprint |
|-----------|-----------|
| Main model (122B MoE 4-bit) | ~61GB |
| Haiku model (8B 4-bit) | ~5GB |
| KV cache (17k context) | ~0.5GB |
| macOS + apps | ~10GB |
| **Total** | **~77GB** |

KV cache: ~24 KB/token. 128k context uses ~3.2 GB.

## Limitations

- Single request at a time per model process. UMA means parallel inference competes for the same memory bandwidth.
- Normalization patterns are tuned for Claude Code's prompt structure. Other clients may need pattern updates in `kevlar/preprocessing/patterns.py`.
- Requires `apply_chat_template` support. Tool calling parses Qwen XML (`<function=Name>`) and bare JSON. Other formats need parser additions in `kevlar/utils/tokenizer.py`.
- Models without multi-tool support (e.g. Llama 3.3) won't work with Claude Code's 26-tool setup.
- No vision/multimodal.
