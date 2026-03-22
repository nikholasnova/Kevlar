# Kevlar

Local MLX inference server for Apple Silicon. Exposes an Anthropic-compatible `/v1/messages` endpoint so Claude Code (or anything that speaks the Anthropic Messages API) can hit local models instead of the cloud.

The main thing it does differently: **smart KV cache management**. Claude Code dynamically injects timestamps, file trees, and system reminders at the top of every prompt. Standard inferencers see this as a brand new conversation and throw away the entire KV cache, forcing a full re-prefill on every turn. On a 50k token context thats 15-30 seconds before the first response token.

Kevlar fixes this by normalizing prompts before they hit the cache -- volatile content gets moved after stable content so the token prefix stays constant across turns. Cache hit rate goes from ~0% to ~90%+.

## How it works

1. Request comes in as Anthropic Messages format
2. Prompt preprocessor extracts volatile sections (timestamps, `<system-reminder>` blocks, file trees) and relocates them after the stable context
3. Cache manager checks for prefix match in memory LRU, then SSD
4. Only the changed tokens get prefilled -- the rest comes from cache
5. Thinking/reasoning traces are stripped from model output before sending the response (model still thinks internally for better quality)
6. Response streams back as Anthropic SSE events

## Install

```
git clone <repo-url>
cd Kevlar
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires macOS with Apple Silicon. Needs `mlx` and `mlx-lm` which are Apple Silicon only.

## Usage

### CLI

```bash
# start server (default model: Qwen 3.5 122B-A10B 4-bit)
kevlar serve

# pick your model
kevlar serve --model mlx-community/Qwen2.5-Coder-32B-Instruct-4bit

# all options
kevlar serve --model <hf-id-or-path> \
             --host 127.0.0.1 \
             --port 8080 \
             --cache-dir ~/.kevlar/cache \
             --max-cache-gb 10 \
             --max-tokens 4096 \
             --prefill-step-size 4096 \
             --no-normalize
```

### Menu bar app

```bash
kevlar gui
```

Puts a "K" icon in your macOS menu bar. Click it to start/stop the server, switch models, manage cache. Models are saved in `~/.kevlar/models.json`.

### Other commands

```bash
kevlar run                 # start server + launch Claude Code (default command)
kevlar status              # check if server is running, show model/cache stats
kevlar cache clear         # wipe SSD cache (with confirmation)
kevlar cache clear -f      # wipe without confirmation
```

### Use with Claude Code

```bash
# quickest way -- starts server if needed, launches Claude Code with correct env
kevlar run

# or manually
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=sk-anything
claude
```

`kevlar run` is also the default command -- bare `kevlar` with no arguments runs it.

The API key is required by Claude Code's client but Kevlar ignores it -- set it to anything.

### Verify with curl

```bash
curl http://localhost:8080/health

curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"x","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}'

curl http://localhost:8080/v1/status
```

## Architecture

```
kevlar/
  api/            FastAPI server, Anthropic message models, SSE streaming
  engine/         MLX model loading, generation loop, sampling
  preprocessing/  Prompt normalization (the cache fix)
  cache/          LRU memory cache, prefix matching, SSD persistence
  cli/            Rich console output (banner, stats, status panels)
  utils/          Chat template translation, thinking strip
  menubar.py      macOS menu bar app (rumps)
  menubar_models.py  Model list persistence
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic Messages API (streaming + non-streaming) |
| `/v1/status` | GET | Server status, model info, cache stats, uptime |
| `/health` | GET | Health check |

### Thinking mode

Models like Qwen3.5 generate chain-of-thought reasoning by default. Kevlar lets the model think (produces better results) but strips the `<think>...</think>` block from the response so Claude Code only sees the actual answer. The thinking tokens still count toward generation but are invisible to the client.

### Cache strategy

RoPE positional embeddings get baked into KV cache entries at insertion time. You cant rearrange cached blocks without corrupting attention scores. True radix/paged attention would need custom MLX kernels.

Instead we solve it upstream: normalize the prompt so the prefix never changes. The volatile stuff (dates, working directory, file trees) gets moved to the end of the system prompt. Same information, same model behavior, but now the first 95% of tokens are identical across turns and the cache hits.

MoE models (like Qwen3.5-122B-A10B) use a mixed cache -- `ArraysCache` for linear attention layers, `KVCache` for standard attention layers. Prefix matching works by trimming the KVCache layers while preserving the ArraysCache accumulated state.

### Memory budget (128GB M-series example)

| Model | Weights | KV budget | Decode speed |
|-------|---------|-----------|-------------|
| 32B dense 4-bit | ~16GB | ~104GB | ~30 tok/s |
| 72B dense 4-bit | ~36GB | ~84GB | ~15 tok/s |
| 122B MoE 4-bit (10B active) | ~61GB | ~59GB | ~60 tok/s |

### SSD persistence

KV caches checkpoint to disk as safetensors. On an M-series NVMe (~7GB/s read) a 3GB quantized cache loads in under 500ms. Useful for resuming sessions or switching between projects. Cached in `~/.kevlar/cache/` with LRU eviction.

## Limitations

- Single request at a time. UMA means parallel inference just fights over the same memory bandwidth.
- Header normalization patterns are tuned for Claude Code. Other clients with different dynamic headers may need pattern updates in `kevlar/preprocessing/patterns.py`.
- Model must support `apply_chat_template` for tool calling to work correctly.
- Thinking token budget: models with chain-of-thought can spend hundreds of tokens reasoning before answering. Set `max_tokens` high enough to accommodate both thinking and the actual response.
- No vision/multimodal support.
