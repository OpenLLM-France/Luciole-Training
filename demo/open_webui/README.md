# Tool-calling demo with Open WebUI

Showcase the Luciole tool-calling model in a polished streaming chat UI,
without hand-rolling a frontend. [Open WebUI](https://github.com/open-webui/open-webui)
connects to any OpenAI-compatible server (vLLM) or to Ollama, streams
responses, and renders tool-call steps for you.

The tool scripts live in [`tools/`](./tools/). Two demo tools are provided in
[`tools/luciole_demo_tools.py`](./tools/luciole_demo_tools.py): `calculator` and
`get_weather`. Additional ready-to-use tools in the same folder cover Python
execution, web search, Wikipedia, and an explicit "no tool" option.

---

## Quick start (Docker Compose)

Brings up **vLLM + Open WebUI together**, already wired to each other. Needs an
NVIDIA GPU + `nvidia-container-toolkit` on the host.

```bash
cp .env.example .env     # edit MODEL, TOOL_PARSER, GPU_COUNT, ...
docker compose up -d
# open http://localhost:3000
```

Then jump to [step 4 (add the tools)](#4-add-the-demo-tools) and
[step 5 (enable native function calling)](#5-enable-native-function-calling) —
the model connection is already configured by compose.

To run the pieces by hand instead (or use Ollama), follow the numbered steps
below.

---

## 1. Serve the model (OpenAI-compatible)

Pick **one** backend.

### Option A — vLLM (recommended for a fine-tuned checkpoint)

Tool calling must be enabled explicitly, and the tool-call parser must match
the model's chat template:

```bash
vllm serve <path-or-hf-id-of-your-luciole-model> \
    --served-model-name luciole \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --port 8000
```

- `--tool-call-parser` must match how your model emits calls. Common values:
  `hermes` (Qwen2.5-style), `llama3_json` (Llama 3.1), `mistral`. If your
  Luciole tool-calling fine-tune uses a custom template, you may need a
  matching parser (or a custom one) — this is the piece most likely to need
  tuning. A mismatch shows up as tool calls being emitted as plain text
  instead of being parsed.
- The OpenAI-compatible endpoint will be at `http://localhost:8000/v1`.

### Option B — Ollama (quickest for an off-the-shelf model)

```bash
ollama serve
ollama pull qwen2.5:7b      # any tool-calling-capable model
```

Ollama listens on `http://localhost:11434`. Open WebUI auto-detects it.

---

## 2. Run Open WebUI

```bash
docker run -d \
    --name open-webui \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

Open <http://localhost:3000> and create the first (admin) account — it's local.

> From inside the container, reach a backend running on the host via
> `host.docker.internal` (e.g. `http://host.docker.internal:8000/v1`).

---

## 3. Connect Open WebUI to your backend

**Settings → Admin Settings → Connections.**

- **vLLM:** add an *OpenAI API* connection
  - Base URL: `http://host.docker.internal:8000/v1`
  - API key: any non-empty string (e.g. `EMPTY`) — vLLM ignores it.
- **Ollama:** usually auto-detected. If not, add an *Ollama* connection with
  URL `http://host.docker.internal:11434`.

Your model (`luciole`, or the Ollama tag) should now appear in the model
picker.

---

## 4. Add the demo tools

**Workspace → Tools → “+”.** Paste the contents of
[`tools/luciole_demo_tools.py`](./tools/luciole_demo_tools.py), then **Save**.

Open WebUI builds each tool's JSON schema from the method's type hints and
docstring, so keep those accurate when you add your own tools.

---

## 5. Enable native function calling

This is the key setting — by default Open WebUI uses its own prompt-based tool
loop. To use the **model's own** tool-calling ability (what you trained):

- Open a chat, or edit the model in **Workspace → Models**.
- **Advanced Params → Function Calling → `Native`.**
- Enable the **Luciole Demo Tools** for the chat/model (the 🔧 tools toggle
  below the message box, or the model's Tools list).

---

## 6. Try it

```
What's the weather in Tokyo, and what's that in Fahrenheit?
Compute (128 * 4) + 17 for me.
What's the weather in Paris? Then multiply the temperature by 3.
```

You should see the tool-call steps and results rendered inline, then the
model's final answer.

---

## Notes

- **The model must actually be able to call tools.** Native function calling
  only works if the checkpoint was trained/templated for it — which is exactly
  what the tool-calling pipeline in this repo
  (`data/processing/tool_calling/`, `finetune/nemo-rl/configs/tool_calling/`)
  is for. To sanity-check the UI plumbing first, point Open WebUI at a known
  tool-calling model (e.g. `qwen2.5:7b` via Ollama).
- **Adding real tools:** replace the demo methods in `tools/luciole_demo_tools.py`
  with your own. Each public method on the `Tools` class = one tool; return a
  string (JSON is convenient). Add `Valves`/`UserValves` (pydantic) if a tool
  needs configurable secrets/endpoints.
- **No WebRTC, no aiortc** — unlike the audio demo, this is plain HTTP
  streaming, which is all a text+tools chat needs.
```
