<div align="center">

# Sowelni · سولني

**A real-time voice assistant that actually understands Moroccan Darija.**

*sowelni* (سولني) — Darija for **"ask me."**

</div>

---

## Why this exists

Moroccan Darija is the first language of roughly 40 million people. It's also one of the most underserved dialects in commercial speech recognition. Google, Amazon, and Microsoft all offer "Arabic" — but they offer it as Modern Standard Arabic with a side of Gulf, and the result on Darija is somewhere between frustrating and unusable. The dialect's French, Amazigh, and Spanish loanwords, its lack of orthographic standardization, and its strong regional variation all break models trained on textbook Arabic.

Sowelni closes that gap. It's a complete, locally-runnable voice assistant built on a fine-tuned [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) model, wrapped in a streaming WebSocket pipeline, and surfaced through a Moroccan-inflected Angular interface. You speak Darija into your browser, the audio rides a WebSocket to a Python backend running the ASR model on your GPU, the transcript flows through an LLM, and the answer streams back token by token. The full audio path never leaves your machine.

---

## What you get

```
 ╭────────────────────╮                    ╭─────────────────────────────────╮
 │   Browser (Ng 19)  │   WebSocket        │       Python Backend            │
 │                    │   ◄═══════════►    │                                 │
 │  ┌──────────────┐  │   audio chunks     │  ┌───────────┐   ┌───────────┐ │
 │  │ MediaRecorder │──┼──────────────────►─┤  │  pydub /  │──►│ Qwen3-ASR │ │
 │  │ + AnalyserNode│  │                    │  │  FFmpeg   │   │  0.6B GPU │ │
 │  └──────────────┘  │   transcription     │  └───────────┘   └─────┬─────┘ │
 │                    │   ◄────────────────┤                         │       │
 │  ┌──────────────┐  │   streamed tokens   │  ┌───────────┐         │       │
 │  │  Chat UI     │◄─┼────────────────────┤  │  Gemini / │◄────────┘       │
 │  │  (Signals)   │  │                    │  │  Local LLM│                 │
 │  └──────────────┘  │                    │  └───────────┘                 │
 ╰────────────────────╯                    ╰─────────────────────────────────╯
```

**Features:**

- **Darija-aware ASR** — Fine-tuned Qwen3-ASR-0.6B with proper label masking and gradient clipping (we'll explain what went wrong the first time)
- **Streaming end-to-end** — Audio streams up, transcriptions stream back, LLM tokens appear character by character
- **Local-first** — The ASR model runs on your GPU. Audio never goes to a third party. Even the LLM can be local.
- **Live waveform visualization** — A 24-bar frequency analyzer wrapped around the microphone button gives you immediate sensory feedback
- **Multi-turn conversation** — Past exchanges carry forward, so follow-up questions work
- **Docker-first deployment** — `docker compose up` and you're running

---

## Tech stack

| Layer | Choice | Why this and not that |
|-------|--------|----------------------|
| Frontend framework | Angular 19 (standalone components, Signals) | No NgModule overhead, fine-grained reactivity, no Zone juggling for streaming UI |
| Real-time transport | Socket.IO over WebSocket | Auto-reconnection, binary frame support, event-based — better than raw WS for audio chunks |
| Backend framework | FastAPI + python-socketio + Uvicorn | Async ASGI all the way down, composes cleanly with Socket.IO |
| ASR runtime | `qwen-asr` PyPI package | The model isn't a standard transformers model — it has its own wrapper |
| Audio decoding | pydub + FFmpeg | Reliable WebM/Opus → PCM, handles every codec ffmpeg supports |
| LLM | Google Gemini 2.0 Flash (or local Qwen3) | Streaming responses, multi-turn, Arabic-fluent. Swappable for a local model. |
| Containerization | Docker + Docker Compose | One command, GPU passthrough handled, model cache persisted |

---

## Project structure (the guided tour)

The repository is organized into two top-level domains — `backend/` and `frontend/` — plus the Docker orchestration that ties them together. Below is what each piece does and why it exists as a separate concern.

```
darija-asr-project/
│
├── docker-compose.yml           ← Orchestration: GPU passthrough, model cache, networking
├── .gitignore                   ← Python + Angular + ML artifacts + secrets
├── README.md                    ← You are here
│
├── backend/                     ── Python service: ASR + LLM + Socket.IO
│   │
│   ├── Dockerfile               ← CUDA base image, FFmpeg, Python deps
│   ├── .env.example             ← Template for secrets
│   ├── .env                     ← Real secrets (gitignored)
│   ├── requirements.txt
│   ├── setup-backend.sh         ← One-shot local install
│   ├── main.py                  ← The composition root
│   │
│   ├── services/                ── Each file is one bounded responsibility
│   │   ├── asr_service.py       ← Qwen3-ASR wrapper, non-blocking
│   │   ├── llm_service.py       ← Gemini streaming, async bridge
│   │   └── audio_processor.py   ← WebM decode + silence gate
│   │
│   └── scripts/                 ── Training-time tools, not part of the runtime
│       ├── train.py             ← Corrected fine-tuning recipe
│       ├── promote_checkpoint.py
│       └── evaluate_wer.py
│
└── frontend/                    ── Angular 19 app: capture + UI + streaming
    │
    ├── Dockerfile               ← node:20-alpine, scaffolds and serves
    ├── fix-angular-json.js      ← Strips Angular's default SSR config
    ├── setup-frontend.sh        ← One-shot local install
    │
    └── src/
        ├── main.ts              ← Bootstrap
        ├── index.html           ← Shell, Google Fonts (Cormorant, Amiri, Manrope)
        ├── styles.scss          ← Design tokens, palette, resets
        │
        ├── environments/
        │   └── environment.ts   ← Socket URL config
        │
        └── app/
            ├── app.ts           ← Root component (renders <app-transcriber>)
            ├── app.config.ts    ← Zone coalescing, no SSR
            │
            ├── services/
            │   ├── websocket.ts       ← Socket + reactive state
            │   └── audio-recording.ts ← Mic capture + waveform
            │
            └── components/transcriber/
                ├── transcriber.ts     ← Component logic
                ├── transcriber.html   ← Chat UI
                └── transcriber.scss   ← Warm palette, animations
```

### The backend, file by file

**`main.py` — The composition root.**
This file does one thing: wire everything together. It loads `.env`, instantiates the `ASRService` and `LLMService` singletons, mounts Socket.IO as the outer ASGI app with FastAPI nested inside (so `/health` works for monitoring while `/socket.io` handles real-time traffic), and registers the event handlers that route audio chunks to the ASR and trigger LLM calls. Per-client session state lives in a plain `dict[str, dict]` keyed by Socket.IO session ID — appropriate for a single-instance deployment, swap for Redis if you ever scale horizontally.

**`services/asr_service.py` — The non-blocking GPU wrapper.**
Qwen3-ASR is published with its own inference package (`qwen-asr`), not as a standard HuggingFace `AutoModel`. This service wraps `Qwen3ASRModel` and solves two real problems that bite production deployments: **GPU contention** and **event-loop blocking**. The `model.transcribe()` call is synchronous and takes 200–800 ms on a 2-second clip. If we awaited it directly, the WebSocket event loop would stall and every other client would freeze. The fix: a single-worker `ThreadPoolExecutor` to host the blocking call, plus an `asyncio.Lock` to serialize GPU access across concurrent clients. The result is that one user can be transcribing while another connects, disconnects, or hits the LLM endpoint — none of them block each other.

**`services/llm_service.py` — The async bridge.**
Google's `google-genai` SDK exposes streaming as a blocking iterator (`for chunk in stream:`). Our event loop needs an `async for`. The bridge: a producer function runs in a background thread, pulls chunks off the blocking iterator, and pushes each one onto an `asyncio.Queue` via `run_coroutine_threadsafe`. The service's `stream()` method is an async generator that yields from the queue. This is a small but crucial pattern — without it, the LLM call would block every other Socket.IO event for the entire duration of the response.

**`services/audio_processor.py` — The boundary between web audio and ML audio.**
The browser sends WebM/Opus blobs. The ASR model wants float32 numpy arrays at 16 kHz, normalized to [-1, 1]. This file is the converter, and it's deliberately small: one function to decode (via pydub → FFmpeg), one function to detect silence. The silence gate (RMS < 0.01) is what prevents the ASR model from hallucinating phantom transcriptions on background noise — a known failure mode of autoregressive speech models that nobody mentions until you've shipped to production.

**`scripts/train.py` — The corrected fine-tuning recipe.**
A previous training run on 40 hours of Darija data produced loss ≈ 8 and gradient norms above 1,000 — pathological values that indicate the model never actually learned anything. The failures were all in the *recipe*, not the data. This script fixes them: it masks prompt tokens out of the loss (so only the assistant's transcription contributes), clips gradients at 1.0, uses cosine LR schedule with warmup, and prefers bf16 over fp16 on Ampere+ hardware. The full diagnosis is in the [Fine-Tuning](#fine-tuning-the-asr-model) section below.

**`scripts/promote_checkpoint.py` — From training artifact to production blob.**
HuggingFace `Trainer` checkpoints contain ~3 GB of optimizer state, scheduler state, RNG state, and trainer logs that are useless for inference. This script copies only the files that matter (`model.safetensors`, configs, tokenizer files) into a clean directory you can ship.

**`scripts/evaluate_wer.py` — The number that matters.**
Runs your fine-tuned model over a validation set and reports word error rate (WER) and character error rate (CER). Below 25% WER on Darija is good; below 20% is excellent.

### The frontend, file by file

**`src/main.ts` — Bootstrap.**
Imports `App` (note the name — Angular 19 renamed `AppComponent` to `App` and changed the filename convention from `app.component.ts` to `app.ts`). Calls `bootstrapApplication`. That's it.

**`src/app/app.ts` — The root component.**
A single-line template: `<app-transcriber />`. Everything interesting lives in the transcriber component. This file's job is to be replaceable — if you ever add routing or multiple top-level features, this is where they slot in.

**`src/app/app.config.ts` — Provider registration.**
Just `provideZoneChangeDetection({ eventCoalescing: true })`. We don't use SSR (this is a real-time WebSocket app, SSR would be actively harmful), so no `provideClientHydration`. We don't use the router (single screen). The config is small on purpose.

**`src/app/services/websocket.ts` — The reactive boundary.**
This service owns the Socket.IO connection and exposes every piece of server-side state as an Angular Signal: `connected`, `currentTranscript`, `currentAnswer`, `isStreamingAnswer`, `turns`, `lastError`. The component reads these signals; the template re-renders automatically when they change. The implementation uses a local-variable pattern during socket setup (`const s = io(...)`) instead of `this.socket = io(...)` — this satisfies TypeScript strict mode without scattering `?.` and `!` operators through every event handler. Conversation history is maintained as a `signal<ConversationTurn[]>` and gets sent to Gemini on every request, enabling multi-turn context.

**`src/app/services/audio-recording.ts` — Mic capture, done correctly.**
This file solves the WebM chunking problem (see below) and drives the live waveform visualization. The MediaRecorder is stopped and restarted every 2 seconds so each emitted blob is a standalone, decodable WebM file. In parallel, a Web Audio `AnalyserNode` reads the same MediaStream, samples its frequency data at the browser's animation frame rate, downsamples 64 frequency bins into 24 visual bars, and pushes them into a Signal. The `requestAnimationFrame` loop runs outside Angular's zone (`NgZone.runOutsideAngular`) — otherwise every frame would trigger a full change-detection pass, tanking performance. The signal update *is* inside the zone, so the template still reacts.

**`src/app/components/transcriber/transcriber.ts` — Component logic.**
Injects both services, exposes their signals, and manages two cross-cutting concerns the services don't handle: lifecycle (connect on init, stop on destroy) and auto-scroll (when new turns or tokens arrive, scroll the conversation container to the bottom via `ngAfterViewChecked`). The `toggle()` method handles the start/stop flow, with a 450 ms grace period after stopping so the final audio chunk's transcription has time to arrive before we trigger the LLM.

**`src/app/components/transcriber/transcriber.html` — The chat UI.**
Built with Angular 19's new `@if` / `@for` control flow (compiled, no runtime overhead like the old `*ngIf` / `*ngFor`). Past exchanges render as bubble pairs; the in-progress turn renders with a pulsing live indicator and (during LLM streaming) a blinking cursor. The microphone button has the waveform visualization mounted inside it as overlaid `<span>` bars, each scaled vertically by its corresponding level signal value.

**`src/app/components/transcriber/transcriber.scss` — The visual language.**
A deliberate departure from the dark-blue-with-purple-gradient aesthetic that has become the default for AI apps. The palette is warm: a deep brown base (`#1a1310`), a saffron-amber accent (`#e8a838`) reminiscent of Moroccan spice markets, and a Cormorant Garamond / Amiri / Manrope type stack. The microphone button has a rotating waveform ring and a pulsing glow when active. CSS animations are used sparingly — only where they tell the user something is happening.

### Docker orchestration

**`docker-compose.yml` — Two services, one command.**
Defines the `backend` and `frontend` services with the right ports (`8000` and `4200`), the `.env` file mount, GPU passthrough for the backend (via `deploy.resources.reservations.devices`), and a named volume (`hf-cache`) mounted at `/root/.cache/huggingface` inside the backend so the 2 GB model download survives container rebuilds. The frontend service depends on `backend` so they come up in the right order.

**`backend/Dockerfile` — CUDA, FFmpeg, Python.**
Built on `nvidia/cuda:12.1-runtime-ubuntu22.04` (or similar). Installs FFmpeg (for pydub), Python, and the requirements. Copies `requirements.txt` before source code so dependency installs are cached when only source changes. The entrypoint runs `python main.py`.

**`frontend/Dockerfile` — Node, Angular CLI, dev server.**
Built on `node:20-alpine`. Runs `setup-frontend.sh` during the image build to scaffold the Angular project and copy in source files. The entrypoint runs `ng serve --host 0.0.0.0 --port 4200` so the dev server is accessible from the host network. For production, you'd swap this for a two-stage build with `ng build` and an nginx serve layer.

---

## Getting started

### With Docker (recommended)

```bash
git clone <your-repo-url> darija-asr-project
cd darija-asr-project

cp backend/.env.example backend/.env
nano backend/.env                    # paste your GOOGLE_API_KEY

docker compose up --build
```

First build downloads the ASR model (~2 GB) and installs everything. Subsequent runs are fast thanks to layer caching and the persisted `hf-cache` volume.

When both containers are healthy:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:4200 |
| Backend API | http://localhost:8000 |
| Backend health | http://localhost:8000/health |

### Without Docker (local development)

**Backend (terminal 1):**
```bash
cd backend
bash setup-backend.sh
nano .env                            # paste your GOOGLE_API_KEY
source .venv/bin/activate
python main.py
```

Wait for `ASR model ready` and `Uvicorn running on http://0.0.0.0:8000`.

**Frontend (terminal 2):**
```bash
cd frontend
bash setup-frontend.sh
cd darija-asr-frontend
ng serve --open
```

The status pill in the header flips from **Offline** to **Ready** when the WebSocket connects.

### Prerequisites

**For Docker:** Docker Engine 20.10+, Docker Compose v2+, NVIDIA Container Toolkit, optional Gemini API key.

**For local:** Python 3.10+, Node 18+, Angular CLI 19, FFmpeg on PATH, CUDA 11.8+ with cuDNN, optional Gemini API key.

---

## Configuration

All backend configuration is in `backend/.env`:

```ini
# Get one at https://aistudio.google.com/apikey
GOOGLE_API_KEY=your_key_here

# Gemini model (flash-lite has more generous free-tier quotas)
GEMINI_MODEL=gemini-2.0-flash

# ASR model — either a HuggingFace ID or a local path
# Testing / first run:
MODEL_DIR=Qwen/Qwen3-ASR-0.6B
# After fine-tuning:
# MODEL_DIR=./darija-asr-production

HOST=0.0.0.0
PORT=8000
```

Frontend Socket.IO endpoint lives in `frontend/src/environments/environment.ts`.

---

## How it actually works

### The WebM chunking problem (and the fix)

`MediaRecorder.start(2000)` looks like it should give you a series of 2-second WebM files. It doesn't. It gives you one WebM stream split into chunks, where only the *first* chunk has the EBML header that ffmpeg needs to decode. Send chunk 2 to ffmpeg in isolation and you get `error code: 187`. Send chunk 47 and you get the same error.

The fix is unintuitive but reliable: **stop and restart the recorder every 2 seconds**. Each cycle produces a complete, standalone WebM file with a valid header. The ~30 ms gap between segments is imperceptible in conversation. The pattern is in `audio-recording.ts`:

```typescript
const rec = new MediaRecorder(this.stream, { mimeType });
rec.ondataavailable = (e) => this.chunks.push(e.data);
rec.onstop = async () => {
  const blob = new Blob(this.chunks, { type: mimeType });
  this.ws.sendAudioChunk(await blob.arrayBuffer());
  if (this.active) this.recordSegment();        // chain to next
};
rec.start();
setTimeout(() => rec.stop(), 2000);
```

### Non-blocking GPU inference

`model.transcribe()` is synchronous and runs on the GPU. If we awaited it directly inside the Socket.IO event handler, every other client would be frozen for the duration. Instead:

```python
async def transcribe(self, audio, sampling_rate=16000):
    async with self._lock:                           # serialize GPU access
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,                          # 1-worker thread pool
            self._transcribe_sync, audio, sampling_rate,
        )
```

The thread pool keeps the event loop free; the lock keeps the GPU sane.

### Token streaming from a blocking SDK

Gemini's Python SDK returns a blocking iterator. We bridge to async with a queue:

```python
def producer():
    chat = client.chats.create(model=model_name, ...)
    for chunk in chat.send_message_stream(user_text):
        asyncio.run_coroutine_threadsafe(queue.put(chunk.text), loop)
    asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

loop.run_in_executor(None, producer)

while True:
    item = await queue.get()
    if item is SENTINEL: break
    yield item                                       # async generator
```

The component-side Signal `currentAnswer.update(prev => prev + token)` appends each token as it arrives, and Angular re-renders the bubble character by character.

---

## WebSocket protocol

| Direction | Event | Payload | When |
|-----------|-------|---------|------|
| → server | `audio_chunk` | Binary `ArrayBuffer` | Every 2s during recording |
| → server | `request_answer` | empty | User pressed Stop |
| → server | `reset` | empty | User cleared the conversation |
| ← client | `ready` | `{sid}` | On socket connection |
| ← client | `transcription` | `{partial, full}` | After each chunk is transcribed |
| ← client | `llm_start` | `{question}` | LLM generation begins |
| ← client | `llm_token` | `{token}` | Per generated token |
| ← client | `llm_done` | `{answer, question}` | Generation complete |
| ← client | `error` | `{message}` | Any server-side failure |

Audio is sent as raw binary (no base64), saving ~33% of bandwidth and the CPU cost of encoding on both ends.

---

## Fine-tuning the ASR model

### What went wrong the first time

The original training run on 40 hours of Darija data produced these numbers at step 16,000:

| Metric | Observed | Should be |
|--------|----------|-----------|
| Final loss | 4.3 – 8.3 (oscillating) | 0.1 – 0.8 (stable) |
| Gradient norm (typical) | 135 – 388 | 0.5 – 5.0 |
| Gradient norm (peak) | 1,002.8 | <10 |
| `best_metric` | `None` | populated by evaluation |

The model never converged. The root causes, in order of severity:

1. **Missing label masking.** The loss was computed across *all* tokens, including the user prompt and audio placeholder. The model was being asked to predict tokens that aren't predictable from the audio input. Impossible task → permanently high loss.
2. **No gradient clipping.** Gradient norms above 1,000 mean the optimizer is making catastrophically large weight updates. The model thrashed instead of learning.
3. **No evaluation during training.** Without `eval_strategy="steps"`, there's no way to identify the best checkpoint or detect overfitting.
4. **fp16 instability.** Mixed precision in fp16 on Ampere hardware introduced numerical issues that bf16 doesn't have.

### The corrected recipe

`scripts/train.py` applies all four fixes. The critical one — label masking — looks like this in the collator:

```python
labels = inputs["input_ids"].clone()
labels[labels == pad_token_id] = -100             # mask padding
for i, prompt_len in enumerate(prompt_lengths):
    labels[i, :prompt_len] = -100                  # mask prompt per-sample
inputs["labels"] = labels
```

The value `-100` is PyTorch's sentinel for "ignore this position in the loss." With this masking, only the assistant's transcription tokens contribute to gradients, which is what we actually want the model to learn.

Other hyperparameters:

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 2e-5 (full FT) / 1e-4 (LoRA) | Empirically stable for this model size |
| `max_grad_norm` | 1.0 | Prevents the gradient explosion seen above |
| `lr_scheduler_type` | cosine | Smooth decay |
| `warmup_ratio` | 0.05 | 5% warmup before cosine kicks in |
| `bf16` | True (on Ampere+) | More stable than fp16 |
| `eval_strategy` | steps, every 500 | Gives us a real `best_metric` |
| `load_best_model_at_end` | True | Keep the best, not the last |
| `gradient_checkpointing` | True | Cuts VRAM in half |

### Running training

Data format: one JSONL row per utterance.
```json
{"audio": "/abs/path/to/clip.wav", "text": "النص ديال الكلام بالدارجة"}
```

Command:
```bash
cd backend && source .venv/bin/activate

python scripts/train.py \
  --model_id   Qwen/Qwen3-ASR-0.6B \
  --train_file /path/to/train.jsonl \
  --val_file   /path/to/val.jsonl \
  --output_dir qwen3-asr-darija \
  --num_epochs 3 --batch_size 2 --grad_accum 8
```

### After training

```bash
# Strip optimizer state, keep weights + configs
python scripts/promote_checkpoint.py \
  --checkpoint qwen3-asr-darija/final \
  --output     darija-asr-production

# Measure quality
python scripts/evaluate_wer.py \
  --model_dir darija-asr-production \
  --val_file  /path/to/val.jsonl

# Point production at the new model
echo "MODEL_DIR=./darija-asr-production" >> .env
```

**Target WER for Darija:** below 25% is good, below 20% is state of the art for open-source.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Frontend stuck on "Offline" | Backend not running | Start backend first, hit `localhost:8000/health` to verify |
| `Decoding failed. ffmpeg error code 187` | Empty or corrupt audio chunk | Harmless — the server skips it and processes the next |
| `RESOURCE_EXHAUSTED` from Gemini | Free tier quota exceeded | Use `gemini-2.0-flash-lite` in `.env`, or enable billing |
| `KeyError: 'qwen3_asr'` from transformers | Wrong loading path | Make sure `qwen-asr` is installed — the model uses its own wrapper, not `AutoModel` |
| High WER on Darija | Using base model without fine-tuning | Run `scripts/train.py` on Darija data |
| GPU out of memory | Model + CUDA context exceeds VRAM | Lower `--batch_size`, close other GPU processes |
| `provideBrowserGlobalErrorListeners` not found | Angular CLI version mismatch | Remove that provider — it's optional and version-specific |
| Mic button does nothing | Browser blocked mic access | Check site permissions, require HTTPS or localhost |

---

## License

MIT for the application code in this repository.

Qwen3-ASR weights are Apache 2.0, released by the Qwen team at Alibaba Cloud. Gemini usage is subject to Google's terms.

---

<div align="center">

**Built for everyone who's been told their language isn't worth understanding.**

</div>
