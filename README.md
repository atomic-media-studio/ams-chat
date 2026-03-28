# openchat

Cross-platform desktop chat for HCI and cognitive science workflows.

## Overview

- Native UI with theming and human vs. agent interaction modes.
- **Ollama** integration: model selection, optional token limits, and local inference.
- **HTTP API** on `127.0.0.1:3000` for inbound agent and evaluator messages, health checks, and an **OpenAI-compatible** surface (`/v1/chat/completions`, `/v1/models`) backed only by **local Ollama** (no cloud).
- **Audit trail**: append-only JSONL (`openchat-audit.jsonl`) with correlation IDs; structured logs via `tracing` (`RUST_LOG`).
- **SQLite** (`openchat.db` in the working directory): durable conversations, messages, per-assistant generation metadata (model / `num_predict`), and session settings.
- **Export**: chat and keyboard logs as JSON or CSV; **Export / Import JSON** in the sidebar for full conversation snapshots.

## Build

```sh
# Development: Builds to 'target/debug/'
cargo run

# Distribution: Builds to 'target/release/'
cargo build --release
```

## API

All endpoints are local-only (bind address in code). Inference for `/v1/*` uses Ollama at `127.0.0.1:11434`.

- `GET /health` — returns `OK` when the server is enabled.
- `POST /` with plain text — message attributed to `API` (ingest-only for Ollama unless you opt in; see below).
- `POST /` with conversation JSON — `sender_name` and `message` (plus optional metadata).
- `POST /` with evaluator JSON — `evaluator_name`, `sentiment`, and `message`.
- **Auto-respond (Ollama):** default is off. Use query `?auto_respond=true` or JSON `"auto_respond": true` on conversation/evaluator payloads (JSON wins when present). When enabled, the message uses the same inference path as typing in the UI.
- **OpenAI-compatible (Ollama):** `GET /v1/models` lists local models. `POST /v1/chat/completions` accepts `model`, `messages` (`role` / `content`), optional `max_tokens` (maps to Ollama `num_predict`), `temperature`, `seed`. Streaming is not supported yet (`stream: true` returns 400).
- Example: `curl -X POST http://127.0.0.1:3000/ -H "Content-Type: application/json" -d '{"sender_name":"Agent 1","message":"Hi","sender_id":1,"receiver_id":0,"receiver_name":"UI","topic":"chat","timestamp":"11:44:50"}'`
