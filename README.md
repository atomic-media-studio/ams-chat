# openchat

Cross-platform desktop chat for HCI and cognitive science workflows.

## Overview

- Native UI with theming and human vs. agent interaction modes.
- **Ollama** integration: model selection, optional token limits, and local inference.
- **HTTP API** on `127.0.0.1:3000` for inbound agent and evaluator messages plus health checks.
- **Audit trail**: append-only JSONL (`openchat-audit.jsonl`) with correlation IDs; structured logs via `tracing` (`RUST_LOG`).
- **Export**: chat and keyboard logs as JSON or CSV for analysis.

## Build

```sh
# Development: Builds to 'target/debug/'
cargo run

# Distribution: Builds to 'target/release/'
cargo build --release
```

## API

- `GET /health` — returns `OK` when the server is enabled.
- `POST /` with plain text — message attributed to `API`.
- `POST /` with conversation JSON — `sender_name` and `message` (plus optional metadata).
- `POST /` with evaluator JSON — `evaluator_name`, `sentiment`, and `message`.
- Example: `curl -X POST http://127.0.0.1:3000/ -H "Content-Type: application/json" -d '{"sender_name":"Agent 1","message":"Hi","sender_id":1,"receiver_id":0,"receiver_name":"UI","topic":"chat","timestamp":"11:44:50"}'`
