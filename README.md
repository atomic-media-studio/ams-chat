# openchat

A cross-platform chat application for HCI and Cognitive Sciences.

## Build

```sh
# Development: Builds to 'target/debug/'
cargo run

# Distribution: Builds to 'target/release/'
cargo build --release
```

## API

- `GET /health` -> returns `OK` when server is enabled.
- `POST /` with plain text body -> creates chat message from `API`.
- `POST /` with JSON `{"sender_name":"Agent 1","message":"hello",...}` -> uses `sender_name` + `message`.
- `POST /` with evaluator JSON `{"evaluator_name":"Agent Evaluator","sentiment":"Positive","message":"...",...}`.
- Example curl: `curl -X POST http://127.0.0.1:3000/ -H "Content-Type: application/json" -d '{"sender_name":"Agent 1","message":"Hi","sender_id":1,"receiver_id":0,"receiver_name":"UI","topic":"chat","timestamp":"11:44:50"}'`

