# Best Practices

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Patterns that keep `spire-core` actors fast, correct, and maintainable.

## Actor construction & wiring

- **Cache service senders at `init`/construction, never per message.** Look up
  `ctx.service::<M>("name")` once in a `ChildActor::init` (or in a constructor
  that receives the registry) and store the sender in a field. The registry is
  `Mutex`-backed — reading it on the hot path serializes every message.
- **Register every long-lived sender under a stable string name** in the
  `ServiceRegistry` at spawn time. Consumers resolve by name + type; a wrong type
  lookup returns `None` (fail fast, no silent `Any`-downcast panics).
- **`ActorSystem::spawn` returns `(Sender, JoinHandle)`** — keep the sender,
  and keep the handle if you need graceful shutdown or task supervision.

## Message passing

- **Use oneshot request/reply for everything that needs an answer.** Pass the
  `oneshot::Sender` inside the message (the `Responder<T>` pattern).
- **Never block the mailbox loop.** Long I/O, HTTP, model loads, or socket waits
  belong in a `tokio::spawn`ed task (see `TransportActor::CallExtension`, which
  spawns the response-waiter so the reader loop keeps draining the socket).
- **Ignore the result of `reply_to.send(...)`** with `let _ =` — the caller may
  have timed out or dropped its receiver.
- **Prefer fire-and-forget `mpsc::Sender::send` over unbounded channels.** The
  default mailbox is bounded (32); design messages to be cheap to enqueue.

## Tools & MCP

- **Let `ToolRouterActor` own dispatch; register tools, don't hardcode them.**
  Modules expose `ListTools`; `spire-code`'s composer registers static tools;
  MCP server tools are discovered dynamically. Adding a new backend is a
  registration, not a match-arm.
- **Keep the MCP client behind `McpClientActor`.** `McpClientManager` is not
  `Sync`; the actor is the single owner of connections.

## Graph & persistence

- **Go through `MemoryGraphActor` / GQL, never the low-level `SharedGraph`**
  (which is confined to `graph.rs`). This keeps ID mapping and schema
  invariants in one place.
- **Use `MergeAttrNode`/`MergeRelationship` for idempotent upserts** and the
  transaction stream (`OpenTransactionStream` + `Commit`/`Rollback`) when a set
  of writes must land atomically.
- **Batch GQL** (`MemoryGraphMessage::BatchGql`) for bulk operations instead of
  N round-trips.

## Embeddings & RAG

- **Share one embedder instance** (`CandleEmbedder` is expensive to construct —
  it loads a model). Register it once as `EmbedderService(Arc<dyn Embedder>)`
  under `"embedder"`; `RagActor::from_registry` resolves it.
- **Always batch embeddings** (`embed_batch`), never a `for` loop of `embed()`.
- **Keep `corpus_version` a pure function of the manifest** so re-ingest is
  idempotent and crash-safe.
- **Fail per source, not per corpus.** Record `SourceStatus` for each source and
  continue; surface skipped sources in `ListSources`.
- **Treat `NoopEmbedder` as an explicit degraded mode.** Log loudly when the
  `"embedder"` service is missing so a silent drop to Jaccard retrieval can't
  go unnoticed.

## Configuration & tests

- **Use the env overrides** (`SPIRE_CONFIG_DIR`, `SPIRE_KNOWLEDGE_DIR`,
  `SPIRE_PLATFORM_DIR`) in tests/CI so you never touch the developer's real
  `~/.spire`.
- **Declare every feature used in `cfg!`/`cfg` in `Cargo.toml`** (e.g.
  `[features] strict-gql = []`) — undeclared features are a compile warning and
  silently evaluate to `false`.

## Filesystem writes

- **Prefer `apply_patch` (strict unified diff) over whole-file writes** for
  surgical edits — it is apply-or-fail and can never corrupt a file with stale
  content.
