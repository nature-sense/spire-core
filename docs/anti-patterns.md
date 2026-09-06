# Anti-Patterns

<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
<!-- Copyright (c) 2026 NatureSense -->

Things that break or silently degrade `spire-core`, and what to do instead.

## 1. `ServiceRegistry::get` on the hot path

`registry.get` locks a `Mutex<HashMap>` and type-checks every call. Doing it
per message serializes the actor and burns cycles.

**Instead:** resolve once in `ChildActor::init` (or in the constructor) and cache
the `mpsc::Sender` in a field. `ChildContext::service` exists exactly for this.

## 2. Blocking the actor's `handle()`

A long synchronous operation inside `handle` stalls the whole mailbox: other
actors' requests queue up, timers fire late, and the system feels dead.

**Instead:** `tokio::spawn` the slow work and reply from the task (like
`TransportActor` does for pending extension calls), or delegate to a dedicated
worker actor.

## 3. Touching the low-level `SharedGraph` directly

`MemoryGraphActor` is the sole data store for a reason: it owns UUID↔`u64` ID
mapping, GQL schema invariants, and snapshot logic. Bypassing it with the raw
SeleneDB API splits that state across the codebase.

**Instead:** always go through `MemoryGraphMessage` (or `GraphDb::execute_gql_*`
inside `graph.rs`).

## 4. `.unwrap()` on oneshot replies

A caller can drop its receiver at any moment (timeout, shutdown). `unwrap` on
`reply_rx.await` or on a `Result<T, ActorError>` turns a normal teardown into a
panic.

**Instead:** handle both levels of failure (`rx.await` **and** the inner
`Result`), and ignore send errors with `let _ = reply_to.send(...)`.

## 5. Registering a bare `Arc<dyn Embedder>` in the registry

`ServiceRegistry::get_service` downcasts via `Any`, which requires `Sized` —
`Arc<dyn Embedder>` won't downcast. It silently returns `None` at lookup time,
and every consumer falls back to `NoopEmbedder`.

**Instead:** register `EmbedderService(Arc<dyn Embedder>)` (the sized wrapper in
`actors::rag`) under `"embedder"`.

## 6. Embedding one chunk at a time

A loop of `embed()` per chunk multiplies model forward passes and dominates
ingest time.

**Instead:** `embed_batch(&chunks)` once (see [`rag.md`](rag.md)).

## 7. Regexes with backreferences

The `regex` crate does not support `\1`-style backreferences. `Regex::new(...)`
returns an `Err`, and an `.unwrap()` turns it into a runtime panic on every call.

**Instead:** split `<script>`/`<style>` stripping into separate regexes (the
pattern `(?is)<(script|style)[^>]*>.*?</\1>` is the historical example).

## 8. `cfg!(feature = "…")` for an undeclared feature

An undeclared feature produces a compile warning and always evaluates to
`false`, silently disabling the gated code (e.g. strict GQL test assertions).

**Instead:** declare `[features] name = []` in `Cargo.toml` for every feature you
`cfg` on.

## 9. Treating RAG as project-scoped

RAG corpora are user-level and domain-scoped (`knowledge_dir()`, keyed by
`rag_domain:<id>`). Mixing them into a project's graph couples corpus lifetime
and cleanup to a project checkout.

**Instead:** keep the KnowledgeStore separate; scope all chunk/entity queries by
the resolved domain id.

## 10. Re-embedding the corpus on every query

Retrieval should embed only the *query*. Re-embedding stored chunks per request
destroys the whole point of persisting vectors at ingest time.

**Instead:** persist vectors with `StoreNodeWithEmbedding`/`MergeAttrNode` at
ingest and vector-search at query time.

## 11. Moving build parsing into `spire-core`

`spire-core` analyzes *structure* (scanner, file tree); build-system-specific
parsing belongs in the MCP build servers (mcp-cargo, mcp-node, …) and in
`spire-code`'s build modules. Duplicating it here drifts the contract.

**Instead:** rely on `build_types` as the serialization contract and let MCP
servers produce it.

## 12. Silently accepting degraded mode

`NoopEmbedder` exists so RAG fails loudly instead of returning zero vectors.
Treating "no embedder registered" as fine means every search silently drops to
lexical Jaccard with no signal.

**Instead:** surface the missing `"embedder"` service at startup (warn/error),
and fail ingest rather than storing un-embeddable chunks.

## 13. Storing coordinates as JSON strings

Spatial queries pre-filter with GQL numeric range predicates
(`WHERE n.latitude >= …`), which only work on **native scalar number
properties**. Hand-serializing coordinates into a JSON string (e.g. a `geo`
property that is one blob) makes the node invisible to every range scan, so the
actor falls back to parsing each candidate — slow and fragile.

**Instead:** use the `AttrNode` spatial helpers (`set_geo_point` /
`set_spatial_geometry`), which write scalar `latitude`/`longitude` (or
`min_lng`/`min_lat`/`max_lng`/`max_lat`) columns **and** keep the optional
`geometry` property for exact `Contains`/`Intersects` refinement. See
[`docs/spatial.md`](spatial.md).
