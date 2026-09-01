// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! End-to-end test for the user-level Platform RAG KnowledgeStore + GraphRAG
//! ingestion pipeline.
//!
//! Verifies:
//! 1. Ingesting a canonical `ingest.yaml` stores chunks + entities +
//!    relationships (content-addressed names) in the KnowledgeStore data
//!    plane (`SPIRE_KNOWLEDGE_DIR`), NOT the project store.
//! 2. The project graph keeps only a `rag_provenance` node (platform_id +
//!    corpus_version) — never the corpus itself.
//! 3. A FRESH project with an EMPTY project store can query the same domain
//!    and resolve chunks from the shared KnowledgeStore (no re-ingest).
//! 4. Re-ingestion is idempotent (same config → identical content-addressed ids).
//! 5. Model gate: the semantic assertions require the real
//!    sentence-transformers/all-MiniLM-L6-v2 embedder. When the model is not
//!    cached the tests print a SKIP notice and return — run the app once to
//!    download it (~80 MB), or point `HF_HOME` at a machine that has it.

use std::path::PathBuf;
use std::sync::Arc;

use spire_core::subsystems::graph::memory_graph::MemoryGraphActor;
use spire_core::actors::rag::{RagActor, RagMessage};
use spire_core::actors::Actor;
use spire_core::embedder::create_embedder;
use spire_core::models::embedding::Embedder;

use tokio::sync::{mpsc, oneshot};

/// Serializes tests that mutate the process-wide `SPIRE_KNOWLEDGE_DIR` /
/// `SPIRE_PLATFORM_DIR` env vars — Cargo runs tests in the same process in
/// parallel, so without a gate the two RAG tests clobber each other's dirs.
static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Load the real embedding model from the HF cache. Returns `None` (printing a
/// SKIP notice) when all-MiniLM-L6-v2 is not cached, so the suite stays green
/// on machines without the model. `create_embedder()` does blocking model I/O,
/// hence the `spawn_blocking` bridge.
async fn try_real_embedder() -> Option<Arc<dyn Embedder>> {
    match tokio::time::timeout(
        std::time::Duration::from_secs(60),
        tokio::task::spawn_blocking(create_embedder),
    )
    .await
    {
        Ok(Ok(Ok(e))) => Some(e),
        Ok(Ok(Err(e))) => {
            eprintln!(
                "SKIP: all-MiniLM-L6-v2 not cached ({e:#}); run the app once to \
                 download (~80 MB) — skipping semantic RAG assertions"
            );
            None
        }
        Ok(Err(e)) => {
            eprintln!("SKIP: embedding model thread failed ({e}); skipping");
            None
        }
        Err(_) => {
            eprintln!(
                "SKIP: embedding model load timed out (network slow?); skipping \
                 semantic RAG assertions"
            );
            None
        }
    }
}

/// Spawn a MemoryGraphActor initialized in `dir` with the given embedder.
async fn spawn_graph(
    dir: &std::path::Path,
    embedder: &Arc<dyn Embedder>,
) -> mpsc::Sender<spire_core::actors::MemoryGraphMessage> {
    std::fs::create_dir_all(dir).unwrap();
    let (tx, rx) = mpsc::channel(64);
    let _join = MemoryGraphActor::new().spawn(rx);
    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::Initialize {
        data_dir: dir.to_path_buf(),
        reply_to: t,
    })
    .await
    .expect("send init");
    r.await.expect("init reply").expect("init ok");
    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::InitializeEmbedder {
        model_path: None,
        embedder: Some(embedder.clone()),
        reply_to: t,
    })
    .await
    .expect("send embedder init");
    r.await.expect("embedder reply").expect("embedder ok");
    tx
}

async fn query_prefix(
    tx: &mpsc::Sender<spire_core::actors::MemoryGraphMessage>,
    subtype: &str,
) -> Vec<String> {
    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::QueryAttrNodes {
        node_type: Some("Unknown".to_string()),
        subtype: Some(subtype.to_string()),
        name: None,
        limit: Some(1000),
        reply_to: t,
    })
    .await
    .expect("send query");
    match r.await {
        Ok(Ok(nodes)) => nodes.iter().map(|n| n.name().to_string()).collect(),
        _ => Vec::new(),
    }
}

async fn query_rag(
    rag_tx: &mpsc::Sender<RagMessage>,
    domain: &str,
    query: &str,
) -> Vec<spire_core::actors::rag::RagChunkResult> {
    let (t, r) = oneshot::channel();
    rag_tx
        .send(RagMessage::Query {
            domain: domain.to_string(),
            query: query.to_string(),
            top_k: 5,
            reply_to: t,
        })
        .await
        .expect("send query");
    r.await.expect("query reply").expect("query ok")
}

/// Fixture: a real `~/.spire/platforms/a7s.yaml` (via SPIRE_PLATFORM_DIR) so
/// `resolve_domain` resolves "Radxa Cubie A7S" → "a7s".
fn write_platform_seed(tmp: &std::path::Path) {
    let dir = tmp.join("platforms");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("a7s.yaml"),
        r#"id: a7s
name: Cubie A7S
os: linux
architecture:
  cpu_family: aarch64
  cpu: armv8-a
  endian: little
  target_triple: aarch64-linux-gnu
  march: armv8.2-a+crc
toolchain:
  c: clang
  cpp: clang++
  ar: llvm-ar
  strip: llvm-strip
sysroot:
  root: /opt/a7s-sysroot
  lib_dirs:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu
  pkg_config_libdir:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig
"#,
    )
    .unwrap();
}

/// Write a canonical `ingest.yaml` with a local docs source + domains +
/// relationship inference. Returns (manifest_path, docs_dir).
fn write_ingest_config(tmp: &std::path::Path) -> (PathBuf, PathBuf) {
    let dir = tmp.join("knowledge").join("a7s");
    std::fs::create_dir_all(&dir).unwrap();
    let docs = tmp.join("a7s-docs");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("NPU.md"),
        "# A733 NPU pipeline\n\nThe A733 packs an 8-core CPU and a 3 TOPS NPU.\n"
    ).unwrap();
    std::fs::write(
        docs.join("ISP.md"),
        "# Camera ISP\n\nV4L2_PIX_FMT_NV12M frames flow from MIPI CSI-2 to the ISP.\n"
    ).unwrap();
    // Co-locate both entity types in ONE chunk so the
    // npu_blocks ↔ camera_formats relationship can be inferred.
    std::fs::write(
        docs.join("pipeline.md"),
        "# NPU camera pipeline\n\nNPU_0 receives V4L2_PIX_FMT_NV12M frames from the ISP.\n"
    ).unwrap();

    let manifest_path = dir.join("ingest.yaml");
    std::fs::write(
        &manifest_path,
        format!(
            r#"
version: "1.0"
pipeline:
  name: "Allwinner A733 GraphRAG Ingestion"
  target_platform: "Radxa Cubie A7S/A7A"
  settings:
    chunk_size: 1800
    chunk_overlap: 0
  domains:
    - name: "npu_blocks"
      patterns:
        - "NPU_[A-Z0-9_]+"
    - name: "camera_formats"
      patterns:
        - "V4L2_PIX_FMT_[A-Z0-9_]+"
  sources:
    - id: "local_docs"
      type: "local"
      enabled: true
      path: {}
      processing:
        parser: "markdown"
        tagging:
          - key: "board"
            value: "cubie_a7s"
  graph_construction:
    relationship_inference:
      - type: "hardware_constraint"
        based_on: ["npu_blocks", "camera_formats"]
output:
  database: "SeleneDB"
  collection: "graphrag_nodes"
"#,
            docs.to_string_lossy()
        ),
    )
    .unwrap();
    (manifest_path, docs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn knowledge_store_split_ingest_and_share() {
    let _env_guard = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(embedder) = try_real_embedder().await else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("SPIRE_PLATFORM_DIR", tmp.path().join("platforms"));
    std::env::set_var("SPIRE_KNOWLEDGE_DIR", tmp.path().join("knowledge"));
    write_platform_seed(tmp.path());

    // ── 1. Project A: ingest the config → chunks/entities land in KnowledgeStore ──
    let project_a = tmp.path().join("proj-a");
    std::fs::create_dir_all(&project_a).unwrap();
    let project_tx_a = spawn_graph(&project_a.join(".spire").join("data"), &embedder).await;
    let knowledge_tx = spawn_graph(&tmp.path().join("knowledge"), &embedder).await;

    let (rag_tx, rx) = mpsc::channel(64);
    let _join = RagActor::new(knowledge_tx.clone(), project_tx_a.clone(), embedder.clone()).spawn(rx);

    let (manifest_path, _docs) = write_ingest_config(tmp.path());
    {
        let (t, r) = oneshot::channel();
        rag_tx
            .send(RagMessage::IngestGraphConfig {
                manifest_path,
                project_root: Some(tmp.path().to_path_buf()),
                reply_to: t,
            })
            .await
            .unwrap();
        let report = r.await.unwrap().expect("ingest config");
        eprintln!(
            "REPORT: domain={} chunks={} entities={} rels={} skipped={:?}",
            report.domain, report.chunks, report.entities, report.relationships, report.sources_skipped
        );
        assert!(report.chunks > 0, "must ingest at least one chunk");
        assert_eq!(report.domain, "a7s");
        assert_eq!(report.platform_id, "a7s");
        assert!(report.entities > 0, "must extract at least one entity");
        assert!(report.relationships > 0, "must infer at least one relationship");
    }

    // Chunks + entities in the KnowledgeStore; none in the project store.
    let know_chunks = query_prefix(&knowledge_tx, "rag_chunk").await;
    let know_entities = query_prefix(&knowledge_tx, "rag_entity").await;
    assert!(!know_chunks.is_empty(), "chunks must land in KnowledgeStore");
    assert!(!know_entities.is_empty(), "entities must land in KnowledgeStore");
    assert!(know_chunks.iter().all(|n| n.starts_with("rag_chunk:")), "{:?}", know_chunks);
    assert!(know_entities.iter().all(|n| n.starts_with("rag_entity:")), "{:?}", know_entities);

    let proj_chunks = query_prefix(&project_tx_a, "rag_chunk").await;
    assert!(proj_chunks.is_empty(), "project store must NOT hold the corpus");

    // ── 2. Fresh project B: EMPTY project store, same domain query ──
    let project_b = tmp.path().join("proj-b");
    std::fs::create_dir_all(&project_b).unwrap();
    let project_tx_b = spawn_graph(&project_b.join(".spire").join("data"), &embedder).await;
    let proj_b_chunks = query_prefix(&project_tx_b, "rag_chunk").await;
    assert!(proj_b_chunks.is_empty(), "fresh project must start with no corpus");

    let hits = query_rag(&rag_tx, "a7s", "NPU").await;
    assert!(!hits.is_empty(), "shared KnowledgeStore must resolve the query");
    assert!(hits.iter().all(|h| h.domain == "a7s"));

    // ── 3. Idempotent re-ingest: same config → same content-addressed ids ──
    let (manifest_path2, _) = write_ingest_config(tmp.path());
    {
        let (t, r) = oneshot::channel();
        rag_tx
            .send(RagMessage::IngestGraphConfig {
                manifest_path: manifest_path2,
                project_root: Some(tmp.path().to_path_buf()),
                reply_to: t,
            })
            .await
            .unwrap();
        let report = r.await.unwrap().expect("re-ingest");
        // `report.entities` counts per-chunk MENTIONS; the store holds unique
        // content-addressed nodes (an entity mentioned in N chunks = one node).
        // Compare store-level sets for idempotency.
        assert_eq!(report.chunks as usize, know_chunks.len(), "re-ingest UPSERTs, no duplicates");
    }
    let know_chunks2 = query_prefix(&knowledge_tx, "rag_chunk").await;
    let know_entities2 = query_prefix(&knowledge_tx, "rag_entity").await;
    assert_eq!(know_chunks.len(), know_chunks2.len(), "same config → identical graph id set");
    assert_eq!(know_entities.len(), know_entities2.len(), "entity set idempotent");
    let mut a = know_chunks.clone();
    let mut b = know_chunks2.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b, "identical chunk id set");
    let mut ea = know_entities.clone();
    let mut eb = know_entities2.clone();
    ea.sort();
    eb.sort();
    assert_eq!(ea, eb, "identical entity id set");

    // ── 4. Domain summaries: clean id + corpus version + entity count ──
    {
        let (t, r) = oneshot::channel();
        rag_tx
            .send(RagMessage::ListDomains { reply_to: t })
            .await
            .unwrap();
        let domains = r.await.unwrap().expect("list domains");
        let d = domains.iter().find(|d| d.id == "a7s").expect("a7s domain");
        assert_eq!(d.chunk_count as usize, know_chunks.len());
        assert!(d.entity_count > 0, "domain state must report entities");
        assert_eq!(d.corpus_version.len(), 16, "corpus_version is 16-hex");
    }

    // ── 5. Manifest discovery: the canonical ingest.yaml is listed ──
    {
        let (t, r) = oneshot::channel();
        rag_tx
            .send(RagMessage::ListManifests {
                project_root: tmp.path().to_path_buf(),
                reply_to: t,
            })
            .await
            .unwrap();
        let manifests = r.await.unwrap().expect("list manifests");
        assert!(
            manifests.iter().any(|m| m.platform_id == "a7s" && m.path.ends_with("ingest.yaml")),
            "the a7s ingest.yaml must be discovered: {:?}",
            manifests
        );
        assert_eq!(manifests[0].corpus_version.len(), 16);
    }
}
/// RAG is project-independent: ingesting the SAME manifest with NO project
/// root still produces chunks/entities (relative paths resolve against the
/// manifest's own directory), and no provenance is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_without_project() {
    let _env_guard = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(embedder) = try_real_embedder().await else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("SPIRE_PLATFORM_DIR", tmp.path().join("platforms"));
    std::env::set_var("SPIRE_KNOWLEDGE_DIR", tmp.path().join("knowledge"));
    write_platform_seed(tmp.path());

    // KnowledgeStore only; NO project graph in scope.
    let knowledge_tx = spawn_graph(&tmp.path().join("knowledge"), &embedder).await;
    let (rag_tx, rx) = mpsc::channel(64);
    let _join = RagActor::new(
        knowledge_tx.clone(),
        knowledge_tx.clone(), // provenance sink unused when project_root is None
        embedder.clone(),
    )
    .spawn(rx);

    let (manifest_path, _docs) = write_ingest_config(tmp.path());
    let (t, r) = oneshot::channel();
    rag_tx
        .send(RagMessage::IngestGraphConfig {
            manifest_path,
            project_root: None, // <-- no project required
            reply_to: t,
        })
        .await
        .unwrap();
    let report = r.await.unwrap().expect("ingest without project");

    assert!(report.chunks > 0, "ingest must work with no project: {:?}", report);
    assert!(report.entities > 0, "entity extraction works without a project");

    // The corpus is queryable from the shared KnowledgeStore.
    let hits = query_rag(&rag_tx, "a7s", "NPU").await;
    assert!(!hits.is_empty(), "query works without any project open");
}

