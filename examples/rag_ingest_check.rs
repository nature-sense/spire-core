// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Throwaway verification: ingest a corpus' REAL installed manifest (under
//! `config::knowledge_dir()`, so the app's own store) against the REAL KnowledgeStore and report what
//! actually lands.
//! Run: `cargo run --example rag_ingest_check -p spire-core [corpus]` (default `a7s`)

use std::sync::Arc;

use spire_core::actors::rag_ingest::{self, IngestContext};
use spire_core::actors::Actor;
use spire_core::models::embedding::Embedder;
use spire_core::subsystems::graph::memory_graph::MemoryGraphActor;
use tokio::sync::{mpsc, oneshot};

#[tokio::main]
async fn main() {
    // The app's own store: per application, and `SPIRE_KNOWLEDGE_DIR` when set — see `rag_domain_check`
    // for why a hand-built `~/.spire/knowledge` is the wrong directory to read.
    let knowledge = spire_core::config::knowledge_dir();
    // Optional argument: the corpus whose *installed* manifest to ingest (default `a7s`). Each corpus
    // has its own `ingest.yaml` under the store, so picking one is a directory name, not a path —
    // `cargo run --example rag_ingest_check -p spire-core spire-idf`.
    let corpus = std::env::args().nth(1).unwrap_or_else(|| "a7s".to_string());
    let manifest = knowledge.join(&corpus).join("ingest.yaml");
    println!("knowledge dir : {}", knowledge.display());
    println!("corpus         : {corpus}");
    println!(
        "manifest       : {} exists={}",
        manifest.display(),
        manifest.exists()
    );
    if !manifest.exists() {
        eprintln!(
            "no manifest at {} — pass a corpus directory under {}",
            manifest.display(),
            knowledge.display()
        );
        std::process::exit(1);
    }
    // The domain the manifest targets (`pipeline.corpus`), which is not the directory name: the QA
    // `ListSources` below has to ask for the domain the ingest actually wrote.
    let domain = rag_ingest::resolve_domain_for_manifest(&manifest)
        .unwrap_or_else(|_| corpus.clone());

    let (tx, rx) = mpsc::channel(64);
    let _join = MemoryGraphActor::new().spawn(rx);
    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::Initialize {
        data_dir: knowledge.clone(),
        reply_to: t,
    })
    .await
    .unwrap();
    r.await.unwrap().unwrap();
    // The REAL embedder (sentence-transformers/all-MiniLM-L6-v2), shared by the graph, the ingest
    // context and the RagActor. It must NOT be `NoopEmbedder`: the pipeline embeds every chunk, so a
    // no-op embedder makes *every* ingest fail with "Embedding model unavailable" — nothing lands.
    let real = match spire_core::embedder::create_embedder() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot load the embedding model: {e:#}");
            eprintln!("run `cargo run --example cache_embedding_model -p spire-core` once first");
            std::process::exit(1);
        }
    };
    let embedder: Arc<dyn Embedder> = real;

    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::InitializeEmbedder {
        model_path: None,
        embedder: Some(embedder.clone()),
        reply_to: t,
    })
    .await
    .unwrap();
    r.await.unwrap().unwrap();

    let ctx = IngestContext {
        knowledge_tx: tx.clone(),
        memory_graph_tx: tx.clone(),
        embedder: embedder.clone(),
    };
    use spire_core::actors::rag::{RagActor, RagMessage};
    let (rag_tx, rag_rx) = mpsc::channel(64);
    let _rag_join = RagActor::new(tx.clone(), tx.clone(), embedder.clone()).spawn(rag_rx);

    let report = rag_ingest::ingest_graph_config(&ctx, &manifest, None).await;
    match report {
        Ok(r) => {
            println!(
                "INGEST OK domain={} chunks={} entities={} rels={} skipped={:?}",
                r.domain, r.chunks, r.entities, r.relationships, r.sources_skipped
            );
            let (t, r) = oneshot::channel();
            tx.send(spire_core::actors::MemoryGraphMessage::QueryAttrNodes {
                node_type: Some("Unknown".to_string()),
                subtype: Some("rag_entity".to_string()),
                name: None,
                limit: Some(1000),
                reply_to: t,
            })
            .await
            .unwrap();
            let nodes = r.await.unwrap().unwrap();
            println!("total rag_entity nodes in store: {}", nodes.len());
            let mut types: std::collections::BTreeMap<String, usize> = Default::default();
            for n in &nodes {
                if let Some(et) = n.get("entity_type").and_then(|v| v.as_str()) {
                    *types.entry(et.to_string()).or_insert(0) += 1;
                }
            }
            for (k, v) in types {
                println!("  {k}: {v}");
            }
        }
        Err(e) => println!("INGEST FAILED: {e}"),
    }

    // QA: rag/list-sources must return the persisted per-source status.
    {
        let (t, r) = oneshot::channel();
        rag_tx
            .send(RagMessage::ListSources {
                domain: domain.clone(),
                reply_to: t,
            })
            .await
            .unwrap();
        match r.await {
            Ok(Ok(sources)) => {
                println!("LIST_SOURCES count={} :", sources.len());
                for s in &sources {
                    println!(
                        "  {} status={} chunks={} files={} reason={}",
                        s.id, s.status, s.chunks, s.files, s.reason
                    );
                }
            }
            Ok(Err(e)) => println!("LIST_SOURCES error: {e}"),
            Err(e) => println!("LIST_SOURCES lost: {e}"),
        }
    }
}
