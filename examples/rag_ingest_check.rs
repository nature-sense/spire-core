// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Throwaway verification: ingest the REAL ~/.spire/knowledge/a7s/ingest.yaml
//! against the REAL KnowledgeStore and report what actually lands.
//! Run: `cargo run --example rag_ingest_check -p spire-core`

use std::sync::Arc;

use spire_core::actors::rag_ingest::{self, IngestContext};
use spire_core::actors::Actor;
use spire_core::embedder::NoopEmbedder;
use spire_core::models::embedding::Embedder;
use spire_core::subsystems::graph::memory_graph::MemoryGraphActor;
use tokio::sync::{mpsc, oneshot};

#[tokio::main]
async fn main() {
    let knowledge = dirs::home_dir().unwrap().join(".spire/knowledge");
    let manifest = knowledge.join("a7s/ingest.yaml");
    println!("knowledge dir : {}", knowledge.display());
    println!(
        "manifest       : {} exists={}",
        manifest.display(),
        manifest.exists()
    );

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
    let (t, r) = oneshot::channel();
    tx.send(spire_core::actors::MemoryGraphMessage::InitializeEmbedder {
        model_path: None,
        embedder: Some(Arc::new(NoopEmbedder) as Arc<dyn Embedder>),
        reply_to: t,
    })
    .await
    .unwrap();
    r.await.unwrap().unwrap();

    let ctx = IngestContext {
        knowledge_tx: tx.clone(),
        memory_graph_tx: tx.clone(),
        embedder: Arc::new(NoopEmbedder),
    };
    use spire_core::actors::rag::{RagActor, RagMessage};
    let (rag_tx, rag_rx) = mpsc::channel(64);
    let _rag_join = RagActor::new(tx.clone(), tx.clone(), Arc::new(NoopEmbedder)).spawn(rag_rx);

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
                domain: "a7s".to_string(),
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
