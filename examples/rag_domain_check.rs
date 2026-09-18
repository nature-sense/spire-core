// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! What a KnowledgeStore actually holds, per domain — read-only, from a fresh process.
//!
//! `rag_ingest_check` (the sibling example) *writes*: it ingests the real `a7s` manifest and reports
//! what landed. This one only reads, which is what makes it useful after a fill or a restart: it opens
//! `~/.spire/knowledge` the way the app does, counts the corpus state per domain, and prints it. If a
//! fill has not survived (a WAL the next process discarded, a store the wrong process wrote), the
//! numbers are missing here and nowhere else.
//!
//! Run: `cargo run --release --example rag_domain_check -p spire-core [domain…]`
//!
//! **Quit the app first.** The KnowledgeStore's `Initialize` takes the store for read-write, and the
//! app's own startup treats any WAL it finds as stale — one process at a time.
//!
//! Two counts are worth not confusing: `chunks` is what retrieval can see, and `src_rows` is the
//! number of configured **sources** with a status row (`rag_source`), one per entry in the manifest —
//! not the number of distinct paths the chunks came from, which is what `rag/list-domains` reports.

use tokio::sync::{mpsc, oneshot};

use spire_core::actors::Actor;
use spire_core::actors::MemoryGraphMessage as MgMsg;
use spire_core::models::memory_graph::AttrNode;
use spire_core::subsystems::graph::memory_graph::MemoryGraphActor;

#[tokio::main]
async fn main() {
    let store = dirs::home_dir()
        .expect("home dir")
        .join(".spire")
        .join("knowledge");
    println!("KnowledgeStore: {}", store.display());

    let (tx, rx) = mpsc::channel(64);
    let _join = MemoryGraphActor::new().spawn(rx);
    let (t, r) = oneshot::channel();
    tx.send(MgMsg::Initialize {
        data_dir: store,
        reply_to: t,
    })
    .await
    .expect("send Initialize");
    r.await.expect("reply").expect("Initialize");

    // The two subtypes every domain writes: the domain row itself, and its chunks.
    let mut domains: Vec<(String, String)> = Vec::new();
    for node in query(&tx, "rag_domain").await {
        let name = node.name().to_string();
        let id = name
            .strip_prefix("rag_domain:")
            .unwrap_or(&name)
            .to_string();
        let description = node
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        domains.push((id, description));
    }
    let chunks = query(&tx, "rag_chunk").await;
    let sources = query(&tx, "rag_source").await;
    let entities = query(&tx, "rag_entity").await;

    // The filter the argument list applies, so a single domain can be inspected.
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    let keep = |domain: &str| wanted.is_empty() || wanted.iter().any(|w| w == domain);

    let mut total = 0usize;
    println!(
        "{:<18} {:>7} {:>8} {:>9}",
        "domain", "chunks", "src_rows", "entities"
    );
    for (id, description) in &domains {
        if !keep(id) {
            continue;
        }
        let count = |nodes: &[AttrNode]| {
            nodes
                .iter()
                .filter(|n| n.get("domain").and_then(|v| v.as_str()) == Some(id.as_str()))
                .count()
        };
        let c = count(&chunks);
        total += c;
        println!(
            "{:<18} {:>7} {:>8} {:>9}   {}",
            id,
            c,
            count(&sources),
            count(&entities),
            description
        );
    }
    println!(
        "{:<18} {:>7}  ({} domains listed)",
        "TOTAL",
        total,
        domains.len()
    );
    if total == 0 {
        eprintln!("nothing found — wrong store, or a fill that did not survive");
        std::process::exit(1);
    }
}

/// All nodes of one subtype, via the open `AttrNode` envelope.
async fn query(tx: &mpsc::Sender<MgMsg>, subtype: &str) -> Vec<AttrNode> {
    let (t, r) = oneshot::channel();
    tx.send(MgMsg::QueryAttrNodes {
        node_type: Some("Unknown".to_string()),
        subtype: Some(subtype.to_string()),
        name: None,
        limit: Some(1_000_000),
        reply_to: t,
    })
    .await
    .expect("send QueryAttrNodes");
    r.await.expect("reply").expect("query")
}
