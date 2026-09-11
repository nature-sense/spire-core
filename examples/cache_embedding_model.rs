// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Download + cache the shared embedding model and validate a real inference.
//!
//! Populates the HF hub cache used by every Spire app:
//!   ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
//! (falling back to a bundled `<exe_dir>/models/all-MiniLM-L6-v2/` dir first,
//! which this example will also pick up if present).
//!
//! Run:
//!   cargo run --release --example cache_embedding_model -p spire-core
//!
//! Exits 0 only when the model loads and produces a valid 384-d embedding.

use spire_core::embedder::create_embedder;

fn main() {
    let started = std::time::Instant::now();
    match create_embedder() {
        Ok(embedder) => {
            println!(
                "model loaded in {:.1}s (metal={})",
                started.elapsed().as_secs_f64(),
                embedder.device_is_metal()
            );
            match embedder.embed_text("Singapore's forested nature reserves and parks") {
                Ok(vec) => {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    println!("dims={} norm={:.4} first={} last={}", vec.len(), norm, vec.first().copied().unwrap_or(0.0), vec.last().copied().unwrap_or(0.0));
                    if vec.len() != 384 {
                        println!("CACHE-FAIL unexpected dimensions");
                        std::process::exit(1);
                    }
                    let home = std::env::var("HOME").unwrap_or_default();
                    println!(
                        "cache: {}/.cache/huggingface/models/all-MiniLM-L6-v2",
                        home
                    );
                    println!("CACHE-OK");
                }
                Err(e) => {
                    println!("CACHE-FAIL inference: {e}");
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            println!("CACHE-FAIL load: {e:#}");
            std::process::exit(1);
        }
    }
}
