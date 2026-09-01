// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! FileWatcherActor — initial project scan + continuous file event watching.
//!
//! This actor is the single point of truth for "what files exist or changed".
//! It combines two complementary responsibilities so that both the bootstrap
//! scan and the incremental change sync feed the same downstream pipeline
//! (AST extraction, project analysis, graph sync):
//!
//! 1. **Initial scan** (`StartWatching`) — calls `scanner::scan_directory` +
//!    `scanner::discover_build_files` and emits a `FileChangeNotification::InitialScan`
//!    containing every file and build config discovered.
//! 2. **Continuous watch** — after the scan completes, starts a
//!    `notify::RecommendedWatcher`. Raw FS events flow through an unbounded
//!    internal channel into a debounce task that emits
//!    `FileChangeNotification::Batch` every 500ms.
//!
//! The actor is intentionally **not wired** into the system yet — consumers
//! attach by sending `StartWatching` with an output channel.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

use crate::actors::Actor;
use crate::analyzer::models::FileInfo;
use crate::analyzer::scanner;

/// Debounce window for batching raw filesystem events.
const DEBOUNCE_MS: u64 = 500;

// ============================================================================
// Output Types
// ============================================================================

/// Classified change kind derived from a `notify::EventKind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileChangeKind {
    Create,
    Modify,
    Remove,
    Rename,
    Other(String),
}

impl FileChangeKind {
    fn from_notify(kind: &notify::EventKind) -> Self {
        use notify::EventKind::*;
        match kind {
            Create(_) => FileChangeKind::Create,
            Modify(_) => FileChangeKind::Modify,
            Remove(_) => FileChangeKind::Remove,
            _ => FileChangeKind::Other(format!("{kind:?}")),
        }
    }
}

/// A single filesystem event observed by the watcher.
#[derive(Debug, Clone)]
pub struct FileEventInfo {
    pub path: PathBuf,
    pub kind: FileChangeKind,
}

impl FileEventInfo {
    fn from_notify(event: &Event) -> Vec<Self> {
        event
            .paths
            .iter()
            .map(|path| FileEventInfo {
                path: path.clone(),
                kind: FileChangeKind::from_notify(&event.kind),
            })
            .collect()
    }
}

/// A debounced batch of events, emitted after each 500ms quiet window.
#[derive(Debug, Clone)]
pub struct FileEventBatch {
    pub events: Vec<FileEventInfo>,
    pub batch_id: u64,
    pub timestamp: DateTime<Utc>,
}

/// Notifications emitted by the FileWatcherActor to its downstream consumer.
#[derive(Debug, Clone)]
pub enum FileChangeNotification {
    /// Emitted once after `StartWatching` completes the initial scan.
    InitialScan {
        root: PathBuf,
        files: Vec<FileInfo>,
        build_configs: Vec<(String, String)>,
    },
    /// Emitted after each debounce window with the changes since the last batch.
    Batch { batch: FileEventBatch },
}

// ============================================================================
// Message Protocol
// ============================================================================

/// Messages for the FileWatcherActor.
#[derive(Debug)]
pub enum FileWatcherMessage {
    /// Scan the root once, then begin watching it for changes.
    /// The output channel receives `InitialScan` immediately, then
    /// `Batch` notifications as files change.
    StartWatching {
        root: PathBuf,
        output: mpsc::Sender<FileChangeNotification>,
        reply_to: oneshot::Sender<Result<(), String>>,
    },
    /// Stop the background watcher + debounce task (keeps the actor alive).
    StopWatching,
    /// Stop everything and end the actor task.
    Shutdown,
}

// ============================================================================
// Actor
// ============================================================================

/// Wraps `notify::RecommendedWatcher` and the scanner, producing the
/// `InitialScan` + debounced `Batch` notifications described above.
pub struct FileWatcherActor {
    root: Option<PathBuf>,
    watcher: Option<RecommendedWatcher>,
    watch_task: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl FileWatcherActor {
    pub fn new() -> Self {
        Self {
            root: None,
            watcher: None,
            watch_task: None,
            shutdown_tx: None,
        }
    }

    /// Perform the initial scan and start the continuous watcher.
    fn start_watching(
        &mut self,
        root: PathBuf,
        output: mpsc::Sender<FileChangeNotification>,
    ) -> Result<(), String> {
        // Stop any previous watcher before re-starting.
        self.stop_watching();

        // 1. Initial scan — same ignore/exclude rules as ProjectSync bootstrap.
        let files = scanner::scan_directory(&root, false);
        let build_configs = scanner::discover_build_files(&root, false);
        let out = output.clone();
        let scan_root = root.clone();
        let _ = tokio::spawn(async move {
            // Deliver the initial scan outside the actor's mailbox so the
            // actor can immediately start watching.
            let _ = out
                .send(FileChangeNotification::InitialScan {
                    root: scan_root,
                    files,
                    build_configs,
                })
                .await;
        });

        // 2. Continuous watch — bridge notify events into an unbounded channel.
        let (event_tx, event_rx) = mpsc::unbounded_channel::<FileEventInfo>();
        let bridge_tx = event_tx.clone();
        let mut watcher = RecommendedWatcher::new(
            move |res: notify::Result<Event>| {
                if let Ok(event) = res {
                    for info in FileEventInfo::from_notify(&event) {
                        let _ = bridge_tx.send(info);
                    }
                }
            },
            notify::Config::default(),
        )
        .map_err(|e| format!("Failed to create file watcher: {e}"))?;

        watcher
            .watch(&root, RecursiveMode::Recursive)
            .map_err(|e| format!("Failed to watch {}: {e}", root.display()))?;

        // 3. Debounce task — collects raw events and emits batches.
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let out = output.clone();
        let handle = tokio::spawn(Self::watch_loop(event_rx, out, shutdown_rx));

        self.root = Some(root);
        self.watcher = Some(watcher);
        self.watch_task = Some(handle);
        self.shutdown_tx = Some(shutdown_tx);

        Ok(())
    }

    /// Stop the watcher + debounce task, keeping the actor alive.
    fn stop_watching(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.watch_task.take() {
            handle.abort();
        }
        self.watcher = None;
        self.root = None;
    }

    /// Debounced event-collection loop. Waits for the first event, then keeps
    /// draining for `DEBOUNCE_MS` and emits one batch per quiet window.
    async fn watch_loop(
        mut rx: mpsc::UnboundedReceiver<FileEventInfo>,
        output: mpsc::Sender<FileChangeNotification>,
        mut shutdown: oneshot::Receiver<()>,
    ) {
        let mut buffer: Vec<FileEventInfo> = Vec::new();
        let mut batch_id = 0u64;

        loop {
            // Wait for the first event of a batch.
            let first = tokio::select! {
                ev = rx.recv() => ev,
                _ = &mut shutdown => return,
            };
            let Some(first) = first else { return };
            buffer.push(first);

            // Debounce window: keep collecting until 500ms of quiet.
            loop {
                tokio::select! {
                    ev = rx.recv() => match ev {
                        Some(e) => buffer.push(e),
                        None => return,
                    },
                    _ = tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)) => break,
                    _ = &mut shutdown => return,
                }
            }

            let batch = FileEventBatch {
                events: std::mem::take(&mut buffer),
                batch_id,
                timestamp: Utc::now(),
            };
            if output.send(FileChangeNotification::Batch { batch }).await.is_err() {
                // Consumer gone — stop.
                return;
            }
            batch_id += 1;
        }
    }
}

impl Default for FileWatcherActor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Actor for FileWatcherActor {
    type Message = FileWatcherMessage;

    async fn handle(&mut self, msg: Self::Message) {
        match msg {
            FileWatcherMessage::StartWatching {
                root,
                output,
                reply_to,
            } => {
                let result = self.start_watching(root, output);
                let _ = reply_to.send(result);
            }
            FileWatcherMessage::StopWatching => {
                self.stop_watching();
            }
            FileWatcherMessage::Shutdown => {
                self.stop_watching();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration as StdDuration;

    /// Helper that spawns one persistent actor and starts watching a dir.
    /// The returned sender keeps the actor (and its watcher) alive.
    async fn spawn_watching(
        root: &std::path::Path,
    ) -> (
        mpsc::Sender<FileWatcherMessage>,
        mpsc::Receiver<FileChangeNotification>,
    ) {
        let (actor_tx, mut actor_rx) = mpsc::channel::<FileWatcherMessage>(8);
        tokio::spawn(async move {
            let mut actor = FileWatcherActor::new();
            while let Some(msg) = actor_rx.recv().await {
                actor.handle(msg).await;
            }
        });

        let (out_tx, out_rx) = mpsc::channel::<FileChangeNotification>(16);
        let (t, r) = oneshot::channel();
        actor_tx
            .send(FileWatcherMessage::StartWatching {
                root: root.to_path_buf(),
                output: out_tx,
                reply_to: t,
            })
            .await
            .unwrap();
        r.await.unwrap().unwrap();
        (actor_tx, out_rx)
    }

    #[tokio::test]
    async fn initial_scan_reports_files_and_build_configs() {
        let tmp = tempfile::tempdir().unwrap();
        // Use a non-hidden project subdir — tempfile's root is `.tmpXXXX`
        // (hidden), which discover_build_files' filter_entry skips.
        let project = tmp.path().join("project");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::File::create(project.join("Cargo.toml"))
            .unwrap()
            .write_all(b"[package]\nname = \"demo\"\n")
            .unwrap();
        std::fs::File::create(project.join("src/main.rs"))
            .unwrap()
            .write_all(b"fn main() {}")
            .unwrap();

        let (_actor, mut out) = spawn_watching(&project).await;

        let notification = tokio::time::timeout(StdDuration::from_secs(5), out.recv())
            .await
            .expect("timed out waiting for InitialScan")
            .expect("channel closed");

        match notification {
            FileChangeNotification::InitialScan {
                files,
                build_configs,
                ..
            } => {
                assert!(
                    files.iter().any(|f| f.relative_path.ends_with("Cargo.toml")),
                    "expected Cargo.toml in scan, got: {:?}",
                    files.iter().map(|f| &f.relative_path).collect::<Vec<_>>()
                );
                assert!(
                    files.iter().any(|f| f.relative_path.ends_with("src/main.rs")),
                    "expected src/main.rs in scan"
                );
                assert!(
                    build_configs.iter().any(|(p, _)| p.ends_with("Cargo.toml")),
                    "expected Cargo.toml as a build config"
                );
            }
            other => panic!("expected InitialScan, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn meson_platform_subdir_with_project_comment_is_not_a_subproject() {
        // Regression: a platform subdir meson.build (e.g. rpi5/meson.build) that
        // references `project(...)` ONLY inside a comment must NOT be treated as
        // a separate buildable subproject. Only the root meson.build (with a real
        // project() call) should be discovered.
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("project");
        std::fs::create_dir_all(project.join("rpi5")).unwrap();

        // Root meson.build — the ONLY project() in the tree.
        std::fs::File::create(project.join("meson.build"))
            .unwrap()
            .write_all(
                b"project('ai-traps', ['c', 'cpp'])\nsubdir('rpi5')\n",
            )
            .unwrap();

        // Platform subdir meson.build — NO project() call, but the comment
        // contains the substring `project(...)` which used to fool the scanner.
        std::fs::File::create(project.join("rpi5/meson.build"))
            .unwrap()
            .write_all(
                b"# No project() call here \xe2\x80\x94 the root project('ai-traps') is the ONLY one.\nexecutable('ai-trap-rpi5', 'main.cpp')\n",
            )
            .unwrap();

        let found = scanner::discover_build_files(&project, false);
        let configs: Vec<&str> = found.iter().map(|(p, _)| p.as_str()).collect();

        assert_eq!(
            configs,
            vec!["meson.build"],
            "expected only the root meson.build, got: {configs:?}"
        );
    }

    #[tokio::test]
    async fn watch_reports_write_event() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("project");
        std::fs::create_dir_all(&project).unwrap();
        let (_actor, mut out) = spawn_watching(&project).await;

        // Consume the initial scan.
        let _ = tokio::time::timeout(StdDuration::from_secs(5), out.recv())
            .await
            .expect("no InitialScan")
            .expect("channel closed");

        // Write a file inside the watched root → expect a debounced batch.
        let target = project.join("hello.txt");
        std::fs::File::create(&target)
            .unwrap()
            .write_all(b"hello")
            .unwrap();

        let notification = tokio::time::timeout(StdDuration::from_secs(5), out.recv())
            .await
            .expect("timed out waiting for watch batch")
            .expect("channel closed");

        // macOS resolves /var → /private/var in notify paths; canonicalize to
        // compare robustly.
        let canonical_target = target
            .canonicalize()
            .unwrap_or_else(|_| target.clone());

        match notification {
            FileChangeNotification::Batch { batch } => {
                assert!(
                    batch.events.iter().any(|e| {
                        e.path
                            .canonicalize()
                            .map(|p| p == canonical_target)
                            .unwrap_or(false)
                    }),
                    "expected event for {} in batch: {:?}",
                    canonical_target.display(),
                    batch.events
                );
            }
            other => panic!("expected Batch, got {other:?}"),
        }
    }
}