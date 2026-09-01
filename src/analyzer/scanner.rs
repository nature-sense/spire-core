// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Filesystem scanner — walks a directory tree and collects file metadata.
//!
//! Two scanning modes:
//! - **Standard** (default): Uses the `ignore` crate to respect `.gitignore`.
//! - **No-ignore**: Uses `walkdir` for a simple recursive listing, skipping
//!   only hidden directories and known non-project directories.

use std::collections::HashMap;
use std::path::Path;

use ignore::WalkBuilder;
use walkdir::WalkDir;

use crate::analyzer::models::FileInfo;

/// Known build config file names used for Stage 1 discovery.
/// These are the primary build manifests that define a project root.
pub const BUILD_CONFIG_FILES: &[&str] = &[
    "Cargo.toml",
    "package.json",
    "pnpm-workspace.yaml",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "go.mod",
    "build.gradle",
    "build.gradle.kts",
    "pom.xml",
    "CMakeLists.txt",
    "Makefile",
    "Gemfile",
    "Package.swift",
];

/// Directories that should always be skipped during discovery (e.g. dependencies, build output).
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    ".pnpm",
    "target",
    "dist",
    "build",
    "out",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".eggs",
    "eggs",
];

/// Scan a directory and return all files, respecting .gitignore.
pub fn scan_directory(root: &Path, no_ignore: bool) -> Vec<FileInfo> {
    let mut files = Vec::new();

    if no_ignore {
        // Use walkdir for simple recursive listing without .gitignore
        for entry in WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| !is_hidden(e.file_name()))
        {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let relative = path
                        .strip_prefix(root)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    // Skip common non-project directories
                    if should_skip(&relative) {
                        continue;
                    }

                    let ext = path
                        .extension()
                        .map(|e| format!(".{}", e.to_string_lossy()))
                        .unwrap_or_default();

                    let ft = entry.file_type();
                    files.push(FileInfo {
                        path: path.to_string_lossy().to_string(),
                        relative_path: relative,
                        extension: ext,
                        size: entry.metadata().map(|m| m.len()).unwrap_or(0),
                        is_dir: ft.is_dir(),
                        is_symlink: ft.is_symlink(),
                    });
                }
                Err(_) => continue,
            }
        }
    } else {
        // Use the `ignore` crate which respects .gitignore
        let walker = WalkBuilder::new(root)
            .standard_filters(true)
            .follow_links(false)
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    let relative = path
                        .strip_prefix(root)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .to_string();

                    if should_skip(&relative) {
                        continue;
                    }

                    let ext = path
                        .extension()
                        .map(|e| format!(".{}", e.to_string_lossy()))
                        .unwrap_or_default();

                    let meta = entry.metadata().ok();
                    let is_dir = entry
                        .file_type()
                        .map(|ft| ft.is_dir())
                        .unwrap_or_else(|| meta.as_ref().map(|m| m.is_dir()).unwrap_or(false));
                    let is_symlink = entry.file_type().map(|ft| ft.is_symlink()).unwrap_or(false);
                    files.push(FileInfo {
                        path: path.to_string_lossy().to_string(),
                        relative_path: relative,
                        extension: ext,
                        size: meta.map(|m| m.len()).unwrap_or(0),
                        is_dir,
                        is_symlink,
                    });
                }
                Err(_) => continue,
            }
        }
    }

    files
}

/// Stage 1: Walk the directory tree looking for build config files.
/// Returns a list of (relative_path_to_build_file, parent_dir_relative) pairs.
/// The parent_dir is the directory containing the build file, relative to root.
pub fn discover_build_files(root: &Path, _no_ignore: bool) -> Vec<(String, String)> {
    let mut results = Vec::new();

    // Use a filtering walker that skips known non-project directories
    let walker = WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Skip hidden dirs
            if name.starts_with('.') {
                return false;
            }
            // Skip known non-project dirs, plus any build-output directory
            // (build, build-native, build-* ...) so Meson/CMake generated
            // artifacts are never mistaken for real projects.
            if e.file_type().is_dir() {
                let name_str = name.as_ref();
                if SKIP_DIRS.contains(&name_str) || name_str.starts_with("build") {
                    return false;
                }
            }
            true
        });

    for entry in walker {
        match entry {
            Ok(entry) => {
                if !entry.file_type().is_file() {
                    continue;
                }
                let path = entry.path();
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();

                // Check if this filename matches a known build config. The
                // authoritative list lives in spire-code (where the build
                // modules are defined) — adding a module never requires editing
                // this core crate.
                let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if crate::build_types::all_config_file_names().iter().any(|c| c == &filename) {
                    // A `meson.build` WITHOUT a `project(...)` call is NOT a
                    // buildable project — it's a platform module / subdir spec
                    // (e.g. `rpi/hal/meson.build` in the ai-traps cross-platform
                    // layout) that the master rpi build includes. Don't spawn a
                    // standalone subproject node for it; its sources/deps belong
                    // to the owning project (`rpi`).
                    if filename == "meson.build" {
                        let content = std::fs::read_to_string(path).unwrap_or_default();
                        // Strip Meson line comments (`#`) before checking for
                        // `project(` — a comment like "the root project('x')"
                        // must not be treated as an actual project declaration.
                        // Mirrors the Meson build module's own comment stripping.
                        let comment_re = regex::Regex::new(r"(?m)#.*$").unwrap();
                        let stripped = comment_re.replace_all(&content, "");
                        if !stripped.contains("project(") {
                            continue;
                        }
                    }
                    // A `Cargo.toml` inside a Cargo WORKSPACE MEMBER crate is
                    // NOT an independent project — the workspace root is the
                    // single project, and Spire's multi-platform scaffold emits
                    // exactly one member `Cargo.toml` per platform leaf
                    // (`core/`, `rpi5/`, `rock3c/`). Treating each member as a
                    // standalone build system previously created a duplicate
                    // "subproject" per Cargo.toml on top of the directory
                    // subprojects (6 entries instead of 3). Mirrors the
                    // meson.build precedent: a member manifest is the platform
                    // module of the owning workspace root.
                    if filename == "Cargo.toml" && is_cargo_workspace_member(path) {
                        continue;
                    }
                    // Get the parent directory
                    let parent = Path::new(&relative)
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| ".".to_string());
                    results.push((relative, parent));
                }
            }
            Err(_) => continue,
        }
    }

    results
}

/// Create a generic, non-parsing `BuildMetadata` for a build config file.
/// This is used by `ProjectSyncActor` to create graph nodes — it performs
/// NO language/build-system-specific parsing. Rich analysis is delegated to
/// the appropriate MCP server via `ProjectAnalyzerActor`.
pub fn generic_build_metadata(filename: &str) -> crate::analyzer::models::BuildMetadata {
    let build_system = build_system_from_filename(filename);
    crate::analyzer::models::BuildMetadata {
        build_system,
        project_type: "unknown".to_string(),
        project_name: None,
        version: None,
        is_workspace: false,
        config_files: vec![filename.to_string()],
        ..Default::default()
    }
}

/// Infer a basic build-system label from a build config filename.
/// Language-specific analysis is delegated to MCP servers; this is only a
/// generic filename→system mapping used for graph node labels (no parsing).
pub fn build_system_from_filename(filename: &str) -> String {
    match filename {
        "Cargo.toml" => "Cargo".to_string(),
        "Package.swift" => "SwiftPM".to_string(),
        "package.json" => "npm".to_string(),
        "pom.xml" => "Maven".to_string(),
        "build.gradle" | "build.gradle.kts" | "settings.gradle" => "Gradle".to_string(),
        "Makefile" | "makefile" => "Make".to_string(),
        "CMakeLists.txt" => "CMake".to_string(),
        "meson.build" => "Meson".to_string(),
        "go.mod" => "Go".to_string(),
        "pyproject.toml" => "Python".to_string(),
        "setup.py" | "setup.cfg" => "Python".to_string(),
        _ => "Unknown".to_string(),
    }
}

/// Group files by their top-level directory.
/// Root-level files (no '/' in path) are grouped under ".".
pub fn group_by_top_dir(files: &[FileInfo]) -> HashMap<String, Vec<&FileInfo>> {
    let mut groups: HashMap<String, Vec<&FileInfo>> = HashMap::new();

    for file in files {
        let top = if file.relative_path.contains('/') {
            file.relative_path
                .split('/')
                .next()
                .unwrap_or(".")
                .to_string()
        } else {
            // Root-level files (e.g. README.md, Cargo.toml) → root group
            ".".to_string()
        };

        groups.entry(top).or_default().push(file);
    }

    groups
}

/// Check if a file path should be skipped (common non-project directories).
fn should_skip(relative: &str) -> bool {
    let parts: Vec<&str> = relative.split('/').collect();
    // Skip hidden directories/files at any level, and known non-project
    // directories (e.g. build output, dependencies) so they don't appear
    // as phantom entries in the file tree. Also skip any directory whose
    // name starts with "build" (build, build-native, build-* ...) so Meson/
    // CMake generated artifacts are never mistaken for source directories.
    parts.iter().any(|p| {
        is_hidden(p.as_ref()) || SKIP_DIRS.contains(p) || p.starts_with("build")
    })
}

/// True when the given `Cargo.toml` belongs to a workspace MEMBER crate, i.e.
/// an ancestor manifest declares `[workspace]` and lists this manifest's
/// directory in `members`, and the manifest itself is not the workspace root
/// (the root declares `[workspace]` and/or has no `[package]` table).
///
/// Mirrors the `meson.build` precedent in `discover_build_files`: a member
/// manifest is the platform module of the owning workspace root, not an
/// independent project. Spire's multi-platform scaffold emits exactly the
/// root workspace `Cargo.toml` plus one member manifest per platform leaf
/// (`core/`, `rpi5/`, `rock3c/`), so without this check every member shows up
/// as a duplicate subproject alongside the directory subprojects.
fn is_cargo_workspace_member(manifest: &Path) -> bool {
    let Some(manifest_name) = manifest.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if manifest_name != "Cargo.toml" {
        return false;
    }
    // The manifest itself must be a package (workspace roots may be virtual
    // and have no [package], but a member ALWAYS has one).
    let own_content = std::fs::read_to_string(manifest).unwrap_or_default();
    if !own_content.contains("[package]") {
        return false;
    }
    // Walk up looking for an ancestor root Cargo.toml with [workspace] whose
    // members list contains this manifest's directory.
    let mut dir = match manifest.parent() {
        Some(d) => d.to_path_buf(),
        None => return false,
    };
    while let Some(d) = dir.parent() {
        let candidate = d.join("Cargo.toml");
        if candidate == manifest {
            dir = d.to_path_buf();
            continue;
        }
        let Ok(root_content) = std::fs::read_to_string(&candidate) else {
            dir = d.to_path_buf();
            continue;
        };
        if !root_content.contains("[workspace]") {
            dir = d.to_path_buf();
            continue;
        }
        // Relative member dir (e.g. "core", "rpi5") from the root manifest.
        let member_rel = manifest
            .parent()
            .and_then(|m| m.strip_prefix(d).ok())
            .unwrap_or_else(|| manifest.parent().unwrap_or(Path::new("")))
            .to_string_lossy()
            .to_string();
        let member_rel = member_rel.trim_matches('/').to_string();
        // members = ["core", "rpi5", "rock3c"] (inline or multi-line).
        let member_re = regex::Regex::new(
            r#"(?m)members\s*=\s*\[([^\]]*)\]"#,
        )
        .unwrap();
        let is_member = member_re
            .captures(&root_content)
            .and_then(|c| c.get(1))
            .map(|m| {
                m.as_str()
                    .split(',')
                    .map(|s| s.trim().trim_matches('"').trim())
                    .any(|m| m == member_rel)
            })
            .unwrap_or(false);
        return is_member;
    }
    false
}

/// Check if a filename is hidden (starts with `.`).
fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// Regression: a Cargo WORKSPACE's member manifests must NOT be treated as
    /// independent projects. Spire's multi-platform scaffold emits a root
    /// `[workspace] Cargo.toml` plus per-platform leaves (`core/`, `rpi5/`,
    /// `rock3c/`). Previously every member `Cargo.toml` produced a duplicate
    /// BuildSystem subproject on top of the directory subprojects (6 entries
    /// instead of 3).
    #[test]
    fn discover_build_files_skips_cargo_workspace_members() {
        let tmp = tempdir().unwrap();
        // tempfile's root is `.tmpXXXX` (hidden), which discover_build_files'
        // filter_entry skips — mirror the meson test by using a non-hidden
        // project subdir.
        let root = tmp.path().join("project");
        fs::create_dir_all(&root).unwrap();
        // Root workspace manifest.
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"core\", \"rpi5\", \"rock3c\"]\n\n[workspace.dependencies]\n",
        )
        .unwrap();
        // Member manifests (each a package inside the workspace).
        for member in ["core", "rpi5", "rock3c"] {
            let dir = root.join(member);
            fs::create_dir_all(dir.join("src")).unwrap();
            fs::write(
                dir.join("Cargo.toml"),
                "[package]\nname = \"demo-foo\"\nversion.workspace = true\nedition.workspace = true\n",
            )
            .unwrap();
            fs::write(dir.join("src/lib.rs"), "// stub\n").unwrap();
        }

        let found = discover_build_files(&root, false);
        let configs: Vec<&str> = found.iter().map(|(p, _)| p.as_str()).collect();
        // Only the workspace ROOT Cargo.toml is a project.
        assert_eq!(configs, vec!["Cargo.toml"], "got: {configs:?}");

        // The workspace root itself is NOT a member (it declares [workspace]).
        assert!(!is_cargo_workspace_member(&root.join("Cargo.toml")));
        // Each leaf IS a member.
        for member in ["core", "rpi5", "rock3c"] {
            assert!(
                is_cargo_workspace_member(&root.join(member).join("Cargo.toml")),
                "{} should be a workspace member",
                member
            );
        }
        // A standalone crate outside any workspace is NOT a member.
        let standalone = root.join("standalone");
        fs::create_dir_all(&standalone).unwrap();
        fs::write(
            standalone.join("Cargo.toml"),
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        assert!(!is_cargo_workspace_member(&standalone.join("Cargo.toml")));
    }
}
