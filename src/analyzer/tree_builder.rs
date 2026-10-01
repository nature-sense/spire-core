// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! File tree builder — assembles a flat file list into a hierarchical tree.
//!
//! The tree is annotated with:
//! - Language detection (from file extension)
//! - Semantic role classification (source, test, config, documentation, etc.)
//! - Estimated lines of code (from file size heuristic)

use std::collections::HashMap;
use std::path::Path;

use crate::analyzer::models::*;
use crate::analyzer::scanner;

/// Build a complete recursive file tree from a scanned directory.
///
/// This walks the flat list of `FileInfo` entries and assembles them into
/// a hierarchical `DirectoryNode` tree, with every file annotated with
/// language, role, and estimated lines.
pub fn build_file_tree(root: &Path, no_ignore: bool) -> DirectoryNode {
    let files = scanner::scan_directory(root, no_ignore);
    let non_dir_files: Vec<&FileInfo> = files.iter().filter(|f| !f.is_dir).collect();

    // Build language lookup
    let lang_map = build_language_map(&non_dir_files);

    // Build the tree recursively
    let root_name = root
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());

    build_directory_node(root_name, ".", &files, &lang_map)
}

/// Recursively build a DirectoryNode from a flat file list.
fn build_directory_node(
    name: String,
    prefix: &str,
    all_files: &[FileInfo],
    lang_map: &HashMap<String, String>,
) -> DirectoryNode {
    let mut directories: Vec<DirectoryNode> = Vec::new();
    let mut files: Vec<FileNode> = Vec::new();
    let mut total_file_count = 0usize;
    let mut total_lines = 0usize;

    // Collect immediate children (files and subdirectories)
    let mut subdirs: HashMap<String, Vec<FileInfo>> = HashMap::new();

    for file in all_files {
        let rel = &file.relative_path;

        // Skip files not under this prefix. Compare component-wise so siblings
        // sharing a string prefix (e.g. `rpi/hal` vs `rpi/halx`) are not
        // mis-grouped into this node — otherwise the unresolved remainder is
        // turned into a phantom subdirectory named after the ancestor
        // (`rpi/hal/rpi`).
        if prefix != "." {
            let inside = rel == prefix
                || rel
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/') && !rest[1..].is_empty());
            if !inside {
                continue;
            }
        }

        // Get the relative path within this directory
        let within = if prefix == "." {
            rel.clone()
        } else {
            rel.strip_prefix(prefix)
                .and_then(|s| s.strip_prefix('/'))
                .unwrap_or(rel)
                .to_string()
        };

        // The entry IS this directory node itself (the scanner emits a dir
        // entry for every directory, including the node being built). Skipping
        // it prevents `strip_prefix(prefix)` from yielding "" and falling back
        // to the full path, which would otherwise be mis-grouped as a phantom
        // subdirectory named after an ancestor (`rpi/hal` → `rpi/hal/rpi`).
        if rel == prefix {
            continue;
        }

        if within.is_empty() || within == "." {
            continue;
        }

        // Check if it's a direct child or nested deeper
        if let Some(slash_pos) = within.find('/') {
            // Nested — belongs to a subdirectory
            let subdir_name = within[..slash_pos].to_string();
            subdirs.entry(subdir_name).or_default().push(file.clone());
        } else if file.is_dir {
            // Directory entry — skip (we'll build from files)
            continue;
        } else {
            // Direct file child
            let ext = file.extension.clone();
            let language = lang_map
                .get(&ext)
                .cloned()
                .unwrap_or_else(|| extension_to_language_name(&ext));
            let role = classify_file_role(&file.relative_path, &language);

            let file_node = FileNode {
                name: within.clone(),
                path: file.relative_path.clone(),
                extension: ext,
                language,
                size: file.size,
                lines_estimated: (file.size / 50) as usize,
                role,
            };
            total_file_count += 1;
            total_lines += file_node.lines_estimated;
            files.push(file_node);
        }
    }

    // Recursively build subdirectory nodes
    for subdir_name in subdirs.keys() {
        let sub_prefix = if prefix == "." {
            subdir_name.clone()
        } else {
            format!("{}/{}", prefix, subdir_name)
        };
        let sub_node = build_directory_node(subdir_name.clone(), &sub_prefix, all_files, lang_map);
        total_file_count += sub_node.total_file_count;
        total_lines += sub_node.total_lines;
        directories.push(sub_node);
    }

    // Sort directories and files alphabetically
    directories.sort_by(|a, b| a.name.cmp(&b.name));
    files.sort_by(|a, b| a.name.cmp(&b.name));

    let role = classify_directory_role(&name, prefix);

    DirectoryNode {
        name,
        path: prefix.to_string(),
        role,
        directories,
        files,
        total_file_count,
        total_lines,
    }
}

/// Build a map of extension → language name from a file list.
fn build_language_map(files: &[&FileInfo]) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    for file in files {
        let lang = extension_to_language_name(&file.extension);
        if lang != "Unknown" {
            map.entry(file.extension.clone()).or_insert_with(|| lang);
        }
    }
    map
}

/// Map a file extension to a language name.
fn extension_to_language_name(ext: &str) -> String {
    match ext {
        ".rs" => "Rust",
        ".ts" | ".tsx" => "TypeScript",
        ".js" | ".jsx" | ".mjs" | ".cjs" => "JavaScript",
        ".py" => "Python",
        ".go" => "Go",
        ".java" => "Java",
        ".kt" | ".kts" => "Kotlin",
        ".swift" => "Swift",
        ".rb" => "Ruby",
        ".c" | ".h" => "C",
        ".cpp" | ".hpp" | ".cc" | ".cxx" => "C++",
        ".cs" => "C#",
        ".scala" => "Scala",
        ".r" | ".R" => "R",
        ".php" => "PHP",
        ".pl" | ".pm" => "Perl",
        ".lua" => "Lua",
        ".ex" | ".exs" => "Elixir",
        ".erl" | ".hrl" => "Erlang",
        ".hs" => "Haskell",
        ".clj" | ".cljs" | ".cljc" => "Clojure",
        ".zig" => "Zig",
        ".dart" => "Dart",
        ".sh" | ".bash" | ".zsh" => "Shell",
        ".ps1" => "PowerShell",
        ".sql" => "SQL",
        ".html" | ".htm" => "HTML",
        ".css" | ".scss" | ".sass" | ".less" => "CSS",
        ".json" => "JSON",
        ".yaml" | ".yml" => "YAML",
        ".toml" => "TOML",
        ".xml" => "XML",
        ".md" | ".markdown" => "Markdown",
        ".dockerfile" => "Docker",
        _ => "Unknown",
    }
    .to_string()
}

/// Classify a file's semantic role based on its path and language.
fn classify_file_role(path: &str, language: &str) -> String {
    let path_lower = path.to_lowercase();

    // Build config files
    if path == "Cargo.toml" || path == "package.json" || path == "meson.build" {
        return "build_file".to_string();
    }

    // Entry points
    if path == "src/main.rs"
        || path == "src/lib.rs"
        || path == "src/main.ts"
        || path == "src/main.js"
        || path == "src/index.ts"
        || path == "src/index.js"
        || path == "main.py"
        || path == "app.py"
        || path == "main.go"
    {
        return "entry_point".to_string();
    }

    // VS Code extension entry points
    if path == "src/extension.ts" || path == "src/extension.js" {
        return "entry_point".to_string();
    }

    // Test files
    if path_lower.contains("test")
        || path_lower.contains("spec")
        || path.ends_with("_test.rs")
        || path.ends_with(".test.ts")
        || path.ends_with(".spec.ts")
        || path.ends_with("_test.go")
    {
        return "test".to_string();
    }

    // Configuration files
    if (path_lower.contains(".json")
        || path_lower.contains(".yaml")
        || path_lower.contains(".yml")
        || path_lower.contains(".toml")
        || path_lower.contains(".ini")
        || path_lower.contains(".cfg")
        || path_lower.contains(".conf"))
        && (path == "Cargo.toml"
            || path == "package.json"
            || path == "tsconfig.json"
            || path == ".gitignore"
            || path == "pnpm-workspace.yaml")
    {
        return "config".to_string();
    }

    // Documentation
    if path_lower.ends_with(".md") || path_lower.ends_with(".txt") || path_lower.ends_with(".rst") {
        return "documentation".to_string();
    }

    // Source code (default for known languages)
    if language != "Unknown"
        && language != "Markdown"
        && language != "JSON"
        && language != "YAML"
        && language != "TOML"
    {
        return "source".to_string();
    }

    "other".to_string()
}

/// Classify a directory's semantic role based on its name and path.
fn classify_directory_role(name: &str, _path: &str) -> String {
    match name {
        "src" => "source_code".to_string(),
        "lib" => "library_code".to_string(),
        "tests" | "test" | "spec" | "specs" => "tests".to_string(),
        "examples" | "example" | "samples" | "sample" => "examples".to_string(),
        "docs" | "doc" | "documentation" => "documentation".to_string(),
        "scripts" | "bin" | "tool" | "tools" => "build_scripts".to_string(),
        "config" | "cfg" | "configuration" | "settings" => "configuration".to_string(),
        "docker" | "container" => "docker".to_string(),
        "ci" | ".github" | ".gitlab" | ".circleci" => "ci_cd".to_string(),
        "assets" | "static" | "public" | "images" | "img" | "media" => "static_assets".to_string(),
        "dist" | "build" | "target" | "out" | "output" => "build_output".to_string(),
        "node_modules" | ".pnpm" | "vendor" | "third_party" | "third-party"
        | "managed_components" => "dependencies".to_string(),
        "benchmarks" | "bench" => "benchmarks".to_string(),
        "data" | "dataset" | "datasets" => "data".to_string(),
        "migrations" | "migrate" => "migrations".to_string(),
        "patches" | "patch" => "patches".to_string(),
        "plugins" | "plugin" | "extensions" | "extension" => "plugins".to_string(),
        "templates" | "template" => "templates".to_string(),
        "i18n" | "locale" | "locales" | "translations" => "localization".to_string(),
        "proto" | "protobuf" | "protos" => "protobuf".to_string(),
        "grafana" | "prometheus" | "monitoring" => "monitoring".to_string(),
        "hack" | "dev" | "development" => "development".to_string(),
        "mcp" => "mcp_servers".to_string(),
        "spire-core" => "core_crate".to_string(),
        "spire-extension" => "vscode_extension".to_string(),
        "subprojects" => "meson_subprojects".to_string(),
        "wrap" | "wraps" => "meson_wraps".to_string(),
        _ => {
            // Check if it looks like a workspace member (has Cargo.toml or package.json)
            // We can't check here without scanning, so default to "directory"
            "directory".to_string()
        }
    }
}
