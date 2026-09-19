// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Shared build-system metadata types.
//!
//! This crate defines the canonical `BuildMetadata` and related types used
//! by the MCP build servers (mcp-cargo, mcp-node, mcp-swift, ...) and by
//! `spire-core` when it deserializes analysis results over MCP.
//!
//! Having a single source of truth prevents the per-server structs from
//! diverging (as they did historically — mcp-swift used a completely
//! different schema, and mcp-cargo/mcp-node each had near-identical copies).

use serde::{Deserialize, Serialize};

/// Normalized metadata for all detected build systems in a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildMetadata {
    /// Stable module identity — the build config's path relative to the scan
    /// root (e.g. "", "ui/swift", "crates/spire-core").
    #[serde(default)]
    pub id: String,
    /// Path to the project directory containing this build system
    #[serde(default)]
    pub project_path: Option<String>,
    /// Detected build system types
    #[serde(default)]
    pub build_types: Vec<String>,
    /// Build system config files found
    #[serde(default)]
    pub config_files: Vec<String>,
    /// Available build commands from package.json scripts (if Node.js project)
    #[serde(default)]
    pub node_scripts: Vec<BuildScript>,
    /// Cargo workspace members (if Rust workspace)
    #[serde(default)]
    pub workspace_members: Vec<WorkspaceMember>,
    /// Entry points detected (main.rs, extension.ts, etc.)
    #[serde(default)]
    pub entry_points: Vec<String>,
    /// Backward-compat: project name (derived from first workspace member or directory)
    #[serde(default)]
    pub project_name: Option<String>,
    /// Human-readable project description (from Cargo.toml, package.json, etc.)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// HAL contract headers (`hal/api/*.hpp`) and their per-platform implementations.
    #[serde(default)]
    pub hal_interfaces: Vec<HalInterface>,
    /// Non-fatal import/analysis diagnostics (e.g. missing HAL implementation).
    #[serde(default)]
    pub issues: Vec<BuildIssue>,
    /// Backward-compat: project type (e.g. "rust_workspace", "vscode_extension")
    #[serde(default)]
    pub project_type: String,
    /// Backward-compat: primary build system (e.g. "Cargo", "npm")
    #[serde(default)]
    pub build_system: String,
    /// Platform targets for a cross-platform build (e.g. Meson
    /// `option('platform', …, values: ['host','rpi5'])`). The first entry is
    /// the default. Empty when the project builds on one platform only.
    #[serde(default)]
    pub platform_targets: Vec<String>,
    /// The project's structural shape (Native / SingleSource / Hal). Filled by
    /// the analyzer; the UI uses it to choose the project-tree presentation
    /// (filesystem subprojects vs virtual component view for HAL projects).
    #[serde(default, skip_serializing_if = "is_native_structure")]
    pub structure: ProjectStructure,
    /// Named, independently-addressable slices of a composite project. For the
    /// Hal shape these are `common` (shared toolkit + HAL contracts) and one
    /// platform domain per target (`rpi5`, `rock3c` — app code + that
    /// platform's HAL implementation). Selecting a domain in the UI sets the
    /// LLM edit context and its editability constraints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domains: Vec<ProjectDomain>,
    /// Backward-compat fields (expected by build parsers)
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub is_workspace: bool,
    #[serde(default)]
    pub scripts: Vec<BuildScript>,
    #[serde(default)]
    pub features: Vec<Feature>,
    #[serde(default)]
    pub targets: Vec<BuildTarget>,
    #[serde(default)]
    pub workspace_member_paths: Vec<String>,
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    #[serde(default)]
    pub raw: Option<serde_json::Value>,
}

/// A single script entry from package.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildScript {
    pub name: String,
    pub command: String,
    pub tool_call: Option<serde_json::Value>,
}

/// A workspace member entry (backward compat).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceMember {
    pub name: String,
    pub path: String,
    pub version: Option<String>,
}

/// A build target (backward compat).
/// `kind` can be a single string ("lib") or an array (["lib"]).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildTarget {
    pub name: String,
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    pub kind: Vec<String>,
    pub source_path: Option<String>,
    /// Source files compiled into this target, relative to the build config's
    /// directory (e.g. platform subdir + shared toolkit sources).
    #[serde(default)]
    pub source_files: Vec<String>,
    /// Dependencies linked into this target (platform-specific + shared).
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    /// Cross-compilation platform this target builds for.
    /// "host" = the development machine (default).
    #[serde(default = "default_host")]
    pub platform: String,
    /// Single = one source set, built with different settings per platform
    /// (Cargo cross-compilation). Composite = the build compiles shared +
    /// platform/app sources together (Meson platform executables).
    #[serde(default, skip_serializing_if = "is_single")]
    pub source_kind: SourceKind,
    /// Explicit shared/HAL/app source composition for composite targets.
    /// Empty for single targets (source_files already describes the set).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_units: Vec<SourceUnit>,
    /// Normalized invocation for this target (command + args + env). When
    /// present the build manager executes it directly; when absent the module
    /// falls back to its existing per-tool logic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_spec: Option<BuildSpec>,
}

/// Whether sources are shared across variants or composed per target.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Same source set, differing build configuration (Cargo --target).
    #[default]
    Single,
    /// Build compiles shared + platform/app sources together (Meson).
    Composite,
}

/// The role a source group plays in a composite build.
///
/// Meson cross-platform projects compose shared toolkit code, a platform
/// implementation of the HAL contract, and platform application code into one
/// executable. Keeping these roles first-class lets the product model separate
/// "the contract" from "this platform's implementation" instead of flattening
/// everything into one soup.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SourceRole {
    /// Platform application logic (e.g. `main.cpp`, `rpi5_detection_pipeline.cpp`).
    #[default]
    App,
    /// The HAL contract header set (e.g. `hal/api/camera_hal.hpp`) — shared, read-only.
    HalInterface,
    /// A platform's implementation of the HAL contract (e.g. `hal/implementations/rpi5/*.cpp`).
    HalImplementation,
    /// Shared toolkit/core code referenced by every platform target.
    Shared,
}

/// The structural shape of a new cross-platform project. Selected in the
/// new-project wizard BEFORE planning — it determines which scaffold the build
/// module emits and how the fill phase classifies sources.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProjectStructure {
    /// Native/host-only project (the legacy single-target scaffold).
    #[default]
    Native,
    /// Meson: one shared source set, per-platform `executable()` differences
    /// via `if platform` — no HAL contract.
    SingleSource,
    /// Meson: shared core + `hal/api/*.hpp` contract + per-platform
    /// implementations + per-target build (the ai-traps shape).
    Hal,
    /// Rust/SwiftUI monorepo built on the Spire framework: a Cargo workspace
    /// (`crates/spire-<name>` crate) plus a host SwiftUI app (`ui/swift`),
    /// with `spire-actor` + `spire-core` as sibling path dependencies. Host
    /// (macOS) only.
    SpireApp,
    /// The **`spire-embedded` container**: a Cargo workspace holding the on-device side of a
    /// firmware family — the **actor framework**, one **BSP** crate per board that needs one, and
    /// the **peripheral drivers** — all written against `embedded-hal`'s traits and, where board
    /// facts are involved, the vendor's HAL (`esp-hal`). There is **one** of these, not one per
    /// application: applications depend on it rather than containing it.
    ///
    /// Scaffolded and maintained **by spire-code**, which is also what adds to it — a BSP crate for
    /// a board with no upstream BSP, a driver for a device with no upstream crate, an actor. The
    /// default is the ecosystem's crate; this project holds what the ecosystem does not.
    ///
    /// Recognized by a **declaration**, not a layout guess: the workspace manifest carries
    /// `[workspace.metadata.spire] structure = "embedded"`. A layout that happens to look
    /// like this one is not this project type, and the layout is a consequence of the structure
    /// rather than its definition.
    Embedded,
    /// Embedded **application**: a Rust binary for one board that *depends on* a
    /// [`ProjectStructure::Embedded`] project — plus the vendor's HAL and whichever driver crates it
    /// uses.
    ///
    /// A structure of its own rather than a flag on `Embedded`, because the two are separate
    /// **projects**, and because the counts differ: the container is a single library, and
    /// applications are many. An application is a crate with a `main`; the container is a workspace
    /// of libraries; an application is built *against* one rather than containing it.
    ///
    /// Recognized by a declaration too, and for the same reason: `[package.metadata.spire]` carries
    /// `structure = "embedded_app"` **and** the `embedded_path` it was built against, so the
    /// dependency is a recorded fact rather than something a later reader has to infer from `../` in
    /// a path.
    EmbeddedApp,
}

impl ProjectStructure {
    /// Stable snake_case key ("native" | "single_source" | "hal" | "spire_app" |
    /// "embedded" | "embedded_app"), matching `#[serde(rename_all = "snake_case")]`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProjectStructure::Native => "native",
            ProjectStructure::SingleSource => "single_source",
            ProjectStructure::Hal => "hal",
            ProjectStructure::SpireApp => "spire_app",
            ProjectStructure::Embedded => "embedded",
            ProjectStructure::EmbeddedApp => "embedded_app",
        }
    }

    /// Parse a wizard/FFI structure key; unknown or empty falls back to the
    /// default (Native), matching how unknown config previously behaved.
    pub fn from_str(s: &str) -> ProjectStructure {
        match s {
            "single_source" => ProjectStructure::SingleSource,
            "hal" => ProjectStructure::Hal,
            "spire_app" => ProjectStructure::SpireApp,
            "embedded" => ProjectStructure::Embedded,
            "embedded_app" => ProjectStructure::EmbeddedApp,
            _ => ProjectStructure::Native,
        }
    }
}

/// Diagnostic found while importing/analyzing a project. Non-fatal — analysis
/// still succeeds, but surfaced to the user so structure problems (e.g. a HAL
/// interface with no implementation for a platform) are visible.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildIssue {
    /// "info" | "warning" | "error".
    pub severity: String,
    /// Stable machine kind, e.g. "missing_implementation", "orphan_implementation",
    /// "unimplemented_for_platform", "duplicate_implementation".
    pub kind: String,
    /// Human-readable message.
    pub message: String,
}

/// A HAL (hardware abstraction layer) contract: one header in the `hal/api`
/// directory plus every platform implementation that satisfies it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HalInterface {
    /// Header stem (e.g. "camera_hal" for `hal/api/camera_hal.hpp`).
    pub name: String,
    /// Path to the interface header, relative to the build module root.
    pub header_path: String,
    /// Per-platform implementation labels (e.g. ["rpi5", "rock3c"]).
    #[serde(default)]
    pub implementations: Vec<String>,
}

/// What the LLM may do inside a domain (used for context-constrained edits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DomainEditability {
    /// Read-only: contract headers — change via the HAL tools, never a raw edit.
    ReadOnly,
    /// Shared: changing this affects every platform (toolkit).
    Shared,
    /// Fillable: safe to implement/modify for this platform only.
    Fillable,
}

/// A named slice of a composite project — the unit the UI selects and the LLM
/// edits within. See `BuildMetadata::domains`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectDomain {
    /// Stable id, e.g. "common", "rpi5", "rock3c".
    pub id: String,
    /// Display name (e.g. "Common", "rpi5").
    pub name: String,
    /// "common" | "platform"
    #[serde(default)]
    pub kind: String,
    /// Files belonging to this domain, relative to the module root.
    #[serde(default)]
    pub files: Vec<String>,
    /// Dependencies that scope to this domain (platform deps, or shared deps
    /// for the common domain).
    #[serde(default)]
    pub dependencies: Vec<Dependency>,
    /// The normalized build invocation when this domain is a platform target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_spec: Option<BuildSpec>,
    /// Editability constraint for LLM modifications within this domain.
    #[serde(default = "default_fillable")]
    pub editability: DomainEditability,
    /// For the common domain: contract stems this domain owns (read-only).
    #[serde(default)]
    pub contracts: Vec<String>,
}

#[inline]
fn default_fillable() -> DomainEditability {
    DomainEditability::Fillable
}

#[inline]
fn is_single(kind: &SourceKind) -> bool {
    *kind == SourceKind::Single
}

#[inline]
fn is_native_structure(s: &ProjectStructure) -> bool {
    *s == ProjectStructure::Native
}

#[inline]
fn default_host() -> String {
    "host".to_string()
}

/// One source group within a composite target.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceUnit {
    /// "app" | "hal_interface" | "hal_implementation" | "shared"
    #[serde(default)]
    pub role: SourceRole,
    /// Relative to the build module root (e.g. "toolkit/src", "rpi5/src").
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub language: String,
}

/// Normalized build invocation (single source of truth for orchestration).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildSpec {
    /// "cargo" | "meson" | "swift"
    pub command: String,
    /// e.g. ["build", "--target", "aarch64-linux-gnu"]
    #[serde(default)]
    pub arguments: Vec<String>,
    /// Relative to the build module root ("" = module root).
    #[serde(default)]
    pub working_dir: String,
    #[serde(default)]
    pub env: Vec<(String, String)>,
}

/// Deserialize a field that can be either a single string or a vec of strings.
fn deserialize_string_or_vec<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Vec<String>, D::Error> {
    use serde::de;
    struct StringOrVec;
    impl<'de> de::Visitor<'de> for StringOrVec {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("string or list of strings")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Vec<String>, E> {
            Ok(vec![v.to_string()])
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<String>, A::Error> {
            let mut v = Vec::new();
            while let Some(s) = seq.next_element::<String>()? {
                v.push(s);
            }
            Ok(v)
        }
    }
    d.deserialize_any(StringOrVec)
}

/// A dependency entry (backward compat).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dependency {
    pub name: String,
    pub version: Option<String>,
    pub version_req: Option<String>,
    pub kind: Option<String>,
    pub source: Option<String>,
    pub source_url: Option<String>,
    pub features: Option<Vec<String>>,
    /// "shared" (built into every platform target) vs "platform" (this target
    /// only). Meson: `core_deps` vs `platform_deps`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

/// A Cargo feature entry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Feature {
    pub name: String,
    pub description: Option<String>,
    pub default: bool,
}

/// MCP server capability mapping.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpServerCapability {
    pub name: String,
    pub capabilities: Vec<String>,
    /// The tool name to call for analysis (e.g. "analyze").
    #[serde(default)]
    pub analyzer_tool: Option<String>,
}

/// The union of build-config file names owned by every registered build module.
///
/// Single source of truth for config-file discovery. Shared: lives in
/// `spire-core` (the analyzer uses it); the build modules in `spire-code` own
/// the per-module definitions. Adding a new build module means adding its
/// config files here (and in the module itself).
pub fn all_config_file_names() -> Vec<&'static str> {
    vec![
        // Cargo
        "Cargo.toml",
        // Node/npm
        "package.json",
        "pnpm-workspace.yaml",
        // SwiftPM
        "Package.swift",
        // Python
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        // Go
        "go.mod",
        // Gradle
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        // Maven
        "pom.xml",
        // CMake
        "CMakeLists.txt",
        // Make
        "Makefile",
        "makefile",
        // Meson
        "meson.build",
        // Ruby
        "Gemfile",
        "Rakefile",
        "*.gemspec",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_structure_keys_roundtrip() {
        for key in ["native", "single_source", "hal", "spire_app", "embedded"] {
            assert_eq!(ProjectStructure::from_str(key).as_str(), key);
        }
        assert_eq!(ProjectStructure::from_str(""), ProjectStructure::Native);
        assert_eq!(
            ProjectStructure::from_str("bogus"),
            ProjectStructure::Native
        );
        assert_eq!(
            ProjectStructure::from_str("SPIRE_APP"),
            ProjectStructure::Native
        );
    }
}
