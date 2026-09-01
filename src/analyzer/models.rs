use serde::{Deserialize, Serialize};

// ============================================================================
// Shared Build Metadata Types
// ============================================================================
//
// Build-system metadata is the contract between spire-core and the MCP build
// servers (mcp-cargo, mcp-node, mcp-swift, ...). It lives in the canonical
// `mcp-build-types` crate so the two sides can never diverge. spire-core only
// deserializes what the MCP servers send — it performs no build parsing itself.

pub use crate::build_types::{
    BuildMetadata, BuildScript, BuildTarget, Dependency, DomainEditability, Feature,
    McpServerCapability, ProjectDomain, WorkspaceMember,
};

// ============================================================================
// File-tree types (filesystem scanning, performed locally by spire-core)
// ============================================================================

/// A single file or directory entry from scanning.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileInfo {
    pub path: String,
    pub relative_path: String,
    pub extension: String,
    pub size: u64,
    pub is_dir: bool,
    pub is_symlink: bool,
}

/// A directory in the hierarchical file tree.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DirectoryNode {
    pub name: String,
    pub path: String,
    pub role: String,
    pub directories: Vec<DirectoryNode>,
    pub files: Vec<FileNode>,
    pub total_file_count: usize,
    pub total_lines: usize,
}

/// A file in the hierarchical file tree.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileNode {
    pub name: String,
    pub path: String,
    pub extension: String,
    pub language: String,
    pub size: u64,
    pub lines_estimated: usize,
    pub role: String,
}
