use serde::{Deserialize, Serialize};

/// A request to analyze a piece of code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysisRequest {
    pub code: String,
    pub language: String,
    pub file_path: Option<String>,
}

/// The result of a code analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAnalysis {
    pub summary: String,
    pub complexity: Option<ComplexityScore>,
    pub symbols: Vec<SymbolInfo>,
    pub suggestions: Vec<String>,
}

/// Complexity scoring for analyzed code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityScore {
    pub cyclomatic: u32,
    pub cognitive: u32,
    pub lines_of_code: u32,
}

/// Information about a symbol found in code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub line: u32,
    pub column: u32,
}

/// The kind of a code symbol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Class,
    Variable,
    Method,
    Interface,
    Enum,
    Struct,
    Trait,
    Module,
    Unknown,
}

/// A graph-backed symbol returned by the `symbols/*` tools.
///
/// Unlike the legacy `SymbolInfo` (name+line+col only), this carries the
/// stable graph node id, file path, and full source span, so the LLM can
/// operate on `symbolId` (a `GraphNode::id()`) rather than brittle
/// `file:line` coordinates. Projection lives in `from_graph_node`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSymbol {
    pub symbol_id: String,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
    pub language: String,
    pub signature: Option<String>,
    pub return_type: Option<String>,
    pub visibility: Option<String>,
    pub body: Option<String>,
}

impl GraphSymbol {
    /// Project an AST graph node onto the tool-facing symbol shape. Returns
    /// None for non-AST nodes so callers can filter cleanly.
    /// Project an open `AttrNode` envelope onto the tool-facing symbol shape.
    /// Returns None for non-AST discriminators.
    pub fn from_attr_node(attr: &crate::models::memory_graph::AttrNode) -> Option<Self> {
        let symbol_id = attr.id().to_string();
        let name = attr.name().to_string();
        let file_path = attr.str_prop("file_path")?;
        let start_line = attr.u32_prop("start_line")?;
        let start_col = attr.u32_prop("start_col")?;
        let end_line = attr.u32_prop("end_line").unwrap_or(start_line);
        let end_col = attr.u32_prop("end_col").unwrap_or(start_col);
        let language = attr.str_prop("language").unwrap_or_default();

        match attr.node_type_str() {
            "astFunction" => Some(GraphSymbol {
                symbol_id,
                name,
                kind: attr.str_prop("kind").unwrap_or_default(),
                file_path,
                start_line,
                start_col,
                end_line,
                end_col,
                language,
                signature: attr.str_prop("signature"),
                return_type: attr.str_prop("return_type"),
                visibility: Some(
                    if attr.bool_prop("is_public").unwrap_or(false) {
                        "public"
                    } else {
                        "private"
                    }
                    .to_string(),
                ),
                body: attr.str_prop("text"),
            }),
            "astClass" => Some(GraphSymbol {
                symbol_id,
                name,
                kind: attr.str_prop("kind").unwrap_or_default(),
                file_path,
                start_line,
                start_col,
                end_line,
                end_col,
                language,
                signature: None,
                return_type: None,
                visibility: Some(
                    if attr.bool_prop("is_public").unwrap_or(false) {
                        "public"
                    } else {
                        "private"
                    }
                    .to_string(),
                ),
                body: attr.str_prop("text"),
            }),
            "astVariable" => Some(GraphSymbol {
                symbol_id,
                name,
                kind: attr.str_prop("kind").unwrap_or_default(),
                file_path,
                start_line,
                start_col,
                end_line,
                end_col,
                language,
                signature: None,
                return_type: attr.str_prop("data_type"),
                visibility: Some(
                    if attr.bool_prop("is_public").unwrap_or(false) {
                        "public"
                    } else {
                        "private"
                    }
                    .to_string(),
                ),
                body: attr.str_prop("text"),
            }),
            "astImport" => {
                let path = attr.str_prop("path").unwrap_or_default();
                let alias = attr.str_prop("alias");
                Some(GraphSymbol {
                    symbol_id,
                    name,
                    kind: "import".to_string(),
                    file_path,
                    start_line,
                    start_col,
                    end_line: start_line,
                    end_col: start_col,
                    language,
                    signature: Some(format!(
                        "use {}",
                        alias
                            .clone()
                            .map(|a| format!("{} as {}", path, a))
                            .unwrap_or_else(|| path.clone())
                    )),
                    return_type: None,
                    visibility: None,
                    body: None,
                })
            }
            _ => None,
        }
    }
}

/// A search result from the codebase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub line: u32,
    pub column: u32,
    pub snippet: String,
    pub score: f64,
    pub context: Option<String>,
}

/// A request to search the codebase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: Option<usize>,
    pub file_pattern: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::memory_graph::AttrNode;

    fn attr_ast_function() -> AttrNode {
        let now = chrono::Utc::now();
        AttrNode {
            id: "sym-1".to_string(),
            node_type: "astFunction".to_string(),
            subtype: None,
            name: "handle".to_string(),
            description: None,
            properties: std::collections::HashMap::from([
                ("kind".to_string(), serde_json::json!("function")),
                ("text".to_string(), serde_json::json!("fn handle() {}")),
                ("start_line".to_string(), serde_json::json!(10)),
                ("start_col".to_string(), serde_json::json!(1)),
                ("end_line".to_string(), serde_json::json!(12)),
                ("end_col".to_string(), serde_json::json!(3)),
                ("file_path".to_string(), serde_json::json!("src/main.rs")),
                ("language".to_string(), serde_json::json!("rust")),
                ("signature".to_string(), serde_json::json!("fn handle()")),
                ("return_type".to_string(), serde_json::json!("()")),
                ("is_public".to_string(), serde_json::json!(true)),
            ]),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 1,
        }
    }

    #[test]
    fn graph_symbol_from_attr_node_projects_ast_fields() {
        let symbol = GraphSymbol::from_attr_node(&attr_ast_function()).expect("symbol");
        assert_eq!(symbol.symbol_id, "sym-1");
        assert_eq!(symbol.name, "handle");
        assert_eq!(symbol.kind, "function");
        assert_eq!(symbol.start_line, 10);
        assert_eq!(symbol.start_col, 1);
        assert_eq!(symbol.file_path, "src/main.rs");
        assert_eq!(symbol.language, "rust");
        assert_eq!(symbol.signature.as_deref(), Some("fn handle()"));
        assert_eq!(symbol.return_type.as_deref(), Some("()"));
        assert_eq!(symbol.visibility.as_deref(), Some("public"));

        // Non-AST discriminators yield no symbol.
        let now = chrono::Utc::now();
        let plain = AttrNode {
            id: "u-1".to_string(),
            node_type: "Unknown".to_string(),
            subtype: None,
            name: "note".to_string(),
            description: None,
            properties: std::collections::HashMap::new(),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 1,
        };
        assert!(GraphSymbol::from_attr_node(&plain).is_none());
    }

    #[test]
    fn attr_node_diagnostic_view() {
        let now = chrono::Utc::now();
        let diag = AttrNode {
            id: "d-1".to_string(),
            node_type: "diagnostic".to_string(),
            subtype: Some("error".to_string()),
            name: "err".to_string(),
            description: None,
            properties: std::collections::HashMap::from([
                ("message".to_string(), serde_json::json!("mismatched types")),
                ("file".to_string(), serde_json::json!("src/main.rs")),
                ("line".to_string(), serde_json::json!(42)),
                ("column".to_string(), serde_json::json!(5)),
                ("severity".to_string(), serde_json::json!("error")),
                ("build_type".to_string(), serde_json::json!("Cargo")),
                ("build_run_id".to_string(), serde_json::json!("run-1")),
            ]),
            embedding_id: None,
            created_at: now,
            updated_at: now,
            version: 1,
        };
        let view = diag.diagnostic().expect("diagnostic view");
        assert_eq!(view.message, "mismatched types");
        assert_eq!(view.file.as_deref(), Some("src/main.rs"));
        assert_eq!(view.line, Some(42));
        assert_eq!(view.column, Some(5));
        assert_eq!(view.build_run_id, "run-1");

        // Non-diagnostic discriminators yield no view.
        assert!(attr_ast_function().diagnostic().is_none());
    }
}
