// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 NatureSense

//! Platform registry — cross-compilation platform definitions (rpi5, rock3c, …).
//!
//! A platform is a declarative description of one cross-compilation target:
//! its CPU architecture, toolchain, and target sysroot. The data‑only types
//! live in [`crate::build_types`]; this module provides:
//!
//! - YAML seed loading (`~/.spire/platforms/*.yaml`, `$SPIRE_PLATFORM_DIR` override)
//! - Meson cross‑file generation from a [`Platform`](crate::build_types::Platform)
//!
//! The **graph is the canonical store** for platforms (each field stored as an
//! individual typed property on a Platform node). The YAML files are only the
//! seed used on startup, mirroring how MCP config is bootstrapped.

use std::fs;
use std::path::{Path, PathBuf};

use crate::build_types::{Platform, PlatformToolchain};
use anyhow::{Context, Result};

/// The ${SYSROOT} placeholder substituted with `sysroot.root` in arg lists.
const SYSROOT_TOKEN: &str = "${SYSROOT}";

#[cfg(test)]
use crate::build_types::{PlatformArchitecture, PlatformSysroot};

impl Platform {
    /// Load a platform definition from a YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Platform> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read platform YAML: {}", path.display()))?;
        let platform: Platform = serde_yaml::from_str(&raw)
            .with_context(|| format!("failed to parse platform YAML: {}", path.display()))?;
        Ok(platform)
    }

    /// Load every platform definition from a directory of `.yaml` files.
    pub fn load_directory(dir: impl AsRef<Path>) -> Result<Vec<Platform>> {
        let dir = dir.as_ref();
        let mut out = Vec::new();
        let mut entries = fs::read_dir(dir)
            .with_context(|| format!("failed to read platform dir: {}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .context("read_dir iterator failed")?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let is_yaml = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "yaml" || e == "yml")
                .unwrap_or(false);
            if !is_yaml {
                continue;
            }
            match Platform::load(&path) {
                Ok(p) => out.push(p),
                Err(e) => {
                    tracing::warn!("skipping invalid platform file {}: {}", path.display(), e);
                }
            }
        }
        Ok(out)
    }

    /// True/OK when no cross-compilation is needed (empty sysroot root = native
    /// host platform) OR the configured `sysroot.root` exists on disk as a
    /// populated target root (contains `usr/`). Used to fail fast in
    /// Meson/Cargo cross-build setup instead of emitting `--sysroot=<missing>`
    /// and letting the tool fail late with a confusing error.
    pub fn sysroot_ok(&self) -> (bool, String) {
        let root = self.sysroot.root.trim();
        if root.is_empty() {
            // Host / non-linux platform: no cross sysroot required.
            return (true, String::new());
        }
        let p = Path::new(root);
        if !p.is_dir() {
            return (false, format!("sysroot root is not a directory: {root}"));
        }
        if !p.join("usr").is_dir() {
            return (
                false,
                format!(
                    "sysroot at {root} is not populated (missing {}/usr)",
                    p.display()
                ),
            );
        }
        (true, String::new())
    }

    /// Discover the seed platform directory: `$SPIRE_PLATFORM_DIR` (for
    /// tests/CI/containers) or the default `~/.spire/platforms`, consistent
    /// with the existing global config location `~/.spire/llm-config.json`.
    pub fn default_platform_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("SPIRE_PLATFORM_DIR") {
            if !dir.trim().is_empty() {
                return PathBuf::from(dir);
            }
        }
        let base = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        base.join(".spire").join("platforms")
    }

    /// Load a single platform by id from the registry
    /// (`$SPIRE_PLATFORM_DIR` or `~/.spire/platforms/*.yaml`).
    pub fn from_registry(id: &str) -> Option<Platform> {
        let dir = Self::default_platform_dir();
        let platforms = Self::load_directory(&dir).ok()?;
        platforms.into_iter().find(|p| p.id == id)
    }

    /// Substitute `${SYSROOT}` in a single arg with the sysroot root.
    fn substitute(&self, arg: &str) -> String {
        arg.replace(SYSROOT_TOKEN, &self.sysroot.root)
    }

    fn substituted(&self, args: &[String]) -> Vec<String> {
        args.iter().map(|a| self.substitute(a)).collect()
    }

    /// Render Cargo's `.cargo/config.toml` for cross-compiling a pure-Rust
    /// project to this platform. Returns `None` for non-`linux` platforms.
    pub fn cargo_config(&self) -> Option<String> {
        use std::fmt::Write;
        if self.os != "linux" {
            return None;
        }
        let triple = &self.architecture.target_triple;
        let sysroot = &self.sysroot.root;

        let mut s = String::new();
        let _ = writeln!(s, "[target.{}]", triple);
        let _ = writeln!(s, "linker = \"{}\"", self.toolchain.c);
        let _ = writeln!(
            s,
            "rustflags = [\"-C\", \"link-arg=--target={triple}\",\n  \"-C\", \"link-arg=--sysroot={sysroot}\"]",
        );
        let _ = writeln!(s);
        let _ = writeln!(s, "[env]");
        let _ = writeln!(s, "PKG_CONFIG_SYSROOT_DIR = \"{sysroot}\"");
        if !self.sysroot.pkg_config_libdir.is_empty() {
            let joined = self
                .sysroot
                .pkg_config_libdir
                .iter()
                .map(|p| self.substitute(p))
                .collect::<Vec<_>>()
                .join(":");
            let _ = writeln!(s, "PKG_CONFIG_LIBDIR = \"{joined}\"");
        }
        let _ = writeln!(s, "CC_{} = \"{}\"", triple.to_uppercase(), self.toolchain.c);
        let _ = writeln!(s, "CXX_{} = \"{}\"", triple.to_uppercase(), self.toolchain.cpp);
        let _ = writeln!(s, "AR_{} = \"{}\"", triple.to_uppercase(), self.toolchain.ar);
        Some(s)
    }

    /// Render this platform as a Meson cross file. Returns `None` for
    /// non-`linux` platforms (e.g. esp-idf/rp2040, which use a different
    /// toolchain model) — those are handled by `cmake_toolchain_args` later.
    pub fn meson_cross_file(&self) -> Option<String> {
        if self.os != "linux" {
            return None;
        }
        let triple = &self.architecture.target_triple;
        let sysroot = &self.sysroot.root;
        let sysroot_arg = format!("--sysroot={}", sysroot);

        // Implicit target args: -target <triple> + --sysroot + optional march.
        let mut target_args = vec![
            "-target".to_string(),
            triple.clone(),
            sysroot_arg.clone(),
        ];
        if let Some(march) = &self.architecture.march {
            target_args.push(format!("-march={}", march));
        }

        let mut c_args = target_args.clone();
        c_args.extend(self.substituted(&self.toolchain.c_args_extra));
        let mut cpp_args = target_args.clone();
        cpp_args.extend(self.substituted(&self.toolchain.cpp_args_extra));

        let mut link_args = vec![
            "-target".to_string(),
            triple.clone(),
            sysroot_arg.clone(),
        ];
        if let Some(ld) = &self.toolchain.ld {
            if ld.ends_with("lld") {
                link_args.push("-fuse-ld=lld".to_string());
            }
        }
        if let Some(march) = &self.architecture.march {
            link_args.push(format!("-march={}", march));
        }
        link_args.extend(self.substituted(&self.toolchain.linker_args_extra));

        let quote = |s: &str| format!("'{}'", s);
        let fmt_list = |paths: &[String]| {
            let items = paths
                .iter()
                .map(|p| self.substitute(p))
                .map(|p| format!("'{}'", p))
                .collect::<Vec<_>>()
                .join(", ");
            if items.is_empty() {
                "[]".to_string()
            } else {
                format!("[{}]", items)
            }
        };

        let mut s = String::new();
        s.push_str("[host_machine]\n");
        s.push_str(&format!("system = '{}'\n", self.os));
        s.push_str(&format!("cpu_family = '{}'\n", self.architecture.cpu_family));
        s.push_str(&format!("cpu = '{}'\n", self.architecture.cpu));
        s.push_str(&format!("endian = '{}'\n", self.architecture.endian));
        s.push('\n');
        s.push_str("[target_machine]\n");
        s.push_str(&format!("system = '{}'\n", self.os));
        s.push_str(&format!("cpu_family = '{}'\n", self.architecture.cpu_family));
        s.push_str(&format!("cpu = '{}'\n", self.architecture.cpu));
        s.push_str(&format!("endian = '{}'\n", self.architecture.endian));
        s.push('\n');
        s.push_str("[binaries]\n");
        s.push_str(&format!("c = {}\n", quote(&self.toolchain.c)));
        s.push_str(&format!("cpp = {}\n", quote(&self.toolchain.cpp)));
        s.push_str(&format!("ar = {}\n", quote(&self.toolchain.ar)));
        s.push_str(&format!("strip = {}\n", quote(&self.toolchain.strip)));
        if let Some(ld) = &self.toolchain.ld {
            s.push_str(&format!("ld = {}\n", quote(ld)));
        }
        if let Some(pkg) = &self.toolchain.pkgconfig {
            s.push_str(&format!("pkgconfig = {}\n", quote(pkg)));
        }
        s.push('\n');
        s.push_str("[built-in options]\n");
        s.push_str(&format!("c_args = {}\n", fmt_list(&c_args)));
        s.push_str(&format!("c_link_args = {}\n", fmt_list(&link_args)));
        s.push_str(&format!("cpp_args = {}\n", fmt_list(&cpp_args)));
        s.push_str(&format!("cpp_link_args = {}\n", fmt_list(&link_args)));
        s.push('\n');
        s.push_str("[properties]\n");
        s.push_str(&format!("sys_root = '{}'\n", sysroot));
        s.push_str(&format!("needs_exe_wrapper = {}\n", self.toolchain.needs_exe_wrapper));
        if !self.sysroot.lib_dirs.is_empty() {
            s.push_str(&format!("lib_dirs = {}\n", fmt_list(&self.sysroot.lib_dirs)));
        }
        if !self.sysroot.pkg_config_libdir.is_empty() {
            s.push_str(&format!(
                "pkg_config_libdir = {}\n",
                fmt_list(&self.sysroot.pkg_config_libdir)
            ));
        }
        Some(s)
    }
}

impl From<&Platform> for PlatformToolchain {
    fn from(p: &Platform) -> Self {
        p.toolchain.clone()
    }
}

/// Pre-rendered cross-compilation settings for one platform id, computed once
/// so Cargo/Meson/Swift modules share a single lookup instead of each calling
/// `Platform::from_registry` + re-deriving the triple / config fragments.
#[derive(Debug, Clone, Default)]
pub struct CrossSpec {
    /// Rendered `.cargo/config.toml` content (None for host/unknown/non-linux).
    pub cargo_config: Option<String>,
    /// Rendered Meson cross file content (None for host/unknown/non-linux).
    pub meson_cross_file: Option<String>,
    /// The target triple (e.g. "aarch64-linux-gnu"). Empty for host/unknown.
    pub target_triple: String,
}

impl CrossSpec {
    /// Resolve a platform id (e.g. "rpi5", "rock3c") to its pre-rendered
    /// cross-compilation spec. Returns `None` for "host", unknown ids, or
    /// platforms without a rendered config (non-linux).
    pub fn for_platform(id: &str) -> Option<CrossSpec> {
        let platform = Platform::from_registry(id)?;
        Some(CrossSpec {
            cargo_config: platform.cargo_config(),
            meson_cross_file: platform.meson_cross_file(),
            target_triple: platform.architecture.target_triple.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_yaml(dir: &Path, name: &str, content: &str) {
        fs::create_dir_all(dir).unwrap();
        let mut f = fs::File::create(dir.join(name)).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn load_cubie_a7s_platform_from_user_registry() {
        // The real user-level seed (write by the platform tooling). Kept as a
        // smoke test so an invalid a7s.yaml fails loudly at test time.
        let path = std::path::Path::new(std::env::var("HOME").unwrap_or_default().as_str())
            .join(".spire")
            .join("platforms")
            .join("a7s.yaml");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let p = Platform::load(&path).expect("a7s platform YAML must parse");
        assert_eq!(p.id, "a7s");
        assert_eq!(p.name, "Cubie A7S");
        assert_eq!(p.os, "linux");
        assert_eq!(p.architecture.cpu_family, "aarch64");
        assert_eq!(p.architecture.target_triple, "aarch64-linux-gnu");
        assert_eq!(p.architecture.march.as_deref(), Some("armv8.2-a+crc"));
        // The linker path is machine-specific (Homebrew vs cross-toolchain);
        // only assert that a linker is declared.
        assert!(p.toolchain.ld.is_some(), "a7s must declare a linker");
        // The sysroot path is machine-specific ("sysroots/a7s" or
        // "/opt/cross/sysroot/cubie-a7s"); assert it references the platform.
        assert!(p.sysroot.root.contains("a7s"), "sysroot must reference a7s");
        // The SYSROOT token must be substituted when rendering the meson cross file.
        let cross = p.meson_cross_file().expect("linux cross file");
        assert!(
            cross.contains(&format!("-I{}/usr/include/c++/12", p.sysroot.root)),
            "cpp include must substitute the SYSROOT token"
        );
    }

    #[test]
    fn load_rock3c_platform_from_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        write_yaml(
            tmp.path(),
            "rock3c.yaml",
            r#"
id: rock3c
name: Rock 3C
os: linux
architecture:
  cpu_family: aarch64
  cpu: armv8-a
  endian: little
  target_triple: aarch64-linux-gnu
  march: armv8.2-a+crc
toolchain:
  c: clang
  cpp: clang++
  ar: /opt/homebrew/opt/llvm/bin/llvm-ar
  strip: /opt/homebrew/opt/llvm/bin/llvm-strip
  ld: /opt/homebrew/bin/ld.lld
  pkgconfig: /Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/bin/aarch64-pkg-config
sysroot:
  root: /Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c
  lib_dirs:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu
  pkg_config_libdir:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig
    - ${SYSROOT}/usr/share/pkgconfig
"#,
        );

        let p = Platform::load(tmp.path().join("rock3c.yaml")).unwrap();
        assert_eq!(p.id, "rock3c");
        assert_eq!(p.os, "linux");
        assert_eq!(p.architecture.target_triple, "aarch64-linux-gnu");
        assert_eq!(p.architecture.march.as_deref(), Some("armv8.2-a+crc"));
        assert_eq!(p.toolchain.ld.as_deref(), Some("/opt/homebrew/bin/ld.lld"));
        assert_eq!(
            p.sysroot.lib_dirs,
            vec!["${SYSROOT}/usr/lib/aarch64-linux-gnu".to_string()]
        );
    }

    #[test]
    fn meson_cross_file_for_linux() {
        let tmp = tempfile::tempdir().unwrap();
        write_yaml(
            tmp.path(),
            "rock3c.yaml",
            r#"
id: rock3c
name: Rock 3C
os: linux
architecture:
  cpu_family: aarch64
  cpu: armv8-a
  endian: little
  target_triple: aarch64-linux-gnu
  march: armv8.2-a+crc
toolchain:
  c: clang
  cpp: clang++
  ar: /opt/homebrew/opt/llvm/bin/llvm-ar
  strip: /opt/homebrew/opt/llvm/bin/llvm-strip
  ld: /opt/homebrew/bin/ld.lld
  pkgconfig: /Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/bin/aarch64-pkg-config
  cpp_args_extra:
    - -I${SYSROOT}/usr/include/c++/12
    - -I${SYSROOT}/usr/include/aarch64-linux-gnu/c++/12
    - -I${SYSROOT}/usr/include
sysroot:
  root: /Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c
  lib_dirs:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu
  pkg_config_libdir:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig
    - ${SYSROOT}/usr/share/pkgconfig
"#,
        );

        let p = Platform::load(tmp.path().join("rock3c.yaml")).unwrap();
        let cross = p.meson_cross_file().expect("linux platform cross file");

        // Target args with -target + sysroot + march on c_args.
        assert!(cross.contains("cpu_family = 'aarch64'"), "missing cpu_family");
        assert!(cross.contains("system = 'linux'"), "missing os");
        assert!(cross.contains("-target"), "missing -target");
        assert!(cross.contains("--sysroot=/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c"), "missing sysroot");
        assert!(cross.contains("-march=armv8.2-a+crc"), "missing march");
        assert!(cross.contains("-fuse-ld=lld"), "missing lld fuse");
        assert!(cross.contains("pkgconfig = '/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/bin/aarch64-pkg-config'"), "missing pkgconfig");
        // ${SYSROOT} substitution in cpp_args_extra.
        assert!(
            cross.contains("-I/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/usr/include/c++/12"),
            "missing substituted cpp include"
        );
        assert!(cross.contains("sys_root = '/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c'"), "missing sys_root");
        assert!(
            cross.contains("pkg_config_libdir = ['/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/usr/lib/aarch64-linux-gnu/pkgconfig', '/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/usr/share/pkgconfig']"),
            "missing pkg_config_libdir"
        );
    }

    #[test]
    fn cargo_config_for_rock3c() {
        let tmp = tempfile::tempdir().unwrap();
        write_yaml(
            tmp.path(),
            "rock3c.yaml",
            r#"
id: rock3c
name: Rock 3C
os: linux
architecture:
  cpu_family: aarch64
  cpu: armv8-a
  endian: little
  target_triple: aarch64-linux-gnu
toolchain:
  c: clang
  cpp: clang++
  ar: /opt/homebrew/opt/llvm/bin/llvm-ar
  strip: /opt/homebrew/opt/llvm/bin/llvm-strip
sysroot:
  root: /Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c
  lib_dirs:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu
  pkg_config_libdir:
    - ${SYSROOT}/usr/lib/aarch64-linux-gnu/pkgconfig
    - ${SYSROOT}/usr/share/pkgconfig
"#,
        );

        let p = Platform::load(tmp.path().join("rock3c.yaml")).unwrap();
        let cfg = p.cargo_config().expect("linux cargo config");

        assert!(cfg.contains("[target.aarch64-linux-gnu]"), "missing target section");
        assert!(cfg.contains("linker = \"clang\""), "missing linker");
        assert!(
            cfg.contains("link-arg=--target=aarch64-linux-gnu"),
            "missing target link-arg"
        );
        assert!(
            cfg.contains("link-arg=--sysroot=/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c"),
            "missing sysroot link-arg"
        );
        assert!(
            cfg.contains("PKG_CONFIG_SYSROOT_DIR = \"/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c\""),
            "missing pkg-config sysroot env"
        );
        assert!(
            cfg.contains("PKG_CONFIG_LIBDIR = \"/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/usr/lib/aarch64-linux-gnu/pkgconfig:/Users/steve/naturesense/ai-traps/tools/native/sysroots/rock3c/usr/share/pkgconfig\""),
            "missing joined pkg-config libdir"
        );
        // Toolchain env vars use the UPPERCASED triple.
        assert!(cfg.contains("CC_AARCH64-LINUX-GNU = \"clang\""), "missing CC env");
        assert!(cfg.contains("CXX_AARCH64-LINUX-GNU = \"clang++\""), "missing CXX env");
        assert!(
            cfg.contains("AR_AARCH64-LINUX-GNU = \"/opt/homebrew/opt/llvm/bin/llvm-ar\""),
            "missing AR env"
        );
    }

    #[test]
    fn non_linux_returns_none() {
        let platform = Platform {
            id: "esp32".into(),
            name: "ESP32-S3".into(),
            os: "esp-idf".into(),
            architecture: PlatformArchitecture {
                cpu_family: "xtensa".into(),
                cpu: "esp32s3".into(),
                endian: "little".into(),
                target_triple: "xtensa-esp32s3-elf".into(),
                march: None,
            },
            toolchain: PlatformToolchain::default(),
            sysroot: PlatformSysroot::default(),
        };
        assert!(platform.meson_cross_file().is_none());
    }

    /// Target-level sysroot sanity: a nonexistent or unpopulated sysroot must be
    /// flagged (the cross-build gate), while a populated one and host platforms
    /// (empty root) pass.
    #[test]
    fn sysroot_ok_detects_missing_and_populated_roots() {
        let tmp = tempfile::tempdir().unwrap();

        // 1. Missing path → blocked.
        let missing = Platform {
            id: "rpi5".into(),
            name: "Raspberry Pi 5".into(),
            os: "linux".into(),
            architecture: PlatformArchitecture {
                cpu_family: "arm".into(),
                cpu: "armv8".into(),
                endian: "little".into(),
                target_triple: "arm-linux-gnueabihf".into(),
                march: None,
            },
            toolchain: PlatformToolchain::default(),
            sysroot: PlatformSysroot {
                root: tmp.path().join("no-such-sysroot").to_string_lossy().to_string(),
                lib_dirs: Vec::new(),
                include_dirs: Vec::new(),
                pkg_config_libdir: Vec::new(),
            },
        };
        let (ok, reason) = missing.sysroot_ok();
        assert!(!ok, "missing root must be blocked");
        assert!(reason.contains("not a directory"), "reason: {reason}");

        // 2. Empty placeholder root (no usr/) → blocked.
        let placeholder_dir = tmp.path().join("empty-sysroot");
        std::fs::create_dir_all(&placeholder_dir).unwrap();
        let placeholder = Platform {
            id: "rpi5".into(),
            name: "Raspberry Pi 5".into(),
            os: "linux".into(),
            architecture: PlatformArchitecture {
                cpu_family: "arm".into(),
                cpu: "armv8".into(),
                endian: "little".into(),
                target_triple: "arm-linux-gnueabihf".into(),
                march: None,
            },
            toolchain: PlatformToolchain::default(),
            sysroot: PlatformSysroot {
                root: placeholder_dir.to_string_lossy().to_string(),
                lib_dirs: Vec::new(),
                include_dirs: Vec::new(),
                pkg_config_libdir: Vec::new(),
            },
        };
        let (ok, reason) = placeholder.sysroot_ok();
        assert!(!ok, "empty root must be blocked");
        assert!(reason.contains("not populated"), "reason: {reason}");

        // 3. Populated root (usr/) → passes.
        let populated_dir = tmp.path().join("ok-sysroot");
        std::fs::create_dir_all(populated_dir.join("usr")).unwrap();
        let populated = Platform {
            id: "rpi5".into(),
            name: "Raspberry Pi 5".into(),
            os: "linux".into(),
            architecture: PlatformArchitecture {
                cpu_family: "arm".into(),
                cpu: "armv8".into(),
                endian: "little".into(),
                target_triple: "arm-linux-gnueabihf".into(),
                march: None,
            },
            toolchain: PlatformToolchain::default(),
            sysroot: PlatformSysroot {
                root: populated_dir.to_string_lossy().to_string(),
                lib_dirs: Vec::new(),
                include_dirs: Vec::new(),
                pkg_config_libdir: Vec::new(),
            },
        };
        assert!(populated.sysroot_ok().0, "populated root must pass");

        // 4. Host platform (empty root) always passes — native builds need no
        // cross sysroot.
        let host = Platform {
            id: "host".into(),
            name: "Host".into(),
            os: "linux".into(),
            architecture: PlatformArchitecture {
                cpu_family: "x86_64".into(),
                cpu: "x86_64".into(),
                endian: "little".into(),
                target_triple: "x86_64-linux-gnu".into(),
                march: None,
            },
            toolchain: PlatformToolchain::default(),
            sysroot: PlatformSysroot::default(),
        };
        assert!(host.sysroot_ok().0, "host must pass");
    }

    #[test]
    fn discover_directory() {
        let tmp = tempfile::tempdir().unwrap();
        write_yaml(
            tmp.path(),
            "rpi5.yaml",
            r#"
id: rpi5
name: Raspberry Pi 5
os: linux
architecture:
  cpu_family: arm
  cpu: armv8
  endian: little
  target_triple: arm-linux-gnueabihf
toolchain:
  c: clang
  cpp: clang++
  ar: llvm-ar
  strip: llvm-strip
  pkgconfig: /usr/bin/pkg-config
sysroot:
  root: /opt/rpi5-sysroot
  lib_dirs:
    - ${SYSROOT}/usr/lib/arm-linux-gnueabihf
"#,
        );
        // A non-yaml file must be ignored.
        write_yaml(tmp.path(), "README.txt", "not a platform");

        let platforms = Platform::load_directory(tmp.path()).unwrap();
        assert_eq!(platforms.len(), 1);
        assert_eq!(platforms[0].id, "rpi5");
    }
}