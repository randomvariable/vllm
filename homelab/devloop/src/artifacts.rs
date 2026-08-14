// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Host-side view of the compiled extensions a build produced.
//!
//! `build_ext --inplace` writes `.so` files into the bind-mounted checkout, so
//! they are readable from the host for free. That matters: `import vllm` costs
//! ~4.2s in this image, and paying it after every build would make a ~10s no-op
//! loop 50% slower, which the fork's hard requirements treat as a defect.
//!
//! What this proves is narrower than the in-container probe, and the split is
//! deliberate. Here: which files exist and when they were written. There: that
//! they actually import. A build is not trusted on this evidence alone -- `test`
//! runs the strict probe before pytest, where the import cost is paid anyway.

use std::fmt::Write as _;
use std::path::Path;
use std::time::SystemTime;

/// One compiled extension found in the checkout.
#[derive(Debug, PartialEq, Eq)]
pub struct Artifact {
    pub name: String,
    pub bytes: u64,
    /// Seconds since the run started, negative meaning written before it.
    pub age_secs: i64,
}

impl Artifact {
    /// `fresh` distinguishes "this build wrote it" from "left by an earlier one",
    /// which is the staleness signal a log reader needs.
    pub fn fresh(&self) -> bool {
        self.age_secs >= 0
    }
}

/// Compiled extensions in `<repo>/vllm`, sorted by name.
///
/// `started` is when the build began, so anything modified after it was written
/// by this run.
pub fn scan(repo: &Path, started: SystemTime) -> Vec<Artifact> {
    let Ok(entries) = std::fs::read_dir(repo.join("vllm")) else {
        return Vec::new();
    };
    let mut artifacts: Vec<Artifact> = entries
        .flatten()
        .filter_map(|entry| {
            if entry.path().extension() != Some(std::ffi::OsStr::new("so")) {
                return None;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let meta = entry.metadata().ok()?;
            let modified = meta.modified().ok()?;
            let age_secs = match modified.duration_since(started) {
                Ok(after) => i64::try_from(after.as_secs()).unwrap_or(i64::MAX),
                Err(before) => -i64::try_from(before.duration().as_secs()).unwrap_or(i64::MAX),
            };
            Some(Artifact {
                name,
                bytes: meta.len(),
                age_secs,
            })
        })
        .collect();
    artifacts.sort_by(|a, b| a.name.cmp(&b.name));
    artifacts
}

/// Render for the provenance banner, flagging what this run rewrote.
pub fn render(artifacts: &[Artifact]) -> String {
    if artifacts.is_empty() {
        return "  (no compiled extensions in the checkout)\n".to_string();
    }
    let mut out = String::new();
    for artifact in artifacts {
        let mark = if artifact.fresh() {
            "rebuilt"
        } else {
            "unchanged"
        };
        // KiB keeps the arithmetic in integers: these are tens of MiB, and an
        // exact size is more useful in a log than a rounded float.
        let kib = artifact.bytes / 1024;
        let _ = writeln!(out, "  {:<39} {kib:>9} KiB  {mark}", artifact.name);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn artifact(name: &str, age_secs: i64) -> Artifact {
        Artifact {
            name: name.to_string(),
            bytes: 4 * 1024 * 1024,
            age_secs,
        }
    }

    /// The point of the age field: a log must show which files this run wrote.
    #[test]
    fn freshness_separates_this_run_from_earlier_ones() {
        assert!(artifact("_C.abi3.so", 2).fresh());
        assert!(!artifact("_C.abi3.so", -900).fresh());
    }

    #[test]
    fn render_marks_rebuilt_and_unchanged_files() {
        let rendered = render(&[artifact("_C.abi3.so", 3), artifact("_rocm_C.abi3.so", -60)]);
        assert!(rendered.contains("_C.abi3.so"), "{rendered}");
        assert!(rendered.contains("rebuilt"), "{rendered}");
        assert!(rendered.contains("unchanged"), "{rendered}");
    }

    /// An empty checkout must read as absence, never as a silent success.
    #[test]
    fn render_says_so_when_nothing_was_built() {
        assert!(render(&[]).contains("no compiled extensions"));
    }

    #[test]
    fn scan_finds_shared_objects_and_ignores_other_files() {
        let dir = std::env::temp_dir().join(format!("devloop-scan-{}", std::process::id()));
        let vllm = dir.join("vllm");
        std::fs::create_dir_all(&vllm).unwrap();
        std::fs::write(vllm.join("_C.abi3.so"), b"binary").unwrap();
        std::fs::write(vllm.join("__init__.py"), b"source").unwrap();

        let started = SystemTime::now() - Duration::from_mins(1);
        let found = scan(&dir, started);
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].name, "_C.abi3.so");
        assert!(found[0].fresh(), "written after the run started");
    }

    /// A missing directory must not panic; `doctor` reports that separately.
    #[test]
    fn scan_of_a_missing_checkout_is_empty() {
        assert!(scan(Path::new("/nonexistent-devloop-path"), SystemTime::now()).is_empty());
    }
}
