// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! In-container provenance probe.
//!
//! The bash harness this replaces reported success while testing a five-day-old
//! installed package, so every run must prove which files it actually loaded:
//! the resolved `vllm.__file__` and whether the compiled extensions import.
//!
//! The probe runs inside the *same* container as the real command. A separate
//! probe container would only prove what some other interpreter could see.

use std::fmt::{self, Write as _};

/// Container path the host checkout is bind-mounted at.
pub const MOUNT: &str = "/src/vllm";

/// Compiled extensions whose absence means a test silently covered the Triton or
/// pure-Python fallback instead of the kernel it names.
pub const EXTENSIONS: [&str; 3] = ["vllm._C", "vllm._rocm_C", "vllm._C_stable_libtorch"];

/// Exit code for a harness or environment fault, as opposed to a genuine build
/// or test failure. Matches Docker's own "could not run the command" code, and
/// pytest never returns it (pytest uses 1-5).
pub const HARNESS_EXIT: u8 = 125;

/// Python source for the probe, fed to `python -` on stdin.
///
/// `strict` in argv makes an unusable environment fail closed with
/// [`HARNESS_EXIT`]; without it the probe only reports, which is what `build`
/// needs because a first-ever build legitimately has no extensions yet.
pub fn probe_source(mount: &str) -> String {
    let mut checks = String::new();
    for module in EXTENSIONS {
        let _ = writeln!(checks, "check({module:?})");
    }
    format!(
        r#"import importlib, importlib.metadata, sys

strict = "strict" in sys.argv[1:]
mount = {mount:?}
failed = []

def check(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"devloop-probe: ext.{{name}}=FAIL {{type(exc).__name__}}")
        failed.append(f"{{name}} does not import ({{type(exc).__name__}})")
    else:
        resolved = getattr(mod, "__file__", None) or "?"
        if not resolved.startswith(mount + "/"):
            print(f"devloop-probe: ext.{{name}}=FAIL outside mount {{resolved}}")
            failed.append(f"{{name}} resolved to {{resolved}}, outside the mounted checkout {{mount}}")
        else:
            print(f"devloop-probe: ext.{{name}}=OK {{resolved}}")


{checks}

try:
    import vllm
except Exception as exc:
    print(f"devloop-probe: vllm_file=FAIL {{type(exc).__name__}}")
    failed.append(f"import vllm failed ({{type(exc).__name__}})")
else:
    resolved = vllm.__file__ or "?"
    print(f"devloop-probe: vllm_file={{resolved}}")
    if not resolved.startswith(mount + "/"):
        failed.append(f"vllm resolved to {{resolved}}, outside the mounted checkout {{mount}}")
try:
    dist = importlib.metadata.distribution("vllm")
    version = dist.version or ""
    metadata_path = str(getattr(dist, "_path", ""))
    print(f"devloop-probe: vllm_metadata=version={{version}} path={{metadata_path}}")
    if not version:
        failed.append("vllm distribution has an empty version")
    if not metadata_path or metadata_path.startswith(mount + "/"):
        failed.append(f"vllm metadata resolved under mounted checkout: {{metadata_path or '?'}}")
except Exception as exc:
    print(f"devloop-probe: vllm_metadata=FAIL {{type(exc).__name__}}")
    failed.append(f"vllm distribution metadata failed ({{type(exc).__name__}})")
print(f"devloop-probe: python={{sys.version.split()[0]}} executable={{sys.executable}}")

if failed and strict:
    for problem in failed:
        print(f"devloop-probe: FATAL {{problem}}", file=sys.stderr)
    print(
        "devloop-probe: refusing to run: the environment would test something "
        "other than this checkout, or would silently exercise a fallback path.",
        file=sys.stderr,
    )
    sys.exit({harness_exit})
"#,
        mount = mount,
        checks = checks.trim_end(),
        harness_exit = HARNESS_EXIT,
    )
}

/// What the probe observed, parsed from its stdout.
///
/// Only `doctor` parses this; `build` and `test` stream the probe's output
/// straight through so pytest's own output is not captured, and rely on the
/// in-container strict exit instead.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ProbeReport {
    /// Resolved `vllm.__file__`, or `None` if `import vllm` failed.
    pub vllm_file: Option<String>,
    /// `(module, Ok(path) | Err(reason))` per compiled extension, probe order.
    pub extensions: Vec<(String, Result<String, String>)>,
    /// Interpreter version reported by the probe.
    pub python: Option<String>,
    /// Distribution version and metadata directory.
    pub vllm_metadata: Option<Result<(String, String), String>>,
}

impl ProbeReport {
    /// Parse `devloop-probe:` lines, ignoring any other output. vLLM logs
    /// warnings to stdout on import, so unrelated lines are expected.
    pub fn parse(stdout: &str) -> Self {
        let mut report = Self::default();
        for line in stdout.lines() {
            let Some(rest) = line.trim().strip_prefix("devloop-probe: ") else {
                continue;
            };
            let Some((key, value)) = rest.split_once('=') else {
                continue;
            };
            match key {
                "vllm_file" => {
                    report.vllm_file =
                        (value != "FAIL" && !value.starts_with("FAIL ")).then(|| value.to_string());
                }
                "python" => {
                    report.python = value.split_whitespace().next().map(str::to_string);
                }
                "vllm_metadata" => {
                    report.vllm_metadata = match value.strip_prefix("FAIL ") {
                        Some(reason) => Some(Err(reason.to_string())),
                        None => {
                            let fields: Vec<_> = value.split_whitespace().collect();
                            match (fields.first(), fields.get(1)) {
                                (Some(version), Some(path)) => Some(Ok((
                                    version.strip_prefix("version=").unwrap_or(version).into(),
                                    path.strip_prefix("path=").unwrap_or(path).into(),
                                ))),
                                _ => Some(Err(value.to_string())),
                            }
                        }
                    };
                }
                _ => {
                    if let Some(module) = key.strip_prefix("ext.") {
                        let outcome = match value.split_once(' ') {
                            Some(("OK", path)) => Ok(path.to_string()),
                            _ => Err(value.trim_start_matches("FAIL").trim().to_string()),
                        };
                        report.extensions.push((module.to_string(), outcome));
                    }
                }
            }
        }
        report
    }

    /// Reasons this environment must not be trusted to test the checkout.
    /// Empty means the run provably loaded this checkout's compiled extensions.
    pub fn faults(&self, mount: &str) -> Vec<String> {
        let mut faults = Vec::new();
        match &self.vllm_file {
            None => faults.push("import vllm failed inside the container".to_string()),
            Some(path) if !path.starts_with(&format!("{mount}/")) => faults.push(format!(
                "vllm resolved to {path}, outside the mounted checkout {mount}"
            )),
            Some(_) => {}
        }
        match &self.vllm_metadata {
            None => faults.push("vllm distribution metadata was not probed".into()),
            Some(Err(reason)) => {
                faults.push(format!("vllm distribution metadata failed ({reason})"))
            }
            Some(Ok((version, path))) if version.is_empty() => {
                faults.push("vllm distribution has an empty version".into())
            }
            Some(Ok((_, path))) if path.is_empty() || path.starts_with(&format!("{mount}/")) => {
                faults.push(format!(
                    "vllm metadata resolved under mounted checkout: {}",
                    if path.is_empty() { "?" } else { path }
                ))
            }
            Some(Ok(_)) => {}
        }
        for expected in EXTENSIONS {
            match self.extensions.iter().find(|(m, _)| m == expected) {
                None => faults.push(format!("{expected} was not probed")),
                Some((_, Err(reason))) => {
                    let reason = if reason.is_empty() {
                        "import failed"
                    } else {
                        reason
                    };
                    faults.push(format!("{expected} does not import ({reason})"));
                }
                Some((_, Ok(path))) if !path.starts_with(&format!("{mount}/")) => faults.push(
                    format!("{expected} resolved to {path}, outside the mounted checkout {mount}"),
                ),
                Some((_, Ok(_))) => {}
            }
        }
        faults
    }
}

impl fmt::Display for ProbeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Widened to the longest label so the values line up in a log.
        writeln!(
            f,
            "  {:<23} {}",
            "vllm.__file__",
            self.vllm_file
                .as_deref()
                .unwrap_or("FAIL (import vllm failed)")
        )?;
        for (module, outcome) in &self.extensions {
            match outcome {
                Ok(path) => writeln!(f, "  {module:<23} {path}")?,
                Err(reason) => writeln!(f, "  {module:<23} FAIL {reason}")?,
            }
        }
        if let Some(python) = &self.python {
            writeln!(f, "  {:<23} {python}", "python")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real probe output, including the amdsmi warning vLLM prints to stdout.
    const HEALTHY: &str = "\
WARNING 07-30 11:45:49 [rocm.py:42] Failed to import from amdsmi
devloop-probe: vllm_file=/src/vllm/vllm/__init__.py
devloop-probe: vllm_metadata=version=0.1 path=/opt/venv/lib/python3.12/site-packages/vllm-0.1.dist-info
devloop-probe: ext.vllm._C=OK /src/vllm/vllm/_C.abi3.so
devloop-probe: ext.vllm._rocm_C=OK /src/vllm/vllm/_rocm_C.abi3.so
devloop-probe: ext.vllm._C_stable_libtorch=OK /src/vllm/vllm/_C_stable_libtorch.abi3.so
devloop-probe: python=3.12.11 executable=/opt/venv/bin/python
";

    #[test]
    fn parses_healthy_probe_and_ignores_unrelated_log_lines() {
        let report = ProbeReport::parse(HEALTHY);
        assert_eq!(
            report.vllm_file.as_deref(),
            Some("/src/vllm/vllm/__init__.py")
        );
        assert_eq!(report.python.as_deref(), Some("3.12.11"));
        assert_eq!(report.extensions.len(), 3);
        assert!(report.faults(MOUNT).is_empty());
    }

    /// The exact failure the bash harness hid: sources load, extensions do not,
    /// so a kernel test silently exercises the Triton fallback and passes.
    #[test]
    fn missing_extension_is_a_fault() {
        let stdout = "\
devloop-probe: vllm_file=/src/vllm/vllm/__init__.py
devloop-probe: ext.vllm._C=OK /src/vllm/vllm/_C.abi3.so
devloop-probe: ext.vllm._rocm_C=FAIL ImportError
devloop-probe: ext.vllm._C_stable_libtorch=OK /src/vllm/vllm/_C_stable_libtorch.abi3.so
";
        let faults = ProbeReport::parse(stdout).faults(MOUNT);
        assert_eq!(faults.len(), 2, "{faults:?}");
        assert!(
            faults.iter().any(|fault| fault.contains("vllm._rocm_C")),
            "{faults:?}"
        );
    }

    #[test]
    fn vllm_import_failure_still_reports_extension_records() {
        let stdout = "\
devloop-probe: vllm_file=FAIL ImportError
devloop-probe: ext.vllm._C=FAIL ImportError
devloop-probe: ext.vllm._rocm_C=OK /src/vllm/vllm/_rocm_C.abi3.so
devloop-probe: ext.vllm._C_stable_libtorch=FAIL ImportError
";
        let report = ProbeReport::parse(stdout);

        assert_eq!(report.extensions.len(), EXTENSIONS.len());
        assert_eq!(report.faults(MOUNT).len(), 4);
    }

    /// Testing an installed copy instead of the checkout is the other half of
    /// the same bug, and passes every extension check while proving nothing.
    #[test]
    fn vllm_outside_the_mount_is_a_fault() {
        let stdout = "\
devloop-probe: vllm_file=/opt/venv/lib/python3.12/site-packages/vllm/__init__.py
devloop-probe: ext.vllm._C=OK /opt/venv/lib/python3.12/site-packages/vllm/_C.abi3.so
devloop-probe: ext.vllm._rocm_C=OK /opt/venv/lib/python3.12/site-packages/vllm/_rocm_C.abi3.so
devloop-probe: ext.vllm._C_stable_libtorch=OK /opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
";
        let faults = ProbeReport::parse(stdout).faults(MOUNT);
        assert_eq!(faults.len(), 5, "{faults:?}");
        assert!(
            faults[0].contains("outside the mounted checkout"),
            "{faults:?}"
        );
    }

    #[test]
    fn extension_outside_the_mount_is_a_fault() {
        let stdout = HEALTHY.replace(
            "/src/vllm/vllm/_rocm_C.abi3.so",
            "/opt/venv/lib/vllm/_rocm_C.abi3.so",
        );
        let faults = ProbeReport::parse(&stdout).faults(MOUNT);
        assert_eq!(faults.len(), 1, "{faults:?}");
        assert!(faults[0].contains("vllm._rocm_C"), "{faults:?}");
    }

    /// Silence must never read as success: no probe output at all is a fault.
    #[test]
    fn empty_output_is_a_fault_not_a_pass() {
        let faults = ProbeReport::parse("").faults(MOUNT);
        assert_eq!(faults.len(), 5, "{faults:?}");
    }

    #[test]
    fn strict_source_exits_with_harness_code_and_covers_every_extension() {
        let source = probe_source(MOUNT);
        assert_eq!(source.matches("check(\"vllm.").count(), EXTENSIONS.len());
        assert!(source.contains(&format!("sys.exit({HARNESS_EXIT})")));
        for module in EXTENSIONS {
            assert!(source.contains(&format!("check({module:?})")), "{module}");
        }
    }
}
