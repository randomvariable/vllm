// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Incremental build and test harness for vLLM on AMD Strix Halo (gfx1151).
//!
//! Replaces `homelab/rocm-dev-test.sh`, which overlaid Python sources onto an
//! installed package and could report success while testing week-old code. The
//! rule that follows from that: an environment which cannot prove it is testing
//! this checkout must fail closed, never pass quietly.
//!
//! Each subcommand is one shot and cold-start safe; there is no container to
//! enter first. Only the two named volumes persist, so runs are always `--rm`.

mod artifacts;
mod docker;
mod probe;

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use docker::{BUILD_CMD, Check, Env, PYTHON};
use probe::{HARNESS_EXIT, MOUNT, ProbeReport};

#[derive(Parser)]
#[command(
    name = "vllm-devloop",
    about = "Incremental build/test loop for vLLM on AMD Strix Halo (gfx1151)",
    long_about = "Runs incremental HIP/C++ builds and pytest inside the \
                  vllm-strix-devtools container, printing provenance so a log \
                  alone proves which files a run loaded.",
    disable_help_subcommand = true
)]
struct Cli {
    /// Container image to use, for testing an alternative devtools build.
    #[arg(long, global = true)]
    image: Option<String>,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Incremental HIP/C++ build; extra args pass to `setup.py build_ext`.
    Build {
        /// Arguments appended to `vllm-hip-build`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run pytest in the container; args pass through unchanged.
    Test {
        /// pytest arguments, e.g. `tests/kernels/... -k sliding_window`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Report what is and is not present, without building.
    Doctor,
    /// Create the image and volumes if absent. Idempotent.
    Setup,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let env = match Env::discover(env!("CARGO_MANIFEST_DIR"), cli.image) {
        Ok(env) => env,
        Err(err) => return fail(&err),
    };

    let result = match cli.command {
        Cmd::Build { args } => build(&env, &args),
        Cmd::Test { args } => test(&env, &args),
        Cmd::Doctor => doctor(&env),
        Cmd::Setup => setup(&env),
    };

    match result {
        Ok(code) => code,
        Err(err) => fail(&err),
    }
}

/// Report a harness fault distinguishably from a build or test failure.
fn fail(err: &anyhow::Error) -> ExitCode {
    eprintln!("vllm-devloop: {err:#}");
    ExitCode::from(HARNESS_EXIT)
}

/// Refuse to start unless every precondition holds.
///
/// Checked up front and all at once: a run that begins against a missing volume
/// or a stale image wastes minutes and can produce a misleading pass.
fn require_ready(env: &Env) -> Result<String> {
    let checks = docker::checks(env);
    let mut remedies = String::new();
    for check in &checks {
        if let Err(remedy) = &check.outcome {
            let _ = writeln!(remedies, "  {}: {remedy}", check.what);
        }
    }
    if !remedies.is_empty() {
        bail!(
            "environment is not ready, refusing to run:\n{}",
            remedies.trim_end()
        );
    }
    docker::image_id(&env.image).context("image disappeared between checks")
}

/// Provenance banner. Requirement: a log alone must identify the run's inputs.
fn banner(env: &Env, image_id: &str, action: &str) {
    println!("vllm-devloop: {action}");
    println!("  image             {} ({image_id})", env.image);
    println!("  checkout          {} -> {MOUNT} (rw)", env.repo.display());
    println!("  volumes           {}", env.volumes().join(", "));
}

/// Held for the run's duration; dropping it releases the lock. Opaque because
/// fd-lock's read and write guards are different types.
type LockGuard = Box<dyn std::any::Any>;

/// Path of the lock guarding one checkout's shared state.
///
/// Derived from the checkout path so separate checkouts never block each other,
/// while every lane against one checkout shares a single lock.
fn lock_path(repo: &std::path::Path) -> PathBuf {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in repo.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    std::env::temp_dir().join(format!("vllm-devloop-{hash:016x}.lock"))
}

/// Lock guarding the shared cmake tree and the in-place `.so` files.
///
/// Concurrent lanes are expected, so this serialises rather than racing. Two
/// builds in one cmake tree corrupt ninja state; a test importing a `.so` that a
/// build is rewriting fails in ways that look like a kernel bug. Both build and
/// test lanes take the lock exclusively because tests perform a freshness build.
///
/// The `RwLock` is leaked so the returned guard can own the lock outright: the
/// guard borrows it, and this process is short-lived, with the kernel releasing
/// the lock when the fd closes at exit.
fn lock(env: &Env, exclusive: bool) -> Result<LockGuard> {
    let path = lock_path(&env.repo);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("cannot open lock file {}", path.display()))?;
    // Advisory only, on its own descriptor so the real lock below is borrowed
    // exactly once. A lane arriving between this check and that acquisition just
    // means the wait happens without the message.
    if let Ok(peek) = std::fs::File::open(&path) {
        let mut peek = fd_lock::RwLock::new(peek);
        let free = if exclusive {
            peek.try_write().is_ok()
        } else {
            peek.try_read().is_ok()
        };
        if !free {
            let kind = if exclusive { "build" } else { "test" };
            println!(
                "vllm-devloop: another run holds {}; waiting ({kind} lane)",
                path.display()
            );
        }
    }

    let lock: &'static mut fd_lock::RwLock<std::fs::File> =
        Box::leak(Box::new(fd_lock::RwLock::new(file)));
    if exclusive {
        Ok(Box::new(lock.write().context("cannot acquire build lock")?))
    } else {
        Ok(Box::new(lock.read().context("cannot acquire test lock")?))
    }
}

/// Run a command in the container, inheriting stdio so output streams live.
fn run_in_container(env: &Env, command: &[String], stdin: bool) -> Result<ExitCode> {
    let args = env.run_args(command, stdin);
    let status = Command::new("docker")
        .args(&args)
        .status()
        .context("failed to spawn docker")?;
    // Exit codes are 0-255; a signalled child reports None, which is a fault.
    let code = status
        .code()
        .and_then(|c| u8::try_from(c).ok())
        .unwrap_or(HARNESS_EXIT);
    // 125-127 are Docker's own "could not run it" range, never the inner
    // command's verdict, so they must not be reported as a test failure.
    if matches!(code, 125..=127) {
        bail!(
            "docker could not run the command (exit {code}); \
             this is an environment fault, not a test failure"
        );
    }
    Ok(ExitCode::from(code))
}

/// Container command that runs the strict probe, then the payload, in ONE
/// container.
///
/// One container, not two: the provenance then describes the very interpreter
/// that runs the payload, rather than what some other interpreter could see.
/// Strict, and first, so an environment that would test something other than
/// this checkout exits [`HARNESS_EXIT`] before the payload starts.
fn with_strict_probe(payload: &[String]) -> Vec<String> {
    let mut command: Vec<String> = [
        "bash",
        "-c",
        "set -eu; py=\"$1\"; probe=\"$2\"; shift 2; \"$py\" -P -c \"$probe\" strict; exec \"$@\"",
        "vllm-devloop",
        PYTHON,
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    command.push(probe::probe_source(MOUNT));
    command.extend(payload.iter().cloned());
    command
}

fn build(env: &Env, args: &[String]) -> Result<ExitCode> {
    let image_id = require_ready(env)?;
    let _guard = lock(env, true)?;
    banner(env, &image_id, "incremental HIP/C++ build");
    println!("  command           {BUILD_CMD} {}", args.join(" "));

    // No in-container probe here, deliberately. `import vllm` costs ~4.2s in this
    // image and a no-op rebuild costs ~10s, so probing every build would make the
    // loop 50% slower for evidence the build does not need: a first-ever build has
    // no extensions to import yet. The artefacts land in the bind-mounted
    // checkout, so the host can report them for free, and `test` pays the import
    // cost once, where pytest would pay it anyway.
    let started = std::time::SystemTime::now();
    let mut command = vec![BUILD_CMD.to_string()];
    command.extend(args.iter().cloned());
    let code = run_in_container(env, &command, false)?;

    if code == ExitCode::SUCCESS {
        println!(
            "vllm-devloop: compiled extensions in {}/vllm",
            env.repo.display()
        );
        print!(
            "{}",
            artifacts::render(&artifacts::scan(&env.repo, started))
        );
        println!("vllm-devloop: run `cargo make test -- <paths>` to verify they import.");
    }
    Ok(code)
}

fn test(env: &Env, args: &[String]) -> Result<ExitCode> {
    let args = strip_cargo_make_separator(args);
    if args
        .iter()
        .any(|arg| arg == "--import-mode" || arg.starts_with("--import-mode="))
    {
        bail!("--import-mode is controlled by vllm-devloop and must not be provided");
    }
    if !has_target(args) {
        bail!(
            "pytest target path or node ID required; flags alone (including `-k foo`) ".to_string()
                + "would run the whole suite; e.g. `cargo make test -- "
                + "tests/kernels/attention -k sliding_window`"
        );
    }
    let image_id = require_ready(env)?;
    // Let Ninja decide whether anything is stale, but never let pytest observe
    // artifacts left by a source change that has not been rebuilt.
    let _guard = lock(env, true)?;
    banner(env, &image_id, "pytest");
    let build_code = run_in_container(env, &[BUILD_CMD.to_string()], false)?;
    if build_code != ExitCode::SUCCESS {
        return Ok(build_code);
    }
    println!(
        "  command           {PYTHON} -P -m pytest --import-mode=importlib {}",
        args.join(" ")
    );

    let mut payload: Vec<String> = [PYTHON, "-P", "-m", "pytest", "--import-mode=importlib"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    payload.extend(args.iter().cloned());
    run_in_container(env, &with_strict_probe(&payload), false)
}

/// Remove cargo-make's argument separator without changing direct invocations.
fn strip_cargo_make_separator(args: &[String]) -> &[String] {
    args.strip_prefix(&["--".to_string()]).unwrap_or(args)
}

/// Require a path-like pytest target or node ID, while allowing flags.
fn has_target(args: &[String]) -> bool {
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--" {
            return args[index + 1..].iter().any(is_target);
        }
        if !arg.starts_with('-') && is_target(arg) {
            return true;
        }
        index += if arg.starts_with('-') && !arg.contains('=') && option_consumes_next(arg) {
            2
        } else {
            1
        };
    }
    false
}

fn is_target(arg: &String) -> bool {
    arg.contains("::")
        || arg.ends_with(".py")
        || arg.contains('/')
        || arg.starts_with("./")
        || arg.starts_with("../")
        || std::path::Path::new(arg).is_dir()
}

fn option_consumes_next(arg: &str) -> bool {
    if arg.starts_with('-') && !arg.starts_with("--") && arg.len() > 2 {
        return false;
    }
    !matches!(
        arg,
        "-q" | "-v"
            | "-s"
            | "-x"
            | "-l"
            | "-f"
            | "-h"
            | "--quiet"
            | "--verbose"
            | "--capture"
            | "--no-capture"
            | "--exitfirst"
            | "--showlocals"
            | "--failed-first"
            | "--last-failed"
            | "--strict-markers"
            | "--strict-config"
            | "--disable-warnings"
            | "--no-header"
            | "--no-summary"
            | "--collect-only"
            | "--trace-config"
            | "--debug"
            | "--version"
            | "--help"
            | "--pdb"
            | "--pdbcls"
            | "--setup-plan"
            | "--setup-show"
    )
}

/// Run the probe alone and capture it, for `doctor` and post-build reporting.
fn probe(env: &Env) -> Result<ProbeReport> {
    let command: Vec<String> = vec![
        PYTHON.to_string(),
        "-P".to_string(),
        "-c".to_string(),
        probe::probe_source(MOUNT),
    ];
    let output = Command::new("docker")
        .args(env.run_args(&command, false))
        .stderr(Stdio::inherit())
        .output()
        .context("failed to spawn docker")?;
    if !output.status.success() {
        bail!("probe container exited {}", output.status);
    }
    Ok(ProbeReport::parse(&String::from_utf8_lossy(&output.stdout)))
}

fn doctor(env: &Env) -> Result<ExitCode> {
    println!("vllm-devloop: environment check");
    println!("  checkout          {}", env.repo.display());
    let checks = docker::checks(env);
    let mut blocked = false;
    for Check { what, outcome } in &checks {
        match outcome {
            Ok(detail) => println!("  ok    {what:<34} {detail}"),
            Err(remedy) => {
                blocked = true;
                println!("  FAIL  {what:<34} {remedy}");
            }
        }
    }
    if blocked {
        println!("\nvllm-devloop: environment incomplete; fix the FAIL lines above.");
        return Ok(ExitCode::from(1));
    }

    // Only meaningful once the image and volumes exist, so it runs last.
    let report = probe(env)?;
    println!("\nvllm-devloop: in-container provenance");
    print!("{report}");
    let faults = report.faults(MOUNT);
    if faults.is_empty() {
        println!("\nvllm-devloop: ready; this checkout's compiled extensions load.");
        return Ok(ExitCode::SUCCESS);
    }
    for fault in &faults {
        println!("  FAIL  {fault}");
    }
    println!("\nvllm-devloop: tests would not exercise this checkout. Run `cargo make build`.");
    Ok(ExitCode::from(1))
}

fn setup(env: &Env) -> Result<ExitCode> {
    let mut missing_volumes = Vec::new();
    let mut needs_image = false;
    for check in docker::checks(env) {
        let Err(remedy) = check.outcome else { continue };
        if let Some(volume) = check.what.strip_prefix("volume ") {
            missing_volumes.push(volume.to_string());
        } else if check.what.starts_with("image ") {
            needs_image = true;
        } else {
            bail!("cannot fix `{}` automatically: {remedy}", check.what);
        }
    }

    for volume in &missing_volumes {
        println!("vllm-devloop: creating volume {volume}");
        let status = Command::new("docker")
            .args(["volume", "create", volume])
            .status()
            .context("failed to spawn docker")?;
        if !status.success() {
            bail!("docker volume create {volume} failed");
        }
    }

    if needs_image {
        println!(
            "vllm-devloop: building image {} (first build takes ~40 min)",
            env.image
        );
        let status = Command::new("docker")
            .args([
                "build",
                "-f",
                "homelab/strix.Dockerfile",
                "--target",
                "devtools",
                "-t",
                &env.image,
                ".",
            ])
            .current_dir(&env.repo)
            .status()
            .context("failed to spawn docker")?;
        if !status.success() {
            bail!("docker build failed");
        }
    }

    if missing_volumes.is_empty() && !needs_image {
        println!("vllm-devloop: image and volumes already present, nothing to do");
    }
    println!("vllm-devloop: setup complete; run `cargo make doctor` to verify");
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    /// pytest flags must reach pytest, not be eaten as devloop flags.
    #[test]
    fn pytest_flags_pass_through_untouched() {
        let cli = Cli::parse_from([
            "vllm-devloop",
            "test",
            "tests/kernels/attention",
            "-k",
            "sliding_window",
            "-x",
        ]);
        let Cmd::Test { args } = cli.command else {
            panic!("expected test")
        };
        assert_eq!(
            args,
            ["tests/kernels/attention", "-k", "sliding_window", "-x"]
        );
    }

    /// `build_ext` flags such as `-j` must survive too.
    #[test]
    fn build_arguments_pass_through_untouched() {
        let cli = Cli::parse_from(["vllm-devloop", "build", "-j", "4"]);
        let Cmd::Build { args } = cli.command else {
            panic!("expected build")
        };
        assert_eq!(args, ["-j", "4"]);
    }

    /// A bare `test` would run the entire suite on a dev box; refuse instead.
    #[test]
    fn test_without_paths_is_rejected() {
        let env = Env::discover("/home/dev/vllm/homelab/devloop", None).unwrap();
        let err = test(&env, &[]).expect_err("empty args must be refused");
        assert!(
            err.to_string().contains("pytest target path or node ID"),
            "{err}"
        );
    }

    #[test]
    fn option_only_pytest_invocation_is_rejected() {
        let args = ["-k", "foo"].map(String::from);
        assert!(!has_target(&args));
    }

    #[test]
    fn path_like_pytest_option_values_are_not_targets() {
        for args in [
            ["--basetemp", "/tmp/run"],
            ["-k", "tests/foo"],
            ["--junitxml", "report.xml"],
        ] {
            let args = args.into_iter().map(String::from).collect::<Vec<_>>();
            assert!(!has_target(&args), "{args:?}");
        }
    }

    #[test]
    fn arbitrary_option_values_and_equals_forms_are_not_targets() {
        for args in [vec!["--ignore", "/tmp/tests"], vec!["--foo=tests/foo"]] {
            let args = args.into_iter().map(String::from).collect::<Vec<_>>();
            assert!(!has_target(&args), "{args:?}");
        }
    }

    #[test]
    fn combined_short_flags_do_not_consume_targets() {
        for flag in ["-vv", "-qf", "-xq"] {
            let args = [flag, "tests/foo.py"].map(String::from);
            assert!(has_target(&args), "{args:?}");
        }
    }

    #[test]
    fn setup_plan_does_not_consume_target() {
        let args = ["--setup-plan", "tests/foo.py"].map(String::from);
        assert!(has_target(&args), "{args:?}");
    }

    #[test]
    fn setup_show_does_not_consume_target() {
        let args = ["--setup-show", "tests/foo.py"].map(String::from);
        assert!(has_target(&args), "{args:?}");
    }

    #[test]
    fn cargo_make_separator_is_stripped() {
        let args = ["--", "tests/kernels/attention", "-k", "foo"].map(String::from);
        assert_eq!(strip_cargo_make_separator(&args), &args[1..]);
    }

    #[test]
    fn value_only_pytest_invocation_is_rejected() {
        let args = ["--junitxml", "report.xml"].map(String::from);
        assert!(!has_target(&args));
    }

    #[test]
    fn pytest_target_and_flags_are_accepted() {
        let args = ["tests/kernels/attention", "-k", "foo"].map(String::from);
        assert!(has_target(&args));
    }

    #[test]
    fn pytest_node_id_is_accepted() {
        let args = ["tests/test_engine.py::test_start", "-q"].map(String::from);
        assert!(has_target(&args));
    }

    /// Both lanes must fail closed, not fall back to a default image.
    #[test]
    fn image_override_reaches_the_environment() {
        let cli = Cli::parse_from(["vllm-devloop", "--image", "bogus:tag", "doctor"]);
        let env = Env::discover("/home/dev/vllm/homelab/devloop", cli.image).unwrap();
        assert_eq!(env.image, "bogus:tag");
    }

    /// The probe must gate pytest, or a run could test a stale installed package
    /// and pass, the exact failure this harness exists to prevent.
    #[test]
    fn test_runs_the_strict_probe_before_pytest() {
        let payload = [PYTHON, "-m", "pytest", "tests/x.py"].map(String::from);
        let command = with_strict_probe(&payload);
        let script = &command[2];
        assert!(script.contains("strict"), "{script}");
        assert!(script.contains("exec"), "{script}");
        assert_eq!(&command[command.len() - payload.len()..], payload);
    }

    /// Probe and pytest share one container: two would add ~5s per run, and the
    /// provenance would describe a different interpreter than the one testing.
    #[test]
    fn probe_shares_the_pytest_container() {
        let command = with_strict_probe(&[PYTHON.to_string(), "-m".into(), "pytest".into()]);
        assert_eq!(command[0], "bash");
        assert_eq!(command.iter().filter(|a| *a == "bash").count(), 1);
    }
}
