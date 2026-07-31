// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Container invocation and environment checks.
//!
//! The flags here are the verified warm invocation for the `devtools` stage of
//! `homelab/strix.Dockerfile`; see that stage's VOLUME CONTRACT comment. The
//! image bakes every `VLLM_*`, `ROCM_*` and ccache variable, so this harness
//! deliberately passes no `-e` flags: a variable set here would either be a
//! silent no-op or would shadow a value the image chose for a reason.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Image built by `--target devtools`.
pub const IMAGE: &str = "vllm-strix-devtools:local";
/// Kernel interfaces required for HSA to see the GPU.
pub const DEVICES: [&str; 2] = ["/dev/kfd", "/dev/dri"];
/// In-container build entry point; keeps the cmake tree on the build volume.
pub const BUILD_CMD: &str = "vllm-hip-build";
/// The release venv, whose torch has the InitDma-safe HSA runtime.
pub const PYTHON: &str = "/opt/venv/bin/python";

/// Where the container looks for things, and where the harness runs.
#[derive(Debug, Clone)]
pub struct Env {
    /// Host checkout, bind-mounted read-write at [`probe::MOUNT`].
    pub repo: PathBuf,
    pub image: String,
}

impl Env {
    /// Stable, Docker-safe key for state belonging to this checkout.
    pub fn checkout_key(&self) -> String {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.repo.to_string_lossy().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        format!("{hash:016x}")
    }

    /// Persistent volumes are checkout-scoped so CMake never shares a cache
    /// containing absolute paths from another clone.
    pub fn volumes(&self) -> [String; 2] {
        let key = self.checkout_key();
        [
            format!("vllm-strix-build-{key}"),
            format!("vllm-strix-ccache-{key}"),
        ]
    }

    /// Resolve the repo root from this binary's location rather than the cwd, so
    /// `cargo make` and a bare `vllm-devloop` behave identically.
    pub fn discover(manifest_dir: &str, image: Option<String>) -> anyhow::Result<Self> {
        let repo = Path::new(manifest_dir)
            .ancestors()
            .nth(2)
            .ok_or_else(|| anyhow::anyhow!("cannot derive repo root from {manifest_dir}"))?
            .to_path_buf();
        Ok(Self {
            repo,
            image: image.unwrap_or_else(|| IMAGE.to_string()),
        })
    }

    /// `docker run` argv for a command in the devtools container.
    ///
    /// `--rm` because only the two named volumes persist. `seccomp=unconfined`
    /// is required for HSA; `--ipc=host` avoids shared-memory limits in
    /// multi-process tests. The mount is read-write because `build_ext
    /// --inplace` writes `.so` files into the checkout.
    pub fn run_args(&self, command: &[String], stdin: bool) -> Vec<String> {
        let volumes = self.volumes();
        let mut args: Vec<String> = ["run", "--rm"].iter().map(ToString::to_string).collect();
        if stdin {
            args.push("-i".to_string());
        }
        args.extend(
            [
                "--device",
                DEVICES[0],
                "--device",
                DEVICES[1],
                "--group-add",
                "video",
                "--security-opt",
                "seccomp=unconfined",
                "--ipc=host",
                "-v",
            ]
            .iter()
            .map(ToString::to_string),
        );
        args.push(format!("{}:{}", self.repo.display(), crate::probe::MOUNT));
        args.extend(
            [
                "-v",
                &format!("{}:/vllm-build", volumes[0]),
                "-v",
                &format!("{}:/ccache", volumes[1]),
                &self.image,
            ]
            .iter()
            .map(ToString::to_string),
        );
        args.extend(command.iter().cloned());
        args
    }
}

/// One environment precondition and whether it holds.
#[derive(Debug, PartialEq, Eq)]
pub struct Check {
    pub what: String,
    /// Detail on success, actionable remedy on failure.
    pub outcome: Result<String, String>,
}

impl Check {
    fn ok(what: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            outcome: Ok(detail.into()),
        }
    }

    fn fail(what: impl Into<String>, remedy: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            outcome: Err(remedy.into()),
        }
    }
}

/// Trimmed stdout of a successful `docker` invocation.
fn docker_stdout(args: &[&str]) -> Option<String> {
    let out = Command::new("docker").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Image ID, for the provenance banner and as an existence check.
pub fn image_id(image: &str) -> Option<String> {
    docker_stdout(&["image", "inspect", image, "--format", "{{.Id}}"]).filter(|s| !s.is_empty())
}

/// Every precondition, in the order a cold machine would satisfy them.
///
/// Reported as data rather than as early exits so `doctor` can show the whole
/// picture and `setup` can act on exactly what is missing.
pub fn checks(env: &Env) -> Vec<Check> {
    let mut checks = vec![
        match docker_stdout(&["version", "--format", "{{.Server.Version}}"]) {
            Some(v) if !v.is_empty() => Check::ok("docker daemon", format!("server {v}")),
            _ => Check::fail("docker daemon", "start docker: `systemctl start docker`"),
        },
    ];

    for device in DEVICES {
        checks.push(if Path::new(device).exists() {
            Check::ok(format!("device {device}"), "present")
        } else {
            Check::fail(
                format!("device {device}"),
                "no usable AMD GPU on this host; kernel launches would fail",
            )
        });
    }

    let manifest = env.repo.join("homelab/strix.Dockerfile");
    checks.push(if manifest.is_file() {
        Check::ok("checkout", env.repo.display().to_string())
    } else {
        Check::fail(
            "checkout",
            format!("{} is not a vLLM checkout", env.repo.display()),
        )
    });

    checks.push(match image_id(&env.image) {
        Some(id) => Check::ok(format!("image {}", env.image), id),
        None => Check::fail(
            format!("image {}", env.image),
            "run `cargo make setup` (docker build -f homelab/strix.Dockerfile \
             --target devtools -t vllm-strix-devtools:local .)"
                .to_string(),
        ),
    });

    let present = docker_stdout(&["volume", "ls", "--format", "{{.Name}}"]).unwrap_or_default();
    for volume in env.volumes() {
        checks.push(if present.lines().any(|l| l.trim() == volume) {
            Check::ok(format!("volume {volume}"), "present")
        } else {
            Check::fail(
                format!("volume {volume}"),
                format!("run `cargo make setup` (docker volume create {volume})"),
            )
        });
    }
    checks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        Env {
            repo: PathBuf::from("/home/dev/vllm"),
            image: IMAGE.to_string(),
        }
    }

    #[test]
    fn volumes_are_stable_and_checkout_scoped() {
        let volumes = env().volumes();
        assert!(volumes[0].starts_with("vllm-strix-build-"));
        assert!(volumes[1].starts_with("vllm-strix-ccache-"));
        assert_eq!(volumes, env().volumes());
        let other = Env {
            repo: PathBuf::from("/home/dev/other-vllm"),
            image: IMAGE.to_string(),
        };
        assert_ne!(volumes, other.volumes());
    }

    fn args_of(command: &[&str], stdin: bool) -> Vec<String> {
        let command: Vec<String> = command.iter().map(ToString::to_string).collect();
        env().run_args(&command, stdin)
    }

    /// Guards the verified invocation: dropping any of these flags either hides
    /// the GPU, breaks HSA, or costs a full cold build.
    #[test]
    fn run_args_match_the_verified_invocation() {
        let args = args_of(&[BUILD_CMD], false);
        let joined = args.join(" ");
        for expected in [
            "run --rm",
            "--device /dev/kfd",
            "--device /dev/dri",
            "--group-add video",
            "--security-opt seccomp=unconfined",
            "--ipc=host",
            "-v /home/dev/vllm:/src/vllm",
            "-v vllm-strix-build-",
            "-v vllm-strix-ccache-",
        ] {
            assert!(
                joined.contains(expected),
                "missing {expected:?} in {joined}"
            );
        }
        assert_eq!(args.last().unwrap(), BUILD_CMD);
        assert!(!args.contains(&"-i".to_string()));
    }

    /// The mount must stay writable: `build_ext --inplace` writes `.so` files
    /// into the checkout, and a `:ro` suffix would break the build silently.
    #[test]
    fn checkout_mount_is_read_write() {
        assert!(args_of(&[BUILD_CMD], false).contains(&"/home/dev/vllm:/src/vllm".to_string()));
    }

    /// The image bakes every variable this harness would otherwise set, and a
    /// `VLLM_*` name absent from `vllm/envs.py` is a silent no-op.
    #[test]
    fn no_environment_variables_are_injected() {
        assert!(!args_of(&[BUILD_CMD], false).iter().any(|a| a == "-e"));
    }

    /// The old overlay harness needed `--import-mode=importlib` because its
    /// read-only mount shadowed a separately installed package. In devtools the
    /// `.so` files sit beside the sources, so the default mode is correct.
    #[test]
    fn pytest_import_mode_is_not_forced() {
        let args = args_of(&[PYTHON, "-m", "pytest", "tests/kernels"], false);
        assert!(!args.join(" ").contains("--import-mode"));
    }

    /// The probe is fed to `python -` on stdin, which needs `-i`.
    #[test]
    fn stdin_is_requested_only_when_asked() {
        assert!(args_of(&[PYTHON, "-"], true).contains(&"-i".to_string()));
    }

    #[test]
    fn user_arguments_are_appended_verbatim_after_the_image() {
        let args = args_of(&[PYTHON, "-m", "pytest", "-k", "sliding_window"], false);
        let image_at = args.iter().position(|a| a == IMAGE).expect("image present");
        assert_eq!(
            &args[image_at + 1..],
            [PYTHON, "-m", "pytest", "-k", "sliding_window"]
        );
    }

    #[test]
    fn discover_walks_up_from_the_crate_to_the_repo_root() {
        let env = Env::discover("/home/dev/vllm/homelab/devloop", None).unwrap();
        assert_eq!(env.repo, PathBuf::from("/home/dev/vllm"));
        assert_eq!(env.image, IMAGE);
    }

    #[test]
    fn image_override_is_honoured() {
        let env =
            Env::discover("/home/dev/vllm/homelab/devloop", Some("other:tag".into())).unwrap();
        assert_eq!(env.image, "other:tag");
    }
}
