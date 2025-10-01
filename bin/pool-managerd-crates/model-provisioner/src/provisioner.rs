use anyhow::Result;
use catalog_core::{
    default_model_cache_dir, CatalogEntry, CatalogStore, Digest, FileFetcher, FsCatalog,
    LifecycleState, ModelFetcher, ModelRef, ResolvedModel,
};
use observability_narration_core::{narrate, NarrationFields};
use std::path::PathBuf;
use std::process::Command;

fn ensure_hf_cli() -> Result<&'static str> {
    // Prefer newer 'hf' CLI; fall back to 'huggingface-cli'. Provide an actionable error.
    if which::which("hf").is_ok() {
        Ok("hf")
    } else if which::which("huggingface-cli").is_ok() {
        Ok("huggingface-cli")
    } else {
        Err(anyhow::anyhow!(
            "Hugging Face CLI not found. Install 'python-huggingface-hub' (provides 'hf' and 'huggingface-cli') or provide a local file: path"
        ))
    }
}

#[derive(Clone)]
pub struct ModelProvisioner<C: CatalogStore, F: ModelFetcher> {
    catalog: C,
    fetcher: F,
}

impl ModelProvisioner<FsCatalog, FileFetcher> {
    pub fn file_only(cache_dir: PathBuf) -> Result<Self> {
        let catalog = FsCatalog::new(cache_dir)?;
        let fetcher = FileFetcher;
        Ok(Self { catalog, fetcher })
    }
}

impl<C: CatalogStore, F: ModelFetcher> ModelProvisioner<C, F> {
    pub fn new(catalog: C, fetcher: F) -> Self {
        Self { catalog, fetcher }
    }

    pub fn ensure_present_str(
        &self,
        model_ref: &str,
        expected_digest: Option<Digest>,
    ) -> Result<ResolvedModel> {
        narrate(NarrationFields {
            actor: "model-provisioner",
            action: "resolve",
            target: model_ref.to_string(),
            human: format!("Resolving model reference: {}", model_ref),
            ..Default::default()
        });
        let mr = ModelRef::parse(model_ref)?;
        self.ensure_present(&mr, expected_digest)
    }

    pub fn ensure_present(
        &self,
        mr: &ModelRef,
        expected_digest: Option<Digest>,
    ) -> Result<ResolvedModel> {
        let ref_str = format!("{:?}", mr);
        narrate(NarrationFields {
            actor: "model-provisioner",
            action: "ensure",
            target: ref_str.clone(),
            human: format!("Ensuring model present: {}", ref_str),
            ..Default::default()
        });
        // Try primary fetcher first (file/relative). If unsupported, handle select schemes inline.
        let resolved = match self.fetcher.ensure_present(mr) {
            Ok(r) => r,
            Err(e) => {
                // Minimal hf: support via CLI (prefer 'hf', fallback to 'huggingface-cli').
                match mr {
                    ModelRef::Hf { org, repo, path } => {
                        let cache_dir = default_model_cache_dir();
                        std::fs::create_dir_all(&cache_dir).ok();
                        // Preflight: require HF CLI
                        narrate(NarrationFields {
                            actor: "model-provisioner",
                            action: "download",
                            target: format!("hf:{}/{}", org, repo),
                            human: format!("Downloading from Hugging Face: {}/{}", org, repo),
                            ..Default::default()
                        });
                        let cli = ensure_hf_cli()?;
                        let repo_spec = format!("{}/{}", org, repo);
                        let mut c = Command::new(cli);
                        c.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
                        c.arg("download").arg(&repo_spec);
                        if let Some(p) = path {
                            c.arg(p);
                        }
                        c.arg("--local-dir").arg(&cache_dir);
                        // Add flags specific to the CLI flavor
                        if cli == "hf" {
                            // Clarify repo type to avoid accidental space/type mismatch
                            c.arg("--repo-type").arg("model");
                        } else {
                            // Older CLI supports explicit symlink toggle
                            c.arg("--local-dir-use-symlinks").arg("False");
                        }
                        let st = c.status()?;
                        if !st.success() {
                            narrate(NarrationFields {
                                actor: "model-provisioner",
                                action: "download-failed",
                                target: repo_spec.clone(),
                                human: format!("HF CLI download failed for {}", repo_spec),
                                ..Default::default()
                            });
                            return Err(anyhow::anyhow!(
                                "HF CLI download failed for {}. Try: pip install 'huggingface_hub[cli]' or verify network/CA certificates.",
                                repo_spec
                            ));
                        }
                        narrate(NarrationFields {
                            actor: "model-provisioner",
                            action: "downloaded",
                            target: repo_spec.clone(),
                            human: format!("Successfully downloaded {}", repo_spec),
                            ..Default::default()
                        });
                        let local_path = if let Some(p) = path {
                            cache_dir.join(p)
                        } else {
                            cache_dir.join(repo_spec.replace('/', "_"))
                        };

                        ResolvedModel {
                            id: format!(
                                "hf:{}/{}{}",
                                org,
                                repo,
                                path.as_ref().map(|p| format!("/{}", p)).unwrap_or_default()
                            ),
                            local_path,
                        }
                    }
                    _ => return Err(e.into()),
                }
            }
        };

        // Optional strict verification: compute sha256 and enforce match when configured.
        if let Some(exp) = expected_digest.as_ref() {
            narrate(NarrationFields {
                actor: "model-provisioner",
                action: "verify",
                target: resolved.id.clone(),
                human: format!("Verifying digest for {}", resolved.id),
                ..Default::default()
            });
            if exp.algo.eq_ignore_ascii_case("sha256") {
                // Guard: only file paths can be hashed
                if !resolved.local_path.is_file() {
                    return Err(anyhow::anyhow!(
                        "digest verification requires a file path; got non-file path: {}. Specify an explicit file (e.g., hf:org/repo/path.gguf) or omit expected_digest.",
                        resolved.local_path.display()
                    ));
                }
                let act = crate::util::compute_sha256_hex(&resolved.local_path)?;
                if act != exp.value {
                    narrate(NarrationFields {
                        actor: "model-provisioner",
                        action: "verify-failed",
                        target: resolved.id.clone(),
                        human: format!("Digest mismatch: expected {}, got {}", exp.value, act),
                        ..Default::default()
                    });

                    return Err(anyhow::anyhow!(
                        "digest mismatch: expected sha256:{}, got {}",
                        exp.value,
                        act
                    ));
                }
                narrate(NarrationFields {
                    actor: "model-provisioner",
                    action: "verified",
                    target: resolved.id.clone(),
                    human: format!("Digest verified: {}", exp.value),
                    ..Default::default()
                });
            }
        }

        let entry = CatalogEntry {
            id: resolved.id.clone(),
            local_path: resolved.local_path.clone(),
            lifecycle: LifecycleState::Active,
            digest: expected_digest,
            last_verified_ms: None,
        };
        self.catalog.put(&entry)?;
        narrate(NarrationFields {
            actor: "model-provisioner",
            action: "complete",
            target: resolved.id.clone(),
            human: format!("Model ready: {} at {}", resolved.id, resolved.local_path.display()),
            ..Default::default()
        });
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_locks::{CWD_LOCK, PATH_LOCK};
    use catalog_core::ModelRef;
    use std::io::Write;

    #[test]
    fn happy_path_file_model() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("tiny.gguf");
        {
            let mut f = std::fs::File::create(&model_path).unwrap();
            writeln!(f, "dummy").unwrap();
        }
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse(&format!("file:{}", model_path.display())).unwrap();
        let resolved = prov.ensure_present(&mr, None).unwrap();
        assert_eq!(resolved.local_path, model_path);
    }

    #[test]
    fn relative_path_resolves_against_cwd_and_absolute_passthrough() {
        let _guard = CWD_LOCK.get_or_init(Default::default).lock().unwrap();
        let old_cwd = std::env::current_dir().unwrap();
        let wd = tempfile::tempdir().unwrap();
        std::env::set_current_dir(wd.path()).unwrap();

        let cache = tempfile::tempdir().unwrap();
        let rel = std::path::PathBuf::from("models/rel.gguf");
        let abs = wd.path().join(&rel);
        std::fs::create_dir_all(abs.parent().unwrap()).unwrap();
        std::fs::write(&abs, b"rel").unwrap();

        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let r1 =
            prov.ensure_present(&ModelRef::parse(rel.to_str().unwrap()).unwrap(), None).unwrap();
        assert_eq!(r1.local_path, abs);
        let abs_ref = format!("file:{}", abs.display());
        let r2 = prov.ensure_present(&ModelRef::parse(&abs_ref).unwrap(), None).unwrap();
        assert_eq!(r2.local_path, abs);
        std::env::set_current_dir(old_cwd).unwrap();
    }

    #[test]
    fn missing_model_errors() {
        let cache = tempfile::tempdir().unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse("file:/non/existent/path.gguf").unwrap();
        let err = prov.ensure_present(&mr, None).unwrap_err();
        let s = format!("{}", err);
        assert!(s.contains("not found") || s.contains("No such file"));
    }

    #[test]
    fn idempotent_ensure_updates_single_catalog_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("idempotent.gguf");
        std::fs::write(&model_path, b"abc").unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse(&format!("file:{}", model_path.display())).unwrap();
        let r1 = prov.ensure_present(&mr, None).unwrap();
        let r2 = prov.ensure_present(&mr, None).unwrap();
        assert_eq!(r1.local_path, r2.local_path);
        let index_path = cache.path().join("index.json");
        let map_val: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index_path).unwrap()).unwrap();
        let obj = map_val.as_object().unwrap();
        assert_eq!(obj.len(), 1);
    }

    #[test]
    fn strict_digest_gating() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let model_path = tmp.path().join("tiny.gguf");
        std::fs::write(&model_path, b"abc").unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse(&format!("file:{}", model_path.display())).unwrap();
        let wrong = Digest { algo: "sha256".into(), value: "deadbeef".into() };
        let err = prov.ensure_present(&mr, Some(wrong)).unwrap_err();
        assert!(format!("{}", err).contains("digest mismatch"));
        let ok = Digest {
            algo: "sha256".into(),
            value: crate::util::compute_sha256_hex(&model_path).unwrap(),
        };
        let resolved = prov.ensure_present(&mr, Some(ok)).unwrap();
        assert_eq!(resolved.local_path, model_path);
    }

    #[test]
    fn hf_path_uses_default_cache_and_path_fake_cli_via_path() {
        let _guard = PATH_LOCK.get_or_init(Default::default).lock().unwrap();
        let old_home = std::env::var("HOME").ok();
        let fake_home = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", fake_home.path());
        let old_path = std::env::var("PATH").ok();
        let bindir = tempfile::tempdir().unwrap();
        let cli_path = bindir.path().join("huggingface-cli");
        std::fs::write(&cli_path, b"#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&cli_path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&cli_path, perms).unwrap();
        }
        std::env::set_var("PATH", bindir.path());

        let cache = tempfile::tempdir().unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse("hf:org/repo/file.gguf").unwrap();
        let resolved = prov.ensure_present(&mr, None).unwrap();
        let expected_base = default_model_cache_dir();
        assert!(resolved.local_path.starts_with(&expected_base));
        assert!(resolved.local_path.ends_with("file.gguf"));

        if let Some(h) = old_home {
            std::env::set_var("HOME", h);
        } else {
            std::env::remove_var("HOME");
        }
        if let Some(p) = old_path {
            std::env::set_var("PATH", p);
        } else {
            std::env::remove_var("PATH");
        }
    }

    #[test]
    fn hf_missing_cli_returns_instructive_error() {
        let _guard = PATH_LOCK.get_or_init(Default::default).lock().unwrap();
        let old_path = std::env::var("PATH").ok();
        std::env::set_var("PATH", "");
        let cache = tempfile::tempdir().unwrap();
        let prov = ModelProvisioner::file_only(cache.path().to_path_buf()).unwrap();
        let mr = ModelRef::parse("hf:org/repo/file.gguf").unwrap();
        let err = prov.ensure_present(&mr, None).unwrap_err();
        let s = format!("{}", err);
        assert!(
            s.contains("Hugging Face CLI not found") || s.contains("huggingface-cli not found")
        );
        if let Some(p) = old_path {
            std::env::set_var("PATH", p);
        } else {
            std::env::remove_var("PATH");
        }
    }
}
