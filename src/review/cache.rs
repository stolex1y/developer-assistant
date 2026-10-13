use anyhow::{Context, Result, bail};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
};

pub(crate) fn prepare_cache_directory(root: &Path, requested: &Path) -> Result<PathBuf> {
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        root.join(requested)
    };
    let relative = requested
        .strip_prefix(root)
        .context("Cache directory должен находиться внутри корня проекта.")?;
    if relative.as_os_str().is_empty() {
        bail!("Cache directory должен находиться в отдельной директории проекта.");
    }

    for component in relative.components() {
        if !matches!(component, Component::Normal(_)) {
            bail!("Cache directory содержит небезопасный путь.");
        }
    }
    ensure_cache_is_ignored(root, relative)?;

    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            bail!("Cache directory содержит небезопасный путь.");
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!("Project cache не может проходить через symlink.");
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => bail!("Элемент пути project cache не является директорией."),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&current)?;
                        if metadata.file_type().is_symlink() || !metadata.is_dir() {
                            bail!("Project cache directory имеет небезопасный тип.");
                        }
                    }
                    Err(error) => {
                        return Err(error).context("Не удалось создать project cache directory.");
                    }
                }
            }
            Err(error) => {
                return Err(error).context("Не удалось проверить project cache directory.");
            }
        }
    }

    let canonical = current
        .canonicalize()
        .context("Не удалось определить project cache directory.")?;
    if !canonical.starts_with(root) {
        bail!("Project cache должен находиться внутри корня проекта.");
    }
    Ok(canonical)
}

fn ensure_cache_is_ignored(root: &Path, relative: &Path) -> Result<()> {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "--no-index", "--quiet", "--"])
        .arg(Path::new(".").join(relative))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("Не удалось проверить Git ignore rule для project cache.")?;
    match status.code() {
        Some(0) => Ok(()),
        Some(1) => bail!("Project cache directory должна быть исключена Git."),
        _ => bail!("Git не смог проверить исключение project cache."),
    }
}
