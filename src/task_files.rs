//! Task-local proposals. Only `apply` writes, after the caller's explicit confirmation.
use crate::{
    git_process::isolate_std_git_environment,
    review::corpus::{contains_secret_like_value, is_excluded_path},
};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path},
    process::Command,
};

const REPLY_BYTES: usize = 32_768;
const PAGE_LINES: usize = 200;

struct Original {
    text: String,
    covered: BTreeSet<usize>,
    #[cfg(unix)]
    identity: (u64, u64, u32),
}

pub struct TaskWorkspace {
    root: std::path::PathBuf,
    directory: File,
    originals: BTreeMap<String, Original>,
    staged: BTreeMap<String, String>,
    exposed: BTreeSet<String>,
}

impl TaskWorkspace {
    pub fn new(root: &Path) -> Result<Self> {
        #[cfg(not(unix))]
        {
            let _ = root;
            bail!("Task workspace requires Unix secure file operations.");
        }
        #[cfg(unix)]
        {
            let root = root
                .canonicalize()
                .map_err(|_| anyhow::anyhow!("Cannot open project root."))?;
            let directory =
                File::open(&root).map_err(|_| anyhow::anyhow!("Cannot open project root."))?;
            if !directory.metadata()?.is_dir() {
                bail!("Project root is not a directory.");
            }
            let mut workspace = Self {
                root,
                directory,
                originals: BTreeMap::new(),
                staged: BTreeMap::new(),
                exposed: BTreeSet::new(),
            };
            let output = workspace.git(&["ls-files", "--stage", "-z"])?;
            if !output.status.success() {
                bail!("Cannot enumerate tracked files.");
            }
            for record in output.stdout.split(|b| *b == 0) {
                let Some(tab) = record.iter().position(|b| *b == b'\t') else {
                    continue;
                };
                let Ok(header) = std::str::from_utf8(&record[..tab]) else {
                    continue;
                };
                let fields: Vec<_> = header.split_whitespace().collect();
                if fields.len() != 3
                    || fields[2] != "0"
                    || !matches!(fields[0], "100644" | "100755")
                {
                    continue;
                }
                let Ok(path) = std::str::from_utf8(&record[tab + 1..]) else {
                    continue;
                };
                if !safe_path(path) || workspace.ignored(path)? {
                    continue;
                }
                let Ok(mut file) = workspace.open_file(path, false, false) else {
                    continue;
                };
                let metadata = file.metadata()?;
                use std::os::unix::fs::MetadataExt;
                if !metadata.is_file() || metadata.nlink() != 1 {
                    continue;
                }
                let mut text = String::new();
                if file.read_to_string(&mut text).is_err() || !safe_content(&text) {
                    continue;
                }
                workspace.originals.insert(
                    path.to_owned(),
                    Original {
                        text,
                        covered: BTreeSet::new(),
                        identity: (metadata.dev(), metadata.ino(), metadata.mode()),
                    },
                );
            }
            Ok(workspace)
        }
    }

    pub fn execute(&mut self, name: &str, arguments: Value) -> Result<String> {
        match name {
            "list_files" => {
                let offset = arguments.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
                let paths: BTreeSet<_> = self.originals.keys().chain(self.staged.keys()).collect();
                let mut page = Vec::new();
                let mut size = 0;
                for path in paths.iter().skip(offset).take(200) {
                    if size + path.len() + 8 > REPLY_BYTES / 2 {
                        break;
                    }
                    size += path.len() + 8;
                    page.push((*path).clone());
                }
                Ok(json!({"paths":page,"next_offset":offset+page.len(),"truncated":offset+page.len()<paths.len()}).to_string())
            }
            "read_file" => {
                let path = string_arg(&arguments, "path")?;
                if !safe_path(path) {
                    bail!("Path is not allowed.");
                }
                let start = usize_arg(&arguments, "start_line")?;
                let end = usize_arg(&arguments, "end_line")?;
                if start == 0 || end < start || end - start >= PAGE_LINES {
                    bail!("Read requires 1-based inclusive bounds of at most 200 lines.");
                }
                let text = self
                    .staged
                    .get(path)
                    .or_else(|| self.originals.get(path).map(|o| &o.text))
                    .context("File is not in the allowed corpus.")?;
                let lines: Vec<_> = text.split_inclusive('\n').collect();
                if start > lines.len().max(1) {
                    bail!("Read starts beyond end of file.");
                }
                let stop = end.min(lines.len());
                let excerpt = if lines.is_empty() {
                    String::new()
                } else {
                    lines[start - 1..stop].concat()
                };
                let result = json!({"path":path,"start_line":start,"end_line":stop,"total_lines":lines.len(),"content":excerpt}).to_string();
                if result.len() > REPLY_BYTES {
                    bail!("Read reply exceeds byte limit; request fewer lines.");
                }
                self.exposed.insert(path.to_owned());
                if !self.staged.contains_key(path)
                    && let Some(original) = self.originals.get_mut(path)
                {
                    original.covered.extend(start..=stop);
                }
                Ok(result)
            }
            "search_text" => {
                let query = string_arg(&arguments, "query")?;
                if query.is_empty() || query.len() > 4096 {
                    bail!("Search query must contain 1 to 4096 bytes.");
                }
                let mut matches = Vec::new();
                let mut size = 0;
                let mut truncated = false;
                let paths: BTreeSet<_> = self.originals.keys().chain(self.staged.keys()).collect();
                'files: for path in paths {
                    let text = self
                        .staged
                        .get(path)
                        .unwrap_or_else(|| &self.originals[path].text);
                    for (index, line) in text.lines().enumerate() {
                        if line.contains(query) {
                            let item = json!({"path":path,"line":index+1,"text":line});
                            let bytes = item.to_string().len();
                            if size + bytes > REPLY_BYTES - 128 || matches.len() >= 100 {
                                truncated = true;
                                break 'files;
                            }
                            size += bytes + 1;
                            matches.push(item);
                        }
                    }
                }
                Ok(json!({"matches":matches,"truncated":truncated}).to_string())
            }
            "propose_file" => {
                let path = string_arg(&arguments, "path")?;
                let content = string_arg(&arguments, "content")?;
                if !safe_path(path) || !safe_content(content) || self.ignored(path)? {
                    bail!("Proposed path or content is not allowed.");
                }
                if let Some(original) = self.originals.get(path) {
                    if !self.exposed.contains(path)
                        || original.covered.len() != original.text.split_inclusive('\n').count()
                    {
                        bail!("Read the entire original file before proposing a replacement.");
                    }
                    if content == original.text {
                        self.staged.remove(path);
                        return Ok("No changes staged.".into());
                    }
                } else {
                    self.ensure_absent(path)?;
                }
                self.staged.insert(path.to_owned(), content.to_owned());
                Ok("Proposal staged; no project files written.".into())
            }
            _ => bail!("Unknown task tool."),
        }
    }

    pub fn changed_paths(&self) -> Vec<String> {
        self.staged.keys().cloned().collect()
    }
    pub fn read_paths(&self) -> Vec<String> {
        self.exposed.iter().cloned().collect()
    }

    /// A deterministic whole-file unified hunk; no external process receives proposed contents.
    pub fn diff(&self) -> Result<String> {
        let mut result = String::new();
        for (path, new) in &self.staged {
            let old = self
                .originals
                .get(path)
                .map(|o| o.text.as_str())
                .unwrap_or("");
            result.push_str(&format!("diff --git a/{path} b/{path}\n"));
            if !self.originals.contains_key(path) {
                result.push_str("new file mode 100644\n");
            }
            let old_lines = old.split_inclusive('\n').count();
            let new_lines = new.split_inclusive('\n').count();
            if old_lines == 0 && new_lines == 0 {
                continue;
            }
            result.push_str(&format!(
                "--- {}\n+++ b/{path}\n",
                if self.originals.contains_key(path) {
                    format!("a/{path}")
                } else {
                    "/dev/null".into()
                }
            ));
            result.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                usize::from(old_lines > 0),
                old_lines,
                usize::from(new_lines > 0),
                new_lines
            ));
            for (prefix, text) in [('-', old), ('+', new.as_str())] {
                for line in text.split_inclusive('\n') {
                    result.push(prefix);
                    result.push_str(line);
                    if !line.ends_with('\n') {
                        result.push_str("\n\\ No newline at end of file\n");
                    }
                }
            }
        }
        Ok(result)
    }

    /// Preflights every target before writing. Not an atomic multi-file transaction.
    pub fn apply(&mut self) -> Result<Vec<String>> {
        #[cfg(not(unix))]
        {
            bail!("Task workspace requires Unix secure file operations.");
        }
        #[cfg(unix)]
        {
            // The plan was derived from this complete snapshot, not only edited files.
            self.validate_root()?;
            for (path, original) in &self.originals {
                let mut file = self.open_file(path, false, false)?;
                self.validate_original(&mut file, original)?;
            }
            for path in self.staged.keys() {
                if !safe_path(path) || self.ignored(path)? {
                    bail!("Target is no longer allowed.");
                }
                if !self.originals.contains_key(path) {
                    self.ensure_absent(path)?;
                }
            }
            let changed = self.changed_paths();
            for (path, content) in &self.staged {
                self.replace_file(path, content)?;
            }
            // A workspace is single-use after apply: prevent stale snapshots from being reused.
            self.staged.clear();
            Ok(changed)
        }
    }

    fn git(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut command = Command::new("git");
        command.arg("-C").arg(&self.root).args(args);
        isolate_std_git_environment(&mut command);
        command
            .output()
            .map_err(|_| anyhow::anyhow!("Cannot run repository safety check."))
    }
    fn ignored(&self, path: &str) -> Result<bool> {
        let output = self.git(&["check-ignore", "--no-index", "-q", "--", path])?;
        match output.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => bail!("Cannot check repository ignore rules."),
        }
    }

    #[cfg(unix)]
    fn validate_root(&self) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        let canonical = self.root.canonicalize().context("Project root changed.")?;
        let current =
            std::fs::symlink_metadata(&self.root).context("Cannot inspect project root.")?;
        let snapshot = self
            .directory
            .metadata()
            .context("Cannot inspect project root snapshot.")?;
        if canonical != self.root
            || !current.is_dir()
            || (current.dev(), current.ino()) != (snapshot.dev(), snapshot.ino())
        {
            bail!("Project root changed.");
        }
        Ok(())
    }

    #[cfg(unix)]
    fn validate_original(&self, file: &mut File, original: &Original) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        let m = file.metadata().context("Cannot inspect target.")?;
        if !m.is_file() || m.nlink() != 1 || (m.dev(), m.ino(), m.mode()) != original.identity {
            bail!("Target changed since task snapshot.");
        }
        file.seek(SeekFrom::Start(0))
            .context("Cannot inspect target.")?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .context("Cannot inspect target.")?;
        if bytes != original.text.as_bytes() {
            bail!("Target changed since task snapshot.");
        }
        Ok(())
    }

    #[cfg(unix)]
    fn replace_file(&self, path: &str, content: &str) -> Result<()> {
        use std::{
            ffi::CString,
            os::{
                fd::{AsRawFd, FromRawFd},
                unix::fs::{MetadataExt, PermissionsExt},
            },
            sync::atomic::{AtomicU64, Ordering},
        };
        static NEXT: AtomicU64 = AtomicU64::new(0);
        self.validate_root()?;
        let (parent, name) = self.parent(path, true)?;
        let temporary = CString::new(format!(
            ".task-stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .expect("fixed temporary name");
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                temporary.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            bail!("Cannot prepare replacement file; apply may be partial.");
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let result = (|| -> Result<()> {
            file.write_all(content.as_bytes())
                .context("Cannot prepare replacement contents; apply may be partial.")?;
            let mode = self
                .originals
                .get(path)
                .map(|o| o.identity.2 & 0o777)
                .unwrap_or(0o644);
            file.set_permissions(std::fs::Permissions::from_mode(mode))
                .context("Cannot set replacement permissions.")?;
            file.sync_all()
                .context("Cannot sync replacement contents.")?;
            // Reopen the root and complete parent chain immediately before publishing.
            self.validate_root()?;
            let (current_parent, _) = self.parent(path, false)?;
            let before = parent.metadata().context("Cannot inspect target parent.")?;
            let now = current_parent
                .metadata()
                .context("Cannot inspect target parent.")?;
            if (before.dev(), before.ino()) != (now.dev(), now.ino()) {
                bail!("Target parent changed.");
            }
            let check_fd = unsafe {
                libc::openat(
                    parent.as_raw_fd(),
                    temporary.as_ptr(),
                    libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                )
            };
            if check_fd < 0 {
                bail!("Replacement file changed.");
            }
            let check = unsafe { File::from_raw_fd(check_fd) };
            let expected = file.metadata().context("Cannot inspect replacement.")?;
            let actual = check.metadata().context("Cannot inspect replacement.")?;
            if !actual.is_file()
                || actual.nlink() != 1
                || (expected.dev(), expected.ino()) != (actual.dev(), actual.ino())
            {
                bail!("Replacement file changed.");
            }
            if let Some(original) = self.originals.get(path) {
                let mut current = self.open_file(path, false, false)?;
                self.validate_original(&mut current, original)?;
                if unsafe {
                    libc::renameat(
                        parent.as_raw_fd(),
                        temporary.as_ptr(),
                        parent.as_raw_fd(),
                        name.as_ptr(),
                    )
                } < 0
                {
                    bail!("Cannot publish replacement; apply may be partial.");
                }
            } else {
                // linkat provides atomic no-clobber publication, even if an untracked file races us.
                if unsafe {
                    libc::linkat(
                        parent.as_raw_fd(),
                        temporary.as_ptr(),
                        parent.as_raw_fd(),
                        name.as_ptr(),
                        0,
                    )
                } < 0
                {
                    bail!("Cannot publish new file; target may already exist.");
                }
            }
            Ok(())
        })();
        // unlinkat cannot follow a symlink; temporary files never survive a normal error path.
        unsafe {
            libc::unlinkat(parent.as_raw_fd(), temporary.as_ptr(), 0);
        }
        result
    }

    #[cfg(unix)]
    fn parent(&self, path: &str, create: bool) -> Result<(File, std::ffi::CString)> {
        use std::{
            ffi::CString,
            os::fd::{AsRawFd, FromRawFd},
        };
        let mut directory = self
            .directory
            .try_clone()
            .context("Cannot access project directory.")?;
        let mut components = path.split('/').peekable();
        while let Some(component) = components.next() {
            let name = CString::new(component).context("Invalid path.")?;
            if components.peek().is_none() {
                return Ok((directory, name));
            }
            let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
            let mut fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
            if fd < 0
                && create
                && std::io::Error::last_os_error().kind() == std::io::ErrorKind::NotFound
            {
                let status = unsafe { libc::mkdirat(directory.as_raw_fd(), name.as_ptr(), 0o755) };
                if status < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::AlreadyExists
                {
                    bail!("Cannot create target directory; apply may be partial.");
                }
                fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
            }
            if fd < 0 {
                bail!("Unsafe or inaccessible target directory.");
            }
            directory = unsafe { File::from_raw_fd(fd) };
        }
        bail!("Invalid path.")
    }
    #[cfg(unix)]
    fn open_file(&self, path: &str, write: bool, create: bool) -> Result<File> {
        use std::os::fd::{AsRawFd, FromRawFd};
        if !safe_path(path) {
            bail!("Path is not allowed.");
        }
        let (parent, name) = self.parent(path, create)?;
        let mut flags = libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK;
        flags |= if write { libc::O_RDWR } else { libc::O_RDONLY };
        if create {
            flags |= libc::O_CREAT | libc::O_EXCL;
        }
        let fd = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags, 0o644) };
        if fd < 0 {
            bail!("Target is unavailable, unsafe, or already exists.");
        }
        let file = unsafe { File::from_raw_fd(fd) };
        use std::os::unix::fs::MetadataExt;
        let m = file.metadata().context("Cannot inspect target.")?;
        if !m.is_file() || m.nlink() != 1 {
            bail!("Target must be a regular file without hard links.");
        }
        Ok(file)
    }
    #[cfg(unix)]
    fn ensure_absent(&self, path: &str) -> Result<()> {
        use std::os::fd::AsRawFd;
        // Missing parents are allowed, but every existing component must be a real directory.
        let mut prefix = String::new();
        for component in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(component);
            let (parent, name) = self.parent(&prefix, false)?;
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            let status = unsafe {
                libc::fstatat(
                    parent.as_raw_fd(),
                    name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            if status < 0 {
                if std::io::Error::last_os_error().kind() == std::io::ErrorKind::NotFound {
                    return Ok(());
                }
                bail!("Cannot inspect proposed target.");
            }
            let stat = unsafe { stat.assume_init() };
            if prefix == path || stat.st_mode & libc::S_IFMT != libc::S_IFDIR {
                bail!("New target already exists or has an unsafe parent.");
            }
        }
        bail!("New target already exists.")
    }
}

fn safe_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.chars().any(char::is_control)
        && !path.starts_with('/')
        && path
            .split('/')
            .all(|s| !s.is_empty() && s != "." && s != "..")
        && Path::new(path)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
        && !is_excluded_path(path)
        && !path.contains('"')
}
fn safe_content(text: &str) -> bool {
    !text
        .chars()
        .any(|ch| ch.is_control() && !matches!(ch, '\n' | '\r' | '\t'))
        && !contains_secret_like_value(text)
}
fn string_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str> {
    args.get(name)
        .and_then(Value::as_str)
        .context("Missing string tool argument.")
}
fn usize_arg(args: &Value, name: &str) -> Result<usize> {
    args.get(name)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .context("Missing nonnegative integer tool argument.")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::{TempDir, tempdir};

    fn fixture() -> TempDir {
        let dir = tempdir().unwrap();
        let status = Command::new("git")
            .arg("-C")
            .arg(dir.path())
            .arg("init")
            .status()
            .unwrap();
        assert!(status.success());
        fs::create_dir(dir.path().join("src")).unwrap();
        fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
        fs::write(dir.path().join("README.md"), "first\nsecond\n").unwrap();
        fs::write(dir.path().join("guide.md"), "guide without newline").unwrap();
        fs::write(dir.path().join(".gitignore"), "ignored/\n").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );
        dir
    }
    fn read(workspace: &mut TaskWorkspace, path: &str, start: usize, end: usize) -> Result<String> {
        workspace.execute(
            "read_file",
            json!({"path":path,"start_line":start,"end_line":end}),
        )
    }
    fn propose(workspace: &mut TaskWorkspace, path: &str, content: &str) -> Result<String> {
        workspace.execute("propose_file", json!({"path":path,"content":content}))
    }
    #[test]
    fn proposals_are_in_memory_and_require_complete_reads() {
        let dir = fixture();
        let mut w = TaskWorkspace::new(dir.path()).unwrap();
        assert!(propose(&mut w, "README.md", "replacement\n").is_err());
        read(&mut w, "README.md", 1, 1).unwrap();
        assert!(propose(&mut w, "README.md", "replacement\n").is_err());
        read(&mut w, "README.md", 2, 2).unwrap();
        read(&mut w, "src/main.rs", 1, 10).unwrap();
        read(&mut w, "guide.md", 1, 10).unwrap();
        assert_eq!(w.read_paths().len(), 3);
        propose(&mut w, "README.md", "replacement\n").unwrap();
        propose(&mut w, "src/extra.rs", "pub fn extra() {}\n").unwrap();
        propose(&mut w, "docs/task.md", "new documentation").unwrap();
        assert!(!dir.path().join("docs").exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "first\nsecond\n"
        );
        assert!(w.diff().unwrap().contains("\\ No newline at end of file"));
        assert!(
            read(&mut w, "README.md", 1, 1)
                .unwrap()
                .contains("replacement")
        );
        assert_eq!(w.apply().unwrap().len(), 3);
        assert_eq!(
            fs::read_to_string(dir.path().join("docs/task.md")).unwrap(),
            "new documentation"
        );
    }
    #[test]
    fn conflict_preflight_prevents_other_writes() {
        let dir = fixture();
        let mut w = TaskWorkspace::new(dir.path()).unwrap();
        read(&mut w, "README.md", 1, 2).unwrap();
        propose(&mut w, "README.md", "replacement").unwrap();
        propose(&mut w, "aaa.md", "new").unwrap();
        fs::write(dir.path().join("README.md"), "external").unwrap();
        assert!(w.apply().is_err());
        assert!(!dir.path().join("aaa.md").exists());
        assert_eq!(
            fs::read_to_string(dir.path().join("README.md")).unwrap(),
            "external"
        );
    }
    #[test]
    fn rejects_escape_secrets_untracked_and_links() {
        use std::os::unix::fs::symlink;
        let dir = fixture();
        let outside = tempdir().unwrap();
        let mut w = TaskWorkspace::new(dir.path()).unwrap();
        fs::write(dir.path().join("untracked.md"), "user").unwrap();
        symlink(outside.path(), dir.path().join("link")).unwrap();
        for path in [
            "../escape",
            "/absolute",
            ".git/config",
            ".env",
            "ignored/file",
            "untracked.md",
            "link/file",
        ] {
            assert!(propose(&mut w, path, "text").is_err(), "{path}");
        }
        assert!(propose(&mut w, "new.md", "-----BEGIN PRIVATE KEY-----").is_err());
        read(&mut w, "README.md", 1, 2).unwrap();
        propose(&mut w, "README.md", "replacement").unwrap();
        fs::remove_file(dir.path().join("README.md")).unwrap();
        fs::write(outside.path().join("victim"), "outside").unwrap();
        symlink(outside.path().join("victim"), dir.path().join("README.md")).unwrap();
        assert!(w.apply().is_err());
        assert_eq!(
            fs::read_to_string(outside.path().join("victim")).unwrap(),
            "outside"
        );
    }
    #[test]
    fn rejects_hardlinks_and_read_bounds_and_removes_noop() {
        let dir = fixture();
        let mut w = TaskWorkspace::new(dir.path()).unwrap();
        assert!(read(&mut w, "README.md", 0, 1).is_err());
        assert!(read(&mut w, "README.md", 1, 201).is_err());
        assert!(read(&mut w, "README.md", 9, 10).is_err());
        read(&mut w, "README.md", 1, 2).unwrap();
        propose(&mut w, "README.md", "replacement").unwrap();
        propose(&mut w, "README.md", "first\nsecond\n").unwrap();
        assert!(w.changed_paths().is_empty());
        propose(&mut w, "README.md", "replacement").unwrap();
        fs::hard_link(dir.path().join("README.md"), dir.path().join("alias")).unwrap();
        assert!(w.apply().is_err());
        assert_eq!(
            fs::read_to_string(dir.path().join("alias")).unwrap(),
            "first\nsecond\n"
        );
    }
    #[test]
    fn oversized_reads_do_not_grant_coverage() {
        let dir = fixture();
        fs::write(dir.path().join("README.md"), "x".repeat(REPLY_BYTES + 1)).unwrap();
        let mut w = TaskWorkspace::new(dir.path()).unwrap();
        assert!(read(&mut w, "README.md", 1, 1).is_err());
        assert!(w.read_paths().is_empty());
        assert!(propose(&mut w, "README.md", "replacement").is_err());
    }
}

#[cfg(all(test, unix))]
mod replacement_tests {
    use super::*;
    use std::{
        fs,
        os::unix::fs::{MetadataExt, PermissionsExt},
    };

    #[test]
    fn replacement_changes_inode_preserves_executable_mode_and_cleans_temporary() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .arg("init")
                .status()
                .unwrap()
                .success()
        );
        let path = dir.path().join("run.rs");
        fs::write(&path, "old\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o751)).unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(["add", "run.rs"])
                .status()
                .unwrap()
                .success()
        );
        let old_inode = fs::metadata(&path).unwrap().ino();
        let mut workspace = TaskWorkspace::new(dir.path()).unwrap();
        workspace
            .execute(
                "read_file",
                json!({"path":"run.rs","start_line":1,"end_line":1}),
            )
            .unwrap();
        workspace
            .execute("propose_file", json!({"path":"run.rs","content":"new\n"}))
            .unwrap();
        workspace.apply().unwrap();
        let metadata = fs::metadata(&path).unwrap();
        assert_ne!(metadata.ino(), old_inode);
        assert_eq!(metadata.mode() & 0o777, 0o751);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new\n");
        assert!(!fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".task-stage-")
        }));
    }
}

#[cfg(not(unix))]
impl TaskWorkspace {
    fn ensure_absent(&self, _path: &str) -> Result<()> {
        bail!("Task workspace requires Unix secure file operations.")
    }
}

#[cfg(all(test, unix))]
mod terminal_safety_tests {
    use super::*;

    #[test]
    fn rejects_terminal_controls_in_originals_and_proposals() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .arg("init")
                .status()
                .unwrap()
                .success()
        );
        std::fs::write(dir.path().join("unsafe.md"), "\u{1b}]52;c;payload\u{7}").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(dir.path())
                .args(["add", "unsafe.md"])
                .status()
                .unwrap()
                .success()
        );
        let mut workspace = TaskWorkspace::new(dir.path()).unwrap();
        assert!(
            workspace
                .execute(
                    "read_file",
                    json!({"path":"unsafe.md","start_line":1,"end_line":1})
                )
                .is_err()
        );
        for content in [
            "\u{1b}[31mred",
            "\u{1b}]52;c;payload\u{7}",
            "\u{9b}31m",
            "bell\u{7}",
        ] {
            assert!(
                workspace
                    .execute("propose_file", json!({"path":"safe.md","content":content}))
                    .is_err()
            );
        }
        assert!(workspace.diff().unwrap().is_empty());
        assert!(safe_content("ordinary\ttext\r\n"));
    }
}
