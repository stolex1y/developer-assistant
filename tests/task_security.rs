#![cfg(unix)]
use developer_assistant::task_files::TaskWorkspace;
use serde_json::json;
use std::{fs, process::Command};

fn repository() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    assert!(
        Command::new("git")
            .args(["init", "--initial-branch=main"])
            .current_dir(root.path())
            .output()
            .unwrap()
            .status
            .success()
    );
    fs::write(root.path().join("README.md"), "project\n").unwrap();
    assert!(
        Command::new("git")
            .args(["add", "README.md"])
            .current_dir(root.path())
            .output()
            .unwrap()
            .status
            .success()
    );
    root
}

#[test]
fn moved_root_is_rejected_before_any_directory_or_file_creation() {
    use std::os::unix::fs::symlink;
    let root = repository();
    let outside = tempfile::tempdir().unwrap();
    let mut workspace = TaskWorkspace::new(root.path()).unwrap();
    workspace
        .execute(
            "propose_file",
            json!({"path":"docs/report.md","content":"report\n"}),
        )
        .unwrap();
    let moved = outside.path().join("moved");
    fs::rename(root.path(), &moved).unwrap();
    symlink(&moved, root.path()).unwrap();
    assert!(workspace.apply().is_err());
    assert!(
        !moved.join("docs").exists(),
        "rejected root change must not create directories outside original root"
    );
}

#[test]
fn git_metadata_directory_cannot_be_used_as_project_root() {
    let root = repository();
    assert!(TaskWorkspace::new(&root.path().join(".git")).is_err());
}
