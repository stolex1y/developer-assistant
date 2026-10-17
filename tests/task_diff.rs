use developer_assistant::task_files::TaskWorkspace;
use serde_json::json;
use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

#[test]
fn sparse_changes_have_separate_context_hunks_and_apply_as_a_patch() {
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
    let old = (1..=100)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    fs::write(root.path().join("guide.md"), &old).unwrap();
    assert!(
        Command::new("git")
            .args(["add", "guide.md"])
            .current_dir(root.path())
            .output()
            .unwrap()
            .status
            .success()
    );
    let mut workspace = TaskWorkspace::new(root.path()).unwrap();
    workspace
        .execute(
            "read_file",
            json!({"path":"guide.md","start_line":1,"end_line":100}),
        )
        .unwrap();
    let new = old
        .replace("line 10\n", "updated ten\n")
        .replace("line 90\n", "updated ninety\n");
    workspace
        .execute("propose_file", json!({"path":"guide.md","content":new}))
        .unwrap();
    workspace
        .execute(
            "propose_file",
            json!({"path":"report.md","content":"new report without newline"}),
        )
        .unwrap();
    workspace
        .execute("propose_file", json!({"path":"empty.md","content":""}))
        .unwrap();
    let patch = workspace.diff().unwrap();
    assert!(patch.contains(" line 7\n"));
    assert!(patch.contains("-line 10\n+updated ten\n"));
    assert!(patch.contains(" line 93\n"));
    assert!(
        !patch.contains("line 50\n"),
        "unchanged middle must not flood the preview"
    );
    let mut child = Command::new("git")
        .args(["apply", "--whitespace=nowarn", "-"])
        .current_dir(root.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(patch.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{patch}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.path().join("guide.md")).unwrap(),
        new
    );
    assert_eq!(
        fs::read_to_string(root.path().join("report.md")).unwrap(),
        "new report without newline"
    );
    assert_eq!(fs::read(root.path().join("empty.md")).unwrap(), b"");
}
