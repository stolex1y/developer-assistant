use serde_json::{Value, json};
use std::{fs, path::Path, process::Stdio, time::Duration};
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    process::Command,
    task::JoinHandle,
    time::timeout,
};

const OLD_DOC: &str = "# Import API\nEndpoint: /old-import\nSee ImportApi.\n";
const NEW_DOC: &str = "# Import API\nEndpoint: /imports\nSee ImportApi.\n";
const SOURCE: &str = "pub struct ImportApi;\nimpl ImportApi {\n    pub const ENDPOINT: &'static str = \"/imports\"; }\n";
const CALLER: &str =
    "use crate::api::ImportApi;\nfn endpoint() -> &'static str {\n    ImportApi::ENDPOINT }\n";
const REPORT: &str = "# ImportApi usages\n- src/api.rs:1: definition\n- src/client.rs:1: import\n- src/client.rs:3: endpoint use\n";

fn project() -> TempDir {
    let temp = tempfile::tempdir().unwrap();
    fs::create_dir_all(temp.path().join("src")).unwrap();
    fs::create_dir_all(temp.path().join("docs")).unwrap();
    fs::write(
        temp.path().join("README.md"),
        "# Import service\nSee docs/api.md.\n",
    )
    .unwrap();
    fs::write(temp.path().join("docs/api.md"), OLD_DOC).unwrap();
    fs::write(temp.path().join("src/api.rs"), SOURCE).unwrap();
    fs::write(temp.path().join("src/client.rs"), CALLER).unwrap();
    for args in [vec!["init", "--initial-branch=main"], vec!["add", "."]] {
        assert!(
            std::process::Command::new("git")
                .args(args)
                .current_dir(temp.path())
                .status()
                .unwrap()
                .success()
        );
    }
    temp
}

fn response(calls: Vec<(&str, Value)>) -> Value {
    let calls: Vec<Value> = calls.into_iter().enumerate().map(|(i, (name, args))| json!({
        "id": format!("{name}-{i}"), "type": "function", "function": {"name": name, "arguments": args.to_string()}
    })).collect();
    let message = if calls.is_empty() {
        json!({"role":"assistant", "content":"Проверил исходники и документацию; подготовленные изменения показаны в diff."})
    } else {
        json!({"role":"assistant", "content":null, "tool_calls":calls})
    };
    json!({"choices":[{"message":message,"finish_reason":if message.get("tool_calls").is_some() {"tool_calls"} else {"stop"}}],
        "usage":{"prompt_tokens":110,"prompt_tokens_details":{"cached_tokens":100}}})
}

fn discovery() -> Vec<Value> {
    vec![
        response(vec![
            ("list_files", json!({})),
            ("search_text", json!({"query":"ImportApi"})),
        ]),
        response(vec![
            (
                "read_file",
                json!({"path":"src/api.rs","start_line":1,"end_line":3}),
            ),
            (
                "read_file",
                json!({"path":"src/client.rs","start_line":1,"end_line":3}),
            ),
            (
                "read_file",
                json!({"path":"docs/api.md","start_line":1,"end_line":3}),
            ),
        ]),
    ]
}

async fn provider(
    responses: Vec<Value>,
    mutate: Option<(std::path::PathBuf, String)>,
) -> (String, JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut requests = vec![];
        let count = responses.len();
        for (index, response) in responses.into_iter().enumerate() {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = vec![];
            let (header_end, content_len) = loop {
                let mut chunk = [0; 8192];
                let n = socket.read(&mut chunk).await.unwrap();
                assert_ne!(n, 0, "incomplete HTTP request");
                bytes.extend_from_slice(&chunk[..n]);
                if let Some(end) = bytes.windows(4).position(|p| p == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&bytes[..end]);
                    let length = header
                        .lines()
                        .find_map(|line| {
                            line.split_once(':')
                                .filter(|(key, _)| key.eq_ignore_ascii_case("content-length"))
                                .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap();
                    break (end + 4, length);
                }
            };
            while bytes.len() < header_end + content_len {
                let mut chunk = [0; 8192];
                let n = socket.read(&mut chunk).await.unwrap();
                assert_ne!(n, 0);
                bytes.extend_from_slice(&chunk[..n]);
            }
            requests.push(
                serde_json::from_slice(&bytes[header_end..header_end + content_len]).unwrap(),
            );
            if index + 1 == count
                && let Some((path, content)) = &mutate
            {
                fs::write(path, content).unwrap();
            }
            let body = response.to_string();
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
        }
        requests
    });
    (url, task)
}

async fn invoke(
    root: &Path,
    responses: Vec<Value>,
    input: &str,
    goal: &str,
    repl: bool,
    mutate: Option<(std::path::PathBuf, String)>,
) -> (std::process::Output, Vec<Value>) {
    let (url, server) = provider(responses, mutate).await;
    let config_dir = tempfile::tempdir().unwrap();
    let config = config_dir.path().join("config.toml");
    fs::write(&config, format!("[providers.local]\nbase_url = \"{url}\"\napi_key_env = \"TASK_ACCEPTANCE_KEY\"\nmodels = [\"test\"]\n[selection]\ndefault_model = \"local/test\"\n")).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_developer-assistant"));
    command
        .current_dir(root)
        .arg("--config")
        .arg(config)
        .env("TASK_ACCEPTANCE_KEY", "synthetic-not-a-secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !repl {
        command.arg("/task").arg(goal);
    }
    let mut child = command.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(input.as_bytes()).await.unwrap();
    drop(stdin);
    let output = timeout(Duration::from_secs(20), child.wait_with_output())
        .await
        .expect("CLI timed out")
        .unwrap();
    if !output.status.success() && !server.is_finished() {
        server.abort();
        panic!(
            "CLI failed before finishing script: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let requests = timeout(Duration::from_secs(3), server)
        .await
        .expect("provider script incomplete")
        .unwrap();
    (output, requests)
}

fn report_script() -> Vec<Value> {
    let mut script = discovery();
    script.push(response(vec![(
        "propose_file",
        json!({"path":"docs/usages.md", "content":REPORT}),
    )]));
    script.push(response(vec![]));
    script
}

#[tokio::test]
async fn goal_search_creates_report_only_after_confirmation() {
    for _ in 0..2 {
        let temp = project();
        let (output, requests) = invoke(
            temp.path(),
            report_script(),
            "yes\n",
            "Найди использования API импорта и подготовь отчёт",
            false,
            None,
        )
        .await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("docs/usages.md")).unwrap(),
            REPORT
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("src/api.rs")).unwrap(),
            SOURCE
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("+"));
        assert!(stdout.contains("docs/usages.md"));
        let messages = requests[2]["messages"].as_array().unwrap();
        let observed: Vec<&str> = messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["content"].as_str())
            .collect();
        let observed = observed.join("\n");
        assert!(observed.contains("/imports"));
        assert!(observed.contains("src/client.rs"));
        assert!(observed.contains("/old-import"));
    }
}

#[tokio::test]
async fn rejected_or_eof_report_does_not_create_any_file() {
    for input in ["no\n", "", "maybe\n"] {
        let temp = project();
        let (output, _) = invoke(
            temp.path(),
            report_script(),
            input,
            "Подготовь отчёт об использовании API",
            false,
            None,
        )
        .await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!temp.path().join("docs/usages.md").exists());
        assert_eq!(
            fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
            OLD_DOC
        );
    }
}

#[tokio::test]
async fn documentation_update_and_repeat_preserve_sources_without_duplicate_sections() {
    let temp = project();
    for _ in 0..2 {
        let mut script = discovery();
        script.push(response(vec![(
            "propose_file",
            json!({"path":"docs/api.md","content":NEW_DOC}),
        )]));
        script.push(response(vec![]));
        let (output, _) = invoke(
            temp.path(),
            script,
            "да\n",
            "Приведи документацию API в соответствие с реализацией",
            false,
            None,
        )
        .await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
            NEW_DOC
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("src/api.rs")).unwrap(),
            SOURCE
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("src/client.rs")).unwrap(),
            CALLER
        );
    }
}

#[tokio::test]
async fn conflict_after_model_read_does_not_overwrite_user_edit_or_create_report() {
    let temp = project();
    let mut script = discovery();
    script.push(response(vec![
        (
            "propose_file",
            json!({"path":"docs/api.md","content":NEW_DOC}),
        ),
        (
            "propose_file",
            json!({"path":"docs/usages.md","content":REPORT}),
        ),
    ]));
    script.push(response(vec![]));
    let user_text = "# User edit\nMust survive confirmation.\n";
    let (output, _) = invoke(
        temp.path(),
        script,
        "yes\n",
        "Исправь документацию и подготовь отчёт",
        false,
        Some((temp.path().join("docs/api.md"), user_text.to_owned())),
    )
    .await;
    assert!(!output.status.success());
    assert_eq!(
        fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
        user_text
    );
    assert!(!temp.path().join("docs/usages.md").exists());
}

#[tokio::test]
async fn source_edits_are_allowed_after_read_and_confirmation() {
    let temp = project();
    let updated = SOURCE.replace("/imports", "/v2/imports");
    let mut script = discovery();
    script.push(response(vec![(
        "propose_file",
        json!({"path":"src/api.rs","content":updated}),
    )]));
    script.push(response(vec![]));
    let (output, _) = invoke(
        temp.path(),
        script,
        "yes\n",
        "Обнови endpoint API импорта",
        false,
        None,
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("src/api.rs")).unwrap(),
        updated
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
        OLD_DOC
    );
}

#[tokio::test]
async fn truncated_completion_discards_previously_staged_changes() {
    let temp = project();
    let mut script = report_script();
    *script.last_mut().unwrap() = json!({"choices":[{"message":{"content":"unfinished"},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"prompt_tokens_details":{"cached_tokens":0}}});
    let (output, _) = invoke(temp.path(), script, "yes\n", "Подготовь отчёт", false, None).await;
    assert!(!output.status.success());
    assert!(!temp.path().join("docs/usages.md").exists());
}

#[tokio::test]
async fn repl_task_confirmation_keeps_following_command_available() {
    let temp = project();
    let (output, _) = invoke(
        temp.path(),
        report_script(),
        "/task Найди использования API и подготовь отчёт\nyes\n/models\n/exit\n",
        "",
        true,
        None,
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("docs/usages.md")).unwrap(),
        REPORT
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("local/test"));
    assert!(!stdout.contains("synthetic-not-a-secret"));
}

#[tokio::test]
async fn untrusted_diff_controls_are_visible_before_confirmation() {
    let temp = project();
    let content = "visible\rhidden\u{202e}reordered\n";
    let mut script = discovery();
    script.push(response(vec![(
        "propose_file",
        json!({"path":"docs/control.md","content":content}),
    )]));
    script.push(response(vec![]));
    let (output, _) = invoke(temp.path(), script, "no\n", "Подготовь отчёт", false, None).await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains('\r'));
    assert!(!stdout.contains('\u{202e}'));
    assert!(stdout.contains("\\u{d}"));
    assert!(stdout.contains("\\u{202e}"));
    assert!(!temp.path().join("docs/control.md").exists());
}

#[tokio::test]
async fn model_can_verify_staged_content_by_rereading_same_range() {
    let temp = project();
    let mut script = discovery();
    script.push(response(vec![(
        "propose_file",
        json!({"path":"docs/api.md","content":NEW_DOC}),
    )]));
    script.push(response(vec![(
        "read_file",
        json!({"path":"docs/api.md","start_line":1,"end_line":3}),
    )]));
    script.push(response(vec![]));
    let (output, requests) = invoke(
        temp.path(),
        script,
        "yes\n",
        "Обнови документацию API и проверь предложенный результат",
        false,
        None,
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
        NEW_DOC
    );
    let last_result = requests.last().unwrap()["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|message| message["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    let read: Value = serde_json::from_str(last_result).unwrap();
    assert_eq!(read["path"], "docs/api.md");
    assert_eq!(read["content"], NEW_DOC);
}

#[tokio::test]
async fn task_summary_renders_markdown_and_redirected_diff_has_no_ansi() {
    let temp = project();
    let mut script = discovery();
    script.push(response(vec![(
        "propose_file",
        json!({"path":"docs/api.md","content":NEW_DOC}),
    )]));
    script.push(response(vec![]));
    script.last_mut().unwrap()["choices"][0]["message"]["content"] = json!(
        "# Результат\n\n**Документация** проверена.\n\n- Endpoint: `/imports`\n\n```rust\nImportApi::ENDPOINT\n```"
    );
    let (output, _) = invoke(
        temp.path(),
        script,
        "no\n",
        "Обнови документацию API",
        false,
        None,
    )
    .await;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Документация"));
    assert!(stdout.contains("ImportApi::ENDPOINT"));
    assert!(!stdout.contains("# Результат"));
    assert!(!stdout.contains("**Документация**"));
    assert!(!stdout.contains("```rust"));
    assert!(!stdout.contains('\x1b'));
    assert!(stdout.contains("-Endpoint: /old-import\n+Endpoint: /imports"));
    assert_eq!(
        fs::read_to_string(temp.path().join("docs/api.md")).unwrap(),
        OLD_DOC
    );
}
